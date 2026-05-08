import asyncio
import queue
import gc
import os
import types
import time
import threading
from typing import List, Dict, Optional, Any, Tuple
import torch
from collections import deque

from albatross_engine.task import Task, ModelLoadConfig, RequestStatus, FinishReason
from albatross_engine.sampling import sample_next_tokens_batch
from albatross_engine.profiling import ProfileAccumulator
from albatross_engine.throughput import ThroughputReporter, get_log_interval_from_env
# from albatross_engine.rapid_sampling_wrapper import load_rapid_sampling

# 定义TaskData的类型结构
from typing_extensions import TypedDict

from albatross.rwkv7 import RWKV_x070 as RWKV_x070_ORIGINAL
from albatross.utils import TRIE_TOKENIZER

from collections import defaultdict

from enum import IntEnum, auto


# For 3.14
def ensure_instance_annotations(cls):
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        if not hasattr(self, "__annotations__"):
            self.__annotations__ = {}
        if init is not object.__init__:
            init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


RWKV_x070 = ensure_instance_annotations(RWKV_x070_ORIGINAL)


def min_swaps_to_target_fast(lst, elements: list[int]):
    swaps: List[Tuple[int, int]] = []

    # 构建每个字符在 target 中的位置队列
    positions = defaultdict(list)
    for idx, val in enumerate(lst):
        positions[val].append(idx)

    offsets: List[Tuple[int, int]] = []
    offset = 0

    for target in elements:
        if target not in positions:
            offsets.append((offset, offset))
            continue

        pos = positions[target]
        target_count = len(pos)

        offsets.append((offset, offset + target_count))

        target_should_move_back_id = [i for i in pos if i >= target_count + offset]
        pos_set = set(pos)
        target_avaliable_id = [i for i in range(offset, target_count + offset) if i not in pos_set]

        for k, v in enumerate(target_should_move_back_id):
            swap = (target_avaliable_id[k], v)
            swaps.append((target_avaliable_id[k], v))
            lst[swap[0]], lst[swap[1]] = lst[swap[1]], lst[swap[0]]

        offset += target_count
        positions = defaultdict(list)
        for idx, val in enumerate(lst[offset:]):
            positions[val].append(idx + offset)

    return swaps, offsets


# Global model initialization lock. Avoid initializing TorchScript/model state concurrently.
_MODEL_INIT_LOCK = threading.Lock()
_MODEL_INIT_DONE = False


class StateCategory(IntEnum):
    FORWARD_ONE_DECODE = auto()
    FORWARD_ONE_PREFILL = auto()
    FORWARD_ONE_SUSPENDED = auto()
    FORWARD_SEQ = auto()
    FINISHED = auto()
    EMPTY = auto()


class TaskData(TypedDict):
    task: Optional[Task]
    # state_pos: int
    new_token: Optional[int]
    next_input_token: Optional[int]
    is_prefilling: Optional[bool]
    state_category: StateCategory
    prefilled_tokens: List[int]
    prefill_cached: bool


class Worker:
    """
    Worker 类用于处理 Task，实现 continuous batching。

    Worker 在独立线程中运行，维护一个任务池并执行模型推理。
    """

    def __init__(
        self,
        worker_id: str,
        gpu_id: List[int],
        model_config: ModelLoadConfig,
        task_queue: queue.Queue,
        master_event_queue: queue.Queue,
        worker_event_queue: queue.Queue,
        batch_size: int = 32,
    ):
        """
        初始化 Worker

        Args:
            gpu_id: 分配给 Worker 的 GPU ID 列表
            model_config: 模型加载配置
            task_queue: 任务队列，Worker 消费该队列
            master_event_queue: 事件队列，包含调度要求
            batch_size: 批处理大小
        """
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.model_config = model_config
        self.task_queue = task_queue
        self.master_event_queue = master_event_queue
        self.worker_event_queue = worker_event_queue

        self.real_state_size = batch_size
        self.max_batch_size = batch_size - 1
        self.max_prefill_count = max(int(batch_size * 0.125), 1)

        # Worker 内部数据
        # self.task_pool: List[TaskData] = []
        self.state_slot: dict[int, TaskData] = {
            i: {
                "task": None,
                "is_prefilling": None,
                "new_token": None,
                "next_input_token": None,
                "state_category": StateCategory.EMPTY,
            }
            for i in range(self.max_batch_size)
        }

        self.model: RWKV_x070_ORIGINAL = None

        self.batch_state: list[torch.Tensor] = None
        self.occurrence: torch.Tensor = None
        self.alpha_presence_vector: torch.Tensor = None

        # 解码参数预处理 tensor
        self.temperature_tensor: torch.Tensor = None
        self.top_p_tensor: torch.Tensor = None
        self.top_k_tensor: torch.Tensor = None
        self.frequency_penalty_tensor: torch.Tensor = None
        self.penalty_decay_tensor: torch.Tensor = None
        self.presence_penalty_tensor: torch.Tensor = None
        self.slot_indices: torch.Tensor = None
        self.no_penalty_token_mask: torch.Tensor = None

        self.no_penalty_token_ids = {33, 10, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58}

        # seq foward
        self.min_forward_seq_len = 10
        self.max_forward_seq_len_per_forward = 100
        self.seq_forward_count_down = 0
        self.decode_prefill_ratio: int = 5

        self.shutdown_flag = False
        self.tokenizer: TRIE_TOKENIZER = None

        self.loop_time_recorder = deque(maxlen=10)
        self.profile = ProfileAccumulator(enabled=os.environ.get("ALBATROSS_PROFILE") == "1")
        self.throughput_reporter = ThroughputReporter(
            worker_id=self.worker_id,
            interval_seconds=get_log_interval_from_env(),
        )

    def _send_worker_loaded_message(self):
        """发送 Worker 加载成功信息"""
        try:
            message = (
                self.worker_id,
                "worker_loaded",
                {
                    "status": "success",
                    "worker_id": self.worker_id,
                    "gpu_id": self.gpu_id,
                    "model_path": self.model_config.model_path,
                },
            )
            self.worker_event_queue.put_nowait(message)
            print(f"[{self.worker_id}] Worker loaded event sent")
        except Exception as e:
            print(f"[{self.worker_id}] Failed to send worker loaded event: {e}")

    def _load_model(self):
        """加载模型"""
        global _MODEL_INIT_LOCK, _MODEL_INIT_DONE

        # Ensure only one worker performs the first model initialization path at a time.
        with _MODEL_INIT_LOCK:
            # After the first initialization path, create a new model instance directly.
            if _MODEL_INIT_DONE:
                args = types.SimpleNamespace()
                args.vocab_size = self.model_config.vocab_size
                args.head_size = self.model_config.head_size
                if self.model_config.model_path.endswith(".pth"):
                    args.MODEL_NAME = self.model_config.model_path[:-4]
                else:
                    args.MODEL_NAME = self.model_config.model_path

                self.model = RWKV_x070(args)
                self.tokenizer = TRIE_TOKENIZER(self.model_config.vocab_path)

                # 发送成功加载信息
                self._send_worker_loaded_message()
                return

            # First load initializes model weights, TorchScript wrappers, tokenizer, and CUDA context.
            print(f"[{self.worker_id}] Loading and initializing model...")
            try:
                args = types.SimpleNamespace()
                args.vocab_size = self.model_config.vocab_size
                args.head_size = self.model_config.head_size
                if self.model_config.model_path.endswith(".pth"):
                    args.MODEL_NAME = self.model_config.model_path[:-4]
                else:
                    args.MODEL_NAME = self.model_config.model_path

                self.model = RWKV_x070(args)
                self.tokenizer = TRIE_TOKENIZER(self.model_config.vocab_path)

                # Mark the one-time model initialization path as complete.
                _MODEL_INIT_DONE = True
                print(f"[{self.worker_id}] Model initialization complete")

                # 发送成功加载信息
                self._send_worker_loaded_message()

            except Exception as e:
                print(f"[{self.worker_id}] Model initialization failed: {e}")
                raise

    def _init_worker(self):

        # 设置 GPU
        if self.gpu_id:
            torch.cuda.set_device(self.gpu_id[0])
        self._load_model()

        # 预分配
        self.batch_state = self.model.generate_zero_state(self.real_state_size)

        self.occurrence = torch.zeros(
            (self.real_state_size, self.model_config.vocab_size),
            dtype=torch.float32,
            device=self.batch_state[0].device,
        )
        self.alpha_presence_vector = torch.zeros(
            (self.real_state_size, self.model_config.vocab_size),
            dtype=torch.float32,
            device=self.batch_state[0].device,
        )
        self.temperature_tensor = torch.zeros(
            (self.real_state_size, 1),
            dtype=torch.float16,
            device=self.batch_state[0].device,
        )
        self.top_p_tensor = torch.zeros(
            (self.real_state_size, 1),
            dtype=torch.float16,
            device=self.batch_state[0].device,
        )
        self.top_k_tensor = torch.zeros(
            (self.real_state_size, 1),
            dtype=torch.int32,
            device=self.batch_state[0].device,
        )
        self.frequency_penalty_tensor = torch.zeros(
            (self.real_state_size, 1),
            dtype=torch.float16,
            device=self.batch_state[0].device,
        )
        self.penalty_decay_tensor = torch.zeros(
            (self.real_state_size, 1),
            dtype=torch.float16,
            device=self.batch_state[0].device,
        )
        self.presence_penalty_tensor = torch.zeros(
            (self.real_state_size, 1),
            dtype=torch.float32,
            device=self.batch_state[0].device,
        )
        self.slot_indices = torch.arange(
            self.real_state_size,
            dtype=torch.long,
            device=self.batch_state[0].device,
        )
        self.no_penalty_token_mask = torch.zeros(
            self.model_config.vocab_size,
            dtype=torch.bool,
            device=self.batch_state[0].device,
        )
        self.no_penalty_token_mask[list(self.no_penalty_token_ids)] = True


    def _switch_batch(self, pos_a: int, pos_b: int):
        if pos_a == pos_b:
            return

        with self.profile.time("state_swap"):
            assert (
                pos_a < self.max_batch_size and pos_b < self.max_batch_size
            ), f"pos_a {pos_a}, pos_b {pos_b}, max_batch_size {self.max_batch_size}, real_state_size {self.real_state_size}; pos_a and pos_b shall be less than max_batch_size."

            # switch state

            # cache pos_a
            self.batch_state[0][:, :, [self.real_state_size - 1], :] = self.batch_state[0][:, :, [pos_a], :]
            self.batch_state[1][:, [self.real_state_size - 1], :, :] = self.batch_state[1][:, [pos_a], :, :]
            self.batch_state[2][[self.real_state_size - 1]] = self.batch_state[2][[pos_a]]

            # pos_b -> pos_a
            self.batch_state[0][:, :, [pos_a], :] = self.batch_state[0][:, :, [pos_b], :]
            self.batch_state[1][:, [pos_a], :, :] = self.batch_state[1][:, [pos_b], :, :]
            self.batch_state[2][[pos_a]] = self.batch_state[2][[pos_b]]

            # cached pos_a -> pos_b
            self.batch_state[0][:, :, [pos_b], :] = self.batch_state[0][:, :, [self.real_state_size - 1], :]
            self.batch_state[1][:, [pos_b], :, :] = self.batch_state[1][:, [self.real_state_size - 1], :, :]
            self.batch_state[2][[pos_b]] = self.batch_state[2][[self.real_state_size - 1]]

            self.occurrence[[self.real_state_size - 1], :] = self.occurrence[[pos_a], :]
            self.occurrence[[pos_a], :] = self.occurrence[[pos_b], :]
            self.occurrence[[pos_b], :] = self.occurrence[[self.real_state_size - 1], :]

            self.alpha_presence_vector[[self.real_state_size - 1], :] = self.alpha_presence_vector[[pos_a], :]
            self.alpha_presence_vector[[pos_a], :] = self.alpha_presence_vector[[pos_b], :]
            self.alpha_presence_vector[[pos_b], :] = self.alpha_presence_vector[[self.real_state_size - 1], :]

            self.frequency_penalty_tensor[[self.real_state_size - 1], :] = self.frequency_penalty_tensor[[pos_a], :]
            self.frequency_penalty_tensor[[pos_a], :] = self.frequency_penalty_tensor[[pos_b], :]
            self.frequency_penalty_tensor[[pos_b], :] = self.frequency_penalty_tensor[[self.real_state_size - 1], :]

            self.penalty_decay_tensor[[self.real_state_size - 1], :] = self.penalty_decay_tensor[[pos_a], :]
            self.penalty_decay_tensor[[pos_a], :] = self.penalty_decay_tensor[[pos_b], :]
            self.penalty_decay_tensor[[pos_b], :] = self.penalty_decay_tensor[[self.real_state_size - 1], :]

            # sample params
            self.temperature_tensor[[self.real_state_size - 1], :] = self.temperature_tensor[[pos_a], :]
            self.temperature_tensor[[pos_a], :] = self.temperature_tensor[[pos_b], :]
            self.temperature_tensor[[pos_b], :] = self.temperature_tensor[[self.real_state_size - 1], :]

            self.top_p_tensor[[self.real_state_size - 1], :] = self.top_p_tensor[[pos_a], :]
            self.top_p_tensor[[pos_a], :] = self.top_p_tensor[[pos_b], :]
            self.top_p_tensor[[pos_b], :] = self.top_p_tensor[[self.real_state_size - 1], :]

            self.top_k_tensor[[self.real_state_size - 1], :] = self.top_k_tensor[[pos_a], :]
            self.top_k_tensor[[pos_a], :] = self.top_k_tensor[[pos_b], :]
            self.top_k_tensor[[pos_b], :] = self.top_k_tensor[[self.real_state_size - 1], :]

            self.presence_penalty_tensor[[self.real_state_size - 1], :] = self.presence_penalty_tensor[[pos_a], :]
            self.presence_penalty_tensor[[pos_a], :] = self.presence_penalty_tensor[[pos_b], :]
            self.presence_penalty_tensor[[pos_b], :] = self.presence_penalty_tensor[[self.real_state_size - 1], :]

    def _organize_batch(self):
        """返回 ([start_pos, end_pos),)
        
        按 StateCategory 排序后的偏移量列表：
        - 0: forward one decode（需要 penalty 和采样）
        - 1: forward one prefill（只需要 forward）
        - 2: forward one suspended
        - 3: seq prefill
        - 4: finished
        - 5: empty"""
        current_task_list = [
            self.state_slot[slot_pos]["state_category"]
            for slot_pos in range(self.max_batch_size)
        ]

        swarps, offsets = min_swaps_to_target_fast(current_task_list, [i for i in sorted(StateCategory)])

        for pos_a, pos_b in swarps:
            self._switch_batch(pos_a, pos_b)
            self.state_slot[pos_a], self.state_slot[pos_b] = self.state_slot[pos_b], self.state_slot[pos_a]

        return offsets

    def _process_events(self) -> bool:
        """
        处理事件队列中的所有事件

        Returns:
            是否需要关闭 Worker
        """
        # 批量拉取所有可用事件
        while True:
            try:
                event = self.master_event_queue.get_nowait()

                if event.get("type") == "shutdown":
                    self.shutdown_flag = True
                    return True
                # 其他事件类型可以在这里处理

            except queue.Empty:
                break

        return False

    def _handle_forward_seq(self, task_data: TaskData, slot_pos):
        assert task_data["is_prefilling"] == True
        assert task_data["next_input_token"] != None, "next_input_token shall not be None."

        if task_data["task"].cache_prefill and len(task_data["task"].prefill_tokens) == (
            max(task_data["task"].cache_prefill_padding - 1, 0)
        ):
            # print(
            #     "cache_prefill fwd seq",
            #     task_data["prefilled_tokens"],
            #     self.tokenizer.decode(task_data["prefilled_tokens"], utf8_errors="ignore"),
            # )
            task_data["state_category"] = StateCategory.FORWARD_ONE_PREFILL

            if task_data["task"].cache_prefill:
                task_data["task"].output_queue.put_nowait(
                    (
                        "cache_prefill",
                        {
                            "state": [
                                self.batch_state[0][:, :, [slot_pos], :].to(device="cpu", non_blocking=True),
                                self.batch_state[1][:, [slot_pos], :, :].to(device="cpu", non_blocking=True),
                                self.batch_state[2][[slot_pos]].to(device="cpu", non_blocking=True),
                            ],
                            "prefilled_tokens": tuple(task_data["prefilled_tokens"]),
                        },
                    )
                )
                task_data["prefill_cached"] = True

        if len(task_data["task"].prefill_tokens) == 0:
            task_data["state_category"] = StateCategory.FORWARD_ONE_DECODE
            task_data["is_prefilling"] = False

        elif len(task_data["task"].prefill_tokens) < self.min_forward_seq_len:
            task_data["state_category"] = StateCategory.FORWARD_ONE_PREFILL
        else:
            # task_data["state_category"] = StateCategory.FORWARD_SEQ
            pass

    def _handle_forward_one_prefill_phase(self, task_data: TaskData, slot_pos: int):
        """处理 Prefill 阶段"""
        task = task_data["task"]

        task_data["prefilled_tokens"].append(task_data["next_input_token"])
        task_data["next_input_token"] = task.prefill_tokens.pop(0)
        if len(task.prefill_tokens) == 0:
            task_data["is_prefilling"] = False
            task_data["state_category"] = StateCategory.FORWARD_ONE_DECODE

        if (
            task_data["task"].cache_prefill
            and len(task_data["task"].prefill_tokens) == (max(task_data["task"].cache_prefill_padding - 1, 0))
            and not task_data["prefill_cached"]
        ):
            # print("cache_prefill fwd one", task_data["task"].prefill_tokens)
            task.output_queue.put_nowait(
                (
                    "cache_prefill",
                    {
                        "state": [
                            self.batch_state[0][:, :, [slot_pos], :].to(device="cpu", non_blocking=True),
                            self.batch_state[1][:, [slot_pos], :, :].to(device="cpu", non_blocking=True),
                            self.batch_state[2][[slot_pos]].to(device="cpu", non_blocking=True),
                        ],
                        "prefilled_tokens": tuple(task_data["prefilled_tokens"]),
                    },
                )
            )
            task_data["prefill_cached"] = True

    def _handle_forward_one_decode_phase(self, task_data: TaskData, slot_pos: int) -> None:
        """处理 Decode 阶段
        """
        task = task_data["task"]
        new_token = task_data["new_token"]

        if new_token in task.stop_tokens:
            task.request_status = RequestStatus.FINISHED_STOPPED
            return

        with self.profile.time("decode_tokenizer_decode"):
            new_text = self.tokenizer.decode([new_token], utf8_errors="ignore")  # TODO: 处理不完整的 utf8

        task.generated_tokens.append(new_token)
        task.decoded_texts.append(new_text)

        with self.profile.time("decode_output_enqueue"):
            task.output_queue.put_nowait(("token_generated", (new_token, new_text)))

        if len(task.generated_tokens) >= task.max_tokens:
            task.request_status = RequestStatus.FINISHED_LENGTH_CAPPED
            return
        
        task_data["next_input_token"] = new_token
        return

    def _is_task_aborted(self, task_data: TaskData):
        """检查任务是否打断"""
        task = task_data["task"]

        # 检查任务事件队列中是否有 abort 事件
        try:
            # while not task.task_event_queue.empty():
            event_type, payload = task.task_event_queue.get_nowait()
            if event_type == "abort":
                return True
        except queue.Empty:
            pass

        return False

    def _update_penalty_from_tokens(
        self,
        decode_offset: Tuple[int, int],
        new_tokens: torch.Tensor,
    ) -> None:
        """Update penalty tensors directly from sampled device tokens."""
        if new_tokens.numel() == 0:
            return

        decode_slice = slice(decode_offset[0], decode_offset[1])
        slots = self.slot_indices[decode_slice]
        tokens = new_tokens.to(device=self.occurrence.device, dtype=torch.long)
        weights = (~self.no_penalty_token_mask[tokens]).to(dtype=self.occurrence.dtype)

        self.occurrence[slots, tokens] += weights
        presence_values = self.presence_penalty_tensor[slots, 0]
        self.alpha_presence_vector[slots, tokens] = presence_values

    def _store_new_tokens(
        self,
        decode_offset: Tuple[int, int],
        new_tokens: List[int],
    ) -> None:
        for slot_pos, new_token in zip(range(*decode_offset), new_tokens):
            self.state_slot[slot_pos]["new_token"] = new_token

    def _process_accomplished_tasks(self, accomplished_task_slot_pos: List[int]):
        """处理已完成的任务"""

        if not accomplished_task_slot_pos:
            return

        for slot in accomplished_task_slot_pos:
            self.state_slot[slot]["task"].output_queue.put_nowait(
                ("task_completed", self.state_slot[slot]["task"])
            )

            self.state_slot[slot] = {
                "task": None,
                "is_prefilling": None,
                "new_token": None,
                "next_input_token": None,
                "state_category": StateCategory.EMPTY,
                "prefilled_tokens": [],
            }

    def _fill_task_pool(self):
        """填充任务池直到达到 batch_size"""
        prefill_count = 0
        for slot_pos in range(self.max_batch_size):
            if prefill_count >= self.max_prefill_count:
                break

            if self.state_slot[slot_pos]["state_category"] != StateCategory.EMPTY:
                if self.state_slot[slot_pos]["state_category"] == StateCategory.FORWARD_SEQ:
                    prefill_count += 1
                continue
            try:
                prefill_count += 1
                task: Task = self.task_queue.get_nowait()

                # 处理任务状态
                if task.state is None:
                    # 初始化空状态 - 创建与当前 batch_size 兼容的零状态
                    new_state = self.model.generate_zero_state(1)
                else:
                    # 将状态移动到 GPU
                    new_state = [state.cuda() for state in task.state]

                device = torch.device("cuda")

                self.batch_state[0][:, :, [slot_pos], :] = new_state[0]
                self.batch_state[1][:, [slot_pos], :, :] = new_state[1]
                self.batch_state[2][[slot_pos]] = new_state[2]

                self.occurrence[[slot_pos], :] = torch.zeros(
                    (1, self.model_config.vocab_size),
                    dtype=torch.float32,
                    device=self.batch_state[0].device,
                )

                self.alpha_presence_vector[[slot_pos], :] = torch.zeros(
                    (1, self.model_config.vocab_size),
                    dtype=torch.float32,
                    device=self.batch_state[0].device,
                )

                self.temperature_tensor[[slot_pos], :] = torch.tensor(
                    [[task.temperature if task.temperature > 0 else 1.0]],
                    dtype=torch.float16,
                    device=device,
                )
                self.top_p_tensor[[slot_pos], :] = torch.tensor(
                    [[task.top_p]],
                    dtype=torch.float16,
                    device=device,
                )
                self.top_k_tensor[[slot_pos], :] = torch.tensor(
                    [[task.top_k]],
                    dtype=torch.int32,
                    device=device,
                )
                self.frequency_penalty_tensor[[slot_pos], :] = torch.tensor(
                    [[task.frequency_penalty]],
                    dtype=torch.float16,
                    device=device,
                )
                self.penalty_decay_tensor[[slot_pos], :] = torch.tensor(
                    [[task.penalty_decay]],
                    dtype=torch.float16,
                    device=device,
                )
                self.presence_penalty_tensor[[slot_pos], :] = torch.tensor(
                    [[task.presence_penalty]],
                    dtype=torch.float32,
                    device=device,
                )

                # 添加到 task_pool

                next_input_token = task.prefill_tokens.pop(0)

                if len(task.prefill_tokens) == 0:
                    state_category = StateCategory.FORWARD_ONE_DECODE
                    is_prefilling = False
                elif len(task.prefill_tokens) - max((task.cache_prefill_padding - 1), 0) < self.min_forward_seq_len:
                    state_category = StateCategory.FORWARD_ONE_PREFILL
                    is_prefilling = True
                else:
                    state_category = StateCategory.FORWARD_SEQ
                    is_prefilling = True

                task_data: TaskData = {
                    "task": task,
                    "is_prefilling": is_prefilling,
                    "new_token": None,
                    "next_input_token": next_input_token,
                    "state_category": state_category,
                    "prefilled_tokens": [],
                    "prefill_cached": False,
                }
                self.state_slot[slot_pos] = task_data

            except queue.Empty:
                break

    def _run_forward_one(self, decode_offset: Tuple[int, int], one_prefill_offset: Tuple[int, int]):
        """运行模型前向推理，单 token
        
        Args:
            decode_offset: one decode 范围 [start, end)，需要 penalty 和采样
            one_prefill_offset: one prefill 范围 [start, end)，只需要 forward
        
        注意：decode_offset 和 one_prefill_offset 是连续的，即 decode_offset[1] == one_prefill_offset[0]
        """
        
        # 合并范围进行 forward
        combined_start = decode_offset[0]
        combined_end = one_prefill_offset[1]
        combined_count = combined_end - combined_start
        
        if combined_count == 0:
            return

        # 构建批处理输入
        next_tokens = [None] * combined_count

        for slot_pos in range(combined_start, combined_end):
            next_tokens[slot_pos - combined_start] = [self.state_slot[slot_pos]["next_input_token"]]

        combined_slice = slice(combined_start, combined_end)

        forward_state = [
            self.batch_state[0][:, :, combined_slice, :],
            self.batch_state[1][:, combined_slice, :, :],
            self.batch_state[2][combined_slice],
        ]

        # 模型前向传播（对所有 one forward 任务）
        out = self.model.forward_seq_batch(next_tokens, forward_state)

        # 以下只对 decode 范围进行处理
        decode_count = decode_offset[1] - decode_offset[0]
        
        if decode_count > 0:
            decode_slice = slice(decode_offset[0], decode_offset[1])
            decode_out = out[:decode_count]  # 取 decode 部分的输出

            # 处理禁止 token（只对 decode）
            for slot_pos in range(*decode_offset):
                for forbidden_token in self.state_slot[slot_pos]["task"].forbidden_tokens:
                    decode_out[slot_pos - decode_offset[0]][forbidden_token] -= 1e10

            # penalty 计算（只对 decode）
            self.occurrence[decode_slice, :] *= self.penalty_decay_tensor[decode_slice, :]
            decode_out -= (
                self.alpha_presence_vector[decode_slice, :]
                + self.occurrence[decode_slice, :] * self.frequency_penalty_tensor[decode_slice, :]
            )

            # 采样（只对 decode）
            with self.profile.time("sampling"):
                new_tokens = sample_next_tokens_batch(
                    logits=decode_out,
                    occurrence=self.occurrence[decode_slice, :],
                    temperature=self.temperature_tensor[decode_slice, :],
                    top_p=self.top_p_tensor[decode_slice, :],
                    top_k=self.top_k_tensor[decode_slice, :],
                    alpha_presence=self.alpha_presence_vector[decode_slice, :],
                    alpha_frequency=self.frequency_penalty_tensor[decode_slice, :],
                    penalty_decay=self.penalty_decay_tensor[decode_slice, :],
                )

            with self.profile.time("sampling_penalty_update"):
                self._update_penalty_from_tokens(decode_offset, new_tokens)

            with self.profile.time("sampling_token_transfer"):
                new_token_list = new_tokens.cpu().tolist()
            self._store_new_tokens(decode_offset, new_token_list)

        del out

    def _run_forward_seq(self, seq_perfill_offset: Tuple[int, int]):
        """运行模型前向推理，token 序列，适合 prefill 模式"""
        token_seq_len_list = [
            (
                len(self.state_slot[i]["task"].prefill_tokens)
                - max((self.state_slot[i]["task"].cache_prefill_padding - 1), 0)
            )
            for i in range(*seq_perfill_offset)
        ]
        token_seq_len = min(self.max_forward_seq_len_per_forward, *token_seq_len_list)
        assert token_seq_len > 0

        next_tokens: List[List[int]] = [None] * (seq_perfill_offset[1] - seq_perfill_offset[0])
        for slot_pos in range(*seq_perfill_offset):
            slot_next_tokens = [self.state_slot[slot_pos]["next_input_token"]] + self.state_slot[slot_pos][
                "task"
            ].prefill_tokens[: token_seq_len - 1]
            self.state_slot[slot_pos]["prefilled_tokens"].extend(slot_next_tokens)

            next_tokens[slot_pos - seq_perfill_offset[0]] = slot_next_tokens
            self.state_slot[slot_pos]["task"].prefill_tokens = self.state_slot[slot_pos]["task"].prefill_tokens[
                token_seq_len - 1 :
            ]
            self.state_slot[slot_pos]["next_input_token"] = self.state_slot[slot_pos]["task"].prefill_tokens.pop(0)

        seq_forward_state = [
            self.batch_state[0][:, :, seq_perfill_offset[0] : seq_perfill_offset[1], :],
            self.batch_state[1][:, seq_perfill_offset[0] : seq_perfill_offset[1], :, :],
            self.batch_state[2][seq_perfill_offset[0] : seq_perfill_offset[1]],
        ]
        # print("fs", next_tokens)
        out = self.model.forward_batch(next_tokens, seq_forward_state)
        del out

    def start(self):
        """
        启动 Worker 的主循环

        这是 Worker 的核心方法，实现了 continuous batching 的完整流程。
        """
        # 初始化模型和状态
        if self.model is None:
            self._init_worker()

        # 主循环
        while True:
            loop_start_time = time.perf_counter()

            with self.profile.time("process_events"):
                should_shutdown = self._process_events()

            if should_shutdown:
                break

            accomplished_task_slot_pos: list[int] = []
            with self.profile.time("state_slot_scan"):
                for key, task_data in sorted(self.state_slot.items()):

                    assert (
                        task_data["state_category"] != StateCategory.FINISHED
                    ), f"Invalid state category: {task_data['state_category'] }"

                    if task_data["state_category"] == StateCategory.EMPTY:
                        continue

                    with self.profile.time("state_slot_abort_check"):
                        is_aborted = self._is_task_aborted(task_data)

                    if is_aborted:
                        with self.profile.time("state_slot_mark_aborted"):
                            task_data["task"].request_status = RequestStatus.FINISHED_ABORTED
                            task_data["state_category"] = StateCategory.FINISHED

                    elif task_data["state_category"] == StateCategory.FORWARD_SEQ:
                        with self.profile.time("state_slot_handle_forward_seq"):
                            self._handle_forward_seq(task_data, key)

                    elif task_data["state_category"] == StateCategory.FORWARD_ONE_PREFILL:
                        with self.profile.time("state_slot_handle_one_prefill"):
                            self._handle_forward_one_prefill_phase(task_data, key)

                    elif task_data["state_category"] == StateCategory.FORWARD_ONE_DECODE:
                        with self.profile.time("state_slot_handle_one_decode"):
                            self._handle_forward_one_decode_phase(task_data, key)

                    with self.profile.time("state_slot_finished_check"):
                        if RequestStatus.is_finished(task_data["task"].request_status):
                            accomplished_task_slot_pos.append(key)

            with self.profile.time("process_accomplished"):
                self._process_accomplished_tasks(accomplished_task_slot_pos)

            with self.profile.time("fill_task_pool"):
                self._fill_task_pool()

            with self.profile.time("organize_batch"):
                decode_offset, one_prefill_offset, decode_suspended_offset, seq_perfill_offset, accomplished_offset, empty_offset = (
                    self._organize_batch()
                )

            decode_count = max(0, decode_offset[1] - decode_offset[0])
            self.profile.add("worker_loops", 1)
            self.profile.add("decode_tokens_scheduled", decode_count)
            self.profile.add("one_prefill_scheduled", max(0, one_prefill_offset[1] - one_prefill_offset[0]))
            self.profile.add("seq_prefill_scheduled", max(0, seq_perfill_offset[1] - seq_perfill_offset[0]))

            if decode_offset[1] - decode_offset[0] == 0 and one_prefill_offset[1] - one_prefill_offset[0] == 0 and seq_perfill_offset[1] - seq_perfill_offset[0] == 0:
                time.sleep(0.05)
                continue

            # 检查是否有 one forward 任务（decode 或 one prefill）
            one_forward_count = one_prefill_offset[1] - decode_offset[0]
            if one_forward_count > 0:
                with self.profile.time("forward_one"):
                    self._run_forward_one(decode_offset, one_prefill_offset)
                self.throughput_reporter.observe(
                    decode_tokens=decode_count,
                    active_batch=decode_count,
                )
                self.seq_forward_count_down -= 1
            else:
                self.seq_forward_count_down = 0

            if self.seq_forward_count_down < 1 and seq_perfill_offset[1] - seq_perfill_offset[0] > 0:
                with self.profile.time("forward_seq"):
                    self._run_forward_seq(seq_perfill_offset)
                self.seq_forward_count_down = max(1, self.decode_prefill_ratio)

            self.loop_time_recorder.append(time.perf_counter() - loop_start_time)

            if self.worker_event_queue:
                one_prefill_count = one_prefill_offset[1] - one_prefill_offset[0]
                info = (
                    self.worker_id,
                    "worker_performance",
                    {
                        "avg_loop_time": sum(self.loop_time_recorder) / len(self.loop_time_recorder),
                        "state_size": self.real_state_size,
                        "state_offset_details": {
                            "decode_offset": decode_offset,
                            "one_prefill_offset": one_prefill_offset,
                            "decode_suspended_offset": decode_suspended_offset,
                            "seq_perfill_offset": seq_perfill_offset,
                            "accomplished_offset": accomplished_offset,
                        },
                        "task_details": {
                            "decode_count": decode_count,
                            "one_prefill_count": one_prefill_count,
                            "seq_prefill_count": seq_perfill_offset[1] - seq_perfill_offset[0],
                        },
                        "max_allocated_memory_GB": torch.cuda.max_memory_allocated() / 1024**3,
                        "profile": self.profile.snapshot(reset=False),
                    },
                )
                self.worker_event_queue.put_nowait(info)

        self._cleanup()

    def _cleanup(self):
        """清理资源"""
        del self.state_slot
        del self.batch_state
        del self.occurrence
        del self.alpha_presence_vector
        del self.temperature_tensor
        del self.top_p_tensor
        del self.top_k_tensor
        del self.frequency_penalty_tensor
        del self.penalty_decay_tensor
        del self.presence_penalty_tensor
        del self.model
