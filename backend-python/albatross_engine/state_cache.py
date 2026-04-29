from typing import Dict, List, Optional, Union

from collections import OrderedDict

import asyncio

import torch


class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.od = OrderedDict()

    def put(self, key, value):
        """
        登记 key。返回被踢掉的 key 和 value；如果没踢掉返回 None。
        """
        if key in self.od:
            self.od.move_to_end(key)
            return None

        self.od[key] = value
        if len(self.od) > self.capacity:
            return self.od.popitem(last=False)
        return None

    def get(self, key):
        if key in self.od:
            self.od.move_to_end(key)
            return self.od[key]
        return None

    def __contains__(self, key):
        return key in self.od

    def __len__(self):
        return len(self.od)


class TrieNode:
    def __init__(self):
        self.children: Dict[int, TrieNode] = {}
        self.state: bool = False
        self.depend_count: int = 0
        self.prefill_condition: Optional[asyncio.Condition] = None


class SimpleStateCache:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.root = TrieNode()
        self.LRU_cache = LRUCache(max_size)
        self.prefill_lock = asyncio.Lock()

    def check(self, tokens: list[int], return_trie_node: bool = False) -> Union[
        tuple[List[int], Union[List[torch.Tensor] | None]],
        tuple[List[int], Union[List[torch.Tensor] | None], int, TrieNode],
    ]:
        node = self.root

        token_index = 0

        tmp_index = 0
        while tmp_index < len(tokens):
            token = tokens[tmp_index]

            if node.state:
                token_index = tmp_index

            next_node = node.children.get(token)
            if next_node is None or next_node.depend_count == 0:
                break
            node = next_node
            tmp_index += 1

        hashed_tokens = tuple(tokens[:token_index])
        state = self.LRU_cache.get(hashed_tokens)
        if return_trie_node:
            return tokens[token_index:], state, token_index, node
        return tokens[token_index:], state, token_index

    async def check_and_wait_prefill(
        self, tokens: list[int], cache_prefill_padding: int
    ) -> tuple[List[int], Union[List[torch.Tensor] | None], int]:
        async with self.prefill_lock:
            # print("enter")
            real_prefill_tokens, state, cached_token_len, node = self.check(tokens, return_trie_node=True)

            if cached_token_len + cache_prefill_padding == len(tokens):
                # print("leave all hit")
                return real_prefill_tokens, state, cached_token_len

            need_prefill_tokens = tokens[cached_token_len:-cache_prefill_padding]
            # print(need_prefill_tokens)

            for token in need_prefill_tokens:
                if token not in node.children:
                    # print("new node")
                    node.children[token] = TrieNode()
                node = node.children[token]

            if node.prefill_condition is None:
                node.prefill_condition = asyncio.Condition()
                # print("leave prefill")
                return real_prefill_tokens, state, cached_token_len
            # print("leave")
            
            
        # print("挂起等待")
        async with node.prefill_condition:
            await node.prefill_condition.wait()
        # print("放行")
        if node.state:
            return (
                tokens[-cache_prefill_padding:],
                self.LRU_cache.get(tuple(tokens[:-cache_prefill_padding])),
                len(tokens) - cache_prefill_padding,
            )
        else:
            print("prefill failed")
            return real_prefill_tokens, state, cached_token_len

    async def awake_hang_up_prefills(
        self,
        node: TrieNode,
    ) -> bool:
        if node.prefill_condition:
            async with node.prefill_condition:
                node.prefill_condition.notify_all()
            node.prefill_condition = None
            return True
        else:
            return False

    def cache(
        self,
        tokens: tuple[int, ...],
        state: object,
        return_trie_node: bool = False,
    ):
        if not tokens:
            return

        node = self.root

        for token in tokens:
            node.depend_count += 1
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]

        node.depend_count += 1
        node.state = True

        if key := self.LRU_cache.put(tokens, state):
            del_node = self.root
            tmp_index = 0
            while tmp_index < len(key[0]):
                token = key[0][tmp_index]
                del_node.depend_count -= 1

                child_node = del_node.children.get(token)
                assert child_node is not None

                if child_node.depend_count == 1:
                    del del_node.children[token]
                    break
                else:
                    del_node = child_node
                tmp_index += 1

            if tmp_index == len(key[0]):
                del_node.state = False
                del_node.depend_count -= 1

            if isinstance(key[1], list):
                for _ in range(len(key[1])):
                    del key[1][0]
                    # print("remove")
            del key

        if return_trie_node:
            return node

    def remove(self, tokens: list[int]):
        hashed_tokens = tuple(tokens)
        if key := self.LRU_cache.get(hashed_tokens):
            node = self.root

            tmp_index = 0
            while tmp_index < len(tokens):
                token = tokens[tmp_index]
                node.depend_count -= 1

                child_node = node.children.get(token)
                assert child_node is not None

                if child_node.depend_count == 1:
                    del node.children[token]
                    break
                else:
                    node = child_node
                tmp_index += 1

            if tmp_index == len(tokens):
                node.state = False

            if isinstance(key[1], list):
                for _ in range(len(key[1])):
                    del key[1][0]
            del key
            del self.LRU_cache.od[hashed_tokens]


if __name__ == "__main__":
    # 请你写一个测试
    cache = SimpleStateCache(max_size=3)
    cache.cache([1, 2, 3, 4], "state1")
    cache.cache([1, 2, 3, 4, 5, 6, 7], "state1_2")
    cache.cache([1, 2, 3, 6, 5, 6, 7, 8], "state2")

    print(cache.check([1, 2, 3, 4]))  # ([1, 2, 3], None)
    print(cache.check([1, 2, 3, 4, 5]))  # ([5], 'state1')
    print(cache.check([1, 2, 3, 4, 5, 6, 7]))  # ([5, 6, 7], 'state1')
    print(cache.check([1, 2, 3, 4, 5, 6, 7, 8]))  # ([8], 'state1_2')
    print(cache.check([1, 2, 3, 6, 5]))  # ([1, 2, 3, 6, 5], None)
    print(cache.check([1, 2, 3, 6, 5, 6, 7, 8, 9]))  # ([9], 'state2')

    print(cache.LRU_cache.od)
    cache.cache([1, 2, 3, 4, 5], "state1_3")
    print(cache.LRU_cache.od)
    print(cache.check([1, 2, 3, 4, 5]))  # ([1, 2, 3, 4, 5], None)

    def print_tire_tree(
        node: TrieNode, depth=0, mark_depth: list[int] = [], value=None, is_last=True, path: list[int] = []
    ):
        marker = "".join(["│" if i in mark_depth else " " for i in range(depth)])
        print(f"{marker[:-1]}{('└' if is_last else '├')if depth!=0 else''}{value}", end="")

        if node.depend_count > 1:
            print(f"({node.depend_count})", end="")

        if path and node.state:
            print(f"{'' if node.depend_count > 1 else '-'*3}->{cache.LRU_cache.get(tuple(path))}")
        else:
            print()

        if len(node.children) > 1:
            mark_depth.append(depth)
        for k, (value, child) in enumerate(node.children.items()):
            if k == len(node.children) - 1 and len(node.children) > 1:
                mark_depth.pop()
            path.append(value)
            print_tire_tree(child, depth + 1, mark_depth, value, is_last=k == len(node.children) - 1, path=path)
            path.pop()

    print_tire_tree(cache.root, value="root")

    # exit()

    cache.LRU_cache.capacity = 100
    cache.cache([1, 2, 5, 4], "state_4")
    cache.cache([1, 2, 3, 6, 9, 4, 5], "state_5")
    print(cache.LRU_cache.od)
    print_tire_tree(cache.root, value="root")
    cache.remove([1, 2, 3, 4, 5])
    print(cache.LRU_cache.od)
    print_tire_tree(cache.root, value="root")
    cache.remove([1, 2, 3, 4, 5, 6, 7])
    print(cache.LRU_cache.od)
    print_tire_tree(cache.root, value="root")

    import torch
    import psutil

    cache = SimpleStateCache(max_size=3)
    # 记录当前内存使用情况
    mem_before = psutil.Process().memory_info().rss
    mem_delta = psutil.Process().memory_info().rss
    for i in range(100):
        cache.cache(
            f"state_{i}",
            [torch.rand(int(33.5 * 1024 * 1024) // 2, dtype=torch.float16)],
        )
        print(f"已使用内存: {(psutil.Process().memory_info().rss - mem_before) / 1024 / 1024:.2f} MB")
