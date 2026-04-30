from albatross_engine.worker import StateCategory, min_swaps_to_target_fast
from albatross_engine.worker import Worker
import torch


def apply_swaps(values, swaps):
    values = list(values)
    for left, right in swaps:
        values[left], values[right] = values[right], values[left]
    return values


def test_min_swaps_groups_state_categories_and_returns_offsets():
    values = [
        StateCategory.EMPTY,
        StateCategory.FORWARD_SEQ,
        StateCategory.FORWARD_ONE_DECODE,
        StateCategory.EMPTY,
        StateCategory.FORWARD_ONE_PREFILL,
        StateCategory.FORWARD_ONE_DECODE,
    ]
    elements = [category for category in sorted(StateCategory)]

    swaps, offsets = min_swaps_to_target_fast(list(values), elements)
    grouped = apply_swaps(values, swaps)

    assert grouped == [
        StateCategory.FORWARD_ONE_DECODE,
        StateCategory.FORWARD_ONE_DECODE,
        StateCategory.FORWARD_ONE_PREFILL,
        StateCategory.FORWARD_SEQ,
        StateCategory.EMPTY,
        StateCategory.EMPTY,
    ]
    assert offsets == [(0, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 6)]


def test_min_swaps_returns_no_swaps_for_already_grouped_categories():
    values = [
        StateCategory.FORWARD_ONE_DECODE,
        StateCategory.FORWARD_ONE_PREFILL,
        StateCategory.FORWARD_SEQ,
        StateCategory.EMPTY,
    ]
    elements = [category for category in sorted(StateCategory)]

    swaps, offsets = min_swaps_to_target_fast(list(values), elements)

    assert swaps == []
    assert offsets == [(0, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4)]


def test_update_penalty_from_tokens_uses_device_tokens_and_slot_indices():
    worker = Worker.__new__(Worker)
    worker.occurrence = torch.zeros(4, 16)
    worker.alpha_presence_vector = torch.zeros(4, 16)
    worker.presence_penalty_tensor = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    worker.slot_indices = torch.arange(4, dtype=torch.long)
    worker.no_penalty_token_mask = torch.zeros(16, dtype=torch.bool)
    worker.no_penalty_token_mask[10] = True

    worker._update_penalty_from_tokens((1, 4), torch.tensor([10, 11, 12]))

    assert worker.occurrence[1, 10].item() == 0.0
    assert worker.occurrence[2, 11].item() == 1.0
    assert worker.occurrence[3, 12].item() == 1.0
    assert torch.allclose(worker.alpha_presence_vector[1, 10], torch.tensor(0.2))
    assert torch.allclose(worker.alpha_presence_vector[2, 11], torch.tensor(0.3))
    assert torch.allclose(worker.alpha_presence_vector[3, 12], torch.tensor(0.4))
