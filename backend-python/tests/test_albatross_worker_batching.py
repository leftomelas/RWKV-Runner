from albatross_engine.worker import StateCategory, min_swaps_to_target_fast


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
