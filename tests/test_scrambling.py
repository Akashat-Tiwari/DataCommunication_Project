
import pytest

from src.scrambling import scramble_b8zs, scramble_hdb3


def test_b8zs_substitutes_eight_zeros():
    bits = [1] + [0] * 8 + [1]

    assert scramble_b8zs(bits) == [
        1, 0, 0, 0, 1, -1, 0, -1, 1, -1
    ]


def test_b8zs_keeps_ordinary_ami_when_no_eight_zero_run():
    assert scramble_b8zs([1, 1, 0, 1]) == [1, -1, 0, 1]


def test_b8zs_handles_two_eight_zero_runs():
    bits = [1] + [0] * 16
    result = scramble_b8zs(bits)

    assert len(result) == len(bits)
    assert result == [
        1,
        0, 0, 0, 1, -1, 0, -1, 1,
        0, 0, 0, 1, -1, 0, -1, 1,
    ]


def test_hdb3_uses_000v_for_even_pulse_count():
    assert scramble_hdb3([1, 1, 0, 0, 0, 0]) == [
        1, -1, 0, 0, 0, -1
    ]


def test_hdb3_uses_b00v_for_odd_pulse_count():
    assert scramble_hdb3([1, 0, 0, 0, 0]) == [
        1, -1, 0, 0, -1
    ]


def test_hdb3_handles_zero_run_at_start():
    assert scramble_hdb3([0, 0, 0, 0]) == [0, 0, 0, -1]


def test_hdb3_handles_eight_consecutive_zeros():
    bits = [1] + [0] * 8
    result = scramble_hdb3(bits)

    assert len(result) == len(bits)
    assert result == [1, -1, 0, 0, -1, 0, 0, 0, -1]


@pytest.mark.parametrize(
    "encoder",
    [scramble_b8zs, scramble_hdb3],
)
def test_scramblers_preserve_length(encoder):
    bits = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    assert len(encoder(bits)) == len(bits)


@pytest.mark.parametrize(
    "encoder",
    [scramble_b8zs, scramble_hdb3],
)
def test_scramblers_reject_invalid_bits(encoder):
    with pytest.raises(ValueError):
        encoder([0, 1, 2])


@pytest.mark.parametrize(
    "encoder",
    [scramble_b8zs, scramble_hdb3],
)
def test_scramblers_accept_empty_input(encoder):
    assert encoder([]) == []
