import random

import pytest

from src.data_generator import (
    generate_random_bits,
    inject_zero_run,
    find_longest_palindrome,
    generate_bits_with_zero_run
)


def test_generate_random_bits_returns_requested_binary_length():
    bits = generate_random_bits(32, random.Random(7))

    assert len(bits) == 32
    assert all(bit in (0, 1) for bit in bits)


def test_generate_random_bits_accepts_zero_length():
    assert generate_random_bits(0) == []


def test_inject_zero_run_adds_contiguous_zeros_without_mutating_input():
    bits = [1] * 8

    result = inject_zero_run(bits, 3, random.Random(4))

    assert len(result) == len(bits) + 3
    assert any(result[index:index + 3] == [0, 0, 0] for index in range(len(result) - 2))
    assert bits == [1] * 8


def test_inject_zero_run_of_zero_returns_copy():
    bits = [1, 0, 1]

    result = inject_zero_run(bits, 0)

    assert result == bits
    assert result is not bits


def test_negative_lengths_are_rejected():
    with pytest.raises(ValueError):
        generate_random_bits(-1)

    with pytest.raises(ValueError):
        inject_zero_run([1, 0], -1)

def test_find_longest_palindrome_odd_length():
    bits = [0, 1, 0, 1, 1]
    assert find_longest_palindrome(bits) == [0, 1, 0]


def test_find_longest_palindrome_even_length():
    bits = [1, 0, 1, 1, 0, 0]
    assert find_longest_palindrome(bits) == [0, 1, 1, 0]


def test_find_longest_palindrome_returns_first_longest_on_tie():
    bits = [0, 0, 1, 1]

    assert find_longest_palindrome(bits) == [0, 0]


def test_find_longest_palindrome_handles_empty_input():
    assert find_longest_palindrome([]) == []


def test_find_longest_palindrome_rejects_non_binary_values():
    with pytest.raises(ValueError):
        find_longest_palindrome([0, 1, 2])



@pytest.mark.parametrize("run_length", [4, 8])
def test_generate_bits_with_zero_run(run_length):
    bits = generate_bits_with_zero_run(
        32,
        run_length,
        random.Random(21),
    )

    assert len(bits) == 32
    assert all(bit in (0, 1) for bit in bits)
    assert any(
        bits[i:i + run_length] == [0] * run_length
        for i in range(len(bits) - run_length + 1)
    )


def test_generate_bits_with_zero_run_rejects_unsupported_run_length():
    with pytest.raises(ValueError):
        generate_bits_with_zero_run(32, 5)


def test_generate_bits_with_zero_run_rejects_insufficient_length():
    with pytest.raises(ValueError):
        generate_bits_with_zero_run(4, 8)