"""Random bitstream generation helpers."""

from __future__ import annotations

import random
from typing import List, Optional


def generate_random_bits(length: int, rng: Optional[random.Random] = None) -> List[int]:
    """Return exactly ``length`` randomly generated binary values."""
    if length < 0:
        raise ValueError("length must be non-negative")

    generator = rng if rng is not None else random
    return [generator.randint(0, 1) for _ in range(length)]


def inject_zero_run(
    bits: List[int], run_length: int, rng: Optional[random.Random] = None
) -> List[int]:
    """Insert a contiguous zero run and return a new bit list."""
    if run_length < 0:
        raise ValueError("run_length must be non-negative")
    if run_length == 0:
        return bits[:]

    generator = rng if rng is not None else random
    insertion_index = generator.randrange(len(bits) + 1)
    return (
        bits[:insertion_index]
        + [0] * run_length
        + bits[insertion_index:]
    )


def find_longest_palindrome(bits: List[int]) -> List[int]:
    """
    Return the longest contiguous palindrome in a binary sequence.

    Uses Manacher's algorithm with O(n) time and O(n) extra space.
    If multiple longest palindromes exist, return the first one.
    """
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")

    n = len(bits)

    if n == 0:
        return []

    best_start = 0
    best_length = 1

    # Odd-length palindromes: 101, 01010, etc.
    odd = [0] * n
    left, right = 0, -1

    for center in range(n):
        radius = (
            1
            if center > right
            else min(odd[left + right - center], right - center + 1)
        )

        while (
            center - radius >= 0
            and center + radius < n
            and bits[center - radius] == bits[center + radius]
        ):
            radius += 1

        odd[center] = radius

        start = center - radius + 1
        length = 2 * radius - 1

        if (
            length > best_length
            or (length == best_length and start < best_start)
        ):
            best_start = start
            best_length = length

        if center + radius - 1 > right:
            left = center - radius + 1
            right = center + radius - 1

    # Even-length palindromes: 0110, 1001, etc.
    even = [0] * n
    left, right = 0, -1

    for center in range(n):
        radius = (
            0
            if center > right
            else min(
                even[left + right - center + 1],
                right - center + 1,
            )
        )

        while (
            center - radius - 1 >= 0
            and center + radius < n
            and bits[center - radius - 1] == bits[center + radius]
        ):
            radius += 1

        even[center] = radius

        start = center - radius
        length = 2 * radius

        if (
            length > best_length
            or (length == best_length and start < best_start)
        ):
            best_start = start
            best_length = length

        if center + radius - 1 > right:
            left = center - radius
            right = center + radius - 1

    return bits[best_start:best_start + best_length]


def generate_bits_with_zero_run(
    length: int,
    run_length: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Generate random bits containing four or eight consecutive zeros.

    The returned sequence has exactly the requested length.
    """
    if length < 0:
        raise ValueError("length must be non-negative")

    if run_length not in (4, 8):
        raise ValueError("run_length must be either 4 or 8")

    if length < run_length:
        raise ValueError("length must be at least run_length")

    generator = rng if rng is not None else random

    bits = generate_random_bits(length, rng)

    start = generator.randrange(length - run_length + 1)
    bits[start:start + run_length] = [0] * run_length

    return bits