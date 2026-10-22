
"""Pulse Code Modulation (PCM) encoder and decoder."""

from dataclasses import dataclass
import math
from collections.abc import Sequence


@dataclass(frozen=True)
class PCMEncoding:
    """Store PCM output and the parameters needed for decoding."""

    bitstream: list[int]
    sample_codes: list[int]
    quantized_samples: list[float]
    min_value: float
    max_value: float
    bits_per_sample: int
    step_size: float


def _validate_bits_per_sample(bits_per_sample: int) -> None:
    """Validate the number of bits used for each sample."""
    if (
        not isinstance(bits_per_sample, int)
        or isinstance(bits_per_sample, bool)
        or not 1 <= bits_per_sample <= 16
    ):
        raise ValueError("bits_per_sample must be an integer from 1 to 16")


def _validate_bits(bits: Sequence[int]) -> None:
    """Ensure the encoded stream contains only binary values."""
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bitstream must contain only 0 and 1")


def encode_pcm(
    samples: Sequence[float],
    bits_per_sample: int = 8,
    min_value: float | None = None,
    max_value: float | None = None,
) -> PCMEncoding:
    """
    Quantize analog samples and encode their quantization indices in binary.

    The quantization range is inferred from the samples unless explicitly
    supplied. Samples outside an explicitly supplied range are rejected.
    """
    _validate_bits_per_sample(bits_per_sample)

    values = [float(sample) for sample in samples]

    if any(not math.isfinite(value) for value in values):
        raise ValueError("samples must contain only finite values")

    lower = float(
        min_value if min_value is not None else min(values, default=0.0)
    )
    upper = float(
        max_value if max_value is not None else max(values, default=0.0)
    )

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("quantization range must contain finite values")

    if lower > upper:
        raise ValueError("min_value must not exceed max_value")

    if any(value < lower or value > upper for value in values):
        raise ValueError("a sample lies outside the quantization range")

    levels = 2**bits_per_sample

    # A constant input has a zero-width range and needs no quantization step.
    if lower == upper:
        step_size = 0.0
        sample_codes = [0] * len(values)
        quantized_samples = [lower] * len(values)
    else:
        step_size = (upper - lower) / (levels - 1)

        sample_codes = []
        for value in values:
            code = round((value - lower) / step_size)
            code = max(0, min(levels - 1, code))
            sample_codes.append(code)

        quantized_samples = [
            lower + code * step_size for code in sample_codes
        ]

    # Encode each quantization index using most-significant-bit-first order.
    bitstream = [
        (code >> shift) & 1
        for code in sample_codes
        for shift in range(bits_per_sample - 1, -1, -1)
    ]

    return PCMEncoding(
        bitstream=bitstream,
        sample_codes=sample_codes,
        quantized_samples=quantized_samples,
        min_value=lower,
        max_value=upper,
        bits_per_sample=bits_per_sample,
        step_size=step_size,
    )


def decode_pcm(
    bitstream: Sequence[int],
    bits_per_sample: int,
    min_value: float,
    max_value: float,
) -> list[float]:
    """
    Decode PCM bits into reconstructed quantized sample values.

    The decoder uses the bit depth and quantization range, not the original
    analog samples.
    """
    _validate_bits_per_sample(bits_per_sample)
    _validate_bits(bitstream)

    lower = float(min_value)
    upper = float(max_value)

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("quantization range must contain finite values")

    if lower > upper:
        raise ValueError("min_value must not exceed max_value")

    if len(bitstream) % bits_per_sample != 0:
        raise ValueError("bitstream length must be divisible by bits_per_sample")

    codes = []

    for start in range(0, len(bitstream), bits_per_sample):
        code = 0

        for bit in bitstream[start:start + bits_per_sample]:
            code = (code << 1) | bit

        codes.append(code)

    if lower == upper:
        return [lower] * len(codes)

    step_size = (upper - lower) / (2**bits_per_sample - 1)

    return [
        lower + code * step_size
        for code in codes
    ]
