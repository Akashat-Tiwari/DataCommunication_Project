
import pytest

from src.pcm import encode_pcm, decode_pcm


def test_pcm_encodes_quantization_indices_as_binary():
    result = encode_pcm(
        [0.0, 1.0, 2.0, 3.0],
        bits_per_sample=2,
        min_value=0.0,
        max_value=3.0,
    )

    assert result.sample_codes == [0, 1, 2, 3]
    assert result.bitstream == [0, 0, 0, 1, 1, 0, 1, 1]
    assert result.quantized_samples == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_pcm_decoder_reconstructs_quantized_samples():
    result = encode_pcm(
        [0.0, 1.0, 2.0, 3.0],
        bits_per_sample=2,
        min_value=0.0,
        max_value=3.0,
    )

    decoded = decode_pcm(
        result.bitstream,
        result.bits_per_sample,
        result.min_value,
        result.max_value,
    )

    assert decoded == pytest.approx(result.quantized_samples)


def test_pcm_supports_negative_sample_values():
    result = encode_pcm(
        [-1.0, 1.0],
        bits_per_sample=2,
        min_value=-1.0,
        max_value=1.0,
    )

    assert result.sample_codes == [0, 3]
    assert result.quantized_samples == pytest.approx([-1.0, 1.0])


def test_pcm_infers_range_from_samples():
    result = encode_pcm([-2.0, 2.0], bits_per_sample=3)

    assert result.min_value == -2.0
    assert result.max_value == 2.0
    assert result.quantized_samples == pytest.approx([-2.0, 2.0])


def test_pcm_handles_constant_samples():
    result = encode_pcm([2.5, 2.5], bits_per_sample=3)

    assert result.sample_codes == [0, 0]
    assert result.quantized_samples == [2.5, 2.5]

    decoded = decode_pcm(
        result.bitstream,
        result.bits_per_sample,
        result.min_value,
        result.max_value,
    )
    assert decoded == [2.5, 2.5]


def test_pcm_handles_empty_input():
    result = encode_pcm([], bits_per_sample=4)

    assert result.bitstream == []
    assert result.sample_codes == []
    assert result.quantized_samples == []

    assert decode_pcm([], 4, 0.0, 0.0) == []


@pytest.mark.parametrize("bits_per_sample", [0, 17, True])
def test_pcm_rejects_invalid_bit_depth(bits_per_sample):
    with pytest.raises(ValueError):
        encode_pcm([0.0, 1.0], bits_per_sample=bits_per_sample)


def test_pcm_rejects_non_finite_samples():
    with pytest.raises(ValueError):
        encode_pcm([0.0, float("nan")])


def test_pcm_rejects_reversed_range():
    with pytest.raises(ValueError):
        encode_pcm([1.0], min_value=3.0, max_value=0.0)


def test_pcm_rejects_samples_outside_explicit_range():
    with pytest.raises(ValueError):
        encode_pcm([4.0], min_value=0.0, max_value=3.0)


def test_pcm_decoder_rejects_incomplete_sample_code():
    with pytest.raises(ValueError):
        decode_pcm([0, 1, 1], 2, 0.0, 3.0)


def test_pcm_decoder_rejects_invalid_bits():
    with pytest.raises(ValueError):
        decode_pcm([0, 2], 2, 0.0, 3.0)


def test_pcm_decoder_rejects_reversed_range():
    with pytest.raises(ValueError):
        decode_pcm([0, 1], 2, 3.0, 0.0)
