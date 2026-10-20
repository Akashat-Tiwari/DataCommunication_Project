
import pytest

from src.line_codes import (
    encode_ami,
    encode_differential_manchester,
    encode_line_code,
    encode_manchester,
    encode_nrz_i,
    encode_nrz_l,
)


def test_nrz_l():
    assert encode_nrz_l([0, 1, 1, 0]) == [1, -1, -1, 1]


def test_nrz_i():
    assert encode_nrz_i([0, 1, 1, 0]) == [-1, 1, -1, -1]


def test_manchester():
    assert encode_manchester([0, 1]) == [1, -1, -1, 1]


def test_differential_manchester():
    assert encode_differential_manchester([0, 1, 0, 1]) == [
        -1, 1, 1, -1, 1, -1, -1, 1
    ]


def test_ami():
    assert encode_ami([1, 0, 1, 1, 0, 1]) == [1, 0, -1, 1, 0, -1]


@pytest.mark.parametrize(
    "encoder",
    [
        encode_nrz_l,
        encode_nrz_i,
        encode_manchester,
        encode_differential_manchester,
        encode_ami,
    ],
)
def test_encoders_accept_empty_input(encoder):
    assert encoder([]) == []


@pytest.mark.parametrize(
    "encoder",
    [
        encode_nrz_l,
        encode_nrz_i,
        encode_manchester,
        encode_differential_manchester,
        encode_ami,
    ],
)
def test_encoders_reject_invalid_bits(encoder):
    with pytest.raises(ValueError):
        encoder([0, 1, 2])


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        ("NRZ-L", [1, -1]),
        ("NRZ-I", [-1, 1]),
        ("Manchester", [1, -1, -1, 1]),
        ("Differential Manchester", [-1, 1, 1, -1]),
        ("AMI", [0, 1]),
    ],
)
def test_line_code_dispatcher(scheme, expected):
    assert encode_line_code([0, 1], scheme) == expected


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError):
        encode_line_code([0, 1], "UNKNOWN")
