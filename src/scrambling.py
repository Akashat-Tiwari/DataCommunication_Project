
"""B8ZS and HDB3 scrambling for AMI line coding."""

from collections.abc import Sequence


def _validate_bits(bits: Sequence[int]) -> None:
    """Ensure input contains only binary values."""
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must contain only 0 and 1")


def scramble_b8zs(bits: Sequence[int]) -> list[int]:
    """
    Encode binary data using B8ZS.

    Eight consecutive zeros are replaced with 000VB0VB.
    V is a bipolar violation; B is a normal bipolar pulse.

    Returns ternary signal levels: -1, 0, and +1.
    Assumption: the previous pulse polarity initially is -1.
    """
    _validate_bits(bits)

    signal = []
    last_pulse = -1
    i = 0

    while i < len(bits):
        if bits[i:i + 8] == [0] * 8:
            # V1 has the same polarity as the preceding pulse.
            violation = last_pulse
            balancing = -violation

            signal.extend([
                0, 0, 0,
                violation, balancing, 0,
                balancing, violation,
            ])

            # The final pulse in the substitution has this polarity.
            last_pulse = violation
            i += 8
        else:
            bit = bits[i]

            if bit == 1:
                last_pulse = -last_pulse
                signal.append(last_pulse)
            else:
                signal.append(0)

            i += 1

    return signal


def scramble_hdb3(bits: Sequence[int]) -> list[int]:
    """
    Encode binary data using HDB3.

    Four consecutive zeros are replaced by:
    - 000V when the number of nonzero pulses since the
      previous violation is even.
    - B00V when that number is odd.

    Returns ternary signal levels: -1, 0, and +1.
    Assumption: the previous pulse polarity initially is -1.
    """
    _validate_bits(bits)

    signal = []
    last_pulse = -1
    pulses_since_violation = 0
    i = 0

    while i < len(bits):
        if bits[i:i + 4] == [0] * 4:
            if pulses_since_violation % 2 == 0:
                # 000V: the violation has the preceding pulse's polarity.
                violation = last_pulse
                signal.extend([0, 0, 0, violation])
                last_pulse = violation
            else:
                # B00V: B is opposite to the preceding pulse;
                # V has the same polarity as B.
                balancing = -last_pulse
                signal.extend([balancing, 0, 0, balancing])
                last_pulse = balancing

            pulses_since_violation = 0
            i += 4
        else:
            bit = bits[i]

            if bit == 1:
                last_pulse = -last_pulse
                signal.append(last_pulse)
                pulses_since_violation += 1
            else:
                signal.append(0)

            i += 1

    return signal
