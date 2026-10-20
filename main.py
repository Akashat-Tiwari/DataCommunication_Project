
"""Main menu for the Data Communication encoder/decoder project."""

import math
from collections.abc import Callable, Sequence

from pathlib import Path

from src.image_decoder import decode_signal_image

from src.data_generator import (
    generate_random_bits,
    generate_bits_with_zero_run,
    find_longest_palindrome,
)
from src.line_codes import (
    encode_nrz_l,
    encode_nrz_i,
    encode_manchester,
    encode_differential_manchester,
    encode_ami,
)
from src.scrambling import scramble_b8zs, scramble_hdb3
from src.pcm import encode_pcm, decode_pcm
from src.plotting import (
    plot_digital_stream,
    plot_line_coded_signal,
    plot_pcm_reconstruction,
)


Encoder = Callable[[Sequence[int]], list[int]]


LINE_CODERS: dict[str, tuple[str, Encoder, int]] = {
    "1": ("NRZ-L", encode_nrz_l, 1),
    "2": ("NRZ-I", encode_nrz_i, 1),
    "3": ("Manchester", encode_manchester, 2),
    "4": ("Differential Manchester", encode_differential_manchester, 2),
}

def _decode_and_report(bits: Sequence[int], scheme: str) -> None:
    """Decode a saved waveform image and display the recovered bitstream."""
    image_path = (
        Path(__file__).resolve().parent
        / "outputs"
        / "decoder_signal.png"
    )

    try:
        decoded_bits = decode_signal_image(image_path, scheme)
    except (OSError, ValueError) as error:
        print(f"Could not decode waveform image: {error}")
        return

    print("\nIMAGE-BASED DECODING RESULTS")
    print(f"Decoded bitstream: {''.join(map(str, decoded_bits))}")

    # The original bits are used only to verify the result after decoding.
    if decoded_bits == list(bits):
        print("Verification: decoded stream matches the original.")
    else:
        print("Warning: decoded stream differs from the original.")

SCRAMBLERS = {
    "1": ("B8ZS", scramble_b8zs),
    "2": ("HDB3", scramble_hdb3),
}


def _read_choice(prompt: str, valid_choices: Sequence[str]) -> str:
    """Read a menu choice and repeat until it is valid."""
    while True:
        choice = input(prompt).strip()

        if choice in valid_choices:
            return choice

        print(f"Invalid choice. Select one of: {', '.join(valid_choices)}.")


def _read_integer(
    prompt: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    """Read an integer within the specified range."""
    while True:
        try:
            value = int(input(prompt).strip())
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue

        if maximum is not None and value > maximum:
            print(f"Value must not exceed {maximum}.")
            continue

        return value


def _read_yes_no(prompt: str) -> bool:
    """Read a yes/no answer."""
    while True:
        answer = input(prompt).strip().lower()

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please enter yes or no.")


def _read_analog_samples() -> list[float]:
    """Read a comma-separated sequence of finite analog samples."""
    while True:
        raw = input(
            "Enter analog samples separated by commas "
            "(example: 0, 1.5, 2.0, -1): "
        ).strip()

        try:
            parts = raw.split(",")

            if not raw or any(not part.strip() for part in parts):
                raise ValueError

            samples = [float(part.strip()) for part in parts]

            if any(not math.isfinite(sample) for sample in samples):
                raise ValueError

            return samples

        except ValueError:
            print("Enter one or more valid finite numbers separated by commas.")


def run_line_coding(
    bits: Sequence[int],
    show_plots: bool = True,
) -> list[int]:
    """Ask for a line code, encode the bits, and plot the signal."""
    while True:
        print("\nLINE CODING SCHEMES")
        print("1. NRZ-L")
        print("2. NRZ-I")
        print("3. Manchester")
        print("4. Differential Manchester")
        print("5. AMI")

        choice = _read_choice("Select a scheme (1-5): ", ("1", "2", "3", "4", "5"))

        if choice == "5":
            if _read_yes_no("Do you want scrambling with AMI? (y/n): "):
                print("\nSCRAMBLING SCHEMES")
                print("1. B8ZS")
                print("2. HDB3")

                scramble_choice = _read_choice(
                    "Select a scrambling scheme (1-2): ", ("1", "2")
                )
                scheme_name, scrambler = SCRAMBLERS[scramble_choice]

                # B8ZS and HDB3 apply AMI polarity rules as part of scrambling.
                signal = scrambler(bits)
                display_name = f"AMI + {scheme_name}"
            else:
                signal = encode_ami(bits)
                display_name = "AMI"

            samples_per_bit = 1

        else:
            display_name, encoder, samples_per_bit = LINE_CODERS[choice]
            signal = encoder(bits)

        print(f"\nSelected scheme: {display_name}")
        print(f"Encoded signal levels: {signal}")

        plot_line_coded_signal(
            bits,
            signal,
            display_name,
            samples_per_bit=samples_per_bit,
            show=show_plots,
        )

        if _read_yes_no(
            "Decode the signal from the saved waveform image? (y/n): "
        ):
            _decode_and_report(bits, display_name)

        return signal


def run_digital_workflow(show_plots: bool = True) -> list[int]:
    """Generate digital data, find its longest palindrome, and encode it."""
    print("\nDIGITAL INPUT")
    print("1. Generate completely random bits")
    print("2. Generate bits containing four or eight consecutive zeros")

    generation_choice = _read_choice(
        "Select a data-generation option (1-2): ", ("1", "2")
    )

    if generation_choice == "1":
        length = _read_integer("Enter the number of bits to generate: ")
        bits = generate_random_bits(length)

    else:
        run_length = _read_integer(
            "Choose the zero-run length (4 or 8): ", minimum=4, maximum=8
        )

        # The assignment permits only runs of four or eight zeros.
        while run_length not in (4, 8):
            print("Choose either 4 or 8.")
            run_length = _read_integer(
                "Choose the zero-run length (4 or 8): ", minimum=4, maximum=8
            )

        length = _read_integer(
            f"Enter the total number of bits (at least {run_length}): ",
            minimum=run_length,
        )

        bits = generate_bits_with_zero_run(length, run_length)

    palindrome = find_longest_palindrome(bits)

    print("\nGenerated digital bitstream:")
    print("".join(map(str, bits)))
    print(f"Bitstream length: {len(bits)}")

    print("\nLongest contiguous palindrome:")
    print("".join(map(str, palindrome)))
    print(f"Palindrome length: {len(palindrome)}")

    plot_digital_stream(bits, show=show_plots)
    run_line_coding(bits, show_plots=show_plots)

    return bits


def run_analog_workflow(show_plots: bool = True) -> list[float]:
    """Encode analog samples using PCM and reconstruct the signal."""
    print("\nANALOG INPUT — PCM")

    samples = _read_analog_samples()
    bits_per_sample = _read_integer(
        "Enter bits per PCM sample (1-16): ",
        minimum=1,
        maximum=16,
    )

    encoding = encode_pcm(samples, bits_per_sample=bits_per_sample)

    bitstream_text = "".join(map(str, encoding.bitstream))

    print("\nPCM ENCODING RESULTS")
    print(f"Input analog samples: {samples}")
    print(f"Quantization range: {encoding.min_value} to {encoding.max_value}")
    print(f"Bits per sample: {encoding.bits_per_sample}")
    print(f"Quantization codes: {encoding.sample_codes}")
    print(f"Quantized samples: {encoding.quantized_samples}")
    print(f"PCM digital bitstream: {bitstream_text}")

    reconstructed = decode_pcm(
        encoding.bitstream,
        encoding.bits_per_sample,
        encoding.min_value,
        encoding.max_value,
    )

    print(f"Reconstructed samples: {reconstructed}")

    plot_pcm_reconstruction(
        samples,
        reconstructed,
        show=show_plots,
    )

    if _read_yes_no("Apply line coding to the PCM bitstream? (y/n): "):
        run_line_coding(encoding.bitstream, show_plots=show_plots)

    return reconstructed


def main() -> None:
    """Run the main application menu."""
    while True:
        print("\n" + "=" * 45)
        print("DIGITAL SIGNAL ENCODER AND DECODER")
        print("=" * 45)
        print("1. Digital input and line coding")
        print("2. Analog input and PCM")
        print("3. Exit")

        choice = _read_choice("Select an option (1-3): ", ("1", "2", "3"))

        if choice == "1":
            run_digital_workflow()
        elif choice == "2":
            run_analog_workflow()
        else:
            print("Exiting the program.")
            break


if __name__ == "__main__":
    main()
