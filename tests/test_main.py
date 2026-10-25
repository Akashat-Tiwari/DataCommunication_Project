
import main


def test_line_coding_selects_nrz_l(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "1")
    captured = {}

    def fake_plot(bits, signal_levels, scheme, samples_per_bit=1, show=True):
        captured["bits"] = list(bits)
        captured["signal"] = signal_levels
        captured["scheme"] = scheme
        captured["samples_per_bit"] = samples_per_bit

    monkeypatch.setattr(main, "plot_line_coded_signal", fake_plot)

    result = main.run_line_coding([0, 1], show_plots=False)

    assert result == [1, -1]
    assert captured["scheme"] == "NRZ-L"
    assert captured["samples_per_bit"] == 1


def test_ami_can_select_hdb3(monkeypatch):
    answers = iter(["5", "y", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    captured = {}

    def fake_plot(bits, signal_levels, scheme, samples_per_bit=1, show=True):
        captured["signal"] = signal_levels
        captured["scheme"] = scheme

    monkeypatch.setattr(main, "plot_line_coded_signal", fake_plot)

    bits = [1, 0, 0, 0, 0]
    result = main.run_line_coding(bits, show_plots=False)

    assert result == [1, -1, 0, 0, -1]
    assert captured["scheme"] == "AMI + HDB3"


def test_digital_workflow_generates_fixed_zero_run(monkeypatch):
    answers = iter(["2", "4", "8", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    monkeypatch.setattr(main, "plot_digital_stream", lambda *args, **kwargs: None)

    captured = {}

    def fake_plot(bits, signal_levels, scheme, samples_per_bit=1, show=True):
        captured["bits"] = list(bits)
        captured["signal"] = signal_levels

    monkeypatch.setattr(main, "plot_line_coded_signal", fake_plot)

    bits = main.run_digital_workflow(show_plots=False)

    assert len(bits) == 8
    assert any(bits[i:i + 4] == [0, 0, 0, 0] for i in range(5))
    assert len(captured["signal"]) == 8


def test_analog_workflow_encodes_and_reconstructs_pcm(monkeypatch):
    answers = iter(["0, 1, 2, 3", "2", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    captured = {}

    def fake_plot(original, reconstructed, show=True):
        captured["original"] = list(original)
        captured["reconstructed"] = list(reconstructed)

    monkeypatch.setattr(main, "plot_pcm_reconstruction", fake_plot)

    result = main.run_analog_workflow(show_plots=False)

    assert result == [0.0, 1.0, 2.0, 3.0]
    assert captured["original"] == [0.0, 1.0, 2.0, 3.0]
    assert captured["reconstructed"] == [0.0, 1.0, 2.0, 3.0]
