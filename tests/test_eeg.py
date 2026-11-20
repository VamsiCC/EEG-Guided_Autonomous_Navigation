"""Synthetic integration and failure-path tests; no headset or robot required."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import hadamard
from sklearn.exceptions import NotFittedError

from src.eeg.__main__ import collect, predict
from src.eeg.model import load_model, load_trials, split_trials, train_model
from src.eeg.signal import CSPLogPower, DecisionGate, LogBandPower, quality_error, spectral_diagnostics, timing_error
from src.eeg.stream import EEGStream, EEGTimingError, diagnose_timing, list_streams, summarize_timing


def synthetic_signal(label, rng):
    t = np.arange(500) / 125
    x = rng.normal(0, 2, (3, 500))
    amplitude = {"left": [4, 8, 18], "right": [18, 8, 4], "rest": [12, 12, 12]}[label]
    return x + np.asarray(amplitude)[:, None] * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))


def recording(directory, session_id, seed):
    directory.mkdir()
    rng = np.random.default_rng(seed)
    for label in ("left", "right", "rest"):
        for i in range(12):
            np.savez_compressed(
                directory / f"{label}_{i}.npz", samples_uv=synthetic_signal(label, rng),
                timestamps=1000 + np.arange(500) / 125, fs=125.0, label=label,
                channel_names=["C3", "Cz", "C4"], channels=[1, 2, 3],
                subject="test", session_id=session_id,
            )


def test_quality_rejects_flat_missing_and_large_artifacts():
    x = synthetic_signal("left", np.random.default_rng(1))
    assert quality_error(x) is None
    flat = x.copy()
    flat[0] = 0
    assert "flat" in quality_error(flat)
    missing = x.copy()
    missing[0, 0] = np.nan
    assert "non-finite" in quality_error(missing)
    artifact = x.copy()
    artifact[0, 100] = 500
    assert "amplitude" in quality_error(artifact)


def test_timing_rejects_gaps_and_wrong_rate():
    ts = np.arange(500) / 125
    assert timing_error(ts, 125) is None
    assert timing_error(ts, 250) is not None
    ts[250:] += 1
    assert "gap" in timing_error(ts, 125)


def test_bandpower_distinguishes_opposite_lateral_power():
    rng = np.random.default_rng(2)
    features = LogBandPower(125).fit_transform(np.stack([
        synthetic_signal("left", rng), synthetic_signal("right", rng),
    ]))
    assert features.shape == (2, 12)
    assert features[0, 0] < features[0, 2]
    assert features[1, 0] > features[1, 2]


def test_gate_resets_on_rest_low_scores_and_direction_changes_then_latches():
    gate = DecisionGate()
    assert gate.update("left", 0.9) is None
    assert gate.update("rest", 0.99) is None
    assert gate.update("left", 0.9) is None
    assert gate.update("left", 0.7) is None
    assert gate.update("left", 0.9) is None
    assert gate.update("right", 0.9) is None
    assert gate.update("right", 0.9) == "right"
    assert gate.update("left", 1.0) is None


def test_manual_reset_clears_evidence():
    gate = DecisionGate()
    gate.update("left", 0.9)
    gate.reset()
    assert gate.update("left", 0.9) is None


def test_session_training_serialization_and_duplicate_protection(tmp_path):
    first, second = tmp_path / "s1", tmp_path / "s2"
    recording(first, "first", 1)
    recording(second, "second", 2)
    output = tmp_path / "model.pkl"
    report = train_model(first, output, test_session=second, method="bandpower")
    assert report["evaluation"] == "session_holdout"
    assert report["balanced_accuracy"] > 0.9  # Deliberately easy artificial signals.
    assert report["rest_false_candidate_window_rate"] == 0
    assert (tmp_path / "model.json").exists()
    model = load_model(output)
    x = synthetic_signal("left", np.random.default_rng(8))
    assert model["classifier"].predict(x[None])[0] == "left"
    with pytest.raises(ValueError, match="already exists"):
        train_model(first, output)
    rows, _, _ = load_trials(first)
    with pytest.raises(ValueError, match="share EEG trials"):
        split_trials(rows, rows)
    with pytest.raises(ValueError, match="Duplicate"):
        split_trials(rows + [rows[0]])


def test_trial_holdout_keeps_whole_trials_disjoint(tmp_path):
    recording(tmp_path / "s1", "first", 1)
    rows, _, _ = load_trials(tmp_path / "s1")
    train, test, kind = split_trials(rows)
    assert kind == "trial_holdout_only"
    assert not {r[3] for r in train} & {r[3] for r in test}
    assert all(r[0].shape == (3, 500) for r in train + test)


def test_rejects_channel_mismatch_between_sessions(tmp_path):
    recording(tmp_path / "s1", "first", 1)
    recording(tmp_path / "s2", "second", 2)
    path = tmp_path / "s2" / "left_0.npz"
    with np.load(path) as z:
        content = dict(z)
    content["channel_names"] = np.asarray(["C4", "Cz", "C3"])
    np.savez_compressed(path, **content)
    with pytest.raises(ValueError, match="channel order"):
        train_model(tmp_path / "s1", tmp_path / "model.pkl", tmp_path / "s2")


def fake_stream(samples, timestamps, age=0):
    stream = EEGStream.__new__(EEGStream)
    stream.fs, stream.channels, stream.scale = 125, [1, 3], 1
    state = {"now": 1000.0}
    iterator = iter(zip(samples.T, timestamps))

    def pull_sample(timeout):
        sample, ts = next(iterator)
        state["now"] = ts + age
        return sample, ts

    stream.lsl = SimpleNamespace(local_clock=lambda: state["now"])
    stream.inlet = SimpleNamespace(flush=lambda: None, pull_sample=pull_sample)
    return stream


def test_stream_selects_simultaneous_channels_in_wiring_order():
    samples = synthetic_signal("left", np.random.default_rng(1))
    ts = 1000 + np.arange(1, 501) / 125
    x, received = fake_stream(samples, ts).read_window(4)
    np.testing.assert_array_equal(x, samples[[0, 2]])
    np.testing.assert_array_equal(received, ts)


@pytest.mark.parametrize("age", [2, -0.4])
def test_stream_aborts_on_stale_or_future_samples(age):
    samples = synthetic_signal("left", np.random.default_rng(1))
    ts = 1000 + np.arange(1, 501) / 125
    with pytest.raises(EEGTimingError) as error:
        fake_stream(samples, ts, age=age).read_window(4)
    assert f"age={age:+.3f}s" in str(error.value)


def test_stream_aborts_on_missing_samples():
    stream = EEGStream.__new__(EEGStream)
    stream.fs = 125
    stream.lsl = SimpleNamespace(local_clock=lambda: 1000.0)
    stream.inlet = SimpleNamespace(flush=lambda: None, pull_sample=lambda timeout: (None, None))
    with pytest.raises(TimeoutError, match="No EEG"):
        stream.read_window(4)


def spatial_recording(directory, session_id, seed, fs):
    """Opposite source power with similar individual sensor powers.

    Two 10 Hz quadrature sources have different variances by class and mix
    through equal-magnitude spatial weights. CSP must use cross-channel
    covariance to recover them. These are artificial signals, not EEG evidence.
    """
    directory.mkdir()
    rng = np.random.default_rng(seed)
    t = np.arange(round(3 * fs)) / fs
    mixing = hadamard(8)[:, :2] / np.sqrt(8)
    for label in ("left", "right", "rest"):
        amplitude = {"left": [4, 16], "right": [16, 4], "rest": [10, 10]}[label]
        for i in range(12):
            phase = 2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi)
            sources = np.stack([np.sin(phase), np.cos(phase)]) * np.asarray(amplitude)[:, None]
            x = mixing @ sources + rng.normal(0, 0.8, (8, len(t)))
            np.savez_compressed(
                directory / f"{label}_{i}.npz", samples_uv=x,
                timestamps=1000 + t, fs=float(fs), label=label,
                channel_names=["C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"],
                channels=np.arange(1, 9), subject="spatial_test", session_id=session_id,
            )


@pytest.mark.parametrize("fs", [125, 250])
def test_csp_binary_classifier_on_independent_spatial_sessions(tmp_path, fs):
    first, second = tmp_path / "s1", tmp_path / "s2"
    spatial_recording(first, "first", 1, fs)
    spatial_recording(second, "second", 2, fs)
    report = train_model(first, tmp_path / "csp.pkl", test_session=second)
    assert report["method"] == "csp"
    assert report["labels"] == ["left", "right"]
    assert report["balanced_accuracy"] > 0.95
    assert report["train_trials"] == 24
    assert report["excluded_train_rest_trials"] == 12
    assert report["test_rest_trials"] == 12
    assert report["scored_test_trials"] == 24
    model = load_model(tmp_path / "csp.pkl")
    assert model["classifier"].classes_.tolist() == ["left", "right"]
    assert model["classifier"][0].filters_.shape == (2, 8)
    assert model["metadata"]["fs"] == fs
    assert model["metadata"]["n_samples"] == 3 * fs


def test_csp_does_not_fit_rest_or_test_session_data(tmp_path):
    first, second = tmp_path / "s1", tmp_path / "s2"
    spatial_recording(first, "first", 1, 250)
    spatial_recording(second, "second", 2, 250)
    train_model(first, tmp_path / "before.pkl", test_session=second)
    # Alter every held-out trial and all training-rest trials, keeping basic
    # quality acceptable. Neither learned filters nor scaling may change.
    for path in list(second.glob("*.npz")) + list(first.glob("rest_*.npz")):
        with np.load(path) as z:
            content = dict(z)
        content["samples_uv"] *= np.linspace(0.6, 1.4, 8)[:, None]
        np.savez_compressed(path, **content)
    train_model(first, tmp_path / "after.pkl", test_session=second)
    before, after = load_model(tmp_path / "before.pkl"), load_model(tmp_path / "after.pkl")
    np.testing.assert_array_equal(before["classifier"][0].filters_, after["classifier"][0].filters_)
    np.testing.assert_array_equal(before["classifier"][1].mean_, after["classifier"][1].mean_)
    np.testing.assert_array_equal(before["classifier"][-1].coef_, after["classifier"][-1].coef_)


def test_csp_requires_binary_labels_and_calibrated_channels():
    rng = np.random.default_rng(4)
    x = np.stack([synthetic_signal("left", rng), synthetic_signal("right", rng)])
    csp = CSPLogPower(fs=125)
    with pytest.raises(NotFittedError):
        csp.transform(x)
    with pytest.raises(ValueError, match="left and right"):
        csp.fit(x, ["left", "rest"])
    with pytest.raises(ValueError, match="components"):
        CSPLogPower(fs=125, n_components=4).fit(x, ["left", "right"])
    csp.fit(x, ["left", "right"])
    with pytest.raises(ValueError, match="channel count"):
        csp.transform(x[:, :2])
    # Shrinkage keeps the eigensystem usable with linearly dependent channels.
    repeated = np.repeat(x[:, :1], 3, axis=1)
    assert np.isfinite(csp.fit_transform(repeated, ["left", "right"])).all()


def test_binary_csp_can_train_without_rest_but_reports_missing_idle_evaluation(tmp_path):
    spatial_recording(tmp_path / "s1", "first", 1, 250)
    for path in (tmp_path / "s1").glob("rest_*.npz"):
        path.unlink()
    report = train_model(tmp_path / "s1", tmp_path / "csp.pkl")
    assert report["rest_false_candidate_window_rate"] is None
    assert report["test_rest_trials"] == 0
    assert report["balanced_accuracy"] > 0.95


def test_prediction_attempt_expires_without_a_candidate(capsys):
    classifier = SimpleNamespace(classes_=np.asarray(["left", "right"]))
    model = dict(metadata=dict(fs=250, channels=[1, 2], n_samples=750,
                               subject="test", channel_names=["C3", "C4"]), classifier=classifier)
    stream = SimpleNamespace(fs=250, channels=[1, 2])
    args = SimpleNamespace(threshold=0.8, consecutive=2, timeout_seconds=1, ros=False)
    predict(args, stream, model)
    output = capsys.readouterr().out
    assert '"event": "selection_timeout"' in output
    assert '"event": "destination_candidate"' not in output


def fake_info(name, kind, channels, fs):
    return SimpleNamespace(name=lambda: name, type=lambda: kind,
                           channel_count=lambda: channels, nominal_srate=lambda: fs)


def test_stream_discovery_lists_non_eeg_outlets(monkeypatch):
    infos = [fake_info("raw", "EXG", 8, 250), fake_info("markers", "Markers", 1, 0)]
    monkeypatch.setitem(sys.modules, "pylsl", SimpleNamespace(resolve_streams=lambda wait_time: infos))
    assert list_streams(0.1) == [
        dict(name="raw", type="EXG", channels=8, sample_rate_hz=250.0),
        dict(name="markers", type="Markers", channels=1, sample_rate_hz=0.0),
    ]


def test_explicit_stream_name_accepts_raw_non_eeg_type_but_rejects_fft(monkeypatch):
    requested = []
    inlet_options = {}
    info = fake_info("raw", "EXG", 8, 250)

    def resolve(prop, value, timeout):
        requested.append((prop, value))
        return [info]

    inlet = SimpleNamespace(open_stream=lambda timeout: None, time_correction=lambda timeout: 0)

    def make_inlet(*args, **kwargs):
        inlet_options.update(kwargs)
        return inlet

    monkeypatch.setattr(EEGStream, "warm_up", lambda self, seconds: None)
    monkeypatch.setitem(sys.modules, "pylsl", SimpleNamespace(
        resolve_byprop=resolve, StreamInlet=make_inlet,
        proc_clocksync=1, proc_dejitter=2, proc_monotonize=4,
    ))
    stream = EEGStream(name="raw")
    assert requested == [("name", "raw")]
    assert stream.channels == list(range(1, 9))
    assert stream.fs == 250
    assert inlet_options["processing_flags"] == 7
    info = fake_info("fft", "EEG", 60, 200)
    with pytest.raises(ValueError, match="Time Series"):
        EEGStream(name="fft")


def test_timing_diagnostic_preserves_bad_timestamps_but_normal_acquisition_rejects():
    samples = synthetic_signal("left", np.random.default_rng(1))
    ts = 1000 + np.arange(1, 501) / 125
    ts[250] = ts[249] - 0.001
    with pytest.raises(ValueError, match="nonmonotonic timestamps"):
        fake_stream(samples, ts).read_window(4)
    _, diagnostic_ts = fake_stream(samples, ts).read_window(4, strict_timing=False)
    np.testing.assert_array_equal(diagnostic_ts, ts)
    assert "1 nonpositive intervals" in timing_error(diagnostic_ts, 125)


def test_large_timestamp_gap_is_still_rejected_after_timing_fix():
    samples = synthetic_signal("left", np.random.default_rng(1))
    ts = 1000 + np.arange(1, 501) / 125
    ts[250:] += 0.5
    with pytest.raises(ValueError, match="stream gap"):
        fake_stream(samples, ts).read_window(4)


def test_warmup_consumes_samples_and_detects_an_interrupted_stream(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.eeg.stream.time.monotonic", lambda: clock[0])
    stream = EEGStream.__new__(EEGStream)
    received = []

    def pull_sample(timeout):
        clock[0] += 0.1
        received.append(clock[0])
        return [1, 2], 1000 + clock[0]

    stream.inlet = SimpleNamespace(pull_sample=pull_sample)
    stream.warm_up(0.5)
    assert len(received) == 5
    stream.inlet = SimpleNamespace(pull_sample=lambda timeout: (None, None))
    with pytest.raises(TimeoutError, match="warm-up"):
        stream.warm_up(0.5)


def test_timing_summary_separates_delivery_rate_and_future_timestamps():
    ts = 1000 + np.arange(500) / 250
    report = summarize_timing(ts, np.full(500, -0.4), elapsed=2.5, fs=250)
    assert report["received_hz"] == 200
    assert report["timestamp_hz"] == 250
    assert report["future_samples"] == 500
    assert report["stale_samples"] == 0
    assert report["age_ms"]["median"] == -400
    assert report["timing_error"] is None
    assert summarize_timing([], [], 2, 250)["timing_error"] == "invalid timestamps"


def test_timing_diagnostic_compares_both_modes_and_closes_on_failure(monkeypatch):
    clock, inlets = [0.0], []
    info = fake_info("test", "EEG", 8, 250)

    class Inlet:
        def __init__(self, resolved, max_buflen, processing_flags):
            assert resolved is info
            self.flags, self.closed = processing_flags, False
            inlets.append(self)

        def open_stream(self, timeout):
            pass

        def time_correction(self, timeout):
            return 0.001

        def pull_chunk(self, timeout, max_samples):
            if timeout == 0:
                return [], []
            clock[0] += timeout
            # Deliberately future-dated smoothed samples must be reported,
            # rather than hidden by ordinary collection's freshness guard.
            end = 1000 + clock[0] + (0.5 if self.flags == 7 else 0)
            ts = end - np.arange(9, -1, -1) / 250
            return [[0] * 8] * 10, ts.tolist()

        def close_stream(self):
            self.closed = True

    monkeypatch.setattr("src.eeg.stream.time.monotonic", lambda: clock[0])
    monkeypatch.setitem(sys.modules, "pylsl", SimpleNamespace(
        resolve_byprop=lambda *args, **kwargs: [info], StreamInlet=Inlet,
        proc_clocksync=1, proc_dejitter=2, proc_monotonize=4,
        local_clock=lambda: 1000 + clock[0],
    ))
    report = diagnose_timing("test", seconds=0.2)
    assert [inlet.flags for inlet in inlets] == [1, 7]
    assert all(inlet.closed for inlet in inlets)
    assert report["modes"]["clock_only"]["future_samples"] == 0
    smoothed = report["modes"]["smoothed"]
    assert smoothed["future_samples"] == smoothed["samples"] > 0

    def failure(self, timeout, max_samples):
        raise RuntimeError("Lost connection")

    monkeypatch.setattr(Inlet, "pull_chunk", failure)
    with pytest.raises(RuntimeError, match="Lost connection"):
        diagnose_timing("test", seconds=0.2)
    assert all(inlet.closed for inlet in inlets)


def test_spectral_diagnostics_distinguish_mains_noise_from_slow_drift():
    fs = 250
    t = np.arange(1000) / fs
    motor_band = 10 * np.sin(2 * np.pi * 10 * t)
    x = np.stack([
        motor_band + 1000 * np.sin(2 * np.pi * 60 * t),
        motor_band + 2000 * np.sin(2 * np.pi * t),
    ])
    original = x.copy()
    mains, drift = spectral_diagnostics(x, fs)
    assert mains["peak_hz"] == 60
    assert mains["line_60_fraction"] > 0.99
    assert mains["line_50_fraction"] < 0.01
    assert drift["peak_hz"] == 1
    assert drift["below_7_fraction"] > 0.99
    assert mains["band_7_30_std_uv"] < 0.25 * x[0].std()
    assert drift["band_7_30_std_uv"] < 0.25 * x[1].std()
    np.testing.assert_array_equal(x, original)
    # The diagnostic is not permission to train on a raw artifact window.
    assert quality_error(x) is not None


def test_training_rejects_mixed_declared_input_filtering(tmp_path):
    recording(tmp_path / "s1", "first", 1)
    path = tmp_path / "s1" / "left_0.npz"
    with np.load(path) as z:
        content = dict(z)
    content["input_filtering"] = "gui-5-50-notch60"
    np.savez_compressed(path, **content)
    with pytest.raises(ValueError, match="input filtering"):
        load_trials(tmp_path / "s1")


def test_prediction_rejects_declared_input_filter_mismatch():
    stream = SimpleNamespace(fs=250, channels=[1, 2])
    model = dict(metadata=dict(fs=250, channels=[1, 2], input_filtering="unfiltered"))
    args = SimpleNamespace(input_filtering="gui-5-50-notch60")
    with pytest.raises(ValueError, match="input filtering"):
        predict(args, stream, model)


def test_reader_consumes_older_samples_without_flushing_or_labeling_them():
    x = synthetic_signal("left", np.random.default_rng(1))
    older = x[:, :10]
    samples = np.concatenate([older, x], axis=1)
    ts = np.concatenate([999 + np.arange(10) / 125, 1000 + np.arange(1, 501) / 125])
    stream = fake_stream(samples, ts)

    def forbidden_flush():
        raise AssertionError("Must consume queued samples through the timestamp processor")

    stream.inlet.flush = forbidden_flush
    received, timestamps = stream.read_window(4)
    np.testing.assert_array_equal(received, x[[0, 2]])
    np.testing.assert_array_equal(timestamps, ts[10:])


def collection_args(output):
    return SimpleNamespace(channel_names=["C3", "C4"], output=output, seed=149,
                           trials_per_class=1, subject="test", input_filtering="gui-5-50-notch60",
                           rest_seconds=2, cue_delay=0.5, seconds=3, max_ptp_uv=250)


def test_collector_drains_rest_and_retries_a_rejected_trial(tmp_path):
    output = tmp_path / "session"
    output.mkdir()  # An empty folder left by an earlier failed attempt is safe.
    clock, waits, acquisitions = [1000.0], [], []

    def wait(seconds):
        waits.append(seconds)
        clock[0] += seconds

    def read(seconds):
        acquisitions.append(clock[0])
        if len(acquisitions) == 1:
            raise EEGTimingError("observed sample rate differs from nominal rate")
        ts = clock[0] + np.arange(750) / 250
        x = np.stack([10 * np.sin(2*np.pi*10*ts), 12 * np.sin(2*np.pi*12*ts)])
        clock[0] += seconds
        return x, ts

    stream = SimpleNamespace(channels=[1, 2], fs=250, name="test", wait=wait,
                             read_window=read, lsl=SimpleNamespace(local_clock=lambda: clock[0]))
    collect(collection_args(output), stream)
    files = sorted(output.glob("*.npz"))
    assert len(files) == 3
    assert waits == [3] + [2, 0.5] * 4
    labels, attempts = [], []
    for path in files:
        with np.load(path) as trial:
            labels.append(str(trial["label"]))
            attempts.append(int(trial["acquisition_attempt"]))
            assert str(trial["input_filtering"]) == "gui-5-50-notch60"
            assert timing_error(trial["timestamps"], 250) is None
    assert sorted(labels) == ["left", "rest", "right"]
    assert sorted(attempts) == [1, 1, 2]
    assert len((output / "rejected_attempts.jsonl").read_text().splitlines()) == 1


def test_collector_stops_after_repeated_timing_failures(tmp_path):
    output = tmp_path / "session"
    attempts = []

    def read(seconds):
        attempts.append(seconds)
        raise EEGTimingError("stream gap")

    stream = SimpleNamespace(channels=[1, 2], fs=250, name="test", wait=lambda seconds: None,
                             read_window=read, lsl=SimpleNamespace(local_clock=lambda: 1000.0))
    with pytest.raises(RuntimeError, match="Three timing failures"):
        collect(collection_args(output), stream)
    assert len(attempts) == 3
    assert not list(output.glob("*.npz"))
    assert len((output / "rejected_attempts.jsonl").read_text().splitlines()) == 3


def test_collector_preserves_existing_recordings(tmp_path):
    output = tmp_path / "session"
    output.mkdir()
    saved = output / "existing.npz"
    saved.write_bytes(b"preserve this recording")
    with pytest.raises(ValueError, match="contains files"):
        collect(collection_args(output), SimpleNamespace(channels=[1, 2]))
    assert saved.read_bytes() == b"preserve this recording"
