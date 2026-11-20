"""Read OpenBCI GUI Time Series LSL; never open the board's serial port."""

import time

import numpy as np

from .signal import timing_error


class EEGTimingError(ValueError):
    """A received window is unsuitable for labeling or prediction."""


def load_lsl():
    try:
        import pylsl
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "Cannot load pylsl/liblsl. See docs/eeg_pipeline.md installation."
        ) from exc
    return pylsl


def describe_stream(info):
    return dict(name=info.name(), type=info.type(), channels=info.channel_count(),
                sample_rate_hz=float(info.nominal_srate()))


def list_streams(timeout=5.0):
    """List every LSL stream type, including GUI outlets not labeled EEG."""
    return [describe_stream(info) for info in load_lsl().resolve_streams(wait_time=timeout)]


def resolve_eeg_info(pylsl, name, expected_channels):
    prop, value = ("name", name) if name else ("type", "EEG")
    streams = pylsl.resolve_byprop(prop, value, timeout=5.0)
    available = [describe_stream(s) for s in streams]
    streams = [s for s in streams if s.channel_count() == expected_channels and s.nominal_srate() > 64]
    if len(streams) != 1:
        raise ValueError(
            f"Need exactly one {expected_channels}-channel Time Series stream. "
            f"Found: {available}. Set --stream-name to its exact name. "
            "Enable LSL Time Series in the GUI, not FFT or BandPower. "
            "Run 'python -m src.eeg streams' to discover all stream types."
        )
    return streams[0]


def summarize_timing(timestamps, ages, elapsed, fs):
    """Describe received timing without repairing or accepting training data."""
    ts, ages = np.asarray(timestamps, dtype=float), np.asarray(ages, dtype=float)
    report = dict(samples=len(ts), received_hz=round(len(ts) / elapsed, 2),
                  timing_error=timing_error(ts, fs))
    if len(ts) < 2 or not np.isfinite(ts).all() or not np.isfinite(ages).all():
        return report
    dt = np.diff(ts)
    duration = ts[-1] - ts[0]
    edge = min(len(ts), max(1, round(fs)))
    report.update(
        timestamp_hz=round((len(ts) - 1) / duration, 2) if duration > 0 else None,
        nonpositive_intervals=int(np.count_nonzero(dt <= 0)),
        max_interval_ms=round(1000 * float(dt.max()), 3),
        stale_samples=int(np.count_nonzero(ages > 0.75)),
        future_samples=int(np.count_nonzero(ages < -0.25)),
        age_ms=dict(min=round(1000 * float(ages.min()), 2),
                    median=round(1000 * float(np.median(ages)), 2),
                    max=round(1000 * float(ages.max()), 2),
                    first_second_median=round(1000 * float(np.median(ages[:edge])), 2),
                    last_second_median=round(1000 * float(np.median(ages[-edge:])), 2)),
    )
    return report


def diagnose_timing(name=None, expected_channels=8, seconds=20.0):
    """Compare one outlet with and without smoothing, independent of trial guards.

    Both inlets apply clock correction exactly once. No samples are relabeled,
    resampled, saved as training data, or passed to the classifier.
    """
    pylsl = load_lsl()
    info = resolve_eeg_info(pylsl, name, expected_channels)
    modes = dict(clock_only=pylsl.proc_clocksync,
                 smoothed=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize)
    inlets, received = {}, {mode: ([], []) for mode in modes}
    offsets = {}
    try:
        for mode, flags in modes.items():
            inlet = pylsl.StreamInlet(info, max_buflen=5, processing_flags=flags)
            inlets[mode] = inlet
            inlet.open_stream(timeout=5.0)
            offsets[mode] = inlet.time_correction(timeout=5.0)
        print(f"Stream: {info.name()}, nominal {info.nominal_srate():g} Hz. "
              f"Comparing clock-only and smoothed timestamps for {seconds:g}s.", flush=True)
        print("No imagery needed; no training files are written. "
              "Positive age = past; negative age = future. Ages include chunk delivery delay.", flush=True)
        # Consume data queued while the second connection was opening. Keep
        # native smoothing's observation history, but exclude this startup backlog.
        for inlet in inlets.values():
            inlet.pull_chunk(timeout=0.0, max_samples=round(5 * info.nominal_srate()))
        start = time.monotonic()
        deadline, next_progress = start + seconds, start + 5
        while time.monotonic() < deadline:
            for mode, inlet in inlets.items():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                _, timestamps = inlet.pull_chunk(timeout=min(0.05, remaining), max_samples=1024)
                now = pylsl.local_clock()
                received[mode][0].extend(timestamps)
                received[mode][1].extend(now - ts for ts in timestamps)
            if time.monotonic() >= next_progress:
                counts = ", ".join(f"{mode}={len(values[0])}" for mode, values in received.items())
                print(f"  {min(seconds, time.monotonic() - start):.0f}s: {counts} samples", flush=True)
                next_progress += 5
        elapsed = time.monotonic() - start
        return dict(stream=describe_stream(info), measured_seconds=round(elapsed, 3),
                    clock_correction_ms={mode: round(1000 * value, 3) for mode, value in offsets.items()},
                    modes={mode: summarize_timing(ts, ages, elapsed, float(info.nominal_srate()))
                           for mode, (ts, ages) in received.items()})
    finally:
        for inlet in inlets.values():
            inlet.close_stream()


class EEGStream:
    def __init__(self, name=None, expected_channels=8, channels=None, unit="uv", warmup_seconds=5.0):
        pylsl = load_lsl()
        self.lsl = pylsl
        info = resolve_eeg_info(pylsl, name, expected_channels)
        self.name = info.name()
        self.fs = float(info.nominal_srate())
        self.channels = list(channels or range(1, expected_channels + 1))
        if len(self.channels) < 2 or len(set(self.channels)) != len(self.channels) or any(
            ch < 1 or ch > expected_channels for ch in self.channels
        ):
            raise ValueError("Choose at least two unique, valid 1-based EEG channel numbers")
        self.scale = 1e6 if unit == "v" else 1.0
        self.inlet = pylsl.StreamInlet(
            info, max_buflen=5,
            processing_flags=pylsl.proc_clocksync | pylsl.proc_dejitter | pylsl.proc_monotonize,
        )
        self.inlet.open_stream(timeout=5.0)
        self.inlet.time_correction(timeout=5.0)
        try:
            print(f"Warming up LSL timestamp processing for {warmup_seconds:g} seconds...", flush=True)
            self.warm_up(warmup_seconds)
        except Exception:
            self.inlet.close_stream()
            raise

    def warm_up(self, seconds):
        """Feed LSL's timestamp estimator before any labeled trial begins."""
        self.wait(seconds)

    def wait(self, seconds):
        """Wait while continuously consuming unlabeled samples.

        Keep the online timestamp processor fed during rest/cue intervals;
        sleeping then flushing would skip samples in its observation history.
        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            sample, _ = self.inlet.pull_sample(timeout=1.0)
            if sample is None:
                raise TimeoutError("No EEG samples during wait/warm-up; check GUI streaming")

    def close(self):
        self.inlet.close_stream()

    def read_window(self, seconds, strict_timing=True):
        """Take fresh data after this call; fail on stale data or interruptions."""
        n = round(seconds * self.fs)
        if n < 32:
            raise ValueError("Window is too short")
        start = self.lsl.local_clock()
        deadline = time.monotonic() + seconds + 5
        samples, timestamps = [], []
        while len(samples) < n:
            if time.monotonic() > deadline:
                raise TimeoutError("Incomplete EEG window; check GUI streaming")
            sample, ts = self.inlet.pull_sample(timeout=1.0)
            if sample is None:
                raise TimeoutError("No EEG samples for one second; check GUI streaming")
            # Consume older samples through LSL's postprocessor instead of
            # skipping their history with inlet.flush(). Never label them.
            if ts < start:
                continue
            age = self.lsl.local_clock() - ts
            if not np.isfinite(age) or age > 0.75 or age < -0.25:
                raise EEGTimingError(
                    f"EEG timestamps are stale or unsynchronized: age={age:+.3f}s "
                    "(allowed -0.250 to +0.750s; negative means future)"
                )
            samples.append(np.asarray(sample)[np.asarray(self.channels) - 1] * self.scale)
            timestamps.append(ts)
        error = timing_error(timestamps, self.fs)
        if error and strict_timing:
            raise EEGTimingError(error)
        return np.asarray(samples, dtype=float).T, np.asarray(timestamps)
