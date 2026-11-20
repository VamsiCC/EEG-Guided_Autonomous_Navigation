"""EEG quality checks and features shared by training and prediction."""

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

LABELS = ("left", "right", "rest")
DIRECTION_LABELS = ("left", "right")


def quality_error(samples, max_ptp_uv=250.0, min_std_uv=0.1):
    """Return a rejection reason for a channels x samples microvolt window."""
    x = np.asarray(samples)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 32:
        return "need at least two channels and 32 samples"
    if not np.isfinite(x).all():
        return "non-finite samples"
    if np.any(np.ptp(x, axis=-1) > max_ptp_uv):
        return "large input amplitude / possible artifact or electrical interference"
    if np.any(np.std(x, axis=-1) < min_std_uv):
        return "flat channel"
    return None


def timing_error(timestamps, fs):
    """Reject gaps and an observed rate inconsistent with stream metadata."""
    ts = np.asarray(timestamps, dtype=float)
    if ts.ndim != 1 or len(ts) < 2 or not np.isfinite(ts).all() or not np.isfinite(fs) or fs <= 0:
        return "invalid timestamps"
    dt = np.diff(ts)
    backwards = int(np.count_nonzero(dt <= 0))
    gaps = int(np.count_nonzero(dt > 4.0 / fs))
    if backwards or gaps:
        problems = []
        if backwards:
            problems.append(f"nonmonotonic timestamps: {backwards} nonpositive intervals, min={1000 * dt.min():.3f} ms")
        if gaps:
            problems.append(f"stream gap: {gaps} intervals over {4000 / fs:.3f} ms, max={1000 * dt.max():.3f} ms")
        return "; ".join(problems)
    duration = ts[-1] - ts[0]
    if abs(duration - (len(ts) - 1) / fs) > 0.15 * (len(ts) - 1) / fs:
        return f"observed sample rate differs from nominal rate: {(len(ts) - 1) / duration:.2f} vs {fs:g} Hz"
    return None


def bandpass_windows(X, fs):
    """Filter complete windows independently, with no cross-trial statistics."""
    x = np.asarray(X, dtype=float)
    if x.ndim != 3 or not np.isfinite(fs) or fs <= 64:
        raise ValueError("Expected trials x channels x samples, fs > 64 Hz")
    if x.shape[1] < 2 or x.shape[2] < 32 or not np.isfinite(x).all():
        raise ValueError("Need finite EEG windows with at least two channels and 32 samples")
    x = x - np.mean(x, axis=-1, keepdims=True)
    sos = butter(4, [7, 30], fs=fs, btype="bandpass", output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def spectral_diagnostics(samples, fs):
    """Inspect mains/drift and model-band amplitude without changing raw data.

    Fractions refer to Welch power from 0.5 Hz through Nyquist. This is a
    troubleshooting aid, not a source-separation or signal-validity classifier.
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected channels x samples for spectral diagnostics")
    filtered = bandpass_windows(x[np.newaxis], fs)[0]
    freqs, psd = welch(x, fs=fs, nperseg=min(x.shape[-1], round(2 * fs)))
    valid = freqs >= 0.5
    power = psd[:, valid].sum(axis=-1)
    denominator = np.maximum(power, 1e-30)
    fraction = lambda low, high: psd[:, (freqs >= low) & (freqs < high)].sum(axis=-1) / denominator
    line50, line60, slow = fraction(48, 52), fraction(58, 62), fraction(0.5, 7)
    peak = freqs[valid][psd[:, valid].argmax(axis=-1)]
    return [dict(
        peak_hz=float(peak[i]) if power[i] > 0 else None,
        line_50_fraction=float(line50[i]), line_60_fraction=float(line60[i]),
        below_7_fraction=float(slow[i]),
        band_7_30_std_uv=float(filtered[i].std()),
        band_7_30_ptp_uv=float(np.ptp(filtered[i])),
    ) for i in range(len(x))]


class CSPLogPower(TransformerMixin, BaseEstimator):
    """Regularized binary CSP followed by log mean-square component power.

    Implements the two-class CSP eigenproblem using SciPy, rather than loading
    NeuroPype's proprietary pipeline. Class covariance is the mean of per-trial
    covariances (without trace normalization), shrunk toward a scaled identity.
    Extreme eigenvectors alternate between the two ends of the spectrum.
    See https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html
    for the CSP construction. All spatial filters are fitted on training only.
    """

    def __init__(self, fs=250.0, n_components=2, reg=0.1):
        self.fs = fs
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        x = bandpass_windows(X, self.fs)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(x) or set(y) != set(DIRECTION_LABELS):
            raise ValueError("Binary CSP needs left and right training trials only")
        if not isinstance(self.n_components, (int, np.integer)) or not 1 <= self.n_components <= x.shape[1]:
            raise ValueError("CSP components must be between 1 and the number of selected channels")
        if not 0 < self.reg < 1:
            raise ValueError("CSP regularization must be in (0, 1)")
        self.n_channels_in_ = x.shape[1]
        covariances = []
        for label in DIRECTION_LABELS:
            trials = x[y == label]
            centered = trials - trials.mean(axis=-1, keepdims=True)
            cov = (centered @ centered.transpose(0, 2, 1)).mean(axis=0) / (x.shape[-1] - 1)
            scale = np.trace(cov) / self.n_channels_in_
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("No usable covariance in CSP training data")
            covariances.append((1 - self.reg) * cov + self.reg * scale * np.eye(self.n_channels_in_))
        # C_left w = lambda (C_left + C_right) w; the extrema distinguish classes.
        eigenvalues, eigenvectors = eigh(covariances[0], covariances[0] + covariances[1])
        order = []
        for index in range((self.n_channels_in_ + 1) // 2):
            order.append(self.n_channels_in_ - index - 1)
            if index != self.n_channels_in_ - index - 1:
                order.append(index)
        chosen = order[:self.n_components]
        self.filters_ = eigenvectors[:, chosen].T
        self.eigenvalues_ = eigenvalues[chosen]
        return self

    def transform(self, X):
        check_is_fitted(self, "filters_")
        x = bandpass_windows(X, self.fs)
        if x.shape[1] != self.n_channels_in_:
            raise ValueError("CSP input channel count differs from calibration")
        projected = np.einsum("kc,nct->nkt", self.filters_, x)
        return np.log(np.maximum(np.mean(projected ** 2, axis=-1), 1e-12))


class LogBandPower(TransformerMixin, BaseEstimator):
    """A small-data baseline: channel power in four motor imagery bands.

    Every complete window is filtered independently in both offline and live
    use. No fitted statistics or filtering cross a train/test boundary.
    """

    def __init__(self, fs=250.0):
        self.fs = fs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = bandpass_windows(X, self.fs)
        freqs, psd = welch(x, fs=self.fs, nperseg=min(x.shape[-1], round(2 * self.fs)))
        bands = ((8, 12), (12, 16), (16, 22), (22, 30))
        df = freqs[1] - freqs[0]
        features = [
            np.log(np.maximum(psd[..., (freqs >= lo) & (freqs < hi)].sum(-1) * df, 1e-12))
            for lo, hi in bands
        ]
        return np.concatenate(features, axis=-1)


class DecisionGate:
    """Require consecutive, non-overlapping confident windows, then latch."""

    def __init__(self, threshold=0.8, consecutive=2):
        if not 0.5 < threshold <= 1 or consecutive < 1:
            raise ValueError("threshold must be in (0.5, 1], consecutive >= 1")
        self.threshold = threshold
        self.consecutive = consecutive
        self.previous = None
        self.count = 0
        self.latched = False

    def reset(self):
        self.previous = None
        self.count = 0

    def update(self, label, confidence):
        if self.latched:
            return None
        if label not in ("left", "right") or not np.isfinite(confidence) or confidence < self.threshold:
            self.reset()
            return None
        self.count = self.count + 1 if label == self.previous else 1
        self.previous = label
        if self.count >= self.consecutive:
            self.latched = True
            return label
        return None
