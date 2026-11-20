"""Train on whole trials; keep an explicit held-out model evaluation."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .signal import CSPLogPower, DIRECTION_LABELS, LABELS, LogBandPower, quality_error, timing_error


def load_trials(directory, max_ptp_uv=250.0):
    files = sorted(Path(directory).rglob("*.npz"))
    if not files:
        raise ValueError(f"No .npz trials found in {directory}")
    rows, metadata, rejected = [], None, []
    for path in files:
        with np.load(path, allow_pickle=False) as trial:
            x = trial["samples_uv"].astype(float)
            fs = float(trial["fs"])
            label = str(trial["label"])
            names = trial["channel_names"].tolist()
            channels = trial["channels"].tolist()
            subject = str(trial["subject"])
            session_id = str(trial["session_id"])
            if label not in LABELS:
                raise ValueError(f"Unknown label in {path}")
            if not np.isfinite(fs) or fs <= 64 or x.ndim != 2 or len(names) != x.shape[0] or len(channels) != x.shape[0]:
                raise ValueError(f"Invalid EEG metadata in {path}")
            current = dict(fs=fs, channel_names=names, channels=channels,
                           subject=subject, n_samples=x.shape[-1],
                           input_filtering=str(trial["input_filtering"]) if "input_filtering" in trial else "unknown",
                           cue_delay_seconds=float(trial["cue_delay_seconds"]) if "cue_delay_seconds" in trial else 1.0)
            if metadata is not None and current != metadata:
                raise ValueError(f"Subject, sample rate, channel order, input filtering, cue delay or window size differs in {path}")
            metadata = current
            error = quality_error(x, max_ptp_uv) or timing_error(trial["timestamps"], fs)
            if len(trial["timestamps"]) != x.shape[-1]:
                error = "timestamp count differs from sample count"
            if error:
                rejected.append(dict(file=str(path), reason=error))
                continue
            fingerprint = hashlib.sha256(x.tobytes()).hexdigest()
            rows.append((x, label, session_id, fingerprint, str(path)))
    if not rows:
        raise ValueError("All trials failed quality checks; fix the signal before training")
    return rows, metadata, rejected


def split_trials(rows, test_rows=None, required_labels=LABELS):
    """Never split a trial into windows or reuse a trial in both sets."""
    hashes = [r[3] for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate EEG trials in training data")
    if test_rows is not None:
        if set(hashes) & {r[3] for r in test_rows}:
            raise ValueError("Training and test data share EEG trials")
        if {r[2] for r in rows} & {r[2] for r in test_rows}:
            raise ValueError("Session holdout requires different recording sessions")
        if len({r[3] for r in test_rows}) != len(test_rows):
            raise ValueError("Duplicate EEG trials in test data")
        train, test = rows, test_rows
        kind = "session_holdout"
    else:
        labels = [r[1] for r in rows]
        if any(labels.count(label) < 5 for label in set(labels) | set(required_labels)):
            raise ValueError("Need at least 5 accepted trials per class; aim for 20-30")
        indices = np.arange(len(rows))
        a, b = train_test_split(indices, test_size=0.25, stratify=labels, random_state=149)
        train, test = [rows[i] for i in a], [rows[i] for i in b]
        kind = "trial_holdout_only"
    for subset in (train, test):
        if not set(required_labels).issubset(r[1] for r in subset):
            raise ValueError(f"Both train and test sets must contain {', '.join(required_labels)}")
    return train, test, kind


def train_model(directory, output, test_session=None, max_ptp_uv=250.0, threshold=0.8,
                method="csp", csp_components=2):
    output = Path(output)
    if output.suffix != ".pkl":
        raise ValueError("Use a .pkl filename for --output")
    if not 0.5 < threshold <= 1:
        raise ValueError("threshold must be in (0.5, 1]")
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise ValueError("Model or report already exists; choose a new --output")
    rows, metadata, rejected = load_trials(directory, max_ptp_uv)
    test_rows = None
    if test_session:
        test_rows, other_metadata, other_rejected = load_trials(test_session, max_ptp_uv)
        if metadata != other_metadata:
            raise ValueError("Test session must have the same subject, channels, rate, input filtering, cue delay and window size")
        rejected.extend(other_rejected)
    if method not in ("csp", "bandpower"):
        raise ValueError("Choose method csp or bandpower")
    labels = DIRECTION_LABELS if method == "csp" else LABELS
    train, test, kind = split_trials(rows, test_rows, labels)
    fit_rows = [r for r in train if r[1] in labels]
    if method == "csp":
        classifier = make_pipeline(
            CSPLogPower(fs=metadata["fs"], n_components=csp_components),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced", max_iter=1000),
        )
    else:
        classifier = make_pipeline(
            LogBandPower(fs=metadata["fs"]), StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=[1/3] * 3),
        )
    classifier.fit(np.stack([r[0] for r in fit_rows]), [r[1] for r in fit_rows])
    probabilities = classifier.predict_proba(np.stack([r[0] for r in test]))
    predictions = classifier.classes_[probabilities.argmax(axis=1)]
    truth = np.asarray([r[1] for r in test])
    accepted = (probabilities.max(axis=1) >= threshold) & (predictions != "rest")
    rest = truth == "rest"
    scored = np.isin(truth, labels)
    report = dict(
        method=method, evaluation=kind, labels=list(labels),
        train_trials=len(fit_rows), excluded_train_rest_trials=len(train) - len(fit_rows),
        test_trials=len(test), scored_test_trials=int(scored.sum()), test_rest_trials=int(rest.sum()),
        balanced_accuracy=float(balanced_accuracy_score(truth[scored], predictions[scored])),
        confusion_matrix=confusion_matrix(truth[scored], predictions[scored], labels=labels).tolist(),
        classification=classification_report(truth[scored], predictions[scored], labels=list(labels),
                                             output_dict=True, zero_division=0),
        candidate_threshold=threshold,
        candidate_window_coverage=float(accepted.mean()),
        candidate_window_accuracy=float((predictions[accepted] == truth[accepted]).mean()) if accepted.any() else None,
        rest_false_candidate_window_rate=float(accepted[rest].mean()) if rest.any() else None,
        rejected_trials=rejected, train_files=[r[4] for r in train], fit_files=[r[4] for r in fit_rows],
        test_files=[r[4] for r in test],
        note="Window scores are uncalibrated. Validate latched decisions and idle false activations live. "
             "Trial holdout alone does not establish generalization to another session. "
             + ("Binary CSP always predicts left or right; rest is NOT a learned class. "
                "Run only during a deliberate selection attempt." if method == "csp" else "Rest is a learned class."),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        pickle.dump(dict(version=1, method=method, classifier=classifier, metadata=metadata,
                         max_ptp_uv=max_ptp_uv, report=report), f)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_model(path):
    """Only load model files you trained or otherwise trust (pickle format)."""
    with Path(path).open("rb") as f:
        model = pickle.load(f)
    if model["version"] != 1:
        raise ValueError("Unsupported model version")
    return model
