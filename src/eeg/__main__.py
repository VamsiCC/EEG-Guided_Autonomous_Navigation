"""Run from the repository root: python3 -m src.eeg --help."""

import argparse
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .model import load_model, train_model
from .signal import DecisionGate, LABELS, quality_error, spectral_diagnostics, timing_error
from .stream import EEGStream, EEGTimingError, diagnose_timing, list_streams


def check(args, stream):
    print(f"Stream: {stream.name}, nominal {stream.fs:g} Hz, channels {stream.channels}")
    for _ in range(args.windows):
        # Diagnostics report suspect timing; collection and prediction retain
        # strict validation and never train or emit a candidate from it.
        x, ts = stream.read_window(args.seconds, strict_timing=False)
        duration = ts[-1] - ts[0]
        rate_text = f"{(len(ts) - 1) / duration:.1f} Hz" if duration > 0 else "undefined rate"
        print(f"Observed {rate_text}; {x.shape}; quality: {quality_error(x, args.max_ptp_uv) or 'basic checks passed'}")
        print(f"  Timing: {timing_error(ts, stream.fs) or 'basic checks passed'}")
        diagnostics = spectral_diagnostics(x, stream.fs) if args.spectral and np.isfinite(x).all() else None
        for index, (ch, values) in enumerate(zip(stream.channels, x)):
            print(f"  CH{ch}: std={values.std():.2f} uV, peak-to-peak={np.ptp(values):.2f} uV")
            if diagnostics is not None:
                d = diagnostics[index]
                peak_text = f"{d['peak_hz']:.1f} Hz" if d["peak_hz"] is not None else "none"
                print(f"    7-30 Hz: std={d['band_7_30_std_uv']:.2f} uV, "
                      f"peak-to-peak={d['band_7_30_ptp_uv']:.2f} uV; strongest peak={peak_text}; "
                      f"50 Hz={100*d['line_50_fraction']:.1f}%, "
                      f"60 Hz={100*d['line_60_fraction']:.1f}%, below 7 Hz={100*d['below_7_fraction']:.1f}%")


def collect(args, stream):
    if not args.channel_names or len(args.channel_names) != len(stream.channels):
        raise ValueError("Supply --channel-names for every selected channel, in actual electrode wiring order")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Recording folder {output} contains files. Choose a new --output to preserve them.")
    session_id = str(uuid.uuid4())
    rng = np.random.default_rng(args.seed)
    # Each block contains every class once, in a shuffled order.
    schedule = [str(label) for _ in range(args.trials_per_class) for label in rng.permutation(LABELS)]
    print("Keep eyes centered and hands/jaw still. Keep both real hands relaxed.")
    print("Imagine the FEELING of the cued hand's fingers slowly curling and releasing, from inside that hand.")
    print(f"Declared input filtering: {args.input_filtering}. Keep the GUI settings fixed.")
    if any(name.upper() == f"CH{ch}" for name, ch in zip(args.channel_names, stream.channels)):
        print("CH labels identify wires, not scalp positions. Verify C3/C4 coverage and record your wiring.")
    print("LEFT = left hand; RIGHT = right hand; REST = relax without imagery.")
    print("Starting in 3 seconds. Ctrl+C preserves completed trials.")
    stream.wait(3)
    rejected_attempts = 0
    for index, label in enumerate(schedule):
        for attempt in range(1, 4):
            print(f"\n{index + 1}/{len(schedule)}: RELAX for {args.rest_seconds:g}s (attempt {attempt}/3)", flush=True)
            try:
                stream.wait(args.rest_seconds)
                instruction = ("Relax both hands; no imagery." if label == "rest" else
                               f"FEEL your {label.upper()} fingers curl and release; keep the hand still.")
                print(f"TASK: {label.upper()} - {instruction} Keep this task until DONE.", flush=True)
                cue_time = stream.lsl.local_clock()
                # Keep consuming unlabeled samples during the cue delay.
                stream.wait(args.cue_delay)
                x, ts = stream.read_window(args.seconds)
            except (EEGTimingError, TimeoutError) as exc:
                rejected_attempts += 1
                with (output / "rejected_attempts.jsonl").open("a") as f:
                    f.write(json.dumps(dict(trial=index, label=label, attempt=attempt,
                                           session_id=session_id, reason=str(exc))) + "\n")
                print(f"DONE. Timing rejected: {exc}. This attempt was not saved as training data.", flush=True)
                if attempt == 3:
                    raise RuntimeError("Three timing failures on one trial. Completed trials are saved; "
                                       "run 'python -m src.eeg timing --stream-name "
                                       f"{stream.name}' to diagnose timestamps before a new session.") from exc
                print("Relax; the task will be cued again.", flush=True)
                continue
            break
        path = output / f"trial_{index:03d}_{label}.npz"
        with path.open("xb") as f:
            np.savez_compressed(
                f, samples_uv=x, timestamps=ts, fs=stream.fs,
                label=label, subject=args.subject, session_id=session_id,
                channel_names=np.asarray(args.channel_names), channels=np.asarray(stream.channels),
                cue_lsl_time=cue_time, cue_delay_seconds=args.cue_delay, stream_name=stream.name,
                input_filtering=args.input_filtering, acquisition_attempt=attempt,
            )
        print(f"DONE. Saved {path.name}; {quality_error(x, args.max_ptp_uv) or 'basic checks passed'}", flush=True)
    print(f"Recording finished: {len(schedule)} trials saved to {output}; {rejected_attempts} timing attempts rejected.")


def publish_candidate(label):
    """Optional ROS boundary: destination suggestion only; never cmd_vel."""
    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError as exc:
        raise RuntimeError("Source your ROS 2 environment to use --ros") from exc
    rclpy.init()
    node = rclpy.create_node("eeg_destination_candidate")
    try:
        pub = node.create_publisher(String, "/bci/destination_candidate", 10)
        deadline = time.monotonic() + 5
        while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            raise RuntimeError("No subscriber for /bci/destination_candidate; candidate was not delivered")
        pub.publish(String(data=label))
        rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def predict(args, stream, model):
    metadata = model["metadata"]
    if stream.fs != metadata["fs"] or stream.channels != metadata["channels"]:
        raise ValueError("Live channel selection and sample rate must match the trained model")
    expected_filtering = metadata.get("input_filtering", "unknown")
    if expected_filtering != "unknown" and expected_filtering != args.input_filtering:
        raise ValueError("Declared live input filtering differs from the calibration. Match GUI settings and --input-filtering.")
    gate = DecisionGate(args.threshold, args.consecutive)
    seconds = metadata["n_samples"] / stream.fs
    print(f"Subject: {metadata['subject']}; electrode order: {metadata['channel_names']}")
    print(f"One selection attempt, up to {args.timeout_seconds:g}s. Ctrl+C cancels.")
    print("Model scores are uncalibrated. Output is a candidate requiring navigation confirmation.")
    if set(model["classifier"].classes_) == {"left", "right"}:
        print("Binary classifier: no learned rest state. Begin left/right imagery now.")
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        if time.monotonic() + seconds > deadline:
            print(json.dumps(dict(event="selection_timeout", destination=None)), flush=True)
            return
        x, _ = stream.read_window(seconds)
        if time.monotonic() > deadline:
            print(json.dumps(dict(event="selection_timeout", destination=None)), flush=True)
            return
        error = quality_error(x, model["max_ptp_uv"])
        if error:
            gate.reset()
            print(f"Rejected: {error}", flush=True)
            continue
        classifier = model["classifier"]
        probabilities = classifier.predict_proba(x[np.newaxis])[0]
        index = int(probabilities.argmax())
        label, confidence = str(classifier.classes_[index]), float(probabilities[index])
        print(json.dumps(dict(prediction=label, score=round(confidence, 3))), flush=True)
        candidate = gate.update(label, confidence)
        if candidate:
            print(json.dumps(dict(event="destination_candidate", destination=candidate,
                                  requires_confirmation=True)), flush=True)
            if args.ros:
                publish_candidate(candidate)
            return


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be positive and finite")
    return value


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("streams", help="List all visible LSL streams, regardless of type")
    p.add_argument("--timeout", type=positive_float, default=5.0)
    p = sub.add_parser("timing", help="Compare clock-only and smoothed timestamps without recording trials")
    p.add_argument("--stream-name", help="Exact GUI LSL Time Series stream name")
    p.add_argument("--expected-channels", type=int, choices=(8, 16), default=8)
    p.add_argument("--seconds", type=positive_float, default=20.0)
    for name in ("check", "collect", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--stream-name", help="Exact GUI LSL Time Series stream name")
        p.add_argument("--expected-channels", type=int, choices=(8, 16), default=8,
                       help="GUI Time Series channel count (default: 8)")
        p.add_argument("--channels", type=positive_int, nargs="+", help="1-based channels in wiring order")
        p.add_argument("--unit", choices=("uv", "v"), default="uv", help="GUI Time Series uses uv")
        p.add_argument("--input-filtering", choices=("gui-5-50-notch60", "unfiltered"),
                       default="gui-5-50-notch60",
                       help="Declare GUI preprocessing; saved with calibration (default: gui-5-50-notch60)")
        p.add_argument("--warmup-seconds", type=positive_float, default=15.0 if name == "collect" else 5.0,
                       help="Consume samples before starting (default: 15 for collect, 5 otherwise)")
        if name != "predict":
            p.add_argument("--seconds", type=positive_float, default=3.0 if name == "collect" else 4.0)
            p.add_argument("--max-ptp-uv", type=positive_float, default=250.0)
    p = sub.choices["check"]
    p.add_argument("--windows", type=positive_int, default=3)
    p.add_argument("--spectral", action="store_true",
                   help="Report 7-30 Hz amplitudes and 50/60 Hz or slow-drift power fractions")
    p = sub.choices["collect"]
    p.add_argument("--output", required=True, help="New recording-session directory")
    p.add_argument("--subject", required=True, help="Anonymous participant ID")
    p.add_argument("--channel-names", nargs="+", required=True,
                   help="Actual scalp positions in channel order; CH1 etc may mark positions not yet mapped")
    p.add_argument("--trials-per-class", type=positive_int, default=20)
    p.add_argument("--rest-seconds", type=positive_float, default=2.0)
    p.add_argument("--cue-delay", type=positive_float, default=0.5,
                   help="Seconds from imagery cue to start of recorded epoch (default: 0.5)")
    p.add_argument("--seed", type=int, default=None)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--test-session", help="Separate, untouched recording-session directory")
    p.add_argument("--output", required=True, help="New .pkl model path")
    p.add_argument("--max-ptp-uv", type=positive_float, default=250.0)
    p.add_argument("--threshold", type=positive_float, default=0.8)
    p.add_argument("--method", choices=("csp", "bandpower"), default="csp",
                   help="csp: binary CSP + logistic regression; bandpower: 3-class LDA")
    p.add_argument("--csp-components", type=positive_int, default=2)
    p = sub.choices["predict"]
    p.add_argument("--model", required=True, help="A trusted, locally trained .pkl model")
    p.add_argument("--threshold", type=positive_float, default=0.8)
    p.add_argument("--consecutive", type=positive_int, default=2)
    p.add_argument("--timeout-seconds", type=positive_float, default=30.0,
                   help="Stop the selection attempt after this duration, without a candidate")
    p.add_argument("--ros", action="store_true", help="Publish one candidate to /bci/destination_candidate")
    args = parser.parse_args()
    stream = None
    try:
        if args.command == "streams":
            streams = list_streams(args.timeout)
            print(json.dumps(streams, indent=2))
            if not streams:
                print("No LSL streams discovered. Check the GUI Networking widget: "
                      "LSL protocol, Time Series data, networking started and headset streaming.")
            return
        if args.command == "timing":
            print(json.dumps(diagnose_timing(args.stream_name, args.expected_channels, args.seconds), indent=2))
            return
        if hasattr(args, "threshold") and not 0.5 < args.threshold <= 1:
            raise ValueError("--threshold must be in (0.5, 1]")
        if args.command == "train":
            report = train_model(args.data, args.output, args.test_session, args.max_ptp_uv,
                                 args.threshold, args.method, args.csp_components)
            print(json.dumps(report, indent=2))
            return
        model = load_model(args.model) if args.command == "predict" else None
        channels = args.channels or (model["metadata"]["channels"] if model else None)
        stream = EEGStream(args.stream_name, args.expected_channels, channels, args.unit,
                           warmup_seconds=args.warmup_seconds)
        if args.command == "check":
            check(args, stream)
        elif args.command == "collect":
            collect(args, stream)
        else:
            predict(args, stream, model)
    except KeyboardInterrupt:
        print("\nStopped. Completed trials remain saved.")
    except (ValueError, RuntimeError, TimeoutError, OSError) as exc:
        parser.exit(1, f"EEG error: {exc}\n")
    finally:
        if stream is not None:
            stream.close()


if __name__ == "__main__":
    main()
