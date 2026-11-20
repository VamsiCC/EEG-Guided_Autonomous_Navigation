# EEG destination selection

For the current setup and copyable collection commands, start with
[EEG setup and data collection](eeg_data_collection.md). For the project-wide
hardware, model, data, and robot integration design, see
[architecture.md](architecture.md). This document retains the detailed EEG
implementation and experiment notes.

## What the classifier is learning

Use **left-hand motor imagery** and **right-hand motor imagery** as the two
commands. Imagine the sensation of repeatedly opening and closing that hand,
without moving it. This is a reproducible task to investigate, not a decoder for
arbitrary thoughts or the meanings of the words "left" and "right". Its usefulness
for this wearer must be measured. Train each person separately.

Before a new calibration, briefly practice the task outside recording: slowly
open and close the left hand a few times, then stop and mentally rehearse how
that movement felt while keeping the real hand relaxed. Repeat for the right
hand. During each imagery cue, mentally repeat that hand's fingers curling and
releasing from a first-person perspective. Maintain the same strategy across
trials, with the actual hands still. REST means no hand imagery.

Kinesthetic and visual imagery can produce different EEG patterns. Emphasizing
movement sensations is a strategy to test, not a proven explanation for the
first model's poor performance or a guarantee of improvement. Preserve earlier
recordings and collect a new session when changing the imagery strategy; do not
retroactively relabel the old trials as if the new instructions were followed.

The path is: Cyton in 8-channel mode → OpenBCI GUI Time Series → LSL → Python →
quality checks → CSP log-power features → logistic regression → stable
destination candidate → confirmation → navigation to destination A or B.

This follows the **classifier architecture** in OpenBCI's motor imagery tutorial:
subject-specific binary CSP followed by logistic regression. The implementation
uses SciPy and scikit-learn, with its parameters described below. It does not
load a pretrained model or NeuroPype's pipeline. The original tutorial is now
deprecated; its accuracy does not establish performance for this implementation
or this wearer.

Keep eyes directed at the same central point for both tasks. Keep the hands,
head and jaw still. Blinking, looking left/right, clenching and actual movement
can produce an apparently successful classifier that is detecting artifacts.
The collector uses centrally displayed text in the same terminal and skips the
first half-second after each cue; this reduces cue transients but does not prove the
remaining signal is motor imagery.

## 1. Check electrode placement and the GUI

You have Cyton + Daisy hardware, but the current experiment uses **Cyton's first
8 channels**. The Daisy can remain physically attached. To switch in the GUI:

1. Stop the data stream and open the System Control Panel.
2. Stop the current session if needed to unlock its configuration.
3. Select CYTON (live), Serial (from Dongle), the same serial port, and
   **8 CHANNELS**.
4. Start the session and start the data stream. Expect channels 1–8.

Eight-channel Cyton over the usual USB radio connection is nominally **250 Hz**.
The CLI reads the LSL rate and verifies the timestamps; it does not hard-code
125 Hz or 250 Hz. Restoring 16-channel Daisy mode later normally changes the
rate to 125 Hz; use `--expected-channels 16` for that stream and collect a new
calibration. Existing 125 Hz/16-channel calibration cannot be mixed with the new
250 Hz/8-channel setup. Changing mode does not fix poor contacts on channels 1–8.

Find which physical channel wires connect to **C3, Cz and C4**. C3 and C4 cover
the left and right sensorimotor areas. Write down the real wiring, reference and
bias arrangement. Do not infer a scalp location from its channel number or wire
color. The model records your selected channel numbers and position names.
Start with a consistent set of clean central electrodes; adding noisy channels
can reduce performance. Do not automatically invert predictions because of the
contralateral motor response: supervised labels already define the output.

### Current motor-imagery wiring

The wearer confirmed **CH1 = C3** and **CH3 = C4** and supplied the OpenBCI
motor-imagery wiring diagram as the intended montage. Check the remaining
connections against the physical headset before recording with these labels.
The wearer subsequently confirmed the physical wiring as correct. That
confirmation is reported by the wearer; the cover obscured the board labels in
the supplied connection photo.

| Electrode | Cyton connection |
| --- | --- |
| C3 | CH1, bottom N1P pin |
| Cz | CH2, bottom N2P pin |
| C4 | CH3, bottom N3P pin |
| P3 | CH4, bottom N4P pin |
| Pz | CH5, bottom N5P pin |
| P4 | CH6, bottom N6P pin |
| O1 | CH7, bottom N7P pin |
| O2 | CH8, bottom N8P pin |
| Ear clip, reference | Bottom SRB pin (SRB2) |
| FPz, bias electrode | Bottom BIAS pin |

The tutorial's table repeats P4 at channel 3; its head diagram shows C4 at
channel 3, consistent with the wearer's confirmed wiring. FPz supplies BIAS in
this montage, so only one ear clip is used for the SRB2 reference. Both the FPz
electrode and the reference ear clip must make electrical contact with skin.
The FPz bias electrode is not one of the eight recorded EEG channels.

The wearer reported impedance readings of approximately 24 kOhm at CH1/C3
and 26 kOhm at CH3/C4. A subsequent Hardware Settings screenshot showed all
eight channels at gain x24, Input Type Normal, Bias Include Yes, SRB2 On and
SRB1 Off. These displayed settings match this montage; they do not verify
physical pin-row connections or establish that C3/C4 contain distinct scalp
signals. In the first calibration's accepted trials, median 7-30 Hz C3/C4
correlation was 0.993, which warrants an acquisition check before another long
calibration. Correlation alone does not identify a wiring or electrode fault.

Use the GUI's Cyton Signal impedance/contact checks before recording, then turn
the impedance test off. Check reference/bias contacts as well as scalp contacts.
Inspect the Time Series with the wearer still and then during a deliberate
blink. This helps establish that you have live biological data. A moving trace
alone does not establish usable EEG.

In the GUI:

1. Keep the Time Series widget open and start the board data stream.
2. Open the Networking widget and choose **LSL**.
3. Select **Time Series**. The current experiment uses Networking **Filters On**
   with the main GUI's **60 Hz notch** and **5–50 Hz bandpass**, following the
   large 60 Hz interference observed during setup. Use these same settings for
   collection and prediction. Python additionally applies its fixed 7–30 Hz
   model-band filtering.
4. Give the stream a recognizable name, such as `OpenBCI_EEG`, if configurable,
   and start the networking stream. Otherwise use its displayed default name.

Only one Time Series outlet is needed. In a GUI with Stream 1/2/3 columns, keep
Stream 1 as TimeSeries and set unused streams to None before restarting LSL.

Use **Time Series**, not the FFT or Band Power output expected by other code.
Each LSL sample must contain 8 simultaneous EEG values by default. The receiver rejects a
stream with the wrong count/rate and refuses to guess between multiple matches.
Leave the GUI in control of the serial port. Python connects to LSL, so it does
not compete with the GUI for the dongle.

## 2. Install and verify Python reception

Run commands from the repository root, using Python 3.10 or later:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eeg.txt
python -m src.eeg check
```

If several streams are active, specify the exact Time Series stream name:

```bash
python -m src.eeg check --stream-name OpenBCI_EEG
```

The stream name here is an example. Replace it if the GUI uses another name.
If `check` says `Found: []`, list every visible LSL outlet:

```bash
python -m src.eeg streams
```

This prints each outlet's exact name, type, channel count and sample rate. If it
finds the correct 8-channel Time Series stream under a type other than `EEG`,
use its exact `name` with `check --stream-name`. If the list is empty, check that
GUI Networking is set to LSL with Time Series selected and networking started;
the headset data stream and the networking stream are separate controls. If
discovery is still empty after that, inspect local LSL/network permission issues.

The check prints the actual stream name, rate, array shape and channel amplitude
statistics. With defaults, each check window should be `(8, 1000)` at approximately
250 Hz. A four-second window may take a little longer than four seconds while
the GUI delivers chunks. Flat channels, very large amplitudes, timestamp gaps,
missing data and old samples cause rejection or an explicit error.

The inlet applies LSL clock correction, timestamp dejittering and monotonic
processing together. It consumes a warm-up before any labeled cue or prediction
attempt: 15 seconds for collection and 5 for diagnostics/prediction by default.
`--warmup-seconds` overrides that duration. This is an initial settling period, not a guarantee that
LSL's estimator has fully converged; the remaining timing checks still apply.
`check` reports bad timing with the number of nonpositive intervals or gaps and
their sizes, while still printing amplitudes for diagnosis. A reported timing
failure is not acceptable calibration data. Collection, prediction and offline
training continue to reject those windows. Stale timestamps and a missing stream
also abort diagnostics. LSL timestamps alone cannot prove that no board samples
were lost; packet-counter/raw-recording checks are needed for stronger validation.

If collection repeatedly reports stale/unsynchronized timestamps or a wrong
observed rate, pause calibration and run this diagnostic first:

```bash
python -m src.eeg timing --stream-name obci_eeg1 --seconds 20
```

Keep the GUI's board and LSL streams running. No imagery is needed and this
command writes no recording files. It reads the same outlet through two inlets:
`clock_only` applies clock correction; `smoothed` additionally applies the
dejittering and monotonic processing used by collection. The report compares
sample counts, delivery rate, timestamp rate, gaps and timestamp ages. A positive
age means a timestamp in the past; a negative age means one in the future.
`first_second_median` and `last_second_median` help reveal drift during the test.
Age includes chunk delivery and diagnostic polling delay (up to about 100 ms),
so it is not a measurement of the headset's physical acquisition latency.

This comparison can distinguish a smoothing problem from a problem already
present in the incoming timestamps or delivery rate. It does not repair timestamps
or establish that the EEG is clean. Do not disable timing guards to collect a
calibration. Ordinary acquisition errors now include the signed age in seconds.

If timing passes but raw amplitudes are very large, inspect the frequency
content before attributing the failure to movement:

```bash
python -m src.eeg check --stream-name obci_eeg1 --windows 1 --spectral
```

This prints each channel's 7–30 Hz amplitude, strongest spectral peak, and power
fractions around 50 Hz, 60 Hz and below 7 Hz. The fractions use the 0.5 Hz–Nyquist
Welch spectrum; the 50/60 Hz bands are ±2 Hz. High 60 Hz power suggests mains
pickup; high slow power suggests drift or low-frequency artifacts. These are
clues, not proof of an electrode fault or of clean EEG. Filtered amplitude is
diagnostic only; the command does not alter the collection/training thresholds.

With GUI Networking Filters Off, Python receives unfiltered data. The GUI's
filtered Time Series/FFT display can consequently look much quieter. Compare
the same filtering conditions when judging amplitudes. Check reference/bias
and scalp contact, stabilize the leads, and move power supplies/cables away.
Testing a laptop on battery can help identify charger-related interference.

The CLI's `--input-filtering gui-5-50-notch60` default records the current GUI
filter configuration in every trial and fitted model. When deliberately using
Networking Filters Off, pass `--input-filtering unfiltered` for collection and
prediction. This is a declaration, not automatic inspection of the GUI. Training
rejects mixed declarations; prediction checks the declared setting against the
model. Check the actual GUI controls too. Older files with no declaration are
marked `unknown`; do not assume they match the current recording setup.

GUI Time Series amplitudes are interpreted as microvolts. `--unit v` is only
for a stream known to contain volts. A unit mistake invalidates both artifact
checks and training. Do not use that option just to make a bad signal pass.

On macOS/Linux, `pylsl` may need a separate native **liblsl** installation. If
the error says it cannot find liblsl, follow the official pylsl installation
instructions linked below and set `PYLSL_LIB` to the installed library's exact
path when necessary. On Apple Silicon, the native library must match the Python
architecture. The Python package alone may not supply that library.

These are basic amplitude/timing checks, not a substitute for electrode contact
inspection, plots or spectral analysis. The 250 µV peak-to-peak and 0.1 µV
standard-deviation thresholds are starting values, not universal physiological
criteria. A broken channel among the selected inputs can reject the whole trial.
Select known good channels rather than raising the thresholds to hide the issue.

## 3. Record your own labeled trials

**Verify electrode wiring before using this example.** The position names below
follow the current motor-imagery montage documented above. Replace names with
the actual positions attached to each numbered channel if the wiring changes.
If you have not mapped positions to wires, `CH1 CH2 ... CH8` can be used as
temporary names without inventing a scalp map. Verify that motor-area electrodes
are present and keep a wiring record before interpreting results as motor imagery.

```bash
python -m src.eeg collect \
  --stream-name obci_eeg1 \
  --subject participant01 \
  --output data/eeg/motor_session01 \
  --input-filtering gui-5-50-notch60 \
  --channels 1 2 3 4 5 6 7 8 \
  --channel-names C3 Cz C4 P3 Pz P4 O1 O2 \
  --warmup-seconds 20 \
  --trials-per-class 30
```

The collector presents one left/right/rest task per randomized block. Each task
has two seconds of relaxation, a half-second cue settling interval, and three
seconds of recording. This follows the tutorial's approximate 0.5–3.5 second
imagery epoch; exact cue/sample timestamps are saved. Thirty trials per class
produces 90 trials in about eight minutes plus startup and stream overhead.
Take breaks between sessions. `--seconds` and `--cue-delay` can change the epoch;
keep them consistent across recording sessions. At 250 Hz, the default training
trial is `(8, 750)` when all eight channels are selected.
For rest trials, relax while maintaining the same eye/head posture.

All selected channels must pass quality checks. If some are unused or unusable,
select only connected, clean channels with the corresponding names. The code
requires at least two inputs and two are enough to test the software path;
use a clean sensorimotor montage for the real CSP experiment, including central
coverage around C3/Cz/C4 when available. Extra noisy inputs can reduce performance.

Each completed trial is saved immediately as an `.npz` with microvolt samples,
LSL timestamps, class, anonymous subject ID, unique recording-session ID,
channel wiring/position names and cue time. Completed trials survive Ctrl+C.
The files store the received stream, which is GUI-filtered in the current setup.
Amplitude artifacts are retained and rejected during training.
The collector continues reading and discarding unlabeled samples throughout
instructions, rest intervals and cue delays. It consumes queued samples through
LSL's timestamp processor instead of sleeping and then flushing them, so pauses
do not skip the processor's observation history.

A timing failure rejects that attempt and presents the same task again after a
fresh relaxation interval. Only valid timing windows become training `.npz`
files. Three consecutive failures on a trial stop collection. The rejected
attempts and reasons are saved to `rejected_attempts.jsonl`; completed trials
remain saved. Repeated failures still require investigation of the actual
stream, including the GUI's packet-loss log, board/dongle connection and battery.
The timing tolerances have not been widened.

An existing empty output directory can be reused. If it contains any files,
choose a new session directory; completed recordings are never overwritten.
EEG recordings and fitted models are excluded from Git by default.

Record another session with exactly the same participant, channels, reference,
GUI settings and trial duration into `data/eeg/motor_session02`. Aim for
20–50 usable examples of each movement in the initial calibration, then measure
whether more sessions help. This is a starting experiment, not a sufficient-data
guarantee. Do not mix different participants into this subject-specific model.

## 4. Train and inspect validation

For an initial check with motor_session01 alone:

```bash
python -m src.eeg train \
  --data data/eeg/motor_session01 \
  --method csp \
  --output models/eeg/participant01_trial_csp.pkl
```

This holds out whole trials, stratified by class. It does **not** demonstrate
performance in a different recording session. For a stronger evaluation:

```bash
python -m src.eeg train \
  --data data/eeg/motor_session01 \
  --test-session data/eeg/motor_session02 \
  --method csp \
  --output models/eeg/participant01_session_csp.pkl
```

Keep test-session data out of `--data`. Choose preprocessing, channel selection
and thresholds on development data, then evaluate once on a new untouched
session. Repeatedly choosing settings based on motor_session02 makes it development
data; collect session03 for the final test. Every trial stays intact, duplicate
trials/shared session IDs across a session holdout are rejected, and learned
scaling/classification use training trials only. The saved model is the same
model that was evaluated; it is not silently refitted on test data.

The default `--method csp` uses:

1. A fourth-order 7–30 Hz Butterworth bandpass on each complete window.
2. Binary CSP fitted on **left/right training trials only**. Class covariance
   is averaged across trials and shrunk 10% toward a scaled identity. SciPy's
   generalized eigensolver supplies two spatial filters by default, alternating
   between largest/smallest eigenvalues. Class covariance is not trace-normalized.
3. Log mean-square power of the projected components.
4. Training-only feature standardization and balanced, regularized logistic
   regression (`C=1`, L-BFGS solver).

`--csp-components` changes the component count. Select it using development data,
not the final test session. The classifier architecture matches the tutorial;
the explicit filtering, covariance regularization and storage choices are this
project's implementation. No equality with NeuroPype's internal outputs is claimed.

**Binary CSP always predicts left or right.** Rest is not a learned class in
this method. The collector still records rest trials so the held-out rest windows
can measure false candidate rates. Training-rest windows are excluded from
fitting. If no rest data exists, idle evaluation is reported as unavailable.

The earlier three-class log-band-power + shrinkage LDA method remains available
as `--method bandpower`, using left/right/rest training trials. Compare direction
performance on the same left/right trials and assess idle behavior separately;
the two methods' headline balanced accuracies cover different class sets.
Neither method has been shown to outperform Sentdex or NeuroPype on your data.

Training writes a `.pkl` and a neighboring `.json` report. Inspect balanced
accuracy, the confusion matrix (rows=true, columns=predicted, with the class order
recorded in `labels`), per-class recall, accepted candidate coverage, and false
candidates on rest windows. For CSP, balanced accuracy and confusion cover
left/right trials only, with a 50% chance reference. Rest is evaluated separately.
The band-power method scores three classes, with a roughly 33% chance reference.
High accepted-window accuracy
with tiny coverage can still be unusable. Model scores are uncalibrated and a
score of 0.8 is not an established 80% chance of correctness.

If left/right discrimination stays near chance, revisit contact quality, central
electrodes, task consistency and motion/eye artifacts before model complexity.
Test on a different session and include long idle periods. Not everyone obtains
useful left/right motor imagery control with a given setup. If necessary, discuss
an alternative BCI task such as visually evoked selection as a change of scope;
do not relabel eye or muscle control as imagined-hand EEG.

Only load `.pkl` files you created or trust: Python pickle can execute code.
Do not load downloaded Sentdex models through this loader. Its format and input
representation are different.

## 5. Test live destination candidates

```bash
python -m src.eeg predict \
  --model models/eeg/participant01_session_csp.pkl
```

Match the headset wiring and GUI settings used during calibration. The CLI
automatically reuses the saved channel selection and window duration, and checks
the rate. It cannot physically verify which electrode you plugged into each
channel. Every inference window uses the same preprocessing as training.

Run this command only when deliberately selecting a destination. The binary
model has no learned idle detector: even relaxed EEG can receive a high left/right
score. The default attempt expires after 30 seconds (`--timeout-seconds` changes
this), emitting `selection_timeout` without a candidate. Cancellation/timeout
is not a third classifier class.

The gate needs two consecutive non-overlapping windows matching the training
duration, with the same non-rest result and score ≥0.8. Low scores and artifacts
reset the count; a learned rest result also resets the count with the optional
band-power model. No samples or stale data abort the attempt. The first candidate
latches and the process exits, so it cannot repeatedly select destinations.
With the default three-second calibration windows, expect at least six seconds
of evidence plus stream/inference overhead. These
windows are not assumed to be statistically independent or proof of correctness.

Example event:

```json
{"event": "destination_candidate", "destination": "left", "requires_confirmation": true}
```

Before integration, measure correct, wrong and rejected **whole selections**,
time to selection, and false activations per minute during idle periods. The
offline per-window report does not measure those online outcomes.

## 6. Connect the navigation boundary

On a laptop with ROS 2 installed and its environment sourced, `--ros` additionally
publishes one candidate as `std_msgs/msg/String` on `/bci/destination_candidate`.
It waits up to five seconds for a subscriber and reports if none is found.
For a transport check, run this listener first:

```bash
ros2 topic echo /bci/destination_candidate std_msgs/msg/String
```

Then run `predict` with the same model and `--ros`. This only demonstrates the
candidate message transport. A subscriber receiving it does not establish that
the robot has accepted or reached a goal.

The existing `plannedcntrl` node plans a cone-avoidance route with a fixed forward
goal. It currently has no EEG subscriber or destination A/B mapping. Add the
mapping after deciding the two destination coordinates and whether the project
will use that controller or Nav2. Keep one owner of robot velocity commands.

The navigation integration should use this sequence:

1. A deliberate selection mode starts one EEG selection attempt.
2. A candidate displays destination A (`left`) or B (`right`).
3. An explicit confirmation commits one destination goal. A keyboard/button
   is sufficient for the first integration; EEG confirmation is a separate task.
4. While navigating, subsequent EEG candidates are ignored.
5. Arrival/cancellation returns to an idle state. A new explicit selection
   attempt rearms the classifier. An independent emergency stop cancels motion.

The current EEG code produces candidates only; it does not implement this
navigation state machine or an emergency stop and never sends velocity commands.
Do not run the current auto-planning cone controller as a passive EEG listener:
it starts planning when enough cone messages arrive, independently of EEG.

## Why the Sentdex model is a reference rather than a drop-in model

The repository describes 16-channel **FFT** frames with 60 frequency values per
channel, collected from its author's setup. Its sample loader reshapes inputs to
`(batch, 16, 60)`. A raw GUI Time Series sample instead contains one voltage for
each channel at a single time. The included code and reported results do not
establish transfer to another wearer's electrodes, task, reference or session.
It is useful as an experiment, but copying its network weights does not resolve
those measurement and validation problems.

## References

- [Sentdex BCI README and input format](https://github.com/Sentdex/BCI)
- [Sentdex training code](https://github.com/Sentdex/BCI/blob/master/training.py)
- [OpenBCI Networking and signal check widgets](https://docs.openbci.com/Software/OpenBCISoftware/GUIWidgets/)
- [OpenBCI board sampling rates](https://docs.openbci.com/FAQ/HowProductsGoTogether/)
- [OpenBCI motor imagery tutorial: CSP + logistic regression](https://docs.openbci.com/Deprecated/MotorImagery/)
- [Neuper et al.: kinesthetic and visual-motor imagery in single-trial EEG](https://pubmed.ncbi.nlm.nih.gov/16236487/)
- [Cyton SDK: 8-channel mode with Daisy attached](https://docs.openbci.com/Cyton/CytonSDK/#select-maximum-channel-number)
- [MNE CSP formulation and logistic regression example](https://mne.tools/stable/auto_tutorials/machine-learning/50_decoding.html#common-spatial-pattern)
- [SciPy generalized eigensolver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html)
- [scikit-learn logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [MNE motor imagery CSP + LDA example](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)
- [scikit-learn LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [pylsl installation and native library setup](https://github.com/labstreaminglayer/pylsl)
- [LSL timing and clock synchronization](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html)

## Development checks

```bash
python -m pytest tests/test_eeg.py
```

The tests use synthetic signals and a simulated inlet. They establish software
behavior only, not biological decoding accuracy or real headset/ROS connectivity.
