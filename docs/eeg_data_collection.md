# EEG setup and data collection

This is the practical guide for connecting OpenBCI, reading EEG in Python,
recording labeled trials, and training the project's classifier. Run terminal
commands from the repository root. For the system design, see
[architecture.md](architecture.md); for additional implementation detail, see
[eeg_pipeline.md](eeg_pipeline.md).

The current experiment uses **left-hand versus right-hand motor imagery**.
Imagine the sensation of opening and closing the cued hand while the actual
hand stays relaxed and still. REST means relaxing without hand imagery.

## 1. Know the two channel configurations

The GUI streams **8 channels at nominally 250 Hz** in both configurations below.
Python chooses which of those channels to save.

| Configuration | Python channels | Position names, in the same order | Three-second trial shape |
| --- | --- | --- | --- |
| Full montage | 1 2 3 4 5 6 7 8 | C3 Cz C4 P3 Pz P4 O1 O2 | `(8, 750)` |
| Focused motor pair | 1 3 | C3 C4 | `(2, 750)` |

Use the focused pair for the workflow in this guide. It avoids rejecting a
trial solely because another channel, such as Cz, is noisy. It does not
guarantee useful left/right classification. Keep the GUI at **8 CHANNELS**;
`--channels 1 3` selects inputs in Python and does not put the board into a
two-channel acquisition mode.

## 2. Connect and wear the hardware

Use the battery-powered Cyton, its USB radio dongle, and the headset. The Daisy
module can remain attached while the GUI is configured for 8-channel Cyton
acquisition. Use the **Cyton's** inputs for channels 1–8. Daisy inputs belong to
the separate 16-channel configuration.

The project's intended montage is:

| Electrode | Cyton connection | Role |
| --- | --- | --- |
| C3 | Bottom N1P pin, CH1 | Left motor-area recording |
| Cz | Bottom N2P pin, CH2 | Central recording |
| C4 | Bottom N3P pin, CH3 | Right motor-area recording |
| P3 | Bottom N4P pin, CH4 | Parietal recording |
| Pz | Bottom N5P pin, CH5 | Parietal recording |
| P4 | Bottom N6P pin, CH6 | Parietal recording |
| O1 | Bottom N7P pin, CH7 | Occipital recording |
| O2 | Bottom N8P pin, CH8 | Occipital recording |
| Ear clip | Bottom SRB pin, **SRB2** | Shared reference |
| FPz electrode | Bottom **BIAS** pin | Body bias connection |

This setup uses **one reference ear clip plus the FPz bias electrode**. Both
must make electrical contact with skin. FPz is not one of the eight recorded
channels. Verify connections using the board labels rather than wire colors
or the apparent height of a connector in a photo.

The motor-imagery tutorial's table repeats P4 at CH3. Its head diagram shows
**C4 at CH3**, matching the wiring used here.

Place scalp electrodes through the hair so their tips contact the scalp
comfortably. Secure the reference clip and FPz contact. Make physical wiring
changes with the board powered off.

## 3. Open and configure OpenBCI GUI

On the current Mac, open `/Applications/OpenBCI_GUI.app`, or run:

```bash
open -a /Applications/OpenBCI_GUI.app
```

The setup used during collection was **OpenBCI GUI v5.0.8**. Newer GUI versions
can place controls differently.

In the **System Control Panel**:

1. Choose **CYTON (live)**.
2. Choose **Serial (from Dongle)**.
3. Select the dongle's serial port. On macOS it can look like
   `/dev/cu.usbserial-...`; use the port actually shown on your laptop.
4. Select **8 CHANNELS**.
5. Click **START SESSION**.

Open the **Time Series** widget. With data streaming stopped, open
**Hardware Settings** and verify the following EEG configuration:

| Setting | Value |
| --- | --- |
| PGA Gain | x24 |
| Input Type | Normal |
| Bias Include | Yes |
| SRB2 | On |
| SRB1 | Off |

If settings were changed, click **Send**, then return to Time Series. Channel
numbers are hardware inputs; GUI head-map defaults do not establish their
physical scalp positions.

### Check contacts, then turn impedance testing off

In this GUI version, the **Ω** button beside a channel enables its impedance
test. Check one channel at a time, note the kOhm reading, and turn the test off
before checking the next channel. Newer versions have a Cyton Signal widget.

**Turn every impedance test OFF before recording EEG.** The test injects a
signal. A channel displaying a kOhm measurement is still being tested; normal
streaming displays its signal statistics in µVrms. A low or balanced impedance
reading alone does not prove that the EEG is useful for motor imagery.

### Set the GUI filters and start the board stream

In the main toolbar, immediately to the right of the data-stream button, use:

- **Notch: 60 Hz**.
- **BP Filt: 5–50 Hz**.

Click **Start Data Stream**. When running, that button reads
**Stop Data Stream** and traces should update. Keep the Time Series widget open.

## 4. Start LSL networking

The board stream and the networking stream have separate start buttons.
Starting one does not establish that the other is running.

Open the **Networking** widget alongside Time Series and configure:

| Networking field | Value |
| --- | --- |
| Protocol | LSL |
| Stream 1 Data Type | TimeSeries |
| Stream 1 Name | obci_eeg1 |
| Stream 1 Type | EEG |
| Stream 1 Filters | **On** |
| Stream 2 and Stream 3 Data Type | None |

Click **Start LSL Stream**. When running, it reads **Stop LSL Stream**.

Select TimeSeries rather than FFT or BandPower. Python expects one simultaneous
voltage value per streamed EEG channel. With Networking Filters On, it receives
the GUI-filtered voltages. Keep these settings fixed between calibration,
validation, and prediction.

The GUI owns the dongle's serial connection. Python reads the GUI's LSL outlet
and does not open the serial port itself.

## 5. Prepare Python

Open the repository folder:

```bash
cd /Users/XXXX/EEG-Guided_Autonomous_Navigation
```

Replace `XXXX` with your macOS username. If you saved the repository elsewhere,
use that folder's path instead. Create the environment on first setup:

```bash
python3 -m venv .venv
```

Activate it in each new terminal and install the EEG dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements-eeg.txt
python -m src.eeg --help
```

The dependencies are NumPy, SciPy, scikit-learn, and pylsl. The local collection
environment used Python 3.13. ROS 2 and NeuroPype are not required for these EEG
collection or offline training commands.

Copy commands without the terminal prompt. For example, use `python3`, not
`:python3` or `❯ python3`.

## 6. Verify reception before collecting

Discover outlets:

```bash
python -m src.eeg streams
```

The expected outlet has name `obci_eeg1`, type `EEG`, 8 channels, and nominal
rate 250 Hz. If its name differs, substitute the exact discovered name in the
commands below.

Check the focused motor pair:

```bash
python -m src.eeg check \
  --stream-name obci_eeg1 \
  --channels 1 3 \
  --warmup-seconds 20 \
  --windows 1 \
  --spectral
```

The default check window is four seconds, so expect `(2, 1000)` with an observed
rate near 250 Hz. Inspect both the quality and timing messages. `check` can print
timing problems without treating the diagnostic command's exit status as a
validation pass.

To inspect every channel, omit `--channels 1 3`:

```bash
python -m src.eeg check --stream-name obci_eeg1 --windows 1 --spectral
```

The current basic quality limits are peak-to-peak amplitude no greater than
250 µV and standard deviation at least 0.1 µV on every selected channel, with
finite data and valid timing. These are screening limits, not proof of clean
EEG. Spectral output helps distinguish large 50/60 Hz pickup from slower drift
or activity in the model's 7–30 Hz band.

For timestamp errors, use:

```bash
python -m src.eeg timing --stream-name obci_eeg1 --seconds 20
```

This compares clock-corrected timestamps with and without smoothing, without
saving training files. Positive timestamp age means past; negative means
future. Clock correction aligns clock origins and is not a measurement of
headset latency. Collection uses clock correction, dejittering, and monotonic
processing together. The 20-second warm-up in this guide is a starting period;
the receiver still rejects invalid timing afterward.

If a biological-signal check is needed, inspect CH7/O1 and CH8/O2 in the FFT
while alternating ten seconds eyes open and ten seconds gently closed. Repeat
three times with Ω tests off and head still. A teammate can watch for a
repeatable alpha-band increase near 8–12 Hz while the eyes are closed. This
supports EEG reception but does not validate left/right decoding; absence of a
clear effect alone does not identify a hardware fault.

## 7. Practice the imagery task before recording

Slowly open and close the left hand a few times outside recording. Stop, relax
the actual hand, and mentally rehearse how that movement felt. Repeat for the
right hand.

During collection:

| Cue | Task |
| --- | --- |
| LEFT | Imagine the sensation of left fingers curling and releasing. |
| RIGHT | Imagine the sensation of right fingers curling and releasing. |
| REST | Relax without hand imagery. |

Keep both actual hands, the head, and jaw still; keep gaze centered for all
conditions. Continue the cued task until **DONE**. Imagery strategy is an
experimental variable, not a guarantee of accuracy.

## 8. Collect training and separate validation recordings

Each output folder must be new or empty. The folders ending in `01` were
already recorded. The examples below use `02` for a future collection; choose
another unused suffix if needed. Keep the same anonymous subject ID only for
the same participant.

### Training: 30 trials per class, 90 total

```bash
python -m src.eeg collect \
  --stream-name obci_eeg1 \
  --subject participant01 \
  --output data/eeg/motor_C3_C4_session02 \
  --input-filtering gui-5-50-notch60 \
  --channels 1 3 \
  --channel-names C3 C4 \
  --warmup-seconds 20 \
  --trials-per-class 30
```

### Validation: 10 trials per class, 30 total

Keep the same headset placement, reference, bias, filters, and imagery strategy:

```bash
python -m src.eeg collect \
  --stream-name obci_eeg1 \
  --subject participant01 \
  --output data/eeg/motor_C3_C4_validation02 \
  --input-filtering gui-5-50-notch60 \
  --channels 1 3 \
  --channel-names C3 C4 \
  --warmup-seconds 20 \
  --trials-per-class 10
```

The collector randomizes one LEFT, RIGHT, and REST task within each block.
Defaults are two seconds RELAX, a 0.5-second settling interval after the cue,
and three seconds of recorded imagery. Delayed chunks can shift the actual
epoch start slightly; cue and sample timestamps are saved. Allow roughly
8–10 minutes for 90 trials and another 3–4 minutes for 30 trials, with extra
time for retries.

Python continues consuming samples during the warm-up, rest, and cue intervals.
A timing failure rejects that attempt and retries the same task, up to three
attempts. Repeated failures stop collection; completed trials remain. Reasons
are recorded in `rejected_attempts.jsonl`.

**A saved trial can still fail amplitude checks.** Those trials are retained
for inspection and excluded during training. `0 timing attempts rejected`
does not mean every saved trial passed signal-quality checks.

To stop early, use Ctrl+C during a RELAX interval. Keep completed files and use
a new folder for a subsequent run; the CLI does not resume a partial session
into a populated directory.

For full-montage collection, replace the channel arguments with:

```text
--channels 1 2 3 4 5 6 7 8
--channel-names C3 Cz C4 P3 Pz P4 O1 O2
```

Those two lines are argument replacements in the collection command, not
standalone shell commands. A full-montage model and focused-pair model have
different input shapes and cannot be interchanged.

## 9. Inspect and train offline

After collection, the headset and GUI can be switched off. The NPZ files contain
the samples, labels, and metadata needed for offline analysis.

Inspect one of the already-recorded focused trials:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

path = sorted(Path('data/eeg/motor_C3_C4_session01').glob('trial_*.npz'))[0]
with np.load(path, allow_pickle=False) as trial:
    print('File:', path)
    print('Shape:', trial['samples_uv'].shape)
    print('Label:', str(trial['label']))
    print('Rate:', float(trial['fs']))
    print('Channels:', trial['channels'].tolist())
    print('Positions:', trial['channel_names'].tolist())
PY
```

For the newly collected `02` recordings, train once using the separate recording
for evaluation:

```bash
python -m src.eeg train \
  --data data/eeg/motor_C3_C4_session02 \
  --test-session data/eeg/motor_C3_C4_validation02 \
  --method csp \
  --csp-components 2 \
  --output models/eeg/participant01_csp_C3_C4_session02.pkl
```

The command writes a model `.pkl` and a JSON evaluation report. It rejects
incompatible metadata and duplicate/shared trials between training and test.
Existing model/report paths are protected; choose a new output name for a rerun.
Inspect `rejected_trials`, the confusion matrix, balanced accuracy, class
recall, rest false-candidate rate, and candidate coverage together.

To reproduce training on the data already collected on November 19, use:

```bash
python -m src.eeg train \
  --data data/eeg/motor_C3_C4_session01 \
  --test-session data/eeg/motor_C3_C4_validation01 \
  --method csp \
  --csp-components 2 \
  --output models/eeg/participant01_csp_C3_C4_reproduction01.pkl
```

That validation recording has already been evaluated. Reproducing its result
does not create a new blind test. For subsequent tuning, use training-data
cross-validation and reserve another prospective recording for final evaluation.

The default CSP model learns **LEFT versus RIGHT**. REST is excluded from its
fit and used to assess false activations. The alternative `--method bandpower`
uses band-power features and a three-class shrinkage-LDA classifier.

### Existing focused data snapshot

| Recording folder | Saved | Accepted LEFT | Accepted RIGHT | Accepted REST |
| --- | ---: | ---: | ---: | ---: |
| `data/eeg/motor_C3_C4_session01` | 90 | 21 | 21 | 17 |
| `data/eeg/motor_C3_C4_validation01` | 30 | 8 | 9 | 8 |

All 120 files were readable. The fixed CSP model trained on 42 direction trials
and scored 8/17 direction trials correctly in the separate validation recording
(47.1% accuracy, 50% balanced accuracy), predicting LEFT for every direction
trial. This is a saved baseline, not reliable destination decoding.

The saved artifacts are:

- `models/eeg/participant01_csp_C3_C4_session01.pkl` — trained model.
- `models/eeg/participant01_csp_C3_C4_session01.json` — evaluation and file lists.
- `models/eeg/participant01_C3_C4_data_manifest.json` — counts and file checksums.

These data/model folders are excluded from Git. Preserve the original files and
manifest; do not assume pushing the source repository backs them up.

## 10. Read live model output during software validation

The saved model is currently an experimental baseline. To observe its Python
output without publishing to ROS, restart the same GUI/LSL setup and run:

```bash
python -m src.eeg predict \
  --stream-name obci_eeg1 \
  --model models/eeg/participant01_csp_C3_C4_session01.pkl \
  --input-filtering gui-5-50-notch60 \
  --warmup-seconds 20 \
  --threshold 0.8 \
  --consecutive 2 \
  --timeout-seconds 30
```

The model supplies channel selection `[1, 3]` automatically. It must match the
physical wiring used for training. The run emits at most one destination
candidate, or times out. Connection/warm-up occurs before the active selection
interval. A score of 0.8 is an uncalibrated model score, not a demonstrated 80%
chance of correct intent.

Keep this workflow separate from robot motion. The optional `--ros` flag sends
only a candidate string; destination confirmation and navigation integration
are still to be implemented. See [the architecture](architecture.md).

## 11. Troubleshooting

| Symptom | Action |
| --- | --- |
| `zsh: command not found: :python3` | Remove the leading colon; use `python3`. |
| No LSL streams / `Found: []` | Verify board power/session, Start Data Stream, and Start LSL Stream. Keep Time Series open. Restart networking if needed, then run `streams`. |
| Several `obci_eeg` outlets | Use Stream 1 only, set Streams 2/3 to None, and pass its exact name. |
| Outlet has wrong shape/type | Choose TimeSeries and 8-channel mode. Select Python channels separately. |
| Nonmonotonic, stale, or wrong-rate timestamps | Run `timing`, inspect its ages/rates and GUI packet-loss log. Keep strict collection checks; do not accept invalid timing by widening limits. |
| Large amplitude | Confirm Ω tests are off; inspect scalp/reference/BIAS contacts, cable movement, and interference. Confirm Networking Filters On. |
| Flat, railed, or nearly identical channels | Check enabled inputs, physical positions, and contact; confirm Normal input mode and the intended reference. Similarity alone does not diagnose a particular fault. |
| Folder/model already exists | Use a new suffix. Preserve completed trials and prior model reports. |
| Prediction fails metadata checks | Match sample rate, selected channels/order, and declared filtering to the calibration. Collect a new calibration when the setup changes. |
| Basic checks pass but accuracy is poor | Inspect task consistency, artifacts, class balance, signal response, and held-out performance. Passing screening limits does not establish motor-imagery decoding. |

## 12. End the recording session

1. Wait for collection to finish, or stop during RELAX with Ctrl+C.
2. Confirm the expected files and completion summary are present.
3. Click **Stop LSL Stream**, then **Stop Data Stream** in the GUI.
4. Power off the Cyton and remove the headset.

Further inspection and model training use the saved data and do not require the
headset to remain on.

## References

- [OpenBCI motor-imagery tutorial](https://docs.openbci.com/Deprecated/MotorImagery/) — source of the montage/task/model approach; now deprecated.
- [OpenBCI GUI widgets](https://docs.openbci.com/Software/OpenBCISoftware/GUIWidgets/) — networking, hardware settings, and impedance tools.
- [OpenBCI EEG setup](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EEGSetup/) — reference/input connections and alpha response.
- [Cyton SDK](https://docs.openbci.com/Cyton/CytonSDK/) — channel configuration and 8/16-channel operation.
- [LSL time synchronization](https://labstreaminglayer.readthedocs.io/info/time_synchronization.html) — clock correction and timestamp processing.
