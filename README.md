# BrainNav

EECS 149 project: select one of two destinations using EEG motor imagery, then
let the navigation system drive the TurtleBot to the selected destination.

The EEG workflow is in [`src/eeg`](src/eeg).

- [EEG setup and data collection](docs/eeg_data_collection.md): hardware wiring,
  OpenBCI GUI/LSL setup, Python commands, recording, troubleshooting, and training.
- [System architecture](docs/architecture.md): project goals, hardware/software,
  model design, data formats, current evidence, and planned robot integration.
- [Detailed EEG implementation guide](docs/eeg_pipeline.md): additional experiment
  and implementation notes.

The default setup is **8-channel Cyton mode** with **binary CSP + logistic
regression**, following the classifier design in OpenBCI's motor imagery
tutorial. It needs participant-specific left/right calibration. Rest trials
measure unwanted selections; they are not a learned class in the binary model.
Each live prediction run is one bounded destination-selection attempt.

```bash
python3 -m src.eeg --help
```

The existing ROS 2 packages in `src/navigation` implement cone perception and
trajectory following. They do not currently consume EEG destination candidates.
The EEG CLI can optionally publish a candidate on `/bci/destination_candidate`
(`std_msgs/msg/String`, `left` or `right`). Mapping a confirmed candidate to
actual destination coordinates remains an integration step.
