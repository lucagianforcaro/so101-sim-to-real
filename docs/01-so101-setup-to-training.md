# SO-101 with LeRobot — From Motor Setup to a Trained Policy

Step-by-step log of my SO-101 project: configuring the arms, collecting
demonstrations with a webcam, and training an imitation-learning policy
(ACT first) with [LeRobot](https://github.com/huggingface/lerobot).

This document is built up as the project progresses. Each step lists the
commands I actually run, plus notes on what happened.

| | |
|---|---|
| **Robot** | SO-101 follower arm (Feetech STS3215 servos); no leader arm yet |
| **Teleoperation** | USB gamepad instead of a leader arm, see [joystick/README.md](../joystick/README.md) |
| **Camera(s)** | USB webcam on a tripod, _being set up_ |
| **Software** | LeRobot `0.6.2` (from source), Python 3.12 |
| **Host** | Windows 11 + WSL2 (Ubuntu), kernel `6.18` |
| **First policy** | ACT (Action Chunking with Transformers) |

## Progress

- [x] 0. Environment setup
- [x] 1. Connect the arms to WSL2
- [x] 2. Find the USB ports
- [x] 3. Set motor IDs and baudrate
- [x] 4. Calibrate the arms (follower)
- [x] 5. Teleoperate (no camera): with the gamepad, see [joystick/README.md](../joystick/README.md)
- [ ] 6. Set up the camera
- [ ] 7. Teleoperate with the camera
- [ ] 8. Record a dataset
- [ ] 9. Visualize and replay the dataset
- [ ] 10. Train the policy
- [ ] 11. Evaluate the policy on the robot

---

## How it works (overview)

```
 leader arm (moved by hand) ──► follower arm copies it
                                      │
            webcam frames + follower joint positions = observation
            leader joint positions                    = action
                                      │
                               LeRobotDataset (episodes)
                                      │
                               lerobot-train (ACT)
                                      │
            policy: observation ──► action, running on the follower
```

The policy learns end to end: images go into a CNN (ResNet18 for ACT)
inside the network. No separate object-detection pipeline is involved.

---

## Variables used in this document

All commands below use these shell variables. Set them once per terminal
session (fill in the values found in steps 2 and 6):

```bash
export FOLLOWER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042597-if00   # stable name, see step 1.3
export LEADER_PORT=/dev/ttyACM1        # TBD: by-id path of the leader board
export FOLLOWER_ID=so101_follower      # calibration name, keep it the same everywhere
export LEADER_ID=so101_leader          # calibration name, keep it the same everywhere
export CAMERA_INDEX=0                  # found in step 6
export HF_USER=$(NO_COLOR=1 hf auth whoami | awk -F': *' 'NR==1 {print $2}')
export TASK_NAME=pick_place_cube       # short name for the task/dataset
```

---

## 0. Environment setup

Install Miniforge (conda), create the environment and install LeRobot from
source with the Feetech motor drivers and training extras.

```bash
# Miniforge
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc

# Environment
conda create -y -n lerobot python=3.12
conda activate lerobot
conda install -y ffmpeg -c conda-forge   # video encoding/decoding

# LeRobot from source
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e ".[feetech,core_scripts,training]"
```

Log in to Hugging Face (needed to push datasets and policies). Create a token
with **write** access at <https://huggingface.co/settings/tokens>.

```bash
hf auth login
hf auth whoami
```

Optional: log in to Weights & Biases to track training curves.

```bash
wandb login
```

**Notes:**
<!-- e.g. versions installed, problems encountered -->

---

## 1. Connect the arms to WSL2

WSL2 is a lightweight virtual machine: the USB ports belong to Windows, so
Ubuntu does not see the arms until they are forwarded with
[usbipd-win](https://github.com/dorssel/usbipd-win). usbipd runs on Windows,
which is why the commands below are typed in **PowerShell**, not in Ubuntu.

### 1.1 Open PowerShell as Administrator

Press the Windows key, type `powershell`, right-click **Windows PowerShell**
→ **Run as administrator** (or press `Ctrl+Shift+Enter` instead of Enter).
The window title starts with "Administrator:" and the prompt is
`PS C:\WINDOWS\system32>`.

Admin rights are needed only for `bind`; `list` and `attach` work in a normal
PowerShell too.

### 1.2 One-time setup (Windows)

```powershell
winget install usbipd     # then close and reopen PowerShell so it is on the PATH
usbipd list               # with the arm plugged in and powered
usbipd bind --busid 1-4   # replace 1-4 with the BUSID shown by `usbipd list`
```

Values like `<BUSID>` are placeholders: write the actual value (e.g. `1-4`),
without angle brackets. The BUSID depends on the physical USB port.

Example `usbipd list` output (the arm's controller board is the CH343 line):

```
Connected:
BUSID  VID:PID    DEVICE                                  STATE
1-4    1a86:55d3  USB-Enhanced-SERIAL CH343 (COM3)        Shared

Persisted:
GUID                                  DEVICE
d66e4cfc-...                          USB-Enhanced-SERIAL CH343 (COM3)
```

| STATE | Meaning |
|---|---|
| `Not shared` | Windows owns the device; `bind` not done yet |
| `Shared` | `bind` done: can be attached to WSL |
| `Attached` | Forwarded to WSL: it disappears from Windows (no more COM3) |

`Persisted` lists the devices already bound: `bind` survives reboots, and a
device listed there but missing from `Connected` is simply unplugged.

### 1.3 Every working session

**Windows (PowerShell, admin not needed):** attach the board to WSL and keep
the window open (minimised is fine):

```powershell
usbipd attach --wsl --hardware-id 1a86:55d3 --auto-attach
```

- `--hardware-id 1a86:55d3` selects the CH343 board whatever USB port it is
  plugged into (works when only one arm is connected; with two arms use
  `--busid` and run one command per arm).
- `--auto-attach` re-attaches the board automatically if it disconnects.
  It keeps running: **closing the window (or Ctrl+C) stops it**. The board
  stays attached until the next disconnection.
- Without `--auto-attach` the command attaches once and returns.
- The same command can be run from a WSL terminal as `usbipd.exe attach ...`.

**Ubuntu (WSL):** check that the port is there and usable:

```bash
ls -l /dev/serial/by-id/      # one line per board, with its serial number
ls -l /dev/ttyACM*            # the actual port name (ttyACM0, ttyACM1, ...)
groups                        # must include "dialout" (see 1.4)
python scripts/check_motors.py $FOLLOWER_PORT   # all six motors must answer
```

Expected output:

```
lrwxrwxrwx 1 root root 13 ... usb-1a86_USB_Single_Serial_5B3D042597-if00 -> ../../ttyACM0
crw-rw---- 1 root dialout 166, 0 ... /dev/ttyACM0
```

**Use the `by-id` path as the port.** The `ttyACM*` name can change: if the
board disconnects while a program still holds the port open, it comes back as
`ttyACM1`, and commands using `/dev/ttyACM0` fail. The
`/dev/serial/by-id/...` link is based on the board's serial number and always
points to the right board, so `FOLLOWER_PORT` uses it (see "Variables used in
this document").

To check whether (and when) the board disconnected:

```bash
sudo dmesg | grep -iE "acm|usb 1-" | tail
```

`cdc_acm ...: ttyACM0: USB ACM device` means attached; `USB disconnect` means
it was lost.

To give the board back to Windows (PowerShell):

```powershell
usbipd detach --busid 1-4
```

### 1.4 Serial port permissions (once)

The port belongs to the `dialout` group, so without it you get
`Permission denied`. Add your user to the group once:

```bash
sudo usermod -aG dialout $USER
```

then restart WSL from PowerShell and reopen Ubuntu (`groups` must list
`dialout`); the restart stops `--auto-attach`, so run it again:

```powershell
wsl --shutdown
usbipd attach --wsl --hardware-id 1a86:55d3 --auto-attach
```

The quick alternative, `sudo chmod 666 /dev/ttyACM0`, is lost every time the
board reconnects.

### 1.5 Troubleshooting

LeRobot reports every port problem as `Could not connect on port '...'`. The
actual cause is in the `OSError:` line a few lines above in the traceback.

| Symptom | Cause / fix |
|---|---|
| `usbipd: command not found` right after install | Reopen PowerShell (PATH not refreshed) |
| Board not listed in `usbipd list` | Not detected by Windows: check cable (data, not charge-only), port, power supply |
| `There is no device with busid` | Board not connected right now, or different BUSID |
| `<` operator error in PowerShell | `<BUSID>` typed literally: replace it with the value |
| `port ''` | Port variable not set in this terminal (`echo $FOLLOWER_PORT`) |
| `OSError: Permission denied` | User not in `dialout` (1.4) |
| `OSError: No such file or directory` | Board not attached to WSL, or renamed `ttyACM1`: re-run the attach, use the `by-id` path |
| Board disconnects by itself every few minutes | Disable Windows USB selective suspend and "allow the computer to turn off this device" on USB root hubs; keep the laptop on AC power; secure the cable |

**Notes:**

| Arm | Board serial (`/dev/serial/by-id/`) |
|---|---|
| Follower | `usb-1a86_USB_Single_Serial_5B3D042597-if00` |
| Leader | _TBD_ |

---

## 2. Find the USB ports

On native Linux, `lerobot-find-port` asks you to unplug the USB cable and
detects which port disappeared:

```bash
lerobot-find-port
```

**On WSL2 this does not work**: unplugging the cable detaches the board from
WSL, so the script sees no difference (`Could not detect the port`). Instead,
connect and attach one arm at a time (step 1): the `/dev/ttyACM*` that appears
is that arm's port.

Then update `FOLLOWER_PORT` and `LEADER_PORT` in the variables section. If a
command reports `port ''`, the variable is not set in the current terminal
(check with `echo $FOLLOWER_PORT`).

| Arm | Port |
|---|---|
| Follower | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042597-if00` (currently `ttyACM0`/`ttyACM1`) |
| Leader | _TBD_ |

---

## 3. Set motor IDs and baudrate

One-time step per arm. All servos ship with the same default ID, so each one
must get a unique ID (1–6) before they are daisy-chained.

The script asks you to connect **one motor at a time** to the controller
board, starting from the gripper:

| Motor | ID |
|---|---|
| shoulder_pan | 1 |
| shoulder_lift | 2 |
| elbow_flex | 3 |
| wrist_flex | 4 |
| wrist_roll | 5 |
| gripper | 6 |

**Follower:**

```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT

export FOLLOWER_PORT=/dev/ttyACM0

lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT

```

**Leader:**

```bash
lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT
```

After both scripts finish, daisy-chain the motors and connect `shoulder_pan`
(ID 1) to the controller board.

> **Troubleshooting:** most communication/timeout errors are physical.
> Check the power supply, the USB cable and the 3-pin cables. Blinking red
> LEDs usually mean overload or the wrong supply voltage (5 V/7.4 V and 12 V
> versions of the arm are not interchangeable).

**Notes:**

---

## 4. Calibrate the arms

Calibration makes leader and follower report the same values in the same
physical pose. The `id` is the name under which the calibration is saved
(`~/.cache/huggingface/lerobot/calibration/`); reuse the same `id` in every
later command.

Procedure: move the arm so that every joint is roughly in the **middle of its
range**, press Enter, then move **each joint through its full range**.

**Follower:**

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID

export FOLLOWER_PORT=/dev/ttyACM0
export FOLLOWER_ID=so101_follower

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID

```

**Leader:**

```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID
```

**Notes:**

---

## 5. Teleoperate (no camera)

Sanity check: the follower should mirror the leader smoothly.

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID
```

**Notes:**

---

## 6. Set up the camera

Find the available cameras. The script prints each camera's index/path and
saves a test image for each one in `outputs/captured_images/`.

```bash
lerobot-find-cameras opencv
```

Then update `CAMERA_INDEX` in the variables section.

**Camera placement guidelines:**

- Fixed and rigid mount: the camera must not move between data collection
  and inference.
- The whole workspace (object, target, gripper) must be visible.
- Constant lighting, a plain background, no people moving in the frame.
- 640×480 at 30 fps is enough (images are resized by the policy anyway).

| Camera | Name in config | Index/path | Position |
|---|---|---|---|
| Webcam | `front` | _TBD_ | _TBD_ |

**Notes:**

---

## 7. Teleoperate with the camera

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true
```

`--display_data=true` opens a [Rerun](https://rerun.io) viewer with the
camera stream and joint values.

**Notes:**

---

## 8. Record a dataset

**Task definition:** _TBD_ (e.g. "Pick up the red cube and place it in the box")

Each episode stores, at 30 fps, the camera frames and follower joint states
(observation) and the leader joint positions (action).

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=$LEADER_PORT \
    --teleop.id=$LEADER_ID \
    --display_data=true \
    --dataset.repo_id=$HF_USER/$TASK_NAME \
    --dataset.single_task="Pick up the red cube and place it in the box" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10
```

**Keyboard controls during recording:**

| Key | Action |
|---|---|
| `→` or `n` | End the current episode early / skip the reset |
| `←` or `r` | Discard and re-record the current episode |
| `Esc` or `q` | Stop, encode the videos and upload the dataset |

**Resume a recording** (add more episodes to the same dataset):
`--dataset.num_episodes` is the number of **additional** episodes.

```bash
lerobot-record ... --resume=true --dataset.num_episodes=10
```

The dataset is saved locally in `~/.cache/huggingface/lerobot/$HF_USER/$TASK_NAME`
and pushed to the Hub at the end (disable with `--dataset.push_to_hub=false`).

**Data collection tips:**

- Start with a constrained task: one object, a few fixed start positions,
  same operator. Extend only once this works.
- Practice the task a few times before recording.
- Be consistent: same strategy and similar speed in every episode.
- Vary the object position slightly across episodes so the policy generalizes.
- 50 episodes is a good first target.

| Dataset | Episodes | Notes |
|---|---|---|
| _TBD_ | | |

---

## 9. Visualize and replay the dataset

Always inspect the data before training (missing frames, blurry images,
inconsistent demonstrations).

Online visualizer:

```bash
echo "https://huggingface.co/spaces/lerobot/visualize_dataset?path=/$HF_USER/$TASK_NAME/episode_0"
```

Replay an episode on the follower:

```bash
lerobot-replay \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --dataset.repo_id=$HF_USER/$TASK_NAME \
    --dataset.episode=0
```

**Notes:**

---

## 10. Train the policy

First policy: **ACT** (fast, low memory, works well with ~50 episodes).

**How many steps?** Think in epochs (5–10 epochs is typical):

```
total_frames    = episodes × episode_seconds × fps     # 50 × 30 × 30 = 45,000
steps_per_epoch = total_frames / batch_size            # 45,000 / 8  ≈ 5,600
total_steps     = epochs × steps_per_epoch             # 5 epochs    ≈ 28,000
```

For ACT on a single task, 30k–80k steps with batch size 8–16 is a good
starting range.

```bash
lerobot-train \
    --dataset.repo_id=$HF_USER/$TASK_NAME \
    --policy.type=act \
    --policy.device=cuda \
    --batch_size=8 \
    --steps=30000 \
    --output_dir=outputs/train/act_$TASK_NAME \
    --job_name=act_$TASK_NAME \
    --wandb.enable=true \
    --policy.repo_id=$HF_USER/act_$TASK_NAME
```

Checkpoints are saved in `outputs/train/act_$TASK_NAME/checkpoints/`.

**Resume an interrupted run:**

```bash
lerobot-train \
    --config_path=outputs/train/act_$TASK_NAME/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

**No local GPU?** Train on Hugging Face Jobs by adding
`--job.target=a10g-small` (list hardware with `hf jobs hardware`), or use
Google Colab.

| Run | Policy | Dataset | Steps | Batch | Hardware | Time | Notes |
|---|---|---|---|---|---|---|---|
| _TBD_ | ACT | | | | | | |

---

## 11. Evaluate the policy on the robot

Run the trained policy on the follower and record evaluation episodes. The
dataset name must start with `eval_`, and the task description must be the
same as in training.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=$FOLLOWER_PORT \
    --robot.id=$FOLLOWER_ID \
    --robot.cameras="{ front: {type: opencv, index_or_path: $CAMERA_INDEX, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=$HF_USER/eval_$TASK_NAME \
    --dataset.single_task="Pick up the red cube and place it in the box" \
    --dataset.num_episodes=10 \
    --policy.path=$HF_USER/act_$TASK_NAME
```

To use a local checkpoint instead of the Hub, set
`--policy.path=outputs/train/act_$TASK_NAME/checkpoints/last/pretrained_model`.

| Run | Episodes | Successes | Success rate | Notes |
|---|---|---|---|---|
| _TBD_ | 10 | | | |

**If the policy fails:**

- It fails at one specific stage → record 10–20 extra episodes focused on that stage.
- It oscillates or hesitates → demonstrations are likely inconsistent, or more training is needed.
- It ignores the object position → check camera placement and add more position variety.

---

## References

- LeRobot SO-101 guide: `lerobot/docs/source/so101.mdx`
- LeRobot imitation learning on real robots: `lerobot/docs/source/il_robots.mdx`
- LeRobot camera guide: `lerobot/docs/source/cameras.mdx`
- ACT paper: Zhao et al., _Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware_, 2023 — <https://arxiv.org/abs/2304.13705>
