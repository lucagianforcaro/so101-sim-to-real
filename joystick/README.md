# Gamepad teleoperation of the SO-101 for collecting demonstrations

> **Status: in progress.** Teleoperation with a gamepad works and its motion
> quality has been measured. Next: the camera, recording demonstrations in the
> LeRobot dataset format, training a policy. This page is updated as each part
> is done (see [Progress](#progress)).

## Contents

1. [Goal](#goal)
2. [Hardware](#hardware)
3. [Getting the devices into WSL2](#getting-the-devices-into-wsl2)
4. [Every working session](#every-working-session)
5. [Gamepad: reading it and finding its mapping](#gamepad-reading-it-and-finding-its-mapping)
6. [Calibration of the scene](#calibration-of-the-scene)
7. [Teleoperation](#teleoperation)
8. [How the teleoperation works](#how-the-teleoperation-works)
9. [Motion quality](#motion-quality)
10. [Files](#files)
11. [Troubleshooting](#troubleshooting)
12. [Next steps](#next-steps)
13. [Progress](#progress)

## Goal

Teach the SO-101 follower to **pick up a cube and put it in a box** by
imitation learning: a human demonstrates the task a few dozen times, the
demonstrations (camera images + joint positions + joint commands) are
recorded in the LeRobot dataset format, and a policy such as
[ACT](https://arxiv.org/abs/2304.13705) learns to reproduce the task from the
camera image alone.

LeRobot's standard way to demonstrate is a second, identical arm (the
*leader*) moved by hand. This project has no leader arm, so the
demonstrations are given with a **USB gamepad** instead:

```
gamepad ──► gripper velocity ──► target gripper position ──► inverse kinematics ──► joint commands ──► SO-101
  (sticks)        (x, y, z)        (+ tilt, roll, gripper)        (MuJoCo model)         (100 Hz)
```

Everything downstream of the joint commands (dataset format, training,
evaluation) is the same as with a leader arm.

Two earlier attempts led here:

- **Phone teleoperation** (iPhone with LeRobot's `phone` teleoperator): the
  inverse kinematics flipped between arm configurations on the 5-joint arm and
  the motion was erratic.
- **Scripted demonstrations** (a program performing the task with known object
  positions): set aside in favour of human demonstrations, which are more
  varied.

## Hardware

| Device | Details | In WSL2 as |
|---|---|---|
| SO-101 follower arm | 12 V, Feetech STS3215 servos, controller board CH343 (USB `1a86:55d3`), calibrated as `so101_follower` | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B3D042597-if00` |
| Gamepad | Trust Predator, generic USB HID gamepad (USB `0810:0003`): 6 axes, 12 buttons | `/dev/input/js0` |
| Webcam | USB, fixed above the work area (to be added) | `/dev/video0` (expected) |
| Clamps | the robot base must be clamped to the table | |

The host is Windows 11 + WSL2 (Ubuntu), without an NVIDIA GPU: training will
run on Google Colab.

## Getting the devices into WSL2

WSL2 is a virtual machine: USB devices belong to Windows until they are
forwarded with [usbipd-win](https://github.com/dorssel/usbipd-win). Each device
goes through the same pipeline:

```
1. Windows sees it          usbipd list                                 (PowerShell)
2. shared, once             usbipd bind --busid <BUSID>                 (PowerShell, as Administrator)
3. attached, every session  usbipd attach --wsl --hardware-id <VID:PID> --auto-attach   (PowerShell, keep open)
4. kernel driver in WSL     dmesg | tail ; ls /dev/...                  (Ubuntu)
5. permissions              the user must be able to read/write the device file
```

### 1–3. Forwarding (Windows)

```powershell
usbipd list
```

Find the device by its `VID:PID` (or unplug it and see which line
disappears). Never attach the Logitech receiver (`046d:c534`): mouse and
keyboard would stop working in Windows.

```powershell
usbipd bind --busid <BUSID>                                   # once per device, as Administrator
usbipd attach --wsl --hardware-id <VID:PID> --auto-attach     # every session
```

- `--hardware-id` selects the device whatever USB port it is plugged into
  (the BUSID changes with the port).
- `--auto-attach` re-attaches it if it disconnects. It keeps running: one
  PowerShell window per device, left open (closing it stops the re-attach).
- While attached to WSL, the device is not available in Windows
  (`usbipd detach --busid <BUSID>` gives it back).

### 4. Kernel drivers (WSL)

| Device | Driver | In the WSL kernel |
|---|---|---|
| Arm controller board (CH343) | `cdc_acm` | yes |
| Gamepad (generic HID) | `usbhid`, `hid-generic`, `joydev` | yes |
| Xbox controllers | `xpad` | **no** (an Xbox controller would not work this way) |
| Webcam | `uvcvideo` | yes |

Check that a device was recognised:

```bash
sudo dmesg | tail          # e.g. "input: USB Gamepad ..." or "cdc_acm ...: ttyACM0"
ls -l /dev/serial/by-id/   # arm
ls -l /dev/input/js0       # gamepad
ls -l /dev/video*          # webcam
```

### 5. Permissions

| Device file | Owner / mode | What is needed |
|---|---|---|
| `/dev/ttyACM0` (arm) | `root:dialout`, `rw-rw----` | the user in group `dialout`, once: `sudo usermod -aG dialout $USER`, then `wsl --shutdown` |
| `/dev/input/js0` (gamepad) | `rw-rw-r--` | nothing: readable by every user |
| `/dev/input/event0` (gamepad) | `rw-rw----`, group `netdev` | not used (pygame/SDL would need it): the gamepad is read from `js0` instead |
| `/dev/video0` (webcam) | usually `root:video` | probably the user in group `video`, once: `sudo usermod -aG video $USER`, then `wsl --shutdown` |

The serial port is used through its `by-id` path, which is stable: the
`ttyACM*` name can change (e.g. `ttyACM1`) when the board reconnects while
the port is still open.

## Every working session

After a reboot of the PC or of WSL (also after a WSL update), the devices must
be attached again.

**Windows, one PowerShell window per device, left open:**

```powershell
usbipd attach --wsl --hardware-id 1a86:55d3 --auto-attach    # arm
usbipd attach --wsl --hardware-id 0810:0003 --auto-attach    # gamepad
```

**Ubuntu:**

```bash
cd ~/robotics
conda activate lerobot
ls -l /dev/serial/by-id/ /dev/input/js0              # both devices present
python scripts/check_motors.py /dev/ttyACM0           # the 6 motors answer
python joystick/teleop_joystick.py                    # teleoperation
```

Before teleoperating: robot base clamped, arm powered, work area clear.

## Gamepad: reading it and finding its mapping

### Reading it without dependencies

The Linux joystick interface sends one 8-byte event per change:

```c
struct js_event {
    __u32 time;     /* timestamp [ms] */
    __s16 value;    /* axis: -32767..32767, button: 0/1 */
    __u8  type;     /* 0x01 button, 0x02 axis, | 0x80 initial state */
    __u8  number;   /* which axis or button */
};
```

[`js_reader.py`](js_reader.py) reads `/dev/input/js0` with these 8-byte events
(non-blocking) and keeps the state of every axis (scaled to −1..1) and button.
On opening, the kernel first sends the state of every control, so the state is
complete right away. If the gamepad disappears, the state goes back to neutral
and `connected` becomes `False`: a disconnected gamepad cannot leave the arm
moving. No pygame and no permission change are needed.

### Finding the mapping

Axis and button numbers differ from one gamepad to another. They are found
with [`js_test.py`](js_test.py), which prints a line whenever a button is
pressed or an axis passes half its travel:

```bash
python joystick/js_test.py
```

Move one control at a time and note the number and sign. For the Trust
Predator:

| Control | Number | Sign |
|---|---|---|
| Left stick up / down | axis 1 | − / + |
| Left stick right / left | axis 0 | + / − |
| Right stick up / down | axis 2 | − / + |
| Right stick right / left | axis 3 | + / − |
| D-pad up / down | axis 5 | − / + (digital: −1, 0, +1) |
| D-pad right / left | axis 4 | + / − (digital) |
| △ / ○ / ✕ / □ | buttons 0 / 1 / 2 / 3 | |
| L2 / R2 / L1 / R1 | buttons 4 / 5 / 6 / 7 | |
| Start | button 9 | |

At rest every axis reads 0.00 (no drift). If the sticks stop responding or
behave differently after a reconnection, check the gamepad's *Analog*/*Mode*
button: these gamepads can come back in another mode. With a different
gamepad, update the constants at the top of `teleop_joystick.py`.

## Calibration of the scene

Both calibrations are done by moving the arm by hand (torque off) and reading
its joint angles: measured by the robot itself, in its own frame. Redo them
whenever the robot is moved or re-clamped.

### Table height: `calibrate_table.py`

The model's `z = 0` is the bottom of the robot base. On this setup the real
table turned out to be about 1 cm lower: with a fixed lower limit of 1 cm the
fingertips stopped 1.3 cm above the table and the cube could only be grasped
by its top.

```bash
python joystick/calibrate_table.py
```

Touch the table with the fingertips, gripper vertical, at 3 points spread over
the work area. The lowest allowed gripper height is the highest touch + 3 mm,
saved to `table.json` (here 0.04 cm). The gripper point used by the kinematics
(`gripperframe` in the SO-101 model) is 0.73 cm above the fingertips when the
gripper points down.

### Ready pose: `teach_ready_pose.py`

The pose △ brings the arm to, at the start of each demonstration.

```bash
python joystick/teach_ready_pose.py
```

Move the arm to the pose by hand and press ENTER. The pose is checked (gripper
inside the workspace box, wrist roll inside its free range, elbow and wrist
away from their limits) and saved to `ready.json`. Without it, the default is
the arm straight ahead, gripper 20 cm in front of the base and 12 cm high,
tilted to −70° (it cannot be vertical that high) and open.

## Teleoperation

```bash
python joystick/teleop_joystick.py              # 100 Hz control loop
python joystick/teleop_joystick.py --log        # also log every cycle (motion quality)
```

| Control | Action |
|---|---|
| Left stick ↑ / ↓ | gripper forward / back |
| Left stick ← / → | gripper left / right |
| Right stick ↑ / ↓ | gripper up / down |
| R2 / L2 (hold) | wrist roll, one direction each |
| R1 / L1 (hold) | close / open the gripper |
| D-pad ↑ / ↓ | gripper tilt (pitch) |
| □ | straighten the gripper to vertical, gradually |
| △ | go to the ready pose (2.5 s) |
| Start | precision mode on / off (30 % speed) |
| Ctrl+C | stop; then hold the arm and press ENTER (torque is released) |

The sticks set the gripper's **velocity** (8 cm/s at full stick), not its
position: half a stick is half speed, a released stick stops the gripper.

**A pick:** △ → left stick above the cube → R2/L2 to align the fingers with
the cube → right stick down (precision mode for the last centimetres) → R1 →
right stick up → left stick above the box → L1 → △.

## How the teleoperation works

### The control loop (100 Hz)

1. read the gamepad;
2. integrate the stick velocities into a target gripper position, clipped to
   the workspace box;
3. solve the inverse kinematics from the current joint angles;
4. accept the solution only if it is safe (below), otherwise keep the previous
   target: the gripper stops;
5. add wrist roll and gripper, send the 6 joint angles through LeRobot.

With the sticks at rest nothing is recomputed: the arm holds exactly where it
is.

### Inverse kinematics: 4 joints for 4 targets

The SO-101 has 5 arm joints. `wrist_roll` turns the gripper about its own axis
(it moves neither the gripper point nor its tilt) and `gripper` opens the
fingers: both are commanded directly. The other 4 joints (`shoulder_pan`,
`shoulder_lift`, `elbow_flex`, `wrist_flex`) are solved for **4 targets**: the
gripper position (x, y, z) and its **pitch**.

`shoulder_lift`, `elbow_flex` and `wrist_flex` rotate about parallel horizontal
axes, so the gripper pitch is exactly (checked on 361 random poses of the
model, error 0.0°):

```
pitch = −(shoulder_lift + elbow_flex + wrist_flex)        (−90° = gripper pointing down)
```

The Jacobian of the 4 targets is the 3×4 position Jacobian (from MuJoCo,
`mj_jacSite`) plus a constant row `[0, −1, −1, −1]` for the pitch. With as many
targets as joints there is one solution near the current pose: no drifting
posture and no jumps between elbow-up and elbow-down configurations (the
problem of the phone teleoperation). It is solved by damped least squares,
warm-started from the current joints, in about 0.1–0.5 ms.

MuJoCo is used here only as a geometry engine (forward kinematics and
Jacobians of the SO-101 model `so101_new_calib`, the same used for the
[lemniscate](../lemniscate/README.md)); no physics is simulated. LeRobot's
`placo`/URDF kinematics gives the same gripper positions.

### Position has priority over pitch

With the gripper vertical, the arm cannot rise above about 9 cm (the wrist
would have to be out of reach). So the IK runs in two attempts:

1. position + desired pitch (pitch weight 0.1): the pitch is enforced;
2. if that fails, position with a soft pitch (weight 0.0005): the gripper
   tilts only as much as needed.

The desired pitch stays vertical, so the gripper straightens by itself where
it can, e.g. going down to the table. In simulation: rising from 7 to 25 cm
tilts it from −90° to −32°; going back down to the table brings it to −90°.

### Safety

| Guard | Value | Why |
|---|---|---|
| Workspace box | x 10–40 cm, y ±25 cm, z table+3 mm – 30 cm | gripper above the table, away from the base and from the stretched (singular) arm |
| Reachability | position error < 3 mm | unreachable targets are not followed |
| Max joint step | 3° per cycle | near a singularity a small gripper step needs a large joint rotation: such jerks are not followed (normal motion: < 2°) |
| Elbow | < 85° | going down with a tilted gripper folds the arm (see below) |
| Wrist roll | 20°–170° | mechanical stop at about +14° (by design); LeRobot wraps the full turn at ±180° |
| LeRobot `max_relative_target` | 10° per command | last line of defence |
| Gamepad disconnected | motion stopped | |

Why the elbow guard: going down to the table at 25 cm while keeping the
gripper tilted at −22° needs the shoulder to swing by almost 100° (from −72° to
+24°) with the elbow at its limit (91°): the arm folds. With the gripper
vertical the same descent moves the shoulder by 7°.

Why the roll range: [`../scripts/test_wrist_roll.py`](../scripts/test_wrist_roll.py)
moves only the wrist roll while logging position and load. Rotating towards
negative angles, the joint stopped at about +14° while its load rose to 100 %
and then dropped to 20 % (the servo's overload protection): a mechanical stop.
In the other direction it turned 150° freely.

### Servo settings

From the [lemniscate experiments](../lemniscate/README.md): acceleration 50
(softer starts and stops, `--acceleration`) and I gain 10 (no sag under
gravity, `--i-gain`), written to the servos' RAM at connection.

## Motion quality

Two sessions with the same picks, logged with `--log` and analysed with
[`analyze_teleop.py`](analyze_teleop.py): only the stretches where a stick was
moving the gripper (about 14 s per session), resampled at 30 Hz (the dataset
rate), gripper positions from forward kinematics of commanded and measured
joints.

| | 30 Hz loop | **100 Hz loop** |
|---|---|---|
| Cycle time (median / 95th percentile) | 33.6 / 33.9 ms | 10.1 / 10.3 ms |
| Lag of the arm behind the command | 167 ms | **133 ms** |
| Shape error (lag removed) | 3.6 mm | **3.3 mm** |
| Jerkiness (measured / commanded acceleration, 1 = as smooth as the command) | 1.45 | **1.09** |
| Stalled intervals | 0 % | 0 % |

At 30 Hz the servo reaches each goal within a few milliseconds and waits for
the next one: the motion is a staircase, 45 % jerkier than the command. At
100 Hz it is only 9 % jerkier. Lag and error are the servos' own (the
lemniscate gave about 130 ms and 2–3 mm) and will be the same when a policy
drives the arm. The teleoperation runs at 100 Hz; the dataset will still be
recorded at 30 fps.

The quality of the *dataset* will also depend on how it is recorded:
consistent demonstrations, few pauses (in these sessions the gripper moved for
14 s out of 70: long idle stretches teach a policy to stop), varied cube
positions, only successful episodes.

## Files

| File | Purpose |
|---|---|
| [`js_reader.py`](js_reader.py) | Dependency-free reader for `/dev/input/js0` |
| [`js_test.py`](js_test.py) | Live view of axes and buttons, to find the mapping |
| [`calibrate_table.py`](calibrate_table.py) | Measures the table height → `table.json` |
| [`teach_ready_pose.py`](teach_ready_pose.py) | Teaches the ready pose → `ready.json` |
| [`teleop_joystick.py`](teleop_joystick.py) | The teleoperation (IK, safety, logging) |
| [`analyze_teleop.py`](analyze_teleop.py) | Motion-quality metrics of the logs in `outputs/` |
| [`../scripts/check_motors.py`](../scripts/check_motors.py) | Checks that the 6 motors answer |
| [`../scripts/test_wrist_roll.py`](../scripts/test_wrist_roll.py) | Wrist roll range and load test |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `FileNotFoundError` on `/dev/input/js0` | gamepad not attached to WSL (e.g. after a reboot or a WSL update): run its `usbipd attach` |
| `Could not connect on port ...` with `OSError: Permission denied` | user not in `dialout` |
| `Could not connect on port ...`, port present | another script holds it: `ps aux \| grep python`. A script stopped with **Ctrl+Z** is suspended, not closed: `fg` then Ctrl+C. Always stop with Ctrl+C |
| "Gamepad disconnected: motion stopped." | USB disconnection; `--auto-attach` re-attaches it: restart the script |
| Sticks dead or swapped after a reconnection | gamepad in another mode: press *Analog*/*Mode*, check with `js_test.py` |
| Gripper stops while moving | edge of the workspace box, unreachable target, or a safety guard: move back, or △ |
| Gripper grasps the cube only by the fingertips | table height not calibrated: `calibrate_table.py` |
| Wrist roll does not turn one way | at the end of its range (20° or 170°): turn the other way |

## Next steps

1. **Camera**: mount it fixed above the work area, attach it to WSL, check it
   with `lerobot-find-cameras opencv`.
2. **Recording**: write demonstrations into a LeRobotDataset at 30 fps (images,
   joint positions, joint commands) while teleoperating at 100 Hz, with
   gamepad buttons to keep or discard an episode; trim idle frames.
3. **Dataset quality**: episode length, idle fraction, success rate, coverage
   of the work area.
4. **Training**: ACT on Colab; **evaluation** on the robot (with command
   interpolation at a higher rate than the policy's 30 Hz).

## Progress

- [x] Gamepad read in WSL2 without extra dependencies, mapping found
- [x] Cartesian teleoperation: 4-joint IK with pitch, position priority, safety guards
- [x] Wrist roll range measured (mechanical stop) and enforced
- [x] Table height and ready pose calibrated by hand
- [x] Motion quality measured: 100 Hz chosen
- [ ] Robot base clamped to the table, calibrations redone
- [ ] Camera mounted and working in WSL2
- [ ] Recording demonstrations into a LeRobotDataset
- [ ] Dataset recorded and inspected
- [ ] Policy trained and evaluated
