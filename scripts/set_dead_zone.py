"""Read or change the dead zone of the SO-101 follower's servos.

The dead zone (CW_Dead_Zone / CCW_Dead_Zone, in encoder ticks) is the error
the servo tolerates before correcting: with the factory value 1, it waits for
a 2-tick error and then catches up at once, which turns a slowly advancing
trajectory into steps. 0 makes it correct every tick (smoother motion), at the
risk of buzzing at standstill.

The registers are in the servo's EEPROM: the value is PERMANENT (survives power
off and is not reset by LeRobot) until written again. Factory value: 1.
Writing requires torque off, so the arm goes limp: hold it.

Usage (from the project root):
    python scripts/set_dead_zone.py                                   # read only
    python scripts/set_dead_zone.py --value 0 --motors shoulder_lift elbow_flex wrist_flex
    python scripts/set_dead_zone.py --value 1                         # restore factory value
"""

import argparse

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

MOTORS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}
REGISTERS = ["CW_Dead_Zone", "CCW_Dead_Zone"]


def read_dead_zones(bus: SerialMotorsBus) -> None:
    for name in MOTORS:
        values = [bus.read(reg, name, normalize=False) for reg in REGISTERS]
        print(f"  {name:<14} CW {values[0]}   CCW {values[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--value", type=int, choices=range(0, 6), help="new dead zone in ticks (omit to only read)")
    parser.add_argument("--motors", nargs="+", default=list(MOTORS), choices=list(MOTORS))
    args = parser.parse_args()

    bus = SerialMotorsBus(
        port=args.port,
        motors={name: Motor(id_, "sts3215", MotorNormMode.RANGE_M100_100) for name, id_ in MOTORS.items()},
    )
    bus.connect()
    try:
        print("Current dead zones [ticks]:")
        read_dead_zones(bus)
        if args.value is None:
            return

        input(f"Hold the arm: torque will be released to write {args.value} on {args.motors}. Press ENTER...")
        # Torque off also opens the Feetech EEPROM lock.
        bus.disable_torque()
        for name in args.motors:
            for reg in REGISTERS:
                bus.write(reg, name, args.value, normalize=False)

        print("New dead zones [ticks] (read back):")
        read_dead_zones(bus)
    finally:
        bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
