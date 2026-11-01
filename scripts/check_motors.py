"""Check which motors answer on an SO-101 arm's bus.

Pings the six expected IDs one by one, then scans the bus for every ID that
answers at the default baud rate (to spot a motor left with a wrong ID).

Usage:
    python scripts/check_motors.py /dev/ttyACM0
"""

import sys

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

EXPECTED = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def main(port: str) -> None:
    motors = {name: Motor(id_, "sts3215", MotorNormMode.RANGE_M100_100) for name, id_ in EXPECTED.items()}
    bus = SerialMotorsBus(port=port, motors=motors)
    # No handshake: it would fail as soon as one motor is missing.
    bus.connect(handshake=False)
    try:
        print(f"Port {port}, baud rate {bus.default_baudrate}\n")
        print("Expected motors:")
        for name, id_ in EXPECTED.items():
            model = bus.ping(id_, num_retry=2)
            status = f"OK (model {model})" if model is not None else "NO ANSWER"
            print(f"  ID {id_}  {name:<14} {status}")

        found = bus._scan(bus._definition("shoulder_pan"))
        print(f"\nAll IDs answering on the bus: {sorted(found) or 'none'}")
        unexpected = sorted(set(found) - set(EXPECTED.values()))
        if unexpected:
            print(f"Unexpected IDs (motor with a wrong ID?): {unexpected}")
    finally:
        bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0")
