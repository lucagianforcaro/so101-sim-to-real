"""Show the joystick's axes and buttons live, to find out which number is which control.

Move one stick or press one button at a time: a log line is printed when a
button is pressed or an axis goes past half of its travel, and the bottom line
shows every axis and the pressed buttons. Ctrl+C to stop.

Usage (from the project root):
    python joystick/js_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joystick.js_reader import Joystick  # noqa: E402


def main() -> None:
    with Joystick() as js:
        print(
            f"{len(js.axes)} axes, {len(js.buttons)} buttons. Move one control at a time (Ctrl+C to stop).\n"
        )
        past_half = dict.fromkeys(js.axes, False)
        try:
            while True:
                for kind, number, value in js.poll():
                    if kind == "button" and value:
                        print(f"\rbutton {number:2d} pressed" + " " * 60)
                    elif kind == "axis":
                        beyond = abs(value) > 0.5
                        if beyond and not past_half[number]:
                            print(f"\raxis {number} -> {'+' if value > 0 else '-'}" + " " * 60)
                        past_half[number] = beyond
                if not js.connected:
                    print("\nJoystick disconnected.")
                    return
                axes = "  ".join(f"a{n}:{v:+.2f}" for n, v in sorted(js.axes.items()))
                pressed = [n for n, p in sorted(js.buttons.items()) if p]
                print(f"\r{axes}   buttons: {pressed}" + " " * 10, end="", flush=True)
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
