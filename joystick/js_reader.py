"""Minimal reader for a Linux joystick device (/dev/input/js0), no dependencies.

The kernel's joystick interface sends one 8-byte event per change:

    struct js_event {
        __u32 time;     /* timestamp in ms */
        __s16 value;    /* axis: -32767..32767, button: 0/1 */
        __u8  type;     /* 0x01 button, 0x02 axis, | 0x80 for the initial state */
        __u8  number;   /* which axis or button */
    };

On open, the kernel first sends the current state of every axis and button
(flagged 0x80), so the state is complete right away. /dev/input/js0 is
readable by every user, unlike /dev/input/event0: no permissions to change.

Usage:
    with Joystick() as js:
        while True:
            js.poll()
            print(js.axes, js.buttons)
"""

import os
import struct

EVENT_FORMAT = "IhBB"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)  # 8 bytes
JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
AXIS_MAX = 32767


class Joystick:
    def __init__(self, path: str = "/dev/input/js0"):
        self.path = path
        self.fd: int | None = None
        self.axes: dict[int, float] = {}  # axis number -> value in [-1, 1]
        self.buttons: dict[int, bool] = {}  # button number -> pressed
        self.connected = False

    def open(self) -> "Joystick":
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NONBLOCK)
        self.connected = True
        self.poll()  # read the initial state
        return self

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.connected = False

    def __enter__(self) -> "Joystick":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def poll(self) -> list[tuple[str, int, float]]:
        """Read every pending event and update the state.

        Returns the events read, as ("axis" | "button", number, value). If the
        device disappears (unplugged, USB detached), `connected` becomes False
        and the state is reset to neutral (no axis deflected, no button pressed).
        """
        events = []
        if self.fd is None:
            return events
        while True:
            try:
                data = os.read(self.fd, EVENT_SIZE)
            except BlockingIOError:
                break  # no more pending events
            except OSError:
                self.close()
                self.axes = dict.fromkeys(self.axes, 0.0)
                self.buttons = dict.fromkeys(self.buttons, False)
                break
            if len(data) < EVENT_SIZE:
                break
            _, value, etype, number = struct.unpack(EVENT_FORMAT, data)
            if etype & JS_EVENT_AXIS:
                self.axes[number] = max(-1.0, min(1.0, value / AXIS_MAX))
                events.append(("axis", number, self.axes[number]))
            elif etype & JS_EVENT_BUTTON:
                self.buttons[number] = bool(value)
                events.append(("button", number, float(value)))
        return events
