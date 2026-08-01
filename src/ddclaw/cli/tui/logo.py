"""Rich-rendered animated cat mascot for the DDclaw TUI."""

from __future__ import annotations

from typing import Literal, cast

from rich.style import Style
from rich.text import Text
from textual.timer import Timer
from textual.widgets import Static

LogoState = Literal[
    "idle",
    "planner",
    "tool",
    "approval",
    "success",
    "error",
]

LOGO_EARS = "        /\\_/\\"
LOGO_FACE = "       ( •ᴗ• )ฅ"
LOGO_TAG = "       /  DD \\"
LOGO_TITLE = "  ━━━ DDclaw ━━━"
LOGO_STAGE = "  MultiAgent Coding Companion"
LOGO_ART = "\n".join(
    (LOGO_EARS, LOGO_FACE, LOGO_TAG, LOGO_TITLE, LOGO_STAGE)
)

_VALID_STATES = {
    "idle",
    "planner",
    "tool",
    "approval",
    "success",
    "error",
}
_BOOT_INTERVAL = 0.1
_BOOT_FRAMES = 20
_MOTION_INTERVAL = 0.28
_IDLE_BLINK_INTERVAL = 5.0
_BLINK_DURATION = 0.16

_CAT_STYLE = Style(color="#f0f6fc", bold=True)
_BRAND_STYLE = Style(color="#58a6ff", bold=True)
_STAGE_STYLE = Style(color="#8b949e", italic=True)
_PLANNER_STYLE = Style(color="#d2a8ff", bold=True)
_TOOL_STYLE = Style(color="#56d4dd", bold=True)
_APPROVAL_STYLE = Style(color="#e3b341", bold=True)
_SUCCESS_STYLE = Style(color="#7ee787", bold=True)
_ERROR_STYLE = Style(color="#ff7b72", bold=True)
_SWEEP_STYLE = Style(color="#ffffff", bold=True)


def render_logo(
    frame: int = _BOOT_FRAMES,
    *,
    state: LogoState = "idle",
    blink: bool = False,
) -> Text:
    """Render a boot frame or a stateful cat-mascot frame with Rich styles."""

    if state not in _VALID_STATES:
        raise ValueError(f"Unsupported DDclaw logo state: {state}")
    frame = max(0, int(frame))
    if frame < _BOOT_FRAMES:
        lines = _boot_lines(frame)
        style = _CAT_STYLE
    else:
        lines = _state_lines(state, motion=frame, blink=blink)
        style = _state_style(state)

    logo = Text()
    for index, line in enumerate(lines):
        if index:
            logo.append("\n")
        line_start = len(logo)
        if index <= 2:
            logo.append(line, style=style)
        elif index == 3:
            logo.append(line, style=_BRAND_STYLE)
        else:
            logo.append(line, style=_STAGE_STYLE)

        if frame < _BOOT_FRAMES and line:
            sweep = min(len(line), max(0, frame * 3 - index * 2))
            highlight_start = line_start + max(0, sweep - 3)
            highlight_end = line_start + min(len(line), sweep + 2)
            logo.stylize(_SWEEP_STYLE, highlight_start, highlight_end)
    return logo


class DDClawLogo(Static):
    """Animated DDclaw cat with boot, idle, and workflow-aware states."""

    DEFAULT_CSS = """
    DDClawLogo {
        width: 100%;
        height: 5;
        content-align: center middle;
        background: $background;
    }
    """

    def __init__(self, *, animate: bool = True, id: str | None = None) -> None:
        initial_frame = 0 if animate else _BOOT_FRAMES
        super().__init__(render_logo(initial_frame), id=id, markup=False)
        self.animate = animate
        self.logo_state: LogoState = "idle"
        self._frame = initial_frame
        self._motion_frame = 0
        self._blinking = False
        self._startup_timer: Timer | None = None
        self._motion_timer: Timer | None = None
        self._idle_timer: Timer | None = None
        self._blink_timer: Timer | None = None

    def on_mount(self) -> None:
        if self.animate:
            self._startup_timer = self.set_interval(
                _BOOT_INTERVAL,
                self._advance_boot,
            )
        else:
            self._start_idle_timer()
            self._render_current_state()

    def on_unmount(self) -> None:
        """Stop every mascot timer explicitly when the TUI closes."""

        for timer in (
            self._startup_timer,
            self._motion_timer,
            self._idle_timer,
            self._blink_timer,
        ):
            if timer is not None:
                timer.stop()

    def set_state(self, state: LogoState) -> None:
        """Switch the mascot to a validated workflow state."""

        if state not in _VALID_STATES:
            raise ValueError(f"Unsupported DDclaw logo state: {state}")
        self._stop_timer("_startup_timer")
        self._frame = _BOOT_FRAMES
        self.logo_state = cast(LogoState, state)
        self._motion_frame = 0
        self._blinking = False
        self._stop_timer("_motion_timer")

        if state == "idle":
            self._start_idle_timer()
        else:
            self._stop_timer("_idle_timer")
            self._motion_timer = self.set_interval(
                _MOTION_INTERVAL,
                self._advance_motion,
            )
        self._render_current_state()

    def _advance_boot(self) -> None:
        self._frame += 1
        if self._frame >= _BOOT_FRAMES:
            self._frame = _BOOT_FRAMES
            self._stop_timer("_startup_timer")
            self._start_idle_timer()
            self._render_current_state()
            return
        self.update(render_logo(self._frame))

    def _advance_motion(self) -> None:
        self._motion_frame += 1
        self._render_current_state()

    def _start_idle_timer(self) -> None:
        if self._idle_timer is None:
            self._idle_timer = self.set_interval(
                _IDLE_BLINK_INTERVAL,
                self._start_blink,
            )
        else:
            self._idle_timer.resume()

    def _start_blink(self) -> None:
        if self.logo_state != "idle" or self._blinking:
            return
        self._blinking = True
        self._render_current_state()
        self._blink_timer = self.set_timer(
            _BLINK_DURATION,
            self._finish_blink,
        )

    def _finish_blink(self) -> None:
        self._blinking = False
        self._blink_timer = None
        self._render_current_state()

    def _render_current_state(self) -> None:
        self.update(
            render_logo(
                _BOOT_FRAMES + self._motion_frame,
                state=self.logo_state,
                blink=self._blinking,
            )
        )

    def _stop_timer(self, attribute: str) -> None:
        timer = getattr(self, attribute)
        if timer is not None:
            timer.stop()
            setattr(self, attribute, None)


def _boot_lines(frame: int) -> tuple[str, str, str, str, str]:
    if frame < 4:
        return (
            LOGO_EARS,
            "       ( -.- )",
            LOGO_TAG,
            "",
            "",
        )
    if frame < 8:
        return (
            LOGO_EARS,
            "       ( •ᴗ• )",
            LOGO_TAG,
            "",
            "",
        )
    if frame < 12:
        return (
            LOGO_EARS,
            LOGO_FACE,
            LOGO_TAG,
            "",
            "",
        )
    if frame < 16:
        return (
            LOGO_EARS,
            f"{LOGO_FACE}  {'╱' * (frame - 11)}",
            LOGO_TAG,
            "  ━━━ " + "DDclaw"[: max(1, (frame - 11) * 2)],
            "",
        )
    return (
        LOGO_EARS,
        LOGO_FACE,
        LOGO_TAG,
        LOGO_TITLE,
        LOGO_STAGE,
    )


def _state_lines(
    state: LogoState,
    *,
    motion: int,
    blink: bool,
) -> tuple[str, str, str, str, str]:
    if state == "idle":
        face = "       ( -ᴗ- )ฅ" if blink else LOGO_FACE
        ears = LOGO_EARS
    elif state == "planner":
        ears = r"        /\_/?"
        face = f"       ( •_• )  {'.' * (motion % 4)}"
    elif state == "tool":
        ears = LOGO_EARS
        keyboard_gap = " " if motion % 2 else ""
        face = f"       ( •ᴗ• )ฅ{keyboard_gap}⌨"
    elif state == "approval":
        ears = LOGO_EARS
        marker = "⚠" if motion % 2 else "!"
        face = f"       ( !ᴗ! )ฅ  {marker}"
    elif state == "success":
        ears = LOGO_EARS
        paw = "ฅ" if motion % 2 else "ฅˊ"
        face = f"       ( ^ᴗ^ ){paw}  ✓"
    else:
        ears = "        /)_ (\\"
        face = "       ( ×﹏× )   ×"
    return (ears, face, LOGO_TAG, LOGO_TITLE, LOGO_STAGE)


def _state_style(state: LogoState) -> Style:
    return {
        "idle": _CAT_STYLE,
        "planner": _PLANNER_STYLE,
        "tool": _TOOL_STYLE,
        "approval": _APPROVAL_STYLE,
        "success": _SUCCESS_STYLE,
        "error": _ERROR_STYLE,
    }[state]


__all__ = [
    "DDClawLogo",
    "LOGO_ART",
    "LOGO_EARS",
    "LOGO_FACE",
    "LOGO_STAGE",
    "LOGO_TAG",
    "LOGO_TITLE",
    "LogoState",
    "render_logo",
]
