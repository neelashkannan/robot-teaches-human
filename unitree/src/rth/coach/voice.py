"""The sensei's voice: macOS ``say`` in the background, or silence."""

from __future__ import annotations

import shutil
import subprocess
import threading

from rth.config import Settings, VoiceSettings
from rth.contracts import Voice


class NullVoice:
    """Silent voice; remembers the last line so the UI can still show it."""

    def __init__(self) -> None:
        self.last_text: str | None = None

    def say(self, text: str) -> None:
        self.last_text = text

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class MacVoice:
    """Speaks with macOS ``say`` without blocking; a new sentence cuts off the previous one."""

    def __init__(self, settings: VoiceSettings, *, executable: str = "say") -> None:
        self.settings = settings
        self.executable = executable
        self.last_text: str | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def command(self, text: str) -> list[str]:
        cmd = [self.executable, "-r", str(int(self.settings.rate))]
        if self.settings.voice:
            cmd += ["-v", self.settings.voice]
        # A leading dash would be read as an option by `say`.
        return cmd + [text.lstrip("- ")]

    def say(self, text: str) -> None:
        text = (text or "").strip()
        self.last_text = text or self.last_text
        with self._lock:
            self._stop_locked()
            if not text:
                return
            try:
                self._proc = subprocess.Popen(
                    self.command(text),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                self._proc = None  # speech is a nice-to-have; never break the demo

    def _stop_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def close(self) -> None:
        self.stop()


def make_voice(settings: Settings) -> Voice:
    """MacVoice when enabled and ``say`` exists, else NullVoice."""
    if settings.voice.enabled and shutil.which("say"):
        return MacVoice(settings.voice)
    return NullVoice()
