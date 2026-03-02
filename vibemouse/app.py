from __future__ import annotations

import threading
import time
from collections.abc import Callable

from vibemouse.audio import AudioRecorder
from vibemouse.config import AppConfig
from vibemouse.mouse_listener import SideButtonListener
from vibemouse.output import TextOutput
from vibemouse.streaming_output import StreamingTextOutput
from vibemouse.transcriber import StreamingResult
from vibemouse.vad_transcriber import VadOfflineTranscriber

# Status callback type: (event, detail) where event is one of:
#   "ready", "recording_start", "recording_stop",
#   "streaming", "transcribed", "error"
StatusCallback = Callable[[str, str], None] | None


class VoiceMouseApp:
    def __init__(
        self,
        config: AppConfig,
        on_status_change: StatusCallback = None,
    ) -> None:
        if config.front_button == config.rear_button:
            raise ValueError("Front and rear side buttons must be different")

        self._config: AppConfig = config
        self._on_status_change = on_status_change

        self._recorder: AudioRecorder = AudioRecorder(
            sample_rate=config.sample_rate,
            channels=config.channels,
            dtype=config.dtype,
            pre_buffer_seconds=config.pre_buffer_seconds,
        )
        self._transcriber = VadOfflineTranscriber(config)
        self._output: TextOutput = TextOutput()
        self._streaming_output: StreamingTextOutput = self._build_streaming_output()

        self._listener: SideButtonListener = SideButtonListener(
            on_front_press=self._on_front_press,
            on_front_release=self._on_front_release,
            on_rear_press=self._on_rear_press,
            front_button=config.front_button,
            rear_button=config.rear_button,
            debounce_s=config.button_debounce_ms / 1000.0,
        )

        self._stop_event: threading.Event = threading.Event()
        self._workers_lock: threading.Lock = threading.Lock()
        self._workers: set[threading.Thread] = set()
        self._session: object | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Signal the app to stop gracefully."""
        self._stop_event.set()

    def run(self) -> None:
        # Pre-load the model BEFORE starting the listener so that button
        # callbacks never block the Win32 hook thread with a long download.
        print("Loading ASR model (first run may download models) ...")
        self._transcriber.ensure_loaded()

        # Start the microphone in hot-standby so the pre-buffer fills
        # before the first button press.
        self._recorder.ensure_hot()

        self._listener.start()
        status_msg = (
            "VibeMouse ready (VAD+offline mode). "
            + f"auto_paste={self._config.auto_paste}, "
            + f"enter_mode={self._config.enter_mode}, "
            + f"debounce_ms={self._config.button_debounce_ms}, "
            + f"front_button={self._config.front_button}, "
            + f"rear_button={self._config.rear_button}. "
            + "Hold side-front to record, release to stop. Side-rear sends Enter."
        )
        print(status_msg)
        self._notify("ready", status_msg)
        try:
            _ = self._stop_event.wait()
        except KeyboardInterrupt:
            self._stop_event.set()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._listener.stop()
        self._recorder.shutdown()

        # Stop any active session so its decode thread stops emitting results
        # that would otherwise be typed into the focused window during exit.
        session = self._session
        self._session = None
        if session is not None:
            try:
                session.stop()
            except Exception:
                pass

        # Erase any partially typed text left on screen.
        try:
            self._streaming_output.cancel()
        except Exception:
            pass

        with self._workers_lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=5)

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_front_press(self) -> None:
        if not self._recorder.is_recording:
            self._start_streaming()

    def _on_front_release(self) -> None:
        if self._recorder.is_recording:
            self._stop_streaming()

    def _on_rear_press(self) -> None:
        try:
            self._output.send_enter(mode=self._config.enter_mode)
            if self._config.enter_mode == "none":
                print("Enter key handling disabled (enter_mode=none)")
            else:
                print("Enter key sent")
        except Exception as error:
            print(f"Failed to send Enter: {error}")

    # ------------------------------------------------------------------
    # Streaming flow
    # ------------------------------------------------------------------

    def _start_streaming(self) -> None:
        # Reset streaming output so a previous finalization cannot corrupt
        # the new session's on-screen text.
        self._streaming_output.finalize()

        try:
            session = self._transcriber.start_session(
                on_result=self._on_streaming_result,
            )
        except Exception as error:
            print(f"Failed to create streaming session: {error}")
            self._notify("error", str(error))
            return

        try:
            self._recorder.start(on_chunk=session.feed_audio)
        except Exception as error:
            # Clean up the session whose decode thread is already running.
            try:
                session.stop()
            except Exception:
                pass
            print(f"Failed to start recording: {error}")
            self._notify("error", str(error))
            return

        self._session = session
        print("Recording started (streaming)")
        self._notify("recording_start")

    def _stop_streaming(self) -> None:
        self._notify("recording_stop")

        session = self._session
        self._session = None
        if session is None:
            self._recorder.cancel()
            return

        worker = threading.Thread(
            target=self._finalize_streaming,
            args=(session,),
            daemon=True,
        )
        with self._workers_lock:
            self._workers.add(worker)
        worker.start()

    def _finalize_streaming(self, session: object) -> None:
        current = threading.current_thread()
        try:
            # Keep the audio stream alive briefly so the OS audio buffer
            # drains into the session, capturing the tail of speech that
            # would otherwise be lost on a fast button release.
            time.sleep(0.2)
        finally:
            self._recorder.cancel()
        try:
            final_text = session.stop()

            if not final_text:
                print("No speech recognized (streaming)")
            else:
                print(f"Streaming done: {len(final_text)} chars")

            self._notify("transcribed", final_text)
        except Exception as error:
            print(f"Streaming finalization failed: {error}")
            self._notify("error", str(error))
        finally:
            # Always reset the streaming output so stale state does not
            # leak into the next session.
            try:
                self._streaming_output.finalize()
            except Exception:
                pass
            with self._workers_lock:
                self._workers.discard(current)

    def _on_streaming_result(self, result: StreamingResult) -> None:
        """Called from the decode thread when new text is available."""
        try:
            self._streaming_output.update(result.text)
            self._notify("streaming", result.text)
        except Exception as error:
            print(f"Streaming output error: {error}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _notify(self, event: str, detail: str = "") -> None:
        if self._on_status_change is not None:
            try:
                self._on_status_change(event, detail)
            except Exception:
                pass

    def _build_streaming_output(self) -> StreamingTextOutput:
        # Reuse the pynput.keyboard module that TextOutput already loaded
        # instead of a redundant importlib call.
        import pynput.keyboard

        from vibemouse.streaming_output import _IS_WINDOWS

        type_fn = None
        backspace_fn = None
        if _IS_WINDOWS:
            from vibemouse.streaming_output import (
                _send_backspaces,
                _send_unicode_string,
            )

            type_fn = _send_unicode_string
            backspace_fn = _send_backspaces

        return StreamingTextOutput(
            keyboard=self._output.keyboard,
            backspace_key=pynput.keyboard.Key.backspace,
            type_fn=type_fn,
            backspace_fn=backspace_fn,
        )
