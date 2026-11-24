"""Local webcam: dissolve started by the V gesture with the index and middle fingers."""

from __future__ import annotations

import argparse
import os
import platform
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from contextlib import ExitStack
from pathlib import Path

if __name__ == "__main__" and not getattr(sys, "frozen", False):
    from mediapipe_log_filter import needs_filter, run_local_filtered
    if needs_filter(sys.argv[1:]):
        raise SystemExit(run_local_filtered(Path(__file__).resolve(), sys.argv[1:]))

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .goku_settings import choose_goku_delay
    from .itachi_effect import CROW_SHEET_PATH, SHARINGAN_PATH, EFFECT_SECONDS as ITACHI_SECONDS, ItachiCrowEffect, ItachiGestureGate, ItachiPoseRecognizer, load_crow_sprites, load_sharingan_art
    from .license_client import LicenseError, activate_license, load_config, show_license_error
    from .goku_effect import (
        GokuGestureGate, GokuPoseRecognizer, GokuTeleport, HOLD_SECONDS,
        TELEPORT_SECONDS,
    )
    from .particles_effect import PARTICLE_SECONDS, ParticleEffect
    from .naruto_effect import (
        EFFECT_SECONDS, LOG_PATH, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from .vanish_sound import SOUND_PATH, TELEPORT_SOUND_PATH, VanishSound
except ImportError:
    from goku_settings import choose_goku_delay
    from itachi_effect import CROW_SHEET_PATH, SHARINGAN_PATH, EFFECT_SECONDS as ITACHI_SECONDS, ItachiCrowEffect, ItachiGestureGate, ItachiPoseRecognizer, load_crow_sprites, load_sharingan_art
    from license_client import LicenseError, activate_license, load_config, show_license_error
    from goku_effect import (
        GokuGestureGate, GokuPoseRecognizer, GokuTeleport, HOLD_SECONDS,
        TELEPORT_SECONDS,
    )
    from particles_effect import PARTICLE_SECONDS, ParticleEffect
    from naruto_effect import (
        EFFECT_SECONDS, LOG_PATH, joined_hands_from_landmarks, landed_log_anchor,
        load_log, overlay_log, render_substitution, substitution_anchor,
    )
    from vanish_sound import SOUND_PATH, TELEPORT_SOUND_PATH, VanishSound


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
WINDOW_NAME = "Webcam - Preview"

SEGMENTER_MODEL = MODEL_DIR / "selfie_segmenter_landscape.tflite"
GESTURE_MODEL = MODEL_DIR / "gesture_recognizer.task"
HAND_MODEL = APP_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"
ALL_MODES = ("normal", "naruto", "particles", "goku", "itachi")
# Firestore licenses the app/device, not individual effects. Report a known
# value until the server rules are updated for newly added local modes.
LICENSE_REPORTED_MODES = frozenset(("normal", "naruto", "particles", "goku"))
EDITION_MODES = {
    "full": ALL_MODES,
    "naruto": ("naruto",),
    "goku": ("goku",),
    "normal_particles": ("normal", "particles"),
}


def edition_modes(edition_file: Path | None = None) -> tuple[str, ...]:
    """Limit a packaged EXE to the modes of the edition named in its bundled edition file."""
    edition_file = edition_file or APP_DIR / "edition.txt"
    if not edition_file.is_file():
        if getattr(sys, "frozen", False):
            raise RuntimeError("Edition settings not found in the EXE.")
        return ALL_MODES
    edition = edition_file.read_text(encoding="utf-8").strip()
    if edition not in EDITION_MODES:
        raise RuntimeError(f"Unknown edition: {edition}")
    return EDITION_MODES[edition]

MODEL_URLS = {
    SEGMENTER_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter_landscape/float16/latest/"
        "selfie_segmenter_landscape.tflite"
    ),
    GESTURE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    ),
    FACE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
}

CALIBRATION_SECONDS = 4.0
BACKGROUND_SAMPLE_SECONDS = 0.55
VANISH_SECONDS = 3.6
APPEAR_SECONDS = 1.0
RELEASE_SECONDS = 0.3
GESTURE_INTERVAL_SECONDS = 0.14
# Phases in which an effect is rendered on the video thread.
HEAVY_PHASES = (
    "vanishing", "naruto_vanishing", "appearing", "goku_vanishing", "goku_appearing",
    "itachi_disappearing", "itachi_appearing",
)


@dataclass
class AppState:
    phase: str = "calibrating"
    calibration_started: float = 0.0
    vanish_started: float | None = None
    last_gesture_check: float = 0.0
    fullscreen: bool = False
    gesture_armed: bool = True
    gesture_free_since: float | None = None

    def gesture_due(self, now: float, interval: float) -> bool:
        if now - self.last_gesture_check < interval:
            return False
        self.last_gesture_check = now
        return True


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError("The frame rate must be between 1 and 120.")
    return fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vanish from the webcam with a hand gesture."
    )
    parser.add_argument(
        "--camera", type=int, default=None,
        help="Camera index; if omitted, you pick the camera in the start menu",
    )
    parser.add_argument("--width", type=int, default=None, help="Video width")
    parser.add_argument("--height", type=int, default=None, help="Video height")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Start in full screen"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Do not mirror the camera in the window"
    )
    parser.add_argument("--fps", type=positive_fps, default=30.0, help="Maximum frame rate of the virtual camera (default: 30)")
    parser.add_argument(
        "--virtual-camera", action=argparse.BooleanOptionalAction, default=None,
        help="Require a virtual camera; by default it is used when available. --no-virtual-camera turns it off",
    )
    parser.add_argument(
        "--virtual-backend", choices=("obs", "unitycapture", "v4l2loopback"),
        help="Driver of the virtual camera to use",
    )
    parser.add_argument("--virtual-device", help="Name of the virtual camera to use")
    parser.add_argument(
        "--mode", choices=edition_modes(),
        help="Skip the start menu and launch the chosen effect",
    )
    parser.add_argument(
        "--gesture-debug", action="store_true",
        help="Print what the recogniser sees to the terminal (nothing on the video)",
    )
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.width is None and args.height is None:
        # A MacBook camera delivers 1280x720 whatever is asked for; using it whole keeps the
        # picture sharp instead of shrinking it to 960x540 and letting the call app blow it up.
        args.width, args.height = (1280, 720) if platform.system() == "Darwin" else (960, 540)
    args.width = args.width or 960
    args.height = args.height or 540
    return args


def choose_mode(available: tuple[str, ...] | None = None, check_license=None) -> str | None:
    """Simple keyboard/mouse menu shown before the clean camera feed."""
    available = available or edition_modes()
    display_modes = available
    title = "Choose effect"
    selected = [None]
    compact = len(display_modes) > 6
    row_step = 94 if len(display_modes) > 5 else 115
    card_height = 80 if compact else row_step - 16 if len(display_modes) > 5 else 94
    rows = (len(display_modes) + 1) // 2 if compact else len(display_modes)
    window = np.zeros((120 + rows * row_step + 25, 700, 3), dtype=np.uint8)
    window[:] = (26, 29, 34)
    cv2.putText(window, "CHOOSE YOUR VANISH", (68, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (236, 239, 242), 2, cv2.LINE_AA)
    labels = {
        "normal": ("NORMAL VANISH", "V sign  /  dissolve"),
        "naruto": ("NARUTO VANISH", "Palms together  /  smoke and a log"),
        "particles": ("PARTICLES", "V sign  /  particles in the wind"),
        "goku": ("GOKU", "Two fingers at the forehead  /  adjustable hold"),
        "itachi": ("ITACHI UCHIHA DISAPPEAR", "Hands at the chest  /  a flock of crows"),
    }
    buttons = tuple(
        (mode, f"{index}  {labels[mode][0]}", labels[mode][1],
         (65 + ((index - 1) % 2) * 294 if compact else 65),
         120 + ((index - 1) // 2 if compact else index - 1) * row_step)
        for index, mode in enumerate(display_modes, 1)
    )
    for mode, heading, detail, left, top in buttons:
        disabled = mode not in available
        if disabled:
            heading = labels[mode][0]
            detail = "Coming soon"
        right = left + (277 if compact else 570)
        cv2.rectangle(window, (left, top), (right, top + card_height),
                      (35, 38, 43) if disabled else (48, 54, 62), -1, cv2.LINE_AA)
        cv2.rectangle(window, (left, top), (right, top + card_height),
                      (66, 71, 78) if disabled else (111, 142, 153), 2, cv2.LINE_AA)
        cv2.putText(window, heading, (left + 18, top + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 if compact else 0.77,
                    (115, 120, 128) if disabled else (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(window, detail, (left + 18, top + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37 if compact else 0.53,
                    (135, 141, 150) if disabled else (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP:
            for mode, _, _, left, top in buttons:
                right = left + (277 if compact else 570)
                if mode in available and left <= x <= right and top <= y <= top + card_height:
                    selected[0] = mode

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            if check_license:
                check_license()
            cv2.imshow(title, window)
            key = cv2.waitKey(30) & 0xFF
            if ord("1") <= key < ord("1") + len(display_modes):
                candidate = display_modes[key - ord("1")]
                if candidate in available:
                    selected[0] = candidate
            elif key in (ord("q"), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


def choose_camera(initial_index: int = 0, mirror: bool = True, check_license=None) -> int | None:
    """Preview camera indices 0-9 before opening the clean video output."""
    title = "Choose camera"
    selected = [max(0, min(9, initial_index))]
    confirmed = [False]
    opened_index = None
    capture = None
    preview_ready = False
    read_failures = 0

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONUP or not 472 <= y <= 535:
            return
        if 70 <= x <= 165:
            selected[0] = max(0, selected[0] - 1)
        elif 180 <= x <= 275:
            selected[0] = min(9, selected[0] + 1)
        elif 445 <= x <= 710 and preview_ready:
            confirmed[0] = True

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while True:
            if check_license:
                check_license()
            if selected[0] != opened_index:
                if capture is not None:
                    capture.release()
                try:
                    capture = open_camera(selected[0], 640, 360)
                except RuntimeError:
                    capture = None
                opened_index = selected[0]
                preview_ready = False
                read_failures = 0

            window = np.full((560, 780, 3), (26, 29, 34), dtype=np.uint8)
            cv2.putText(window, "CHOOSE THE CAMERA", (70, 59),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.86, (236, 239, 242), 2, cv2.LINE_AA)
            cv2.putText(window, "0-9 or A/D to switch  |  ENTER to confirm  |  R to retry", (70, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (179, 195, 201), 1, cv2.LINE_AA)
            cv2.rectangle(window, (69, 94), (711, 456), (111, 142, 153), 2)
            if capture is not None:
                ok, frame = capture.read()
                if ok:
                    if mirror:
                        frame = cv2.flip(frame, 1)
                    window[95:455, 70:710] = cv2.resize(frame, (640, 360))
                    preview_ready = True
                    read_failures = 0
                else:
                    read_failures += 1
                    if read_failures >= 10:
                        capture.release()
                        capture = None
                        preview_ready = False
            if not preview_ready:
                message = ("Waiting for the camera signal..." if capture is not None
                           else "No signal from this camera")
                cv2.putText(window, message, (125, 277),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (179, 195, 201), 2, cv2.LINE_AA)

            for left, right, label in ((70, 165, "-"), (180, 275, "+")):
                cv2.rectangle(window, (left, 472), (right, 535), (48, 54, 62), -1)
                cv2.rectangle(window, (left, 472), (right, 535), (111, 142, 153), 2)
                cv2.putText(window, label, (left + 35, 516), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (240, 245, 247), 2, cv2.LINE_AA)
            cv2.putText(window, f"CAM {selected[0]}", (302, 514),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 245, 247), 2, cv2.LINE_AA)
            button_color = (78, 132, 124) if preview_ready else (48, 54, 62)
            cv2.rectangle(window, (445, 472), (710, 535), button_color, -1)
            cv2.putText(window, "USE THIS CAMERA", (480, 511),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57, (240, 245, 247), 2, cv2.LINE_AA)
            if check_license:
                check_license()
            cv2.imshow(title, window)

            key = cv2.waitKey(30) & 0xFF
            if ord("0") <= key <= ord("9"):
                selected[0] = key - ord("0")
            elif key in (ord("a"), ord("A")):
                selected[0] = max(0, selected[0] - 1)
            elif key in (ord("d"), ord("D")):
                selected[0] = min(9, selected[0] + 1)
            elif key in (ord("r"), ord("R")):
                opened_index = None
            elif key in (10, 13) and preview_ready:
                confirmed[0] = True
            elif key in (ord("q"), 27):
                return None
            if confirmed[0] and preview_ready:
                return selected[0]
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        if capture is not None:
            capture.release()
        cv2.destroyWindow(title)


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam

        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = (
            "Virtual camera not available. Install OBS Studio (with the "
            "OBS Virtual Camera driver) or Unity Capture. If you use OBS Virtual Camera, "
            "stop 'Start Virtual Camera' in OBS. "
            "Start from AVVIA_LOCALE.bat (Windows) or AVVIA_MAC.command (Mac) to install pyvirtualcam. "
            f"Details: {error}"
        )
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print(f"[Webcam] {message}\n[Webcam] Continuing with the local preview.")
        return None
    print(f"[Webcam] Virtual camera active: {camera.device} ({width}x{height}, up to {args.fps:g} FPS)")
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame))


def show_output(args: argparse.Namespace, virtual_camera, effect: np.ndarray, camera_frame: np.ndarray) -> None:
    """Give the effect to the call and keep the window a plain monitor of the camera."""
    if virtual_camera is None:
        cv2.imshow(WINDOW_NAME, effect)
        return
    send_virtual_frame(virtual_camera, effect)
    cv2.imshow(WINDOW_NAME, camera_frame if args.no_mirror else cv2.flip(camera_frame, 1))


def download_models(mode: str = "normal") -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for destination, url in MODEL_URLS.items():
        if mode in ("naruto", "goku", "itachi") and destination == GESTURE_MODEL:
            continue
        if mode != "goku" and destination == FACE_MODEL:
            continue
        if destination.exists() and destination.stat().st_size > 100_000:
            continue

        temporary = destination.with_suffix(destination.suffix + ".part")
        print(f"[Vanish Lab] Downloading {destination.name}...")
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Vanish-Lab/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 256):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = int(received * 100 / total)
                            print(f"\r  {percent:3d}%", end="", flush=True)
            print()
            os.replace(temporary, destination)
        except (OSError, urllib.error.URLError) as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                "Could not download the MediaPipe models. "
                "Check the internet connection and try again."
            ) from error


class SizedCapture:
    """Serve every frame at the size the app asked the camera for."""

    def __init__(self, capture, size: tuple[int, int]):
        self.capture, self.size = capture, size

    def read(self) -> tuple[bool, np.ndarray]:
        ok, frame = self.capture.read()
        if ok and frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return ok, frame

    def __getattr__(self, name):
        return getattr(self.capture, name)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture | SizedCapture:
    if platform.system() == "Windows":
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"Camera {index} is not available. Close other apps that may use it, or try --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    given = (capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if given[0] > width and given[1] > height:
        # AVFoundation keeps the camera format, 1280x720 on a MacBook, whatever was asked.
        capture = SizedCapture(capture, (width, height))
    return capture


class AsyncRecognizer:
    """Run a recogniser on its own thread so a slow check never stalls the video.

    The video thread hands over the newest frame and goes on with the previous answer. A frame
    that arrives while a check is running replaces the waiting one instead of queuing up.
    """

    FIRST_ANSWER = {
        "recognize_gesture": ("No_Hand", 0.0), "recognize_joined_hands": False, "recognize_pose": False,
    }

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.answers = dict(self.FIRST_ANSWER)
        self.pending = queue.Queue(maxsize=1)
        self.failure = None
        self.thread = threading.Thread(target=self._work, name="recogniser", daemon=True)
        self.thread.start()

    def __getattr__(self, name):
        attribute = getattr(self.recognizer, name)
        if name.startswith("_") or not callable(attribute):
            return attribute

        def later(*arguments):
            if self.failure:
                raise self.failure
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put((name, arguments))
            return self.answers.get(name)

        return later

    def _work(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            name, arguments = item
            try:
                self.answers[name] = getattr(self.recognizer, name)(*arguments)
            except Exception as error:  # Show it on the video thread instead of ending silently.
                self.failure = error

    def close(self):
        self.pending.put(None)
        self.thread.join(timeout=5)
        self.recognizer.close()


def create_tasks(mode: str = "normal") -> tuple[vision.ImageSegmenter, AsyncRecognizer]:
    segmenter_options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    if mode == "goku":
        if not HAND_MODEL.is_file() or not FACE_MODEL.is_file():
            raise RuntimeError("Hand or face model missing for Goku.")
        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        face_options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        )
        segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
        try:
            hands = vision.HandLandmarker.create_from_options(hand_options)
            try:
                fallback_hands = vision.HandLandmarker.create_from_options(
                    replace(hand_options, running_mode=vision.RunningMode.IMAGE)
                )
                try:
                    face = vision.FaceDetector.create_from_options(face_options)
                except Exception:
                    fallback_hands.close()
                    raise
            except Exception:
                hands.close()
                raise
        except Exception:
            segmenter.close()
            raise
        return segmenter, AsyncRecognizer(GokuPoseRecognizer(hands, face, fallback_hands))
    if mode in ("naruto", "itachi"):
        if not HAND_MODEL.is_file():
            raise RuntimeError(f"Hand model not found: {HAND_MODEL}")
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.45,
        )
        task_class = vision.HandLandmarker
        wrapper_class = MultiViewHandLandmarker
    else:
        classifier_options = mp.tasks.components.processors.ClassifierOptions(
            score_threshold=0.55,
            category_allowlist=["Victory"],
        )
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(GESTURE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.5,
            canned_gesture_classifier_options=classifier_options,
        )
        task_class = vision.GestureRecognizer
        wrapper_class = MultiViewGestureRecognizer

    segmenter = vision.ImageSegmenter.create_from_options(segmenter_options)
    try:
        primary = task_class.create_from_options(options)
        try:
            alternate = task_class.create_from_options(
                replace(options, running_mode=vision.RunningMode.IMAGE)
            )
        except Exception:
            primary.close()
            raise
    except Exception:
        segmenter.close()
        raise
    if mode == "itachi":
        return segmenter, AsyncRecognizer(ItachiPoseRecognizer(primary, alternate))
    return segmenter, AsyncRecognizer(wrapper_class(primary, alternate))


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def ease_in_out_cubic(value: float) -> float:
    if value < 0.5:
        return 4.0 * value * value * value
    return 1.0 - ((-2.0 * value + 2.0) ** 3) / 2.0


def make_dissolve_fields(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7391)
    y, x = np.mgrid[0:height, 0:width]
    y_normal = y.astype(np.float32) / max(height - 1, 1)
    x_normal = x.astype(np.float32) / max(width - 1, 1)
    low_noise = rng.random((max(2, height // 10), max(2, width // 10)), dtype=np.float32)
    low_noise = cv2.resize(low_noise, (width, height), interpolation=cv2.INTER_CUBIC)
    fine_noise = rng.random((height, width), dtype=np.float32)
    field = (
        (1.0 - y_normal) * 0.72
        + low_noise * 0.17
        + fine_noise * 0.08
        + np.sin(x_normal * 18.0) * 0.025
    )
    return np.clip(field, 0.0, 1.0), fine_noise


def person_mask(
    segmenter: vision.ImageSegmenter,
    image: mp.Image,
    timestamp_ms: int,
    width: int,
    height: int,
) -> np.ndarray:
    result = segmenter.segment_for_video(image, timestamp_ms)
    if not result.confidence_masks:
        return np.zeros((height, width), dtype=np.float32)
    raw = np.squeeze(result.confidence_masks[0].numpy_view()).astype(np.float32)
    resized = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.GaussianBlur(resized, (0, 0), sigmaX=2.2, sigmaY=2.2)
    return np.clip((resized - 0.12) / 0.68, 0.0, 1.0)


def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    # Whole-frame math in uint8: about three times faster than the float version and the same
    # to within two levels on a feathered edge.
    weight = np.repeat((alpha * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    return cv2.add(
        cv2.multiply(person, weight, scale=1 / 255.0),
        cv2.multiply(background, cv2.bitwise_not(weight), scale=1 / 255.0),
    )


def erase_person(frame: np.ndarray, background: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The live frame with the person cut out of it. Only the silhouette's rectangle is touched;
    the rest of the frame already is the camera."""
    x, y, width, height = cv2.boundingRect((mask > 0.01).astype(np.uint8))
    erased = frame.copy()
    if width and height:
        rectangle = np.s_[y:y + height, x:x + width]
        # The stored room where the person is, the camera everywhere else in the rectangle.
        erased[rectangle] = alpha_composite(background[rectangle], frame[rectangle], mask[rectangle])
    return erased


def apply_dissolve(
    frame: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    progress: float,
    field: np.ndarray,
    sparkle_noise: np.ndarray,
    erased: np.ndarray | None = None,
) -> np.ndarray:
    """Dissolve the person over the stored empty room; the rest of the frame stays live.

    `erased` is the frame with the person already cut out of it. Pass it when it was made this
    frame anyway; otherwise it is made here.
    """
    if progress <= 0.002:
        return frame
    if erased is None:
        erased = alpha_composite(background, frame, mask)
    if progress >= 0.999:
        return erased

    points = cv2.findNonZero((mask > 0.01).astype(np.uint8))
    if points is None:
        return erased
    x, y, width, height = cv2.boundingRect(points)
    left, top = max(0, x - 18), max(0, y - 18)
    right = min(mask.shape[1], x + width + 18)
    bottom = min(mask.shape[0], y + height + 18)
    area = np.s_[top:bottom, left:right]
    mask = mask[area]
    field = field[area]
    sparkle_noise = sparkle_noise[area]
    eased = ease_in_out_cubic(progress)
    survival = smoothstep(eased - 0.055, eased + 0.055, field)
    alpha = mask * survival
    output = alpha_composite(frame[area], erased[area], alpha)
    edge = (
        (np.abs(field - eased) < 0.018)
        & (mask > 0.48)
        & (sparkle_noise > 0.54)
    ).astype(np.float32)
    # The glow is soft, so it is blurred on a half-size grid and scaled back: a seventh of the
    # work, and within 0.03 of the full-size glow everywhere.
    area_width, area_height = right - left, bottom - top
    small = cv2.resize(edge, (max(1, area_width // 2), max(1, area_height // 2)), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(
        cv2.GaussianBlur(small, (0, 0), sigmaX=2.5, sigmaY=2.5) * 5.0,
        (area_width, area_height), interpolation=cv2.INTER_LINEAR,
    )
    glow = np.clip(glow + edge, 0.0, 1.0) * 0.8
    # The same glow in integer math: about a third of the time of the float version.
    weight = np.repeat((glow * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    cyan = np.full_like(output, (255, 228, 92))
    result = cv2.add(
        cv2.multiply(output, cv2.bitwise_not(weight), scale=1 / 255.0),
        cv2.multiply(cyan, weight, scale=1 / 255.0),
    )
    canvas = erased
    canvas[area] = result
    return canvas


def victory_landmark_score(points: np.ndarray) -> float:
    """Recognize extended index and middle fingers using 3D distances, not orientation."""
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return 0.0

    straightness = []
    extended = []
    folded = []
    lengths = []
    # MCP, PIP, DIP and tip: index, middle, ring and little fingers.
    for index, base in enumerate((5, 9, 13, 17)):
        finger = points[base : base + 4]
        length = float(np.linalg.norm(np.diff(finger, axis=0), axis=1).sum())
        if length < 1e-6:
            if index < 2:
                return 0.0
            straightness.append(0.0)
            lengths.append(0.0)
            extended.append(False)
            folded.append(True)
            continue
        ratio = float(np.linalg.norm(finger[3] - finger[0])) / length
        tip_distance = float(np.linalg.norm(finger[3] - points[0]))
        joint_distance = float(np.linalg.norm(finger[1] - points[0]))
        straightness.append(ratio)
        lengths.append(length)
        extended.append(ratio >= 0.82 and tip_distance > joint_distance * 1.10)
        folded.append(ratio < 0.78 or tip_distance <= joint_distance * 1.05)

    # From the side, folded fingers can look straight but very short: do not require
    # clearly visible joints for the ring and little fingers.
    reference_length = min(lengths[:2])
    for index, base in ((2, 13), (3, 17)):
        reach = float(np.linalg.norm(points[base + 3] - points[base]))
        folded[index] = folded[index] or reach < reference_length * 0.65
    if not (extended[0] and extended[1] and folded[2] and folded[3]):
        return 0.0
    # The two tips must be separated; the thumb may stay hidden.
    separation = float(np.linalg.norm(points[8] - points[12]))
    if separation < min(lengths[:2]) * 0.20:
        return 0.0
    # Geometric quality of the extended fingers, not the classifier's probability.
    return float(np.clip(min(straightness[:2]), 0.0, 1.0))


def classify_gesture_result(
    result: vision.GestureRecognizerResult,
    image: mp.Image,
) -> tuple[str, float]:
    # The landmarks are returned even when the classifier does not recognize
    # "Victory" (for example with the back of the hand facing the webcam).
    for landmarks in result.hand_world_landmarks:
        points = np.array([(point.x, point.y, point.z) for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    # ALWAYS check the visible geometry too: the estimated depth can be imprecise
    # when the palm and the bent fingers are hidden.
    for landmarks in result.hand_landmarks:
        points = np.array([(point.x * image.width, point.y * image.height, 0.0)
                           for point in landmarks])
        score = victory_landmark_score(points)
        if score:
            return "Victory", score
    for categories in result.gestures:
        for category in categories:
            if category.category_name == "Victory" and category.score >= 0.62:
                return "Victory", float(category.score)
    if result.hand_landmarks or result.hand_world_landmarks:
        return "Hand_Detected", 0.0
    return "No_Hand", 0.0


class MultiViewGestureRecognizer:
    """Retry detection on rotated frames without altering the tracking."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_rotation: int | None = None
        self.preferred_crop: str | None = None
        self.next_retry_ms = 0
        self.retry_cursor = 0

    def recognize_gesture(self, image: mp.Image, timestamp_ms: int) -> tuple[str, float]:
        result = self.primary.recognize_for_video(image, timestamp_ms)
        best = classify_gesture_result(result, image)
        if best[0] == "Victory":
            self.preferred_rotation = None
            self.preferred_crop = None
            return best
        if timestamp_ms < self.next_retry_ms:
            return best

        # IMAGE forces a fresh detection; the VIDEO model stays on the original
        # frame. This avoids inconsistent timestamps/tracking between different rotations.
        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        # The palm in the side pose is often too small in the full frame. The two
        # lower crops overlap and cover both hands, even when the image is mirrored.
        crops = {
            "full": rgb,
            "lower_left": rgb[int(height * 0.28):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.28):, int(width * 0.22):],
        }
        views = [("full", rotation) for rotation in (0, 1, 3, 2)]
        views += [(crop, rotation) for crop in ("lower_left", "lower_right")
                  for rotation in (0, 1, 3)]
        preferred = (self.preferred_crop, self.preferred_rotation)
        selected = []
        if preferred in views:
            selected.append(preferred)
        # Spread the expensive IMAGE attempts over successive video frames.
        while len(selected) < 3:
            view = views[self.retry_cursor % len(views)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(crops[crop], rotation)),
            )
            result = self.alternate.recognize(rotated)
            candidate = classify_gesture_result(result, rotated)
            if candidate[0] == "Victory":
                self.preferred_rotation = rotation
                self.preferred_crop = crop
                self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
                return candidate
            if candidate[0] == "Hand_Detected":
                best = candidate
        self.preferred_rotation = None
        self.preferred_crop = None
        self.next_retry_ms = timestamp_ms + int(GESTURE_INTERVAL_SECONDS * 1000)
        return best

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


class MultiViewHandLandmarker:
    """Track two hands in video and retry difficult views with the hand model."""

    def __init__(self, primary, alternate):
        self.primary = primary
        self.alternate = alternate
        self.preferred_view: tuple[str, int] | None = None
        self.next_retry_ms = 0
        self.last_hand_count = 0
        self.retry_cursor = 0

    def recognize_joined_hands(self, image: mp.Image, timestamp_ms: int) -> bool:
        result = self.primary.detect_for_video(image, timestamp_ms)
        self.last_hand_count = len(result.hand_landmarks)
        if joined_hands_from_landmarks(result.hand_landmarks, image.width, image.height):
            self.preferred_view = None
            return True
        if timestamp_ms < self.next_retry_ms:
            return False

        rgb = image.numpy_view()
        height, width = rgb.shape[:2]
        views = {
            "full": rgb,
            "center": rgb[int(height * 0.18):, int(width * 0.12):int(width * 0.88)],
            "lower_left": rgb[int(height * 0.22):, :int(width * 0.78)],
            "lower_right": rgb[int(height * 0.22):, int(width * 0.22):],
        }
        attempts = [("full", rotation) for rotation in (1, 3, 2)]
        attempts += [("center", rotation) for rotation in (0, 1, 3)]
        attempts += [("lower_left", 0), ("lower_right", 0)]
        selected = []
        if self.preferred_view in attempts:
            selected.append(self.preferred_view)
        while len(selected) < 2:
            view = attempts[self.retry_cursor % len(attempts)]
            self.retry_cursor += 1
            if view not in selected:
                selected.append(view)
        for crop, rotation in selected:
            rotated = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(np.rot90(views[crop], rotation)),
            )
            result = self.alternate.detect(rotated)
            self.last_hand_count = max(self.last_hand_count, len(result.hand_landmarks))
            if joined_hands_from_landmarks(result.hand_landmarks, rotated.width, rotated.height):
                self.preferred_view = (crop, rotation)
                self.next_retry_ms = timestamp_ms + 90
                return True
        self.preferred_view = None
        self.next_retry_ms = timestamp_ms + 80
        return False

    def close(self) -> None:
        try:
            self.primary.close()
        finally:
            self.alternate.close()


def recognize_gesture(
    recognizer: MultiViewGestureRecognizer | AsyncRecognizer,
    image: mp.Image,
    timestamp_ms: int,
) -> tuple[str, float]:
    if hasattr(recognizer, "recognize_gesture"):
        return recognizer.recognize_gesture(image, timestamp_ms)
    return classify_gesture_result(recognizer.recognize_for_video(image, timestamp_ms), image)


def arm_on_release(state: AppState, present: bool, now: float) -> None:
    """The gesture counts again only after it has been let go for a moment."""
    if present:
        state.gesture_free_since = None
    elif state.gesture_free_since is None:
        state.gesture_free_since = now
    elif now - state.gesture_free_since >= RELEASE_SECONDS:
        state.gesture_armed = True


def start_vanish_on_gesture(state: AppState, name: str, score: float, now: float) -> bool:
    """Start the dissolve on the first solid V, or the return while the person is invisible."""
    present = name == "Victory" and score >= 0.62
    arm_on_release(state, present, now)
    if state.phase in ("ready", "vanished") and state.gesture_armed and present:
        state.phase = "vanishing" if state.phase == "ready" else "appearing"
        state.vanish_started = now
        state.gesture_armed = False
        return True
    return False


def start_naruto_on_join(state: AppState, joined: bool, now: float) -> bool:
    arm_on_release(state, joined, now)
    if state.phase in ("ready", "vanished") and state.gesture_armed and joined:
        state.phase = "naruto_vanishing" if state.phase == "ready" else "appearing"
        state.vanish_started = now
        state.gesture_armed = False
        return True
    return False


def set_fullscreen(enabled: bool) -> None:
    mode = cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)


def self_test() -> int:
    """Check bundled resources and model initialization without opening a webcam."""
    load_config()
    modes = edition_modes()
    required = [SEGMENTER_MODEL]
    if any(mode in modes for mode in ("normal", "particles")):
        required.append(GESTURE_MODEL)
    if any(mode in modes for mode in ("naruto", "goku", "itachi")):
        required.append(HAND_MODEL)
    if "goku" in modes:
        required.extend((FACE_MODEL, TELEPORT_SOUND_PATH))
    if "itachi" in modes:
        required.extend((CROW_SHEET_PATH, SHARINGAN_PATH))
    if "naruto" in modes:
        required.extend((LOG_PATH, SOUND_PATH))
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Risorse mancanti: {', '.join(missing)}")
    if "naruto" in modes:
        load_log()
    if "itachi" in modes:
        load_crow_sprites()
        load_sharingan_art()
    for sound_path in (SOUND_PATH, TELEPORT_SOUND_PATH):
        if sound_path not in required:
            continue
        sound = VanishSound(sound_path)
        sound.close()
    for mode in modes:
        segmenter, recognizer = create_tasks(mode)
        try:
            recognizer.close()
        finally:
            segmenter.close()
    print("[Webcam] Self-test complete: models, assets and audio are present.")
    return 0


def run(license_session=None) -> int:
    args = parse_args()
    if getattr(args, "self_test", False):
        return self_test()
    available = edition_modes()
    check = license_session.ensure_valid if license_session is not None else None
    mode = args.mode or (available[0] if len(available) == 1 else (
        choose_mode(available, check_license=check) if check else choose_mode(available)))
    if mode is None:
        return 0
    if license_session is not None:
        license_session.backend.mode = mode if mode in LICENSE_REPORTED_MODES else "full"
    if args.camera is None:
        args.camera = (choose_camera(mirror=not args.no_mirror, check_license=check)
                       if check else choose_camera(mirror=not args.no_mirror))
        if args.camera is None:
            return 0
    if mode == "goku":
        args.goku_delay = choose_goku_delay(check_license=check) if check else choose_goku_delay()
        if args.goku_delay is None:
            return 0
    with ExitStack() as resources:
        if check:
            check()
            return run_camera(args, resources, mode, license_session)
        return run_camera(args, resources, mode)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str, license_session=None) -> int:
    download_models(mode)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    segmenter, recognizer = create_tasks(mode)
    resources.callback(segmenter.close)
    resources.callback(recognizer.close)
    log_sprite = load_log() if mode == "naruto" else None
    sound_path = {"naruto": SOUND_PATH, "goku": TELEPORT_SOUND_PATH}.get(mode)
    vanish_sound = VanishSound(sound_path) if sound_path is not None else None
    if vanish_sound is not None:
        resources.callback(vanish_sound.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    state = AppState(
        calibration_started=time.perf_counter(), fullscreen=args.fullscreen
    )
    if state.fullscreen:
        set_fullscreen(True)

    background: np.ndarray | None = None
    background_samples: list[np.ndarray] = []
    last_mask: np.ndarray | None = None
    last_mask_at = float("-inf")
    dissolve_field: np.ndarray | None = None
    sparkle_noise: np.ndarray | None = None
    log_anchor: tuple[int, int, int] | None = None
    log_cache: dict[int, np.ndarray] = {}
    particle_effect: ParticleEffect | None = None
    itachi_effect: ItachiCrowEffect | None = None
    itachi_gate = ItachiGestureGate() if mode == "itachi" else None
    goku_effect: GokuTeleport | None = None
    goku_delay = getattr(args,"goku_delay",HOLD_SECONDS)
    goku_gate = GokuGestureGate(hold_seconds=goku_delay) if mode == "goku" else None
    goku_countdown_remaining: float | None = None
    virtual_camera = None
    virtual_camera_initialized = False
    calibration_remaining = None
    frame_tick = 0
    last_gesture_report = 0.0
    started = time.perf_counter()
    print("[Webcam] Step out of the frame for 4 seconds: capturing the background.")
    if mode == "goku":
        print(f"[Webcam] Goku: hold two fingers together at the forehead for {goku_delay:g} seconds "
              "to vanish or to return. Let go of the pose before repeating it.")
        print("[Webcam] Keys: SPACE new background, F full screen, Q quit.")
    else:
        print("[Webcam] Keys in the window: SPACE new background, "
              "R reappear, V vanish, F full screen, Q quit.")
        if mode == "itachi":
            print("[Itachi] Touch both hands in front of the centre of your chest. "
                  "Let go of the pose and repeat it to reappear.")
        else:
            print("[Webcam] To reappear, show the gesture again after letting go of it for a moment.")

    def look_for_gesture(image, elapsed_ms: int, now: float) -> bool:
        """One check of the mode's gesture. True when it started the vanish or the return."""
        nonlocal last_gesture_report
        if mode != "naruto":
            name, score = recognize_gesture(recognizer, image, elapsed_ms)
            if args.gesture_debug and now - last_gesture_report >= 1.0:
                print(
                    f"[Gesture] {name}, score={score:.2f}, "
                    f"crop={recognizer.preferred_crop or 'full'}, "
                    f"rotation={recognizer.preferred_rotation or 0}"
                )
                last_gesture_report = now
            return start_vanish_on_gesture(state, name, score, now)
        joined = recognizer.recognize_joined_hands(image, elapsed_ms)
        if args.gesture_debug and now - last_gesture_report >= 1.0:
            print(f"[Hands] detected={recognizer.last_hand_count}, "
                  f"joined={joined}, view={recognizer.preferred_view or 'full'}")
            last_gesture_report = now
        if state.phase not in ("ready", "vanished"):
            arm_on_release(state, joined, now)  # While the effect runs, only note when the palms part.
            return False
        if start_naruto_on_join(state, joined, time.perf_counter()):
            vanish_sound.play()
            return True
        return False

    try:
        while True:
            if license_session is not None:
                license_session.ensure_valid()
            ok, camera_frame = capture.read()
            if not ok:
                raise RuntimeError("The camera stopped delivering frames.")

            height, width = camera_frame.shape[:2]
            if not virtual_camera_initialized:
                virtual_camera = open_virtual_camera(args, width, height)
                virtual_camera_initialized = True
                if virtual_camera is not None:
                    resources.callback(virtual_camera.close)
            if not args.no_mirror and virtual_camera is None:
                # Without a virtual camera the window is the output: the effect, seen as a mirror.
                camera_frame = cv2.flip(camera_frame, 1)
            if mode in ("normal", "particles", "naruto") and (
                    dissolve_field is None or dissolve_field.shape != (height, width)):
                dissolve_field, sparkle_noise = make_dissolve_fields(height, width)
            now = time.perf_counter()

            elapsed_ms = int((now - started) * 1000)
            mp_image = None

            if (mode == "itachi" and background is not None
                    and state.phase in ("ready", "vanished")
                    and now - state.last_gesture_check >= 0.09):
                state.last_gesture_check = now
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                last_mask, last_mask_at = mask, now
                touching = recognizer.recognize_pose(mp_image, elapsed_ms, mask)
                if args.gesture_debug and now - last_gesture_report >= 1.0:
                    print(f"[Itachi] hands={recognizer.last_hand_count}, chest={touching}")
                    last_gesture_report = now
                if itachi_gate.update(touching, now):
                    try:
                        itachi_effect = ItachiCrowEffect(camera_frame, mask)
                    except ValueError:
                        itachi_gate.reset()
                    else:
                        appearing = state.phase == "vanished"
                        state.phase = "itachi_appearing" if appearing else "itachi_disappearing"
                        state.vanish_started = now
                        print("[Itachi] Crows gathering." if appearing
                              else "[Itachi] Scattering into crows.")

            if (mode == "goku" and background is not None
                    and state.phase in ("ready", "vanished")
                    and now - state.last_gesture_check >= 0.10):
                state.last_gesture_check = now
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pose = recognizer.recognize_pose(mp_image, elapsed_ms)
                if args.gesture_debug and now - last_gesture_report >= 1.0:
                    held = 0.0 if goku_gate.hold_started is None else now - goku_gate.hold_started
                    print(f"[Goku] faces={recognizer.last_face_count}, "
                          f"hands={recognizer.last_hand_count}, pose={pose}, "
                          f"held={held:.2f}/{goku_gate.hold_seconds:g} s")
                    last_gesture_report = now
                if goku_gate.update(pose, now):
                    action = "Reappearing" if state.phase == "vanished" else "Vanishing"
                    print(f"[Goku] {action}!")
                    goku_countdown_remaining = None
                    mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    goku_effect = GokuTeleport(camera_frame, background, mask)
                    appearing = state.phase == "vanished"
                    state.phase = "goku_appearing" if appearing else "goku_vanishing"
                    state.vanish_started = time.perf_counter()
                    vanish_sound.play()
                elif pose and goku_gate.armed and goku_gate.hold_started is not None:
                    left = goku_gate.hold_seconds - (now - goku_gate.hold_started)
                    remaining = (max(.1,float(np.ceil(left*10)/10)) if goku_gate.hold_seconds < 1
                                 else max(1,int(np.ceil(left))))
                    if remaining != goku_countdown_remaining:
                        action = "reappear" if state.phase == "vanished" else "vanish"
                        print(f"[Goku] {action.capitalize()} in {remaining:g}...")
                        goku_countdown_remaining = remaining
                elif goku_gate.hold_started is None:
                    goku_countdown_remaining = None

            if state.phase == "calibrating":
                elapsed = now - state.calibration_started
                remaining = max(0, int(np.ceil(CALIBRATION_SECONDS - elapsed)))
                if remaining != calibration_remaining:
                    print(f"[Webcam] Capturing background: {remaining} second{'' if remaining == 1 else 's'}.")
                    calibration_remaining = remaining
                if elapsed >= CALIBRATION_SECONDS - BACKGROUND_SAMPLE_SECONDS:
                    background_samples.append(camera_frame.copy())
                if elapsed >= CALIBRATION_SECONDS:
                    if background_samples:
                        stack = np.stack(background_samples[-18:], axis=0)
                        background = np.median(stack, axis=0).astype(np.uint8)
                    else:
                        background = camera_frame.copy()
                    background_samples.clear()
                    state.phase = "ready"
                    print("[Webcam] Background ready. You can come back into the frame.")
                display = camera_frame

            elif background is None:
                state.phase = "calibrating"
                state.calibration_started = now
                display = camera_frame

            elif mode == "goku" and state.phase in ("goku_vanishing", "goku_appearing"):
                appearing = state.phase == "goku_appearing"
                age = time.perf_counter() - state.vanish_started
                display = goku_effect.render(background, age, appearing)
                if age >= TELEPORT_SECONDS:
                    state.phase = "ready" if appearing else "vanished"

            elif mode == "itachi" and state.phase in ("itachi_disappearing", "itachi_appearing"):
                appearing = state.phase == "itachi_appearing"
                age = max(0.0, time.perf_counter() - state.vanish_started)
                display = itachi_effect.render(background, age, appearing)
                if age >= ITACHI_SECONDS:
                    state.phase = "ready" if appearing else "vanished"
                    if appearing:
                        itachi_effect = None  # The crows are gone; the ink stays while the person is invisible.

            elif state.phase == "vanished":
                # The room stays live: only the person is cut out of the camera picture.
                if last_mask is None or now - last_mask_at >= 0.05:
                    if mp_image is None:
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    last_mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    last_mask_at = now
                display = erase_person(camera_frame, background, last_mask)
                if mode == "naruto" and log_anchor is not None:
                    overlay_log(display, log_sprite, landed_log_anchor(log_anchor, height),
                                cache=log_cache)
                if mode == "itachi" and itachi_effect is not None:
                    display = itachi_effect.blood_view(display)
                if (mode not in ("goku", "itachi")
                        and state.gesture_due(
                            now, 0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)):
                    if mp_image is None:
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if look_for_gesture(mp_image, elapsed_ms, now):
                        print("[Webcam] Reappearing.")

            elif state.phase == "appearing":
                if last_mask is None or now - last_mask_at >= 0.05:
                    if mp_image is None:
                        rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    last_mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    last_mask_at = now
                age = time.perf_counter() - state.vanish_started
                display = apply_dissolve(
                    camera_frame, background, last_mask, 1.0 - min(1.0, age / APPEAR_SECONDS),
                    dissolve_field, sparkle_noise,
                    erased=erase_person(camera_frame, background, last_mask),
                )
                if age >= APPEAR_SECONDS:
                    state.phase = "ready"
                    log_anchor = None
                    last_mask = None
                    last_mask_at = float("-inf")
                    particle_effect = None

            else:
                mask = last_mask
                # While ready, the gesture starts the vanish. While the effect runs, the checks
                # only learn when it is let go, so a repeat shown too early does not count.
                if (state.phase in ("ready", "vanishing", "naruto_vanishing")
                        and mode not in ("goku", "itachi")
                        and state.gesture_due(
                            now, 0.08 if mode == "naruto" else GESTURE_INTERVAL_SECONDS)):
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    if look_for_gesture(mp_image, elapsed_ms, now):
                        print("[Webcam] Vanishing.")

                if state.phase in ("vanishing", "naruto_vanishing"):
                    # The silhouette is tracked at 20 Hz so the live person stays erased
                    # while the effect runs over the stored room.
                    if mask is None or now - last_mask_at >= 0.05:
                        if mp_image is None:
                            rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                        last_mask = mask
                        last_mask_at = now
                    if mode == "naruto" and log_anchor is None:
                        log_anchor = substitution_anchor(mask)
                    erased = erase_person(camera_frame, background, mask)

                if mode == "naruto" and state.phase == "naruto_vanishing":
                    age = max(0.04, time.perf_counter() - state.vanish_started)
                    display = render_substitution(
                        camera_frame, erased, mask, log_sprite, log_anchor, age, log_cache
                    )
                    if age >= EFFECT_SECONDS:
                        state.phase = "vanished"
                elif mode == "particles" and state.phase == "vanishing":
                    if particle_effect is None:
                        particle_effect = ParticleEffect(camera_frame, mask)
                    age = time.perf_counter() - state.vanish_started
                    display = particle_effect.render(erased, age)
                    if age >= PARTICLE_SECONDS:
                        state.phase = "vanished"
                elif state.phase == "ready":
                    display = camera_frame
                else:
                    progress = 0.0
                    if state.phase == "vanishing" and state.vanish_started is not None:
                        progress = min(
                            1.0, (time.perf_counter() - state.vanish_started) / VANISH_SECONDS
                        )
                        if progress >= 1.0:
                            state.phase = "vanished"
                    display = apply_dissolve(
                        camera_frame, background, mask, progress, dissolve_field, sparkle_noise,
                        erased=erased,
                    )

            if license_session is not None:
                license_session.ensure_valid()
            show_output(args, virtual_camera, display, camera_frame)

            frame_tick += 1
            if virtual_camera is not None and state.phase in HEAVY_PHASES and not frame_tick % 2:
                # Servicing the window costs a display refresh on macOS. While an effect renders,
                # the window is only a monitor: it is serviced every other frame and the call
                # gets the time. A key pressed meanwhile waits for the next service.
                key = -1
            else:
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break  # The window was closed with its button: quit, like any other app.
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                state.phase = "calibrating"
                state.calibration_started = now
                state.vanish_started = None
                state.gesture_armed = True
                state.gesture_free_since = None
                log_anchor = None
                last_mask = None
                last_mask_at = float("-inf")
                particle_effect = None
                itachi_effect = None
                if itachi_gate is not None:
                    itachi_gate.reset()
                goku_effect = None
                if goku_gate is not None:
                    goku_gate.reset()
                    goku_countdown_remaining = None
                background_samples.clear()
                calibration_remaining = None
                print("[Webcam] Step out of the frame for 4 seconds: capturing a new background.")
            elif key == ord("r"):
                if background is not None and mode == "itachi" and state.phase == "vanished":
                    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                    try:
                        itachi_effect = ItachiCrowEffect(camera_frame, mask)
                    except ValueError:
                        print("[Itachi] Come back into the frame before reappearing.")
                    else:
                        state.phase = "itachi_appearing"
                        state.vanish_started = time.perf_counter()
                elif background is not None and mode not in ("goku", "itachi"):
                    state.phase = "appearing" if state.phase == "vanished" else "ready"
                    state.vanish_started = time.perf_counter()
                    log_anchor = None
                    last_mask = None
                    last_mask_at = float("-inf")
                    particle_effect = None
            elif key == ord("v") and background is not None and mode == "itachi" and state.phase == "ready":
                rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mask = person_mask(segmenter, mp_image, elapsed_ms, width, height)
                try:
                    itachi_effect = ItachiCrowEffect(camera_frame, mask)
                except ValueError:
                    print("[Itachi] No person detected.")
                else:
                    state.phase = "itachi_disappearing"
                    state.vanish_started = time.perf_counter()
            elif key == ord("v") and background is not None and mode not in ("goku", "itachi"):
                state.phase = "naruto_vanishing" if mode == "naruto" else "vanishing"
                state.vanish_started = time.perf_counter()
                if mode == "naruto":
                    vanish_sound.play()
                    log_anchor = None
                particle_effect = None
            elif key == ord("f"):
                state.fullscreen = not state.fullscreen
                set_fullscreen(state.fullscreen)
            if virtual_camera is not None:
                virtual_camera.sleep_until_next_frame()
    finally:
        resources.close()

    return 0


def main() -> int:
    try:
        if any(flag in sys.argv[1:] for flag in ("--self-test", "--help", "-h")):
            return run()  # Model/resource checks only; never opens a camera.
        if not getattr(sys, "frozen", False):
            print("[Webcam] Local start from Python: no online license check.")
            return run()
        session = activate_license()
        if session is None:
            return 0
        with session:
            return run(session)
    except LicenseError as error:
        print(f"[License] {error}", file=sys.stderr)
        show_license_error(error)
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\n[Vanish Lab] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
