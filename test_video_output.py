import argparse
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from local_app import vanish_cam as app


def arguments(**overrides):
    values = dict(
        camera=0, width=960, height=540, fps=30.0, fullscreen=False,
        no_mirror=True, virtual_camera=None, virtual_backend=None, virtual_device=None,
        gesture_debug=False, mode="normal",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class VirtualCameraTests(unittest.TestCase):
    def test_bgr_format_and_actual_frame_dimensions(self):
        camera = Mock(device="OBS Virtual Camera")
        module = SimpleNamespace(Camera=Mock(return_value=camera), PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIs(app.open_virtual_camera(arguments(), 640, 480), camera)
        module.Camera.assert_called_once_with(
            width=640, height=480, fps=30.0, fmt="BGR", backend=None, device=None,
        )

    def test_unavailable_driver_auto_falls_back_but_explicit_mode_fails(self):
        module = SimpleNamespace(Camera=Mock(side_effect=RuntimeError("driver missing")),
                                 PixelFormat=SimpleNamespace(BGR="BGR"))
        with patch.dict("sys.modules", pyvirtualcam=module), patch("builtins.print"):
            self.assertIsNone(app.open_virtual_camera(arguments(), 640, 480))
            for options in (dict(virtual_camera=True), dict(virtual_backend="obs")):
                with self.assertRaisesRegex(RuntimeError, "driver missing"):
                    app.open_virtual_camera(arguments(**options), 640, 480)

    def test_disabled_camera_does_not_open_device(self):
        module = SimpleNamespace(Camera=Mock())
        with patch.dict("sys.modules", pyvirtualcam=module):
            self.assertIsNone(app.open_virtual_camera(arguments(virtual_camera=False), 640, 480))
        module.Camera.assert_not_called()

    def test_send_preserves_colors_and_handles_noncontiguous_frames(self):
        camera = Mock(width=40, height=24)
        frame = np.zeros((24, 40, 3), np.uint8)
        frame[:] = (10, 70, 200)  # BGR: red and blue must not be swapped.
        app.send_virtual_frame(camera, frame[:, ::-1])
        sent = camera.send.call_args.args[0]
        self.assertTrue(sent.flags.c_contiguous)
        np.testing.assert_array_equal(sent, frame)

    def test_send_resizes_if_input_resolution_changes(self):
        camera = Mock(width=40, height=24)
        app.send_virtual_frame(camera, np.full((48, 80, 3), 42, np.uint8))
        sent = camera.send.call_args.args[0]
        self.assertEqual(sent.shape, (24, 40, 3))
        self.assertTrue((sent == 42).all())


class MenuTests(unittest.TestCase):
    def test_camera_selector_previews_and_switches_devices(self):
        cameras = [Mock(), Mock()]
        for camera in cameras:
            camera.read.return_value = (True, np.zeros((24, 40, 3), np.uint8))
        with patch.object(app, "open_camera", side_effect=cameras) as opening, \
             patch.object(app.cv2, "namedWindow"), \
             patch.object(app.cv2, "setMouseCallback"), \
             patch.object(app.cv2, "imshow"), \
             patch.object(app.cv2, "destroyWindow"), \
             patch.object(app.cv2, "getWindowProperty", return_value=1), \
             patch.object(app.cv2, "waitKey", side_effect=(ord("1"), 13)):
            self.assertEqual(app.choose_camera(), 1)
        opening.assert_has_calls((call(0, 640, 360), call(1, 640, 360)))
        for camera in cameras:
            camera.release.assert_called_once()

    def test_camera_selector_waits_for_an_available_device(self):
        camera = Mock()
        camera.read.return_value = (True, np.zeros((24, 40, 3), np.uint8))
        with patch.object(app, "open_camera", side_effect=(RuntimeError("missing"), camera)), \
             patch.object(app.cv2, "namedWindow"), \
             patch.object(app.cv2, "setMouseCallback"), \
             patch.object(app.cv2, "imshow"), \
             patch.object(app.cv2, "destroyWindow"), \
             patch.object(app.cv2, "getWindowProperty", return_value=1), \
             patch.object(app.cv2, "waitKey", side_effect=(13, ord("1"), 13)):
            self.assertEqual(app.choose_camera(), 1)
        camera.release.assert_called_once()

    def test_camera_selector_waits_for_first_preview_frame(self):
        camera = Mock()
        camera.read.side_effect = (
            (False, None),
            (True, np.zeros((24, 40, 3), np.uint8)),
        )
        with patch.object(app, "open_camera", return_value=camera), \
             patch.object(app.cv2, "namedWindow"), \
             patch.object(app.cv2, "setMouseCallback"), \
             patch.object(app.cv2, "imshow"), \
             patch.object(app.cv2, "destroyWindow"), \
             patch.object(app.cv2, "getWindowProperty", return_value=1), \
             patch.object(app.cv2, "waitKey", side_effect=(13, 13)):
            self.assertEqual(app.choose_camera(), 0)
        self.assertEqual(camera.read.call_count, 2)
        camera.release.assert_called_once()

    def test_packaged_edition_limits_available_modes(self):
        with TemporaryDirectory() as directory:
            edition_file = Path(directory) / "edition.txt"
            for edition, expected in app.EDITION_MODES.items():
                edition_file.write_text(edition, encoding="utf-8")
                self.assertEqual(app.edition_modes(edition_file), expected)
            edition_file.write_text("unknown", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unknown edition"):
                app.edition_modes(edition_file)

    def test_default_size_follows_the_camera_of_the_platform(self):
        for system, expected in (("Darwin", (1280, 720)), ("Windows", (960, 540))):
            with self.subTest(system=system), patch.object(app.sys, "argv", ["vanish_cam.py"]), \
                 patch.object(app.platform, "system", return_value=system):
                args = app.parse_args()
            self.assertEqual((args.width, args.height), expected)
        with patch.object(app.sys, "argv", ["vanish_cam.py", "--width", "640"]), \
             patch.object(app.platform, "system", return_value="Darwin"):
            args = app.parse_args()
        self.assertEqual((args.width, args.height), (640, 540))  # A size given by hand is kept.

    def test_single_mode_edition_starts_without_a_menu(self):
        with patch.object(app, "edition_modes", return_value=("naruto",)), \
             patch.object(app, "parse_args", return_value=arguments(mode=None)), \
             patch.object(app, "choose_mode") as menu, \
             patch.object(app, "choose_camera") as camera_menu, \
             patch.object(app, "run_camera", return_value=0) as camera:
            self.assertEqual(app.run(), 0)
        menu.assert_not_called()
        camera_menu.assert_not_called()
        self.assertEqual(camera.call_args.args[2], "naruto")

    def test_run_uses_selected_camera_when_not_set_on_command_line(self):
        args = arguments(mode=None, camera=None)
        with patch.object(app, "edition_modes", return_value=("goku",)), \
             patch.object(app, "parse_args", return_value=args), \
             patch.object(app, "choose_camera", return_value=2) as menu, \
             patch.object(app, "choose_goku_delay", return_value=0.5), \
             patch.object(app, "run_camera", return_value=0) as camera:
            self.assertEqual(app.run(), 0)
        menu.assert_called_once_with(mirror=False)
        self.assertEqual(camera.call_args.args[0].camera, 2)
        self.assertEqual(camera.call_args.args[0].goku_delay, 0.5)

    def test_combined_edition_menu_has_only_normal_and_particles(self):
        with patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "setMouseCallback"), \
             patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyWindow"), \
             patch.object(app.cv2, "waitKey", return_value=ord("2")), \
             patch.object(app.cv2, "getWindowProperty", return_value=1):
            self.assertEqual(app.choose_mode(("normal", "particles")), "particles")

    def test_particles_can_be_selected_with_three(self):
        with patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "setMouseCallback"), \
             patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyWindow"), \
             patch.object(app.cv2, "waitKey", return_value=ord("3")), \
             patch.object(app.cv2, "getWindowProperty", return_value=1):
            self.assertEqual(app.choose_mode(), "particles")

    def test_goku_can_be_selected_with_four(self):
        with patch.object(app.cv2, "namedWindow"), patch.object(app.cv2, "setMouseCallback"), \
             patch.object(app.cv2, "imshow"), patch.object(app.cv2, "destroyWindow"), \
             patch.object(app.cv2, "waitKey", return_value=ord("4")), \
             patch.object(app.cv2, "getWindowProperty", return_value=1):
            self.assertEqual(app.choose_mode(), "goku")


class VideoLoopTests(unittest.TestCase):
    def run_mock_video(self, send_error=None, gesture=("No_Hand", 0.0), mode="normal", mask=None,
                      window_visible=1.0):
        capture = Mock()
        frames = [np.full((24, 40, 3), color, np.uint8) for color in (10, 40, 80, 120, 160)]
        frame_index = [0]

        def read():
            frame_index[0] += 1
            return True, frames[frame_index[0] - 1]

        capture.read.side_effect = read  # One frame per iteration; past the last one it fails.
        segmenter, recognizer = Mock(), Mock()
        camera = Mock(width=40, height=24)
        camera.send.side_effect = send_error
        shown = []
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "parse_args", return_value=arguments(mode=mode)))
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            opening = mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            segment = mocks.enter_context(patch.object(
                app, "person_mask",
                return_value=np.ones((24, 40), np.float32) if mask is None else mask,
            ))
            mocks.enter_context(patch.object(app, "recognize_gesture", return_value=gesture))
            times = (0, 0, 0.1, 4.1, 4.3, 4.35, 5.0, 5.02, 8.2, 8.23)
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=times))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=window_visible))
            destroy = mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: shown.append(frame.copy())))
            # A key belongs to the frame it was pressed in, whether or not the window was serviced then.
            keys = (0, 0, ord("v") if gesture[0] != "Victory" else 0, 0, ord("q"))
            pump = mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=lambda delay: keys[frame_index[0] - 1]))
            mocks.enter_context(patch.object(app.cv2, "putText", side_effect=AssertionError("Overlay is forbidden")))
            mocks.enter_context(patch("builtins.print"))
            if send_error is None:
                self.assertEqual(app.run(), 0)
                if window_visible >= 1:
                    self.assertEqual(pump.call_count, 4)  # While the effect runs, the window is serviced every other frame.
            else:
                with self.assertRaisesRegex(RuntimeError, "send failed"):
                    app.run()
            opening.assert_called_once()
            self.assertEqual(opening.call_args.args[1:], (40, 24))
            capture.release.assert_called_once()
            segmenter.close.assert_called_once()
            recognizer.close.assert_called_once()
            camera.close.assert_called_once()
            destroy.assert_called_once()
            if mode == "particles":
                self.assertGreaterEqual(segment.call_count, 1)  # The silhouette is tracked during the effect.
        return frames, camera, shown, [call.args[0] for call in camera.send.call_args_list]

    def test_window_is_a_plain_monitor_and_the_call_gets_the_effect(self):
        frames, camera, shown, sent = self.run_mock_video()
        self.assertEqual(len(shown), 5)
        for monitor, raw in zip(shown, frames):
            np.testing.assert_array_equal(monitor, raw)  # The window does not show the effect.
        self.assertEqual(len(sent), 5)
        np.testing.assert_array_equal(sent[0], frames[0])  # Calibration.
        np.testing.assert_array_equal(sent[2], frames[2])  # Ready.
        np.testing.assert_array_equal(sent[4], frames[1])  # Vanished: background only.

    def test_the_monitor_mirrors_the_camera_while_the_call_does_not(self):
        effect = np.full((24, 40, 3), 90, np.uint8)
        camera_frame = np.zeros((24, 40, 3), np.uint8)
        camera_frame[:, :20] = (10, 200, 30)  # Two different halves, so a flip is visible.
        for no_mirror, expected in ((False, camera_frame[:, ::-1]), (True, camera_frame)):
            with self.subTest(no_mirror=no_mirror):
                camera = Mock(width=40, height=24)
                shown = []
                with patch.object(app.cv2, "imshow", side_effect=lambda name, image: shown.append(image.copy())):
                    app.show_output(arguments(no_mirror=no_mirror), camera, effect, camera_frame)
                np.testing.assert_array_equal(shown[0], expected)
                np.testing.assert_array_equal(camera.send.call_args.args[0], effect)

    def test_without_a_virtual_camera_the_window_shows_the_effect(self):
        effect = np.full((24, 40, 3), 90, np.uint8)
        shown = []
        with patch.object(app.cv2, "imshow", side_effect=lambda name, image: shown.append(image.copy())):
            app.show_output(arguments(), None, effect, np.zeros((24, 40, 3), np.uint8))
        np.testing.assert_array_equal(shown[0], effect)

    def test_first_gesture_starts_dissolve_in_the_call(self):
        frames, camera, shown, sent = self.run_mock_video(gesture=("Victory", 0.9))
        np.testing.assert_array_equal(sent[2][:12], frames[2][:12])  # The dissolve starts from the bottom.
        self.assertTrue(np.any(sent[3] != frames[3]))  # Animation in progress.
        self.assertTrue(np.any(sent[3] != frames[1]))  # Not vanished yet.
        np.testing.assert_array_equal(sent[4], frames[1])  # Animation complete.

    def test_particles_use_same_v_gesture_and_virtual_output(self):
        frames, camera, shown, sent = self.run_mock_video(
            gesture=("Victory", 0.9), mode="particles",
        )
        self.assertTrue(np.any(sent[2] != frames[2]))  # Starts at once.
        self.assertTrue(np.any(sent[3] != frames[3]))  # The body dissolves.
        np.testing.assert_array_equal(sent[4], frames[1])

    def test_vanished_room_follows_the_camera_outside_the_person(self):
        half = np.zeros((24, 40), np.float32)
        half[:, :20] = 1.0
        frames, camera, shown, sent = self.run_mock_video(mask=half)
        # Vanished: the stored room where the person is, the live camera everywhere else.
        np.testing.assert_array_equal(sent[4][:, :20], frames[1][:, :20])
        np.testing.assert_array_equal(sent[4][:, 20:], frames[4][:, 20:])

    def test_all_devices_close_on_send_error(self):
        self.run_mock_video(send_error=RuntimeError("send failed"))

    def test_closing_the_window_quits_cleanly(self):
        frames, camera, shown, sent = self.run_mock_video(window_visible=-1.0)
        self.assertEqual(len(shown), 1)  # One frame, then the closed window is seen.

    def test_goku_vanishes_and_reappears_only_after_one_second_holds(self):
        first_hold = tuple(4.3 + index * 0.101 for index in range(11))
        second_hold = tuple(6.4 + index * 0.101 for index in range(11))
        times = (0.1, 4.1) + first_hold + (5.4, 5.64, 5.8, 6.2) \
            + second_hold + (7.5, 7.74, 7.9)
        clock = [0.0]
        frames = [np.full((80, 120, 3), 40 if i < 2 else 160, np.uint8)
                  for i in range(len(times))]
        capture = Mock()
        cursor = [0]

        def read_frame():
            index = cursor[0]
            cursor[0] += 1
            clock[0] = times[index]
            return True, frames[index]

        capture.read.side_effect = read_frame
        segmenter = Mock()
        recognizer = Mock()
        recognizer.recognize_pose.side_effect = (
            [True] * 11 + [False] * 2 + [True] * 11 + [False]
        )
        camera = Mock(width=120, height=80)
        sound = Mock()
        mask = np.zeros((80, 120), np.float32)
        mask[10:70, 30:90] = 1.0
        displayed = []
        keys = [0] * len(times)
        keys[2] = ord("v")  # Manual activation is forbidden in Goku mode.
        keys[14] = ord("r")  # Manual return is forbidden too.
        keys[-1] = ord("q")
        with ExitStack() as mocks:
            mocks.enter_context(patch.object(app, "download_models"))
            mocks.enter_context(patch.object(app, "open_camera", return_value=capture))
            mocks.enter_context(patch.object(app, "create_tasks", return_value=(segmenter, recognizer)))
            mocks.enter_context(patch.object(app, "open_virtual_camera", return_value=camera))
            masks = mocks.enter_context(patch.object(app, "person_mask", return_value=mask))
            sound_factory = mocks.enter_context(patch.object(app, "VanishSound", return_value=sound))
            mocks.enter_context(patch.object(app.time, "perf_counter", side_effect=lambda: clock[0]))
            mocks.enter_context(patch.object(app.cv2, "namedWindow"))
            mocks.enter_context(patch.object(app.cv2, "resizeWindow"))
            mocks.enter_context(patch.object(app.cv2, "getWindowProperty", return_value=1))
            mocks.enter_context(patch.object(app.cv2, "destroyAllWindows"))
            mocks.enter_context(patch.object(app.cv2, "imshow", side_effect=lambda name, frame: displayed.append(frame.copy())))
            mocks.enter_context(patch.object(app.cv2, "waitKey", side_effect=lambda delay: keys[cursor[0] - 1]))
            printed = mocks.enter_context(patch("builtins.print"))
            with ExitStack() as resources:
                self.assertEqual(app.run_camera(arguments(mode="goku"), resources, "goku"), 0)

        self.assertGreaterEqual(masks.call_count, 2)  # The silhouette is tracked while invisible too.
        sound_factory.assert_called_once_with(app.TELEPORT_SOUND_PATH)
        self.assertEqual(sound.play.call_count, 2)
        countdown = [call.args[0] for call in printed.call_args_list
                     if call.args and call.args[0].startswith("[Goku] ")]
        self.assertEqual(countdown, [
            "[Goku] Vanish in 1...",
            "[Goku] Vanishing!",
            "[Goku] Reappear in 1...",
            "[Goku] Reappearing!",
        ])
        sent = [call.args[0] for call in camera.send.call_args_list]
        gone = np.where(mask[..., None] > 0, frames[1], frames[16])  # Live room, person removed.
        for monitor, raw in zip(displayed, frames):
            np.testing.assert_array_equal(monitor, raw)  # The window stays a plain camera.
        np.testing.assert_array_equal(sent[2], frames[2])
        np.testing.assert_array_equal(sent[14], frames[1])
        np.testing.assert_array_equal(sent[16], gone)
        np.testing.assert_array_equal(sent[29][40, 60], frames[29][40, 60])


if __name__ == "__main__":
    unittest.main()
