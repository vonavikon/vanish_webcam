# VanishCam (fork of vanish_webcam for macOS)

VanishCam makes a person disappear from a webcam picture with a hand gesture, in five modes. It shows the result in a preview window and can send it to a virtual camera for call apps. This is a fork of [Nicovic17/vanish_webcam](https://github.com/Nicovic17/vanish_webcam) that runs on macOS; the differences are listed under [What changed in this fork](#what-changed-in-this-fork).

## Effects

Every mode starts with a calibration: step out of the frame for 4 seconds while the app captures the empty room. Instructions go to the terminal; nothing is drawn over the video.

| Mode | `--mode` | Gesture | What happens | How to return |
| --- | --- | --- | --- | --- |
| Normal | `normal` | Index and middle fingers extended in a V, ring finger and little finger folded | The person dissolves over the empty room in 3.6 s, with a soft glow on the moving edge | The same gesture after letting go of it for 0.3 s, which starts a 1 s dissolve. `R` also works |
| Naruto | `naruto` | Palms together, both hands open and visible to the camera | Smoke, then a wooden log falls to the floor and stays there. 1.45 s. A sound plays | The same gesture after letting go of it for 0.3 s, or `R` |
| Particles | `particles` | Same as Normal | The silhouette breaks into colored fragments that drift sideways as if carried by the wind. 2.1 s | Same as Normal |
| Goku | `goku` | Index and middle fingers together at the forehead, held for the chosen time (1 s by default) | Vertical speed lines and a hard cut to the empty room after 0.045 s. 0.25 s in total. A sound plays | Lower the hand for at least 0.35 s, then hold the pose again for the same time. `V` and `R` are off |
| Itachi | `itachi` | Both hands touching in front of the center of your chest | The silhouette breaks into crows, the Sharingan eyes appear briefly and red ink covers the room. 4 s | Let go for at least 0.25 s, then touch the hands again. `R` starts the return once you are back in the frame |

While the person is invisible, the area they cover shows the stored background, because the camera cannot see behind them. The rest of the frame is the live camera.

## What changed in this fork

- **macOS support.** A new launcher, `start_mac.command`, creates `.venv-local` with Python 3.12 or 3.13, installs the requirements, rebuilds the environment when it stopped working and appends the terminal output to `~/Library/Logs/VanishCam.log`. `requirements.txt` pins MediaPipe 1.2.3 on macOS, because 1.0.1 aborts the process there when a hand, face or gesture model starts.
- **Windows launchers.** `start_windows.bat` and its two wrappers, `start_windows_virtual_camera.bat` and `start_windows_no_virtual_camera.bat`, work from the root of the repository: `start_windows.bat` creates `.venv-local` there, installs `requirements.txt` and starts `vanish_cam.py`. The files of the original project lived in a `local_app` folder and went one level up. The launchers were not tested on Windows.
- **Frame rate.** Gesture recognition runs on a worker thread (`AsyncRecognizer`), frames larger than requested are brought to the requested size (`SizedCapture`), the silhouette is tracked at 20 Hz while the Normal, Particles or Naruto effects run and while the person is invisible, and the compositing uses integer math. On macOS the default size is 1280x720, which a MacBook camera delivers whatever size is asked for.
- **Live room.** While the person is invisible, the camera picture is shown with the person cut out (`erase_person`) instead of a frozen photo. In Normal, Particles and Naruto the same gesture brings the person back with a short dissolve; the gesture has to be let go first.
- **Virtual camera.** When one is active, the effect goes to the call only, unmirrored, and the app's window is a mirror of the camera. While an effect renders, that window is refreshed every other frame.
- **English.** The interface, the messages and the comments are in English.
- **Launcher names.** The launcher files have English names now; the original project used Italian names. They are `start_mac.command`, `start_windows.bat`, `start_windows_virtual_camera.bat` and `start_windows_no_virtual_camera.bat`.
- **Window close.** Closing the window quits the app and releases the cameras.
- **Removed files.** `run3.py` and `face_landmarker.task`, two unused files of the original author's other app, are gone.

## Requirements

- A webcam, and macOS or Windows. Only macOS on Apple Silicon was tested with this fork.
- Python 3.12 or 3.13. The macOS launcher looks for these two versions only, because `pygame` has no packages for Python 3.14 and later.
- Internet access for the first start: the packages come from PyPI and the MediaPipe models from `storage.googleapis.com`.
- Optional: a virtual camera, when the effect has to reach a call app (see [Virtual camera](#virtual-camera)).

`requirements.txt` lists these packages:

| Package | Version |
| --- | --- |
| `mediapipe` | 1.2.3 on macOS, 1.0.1 on other systems |
| `opencv-contrib-python` | >= 4.14, < 6 |
| `numpy` | >= 2.0, < 3 |
| `pyvirtualcam` | >= 0.15, < 1 |
| `pygame` | >= 2.6, < 3 |

## Install

### macOS

```sh
git clone https://github.com/vonavikon/vanish_webcam.git
cd vanish_webcam
./start_mac.command --mode normal --camera 0
```

The options skip both menus. Double-clicking `start_mac.command` in Finder does the same without options, so the mode menu and the camera menu open. The launcher creates `.venv-local` with Python 3.12 or 3.13, installs the requirements when the packages cannot be imported and starts `vanish_cam.py` without a license check. If the environment stops working, for example after a Python upgrade, the launcher rebuilds it. The terminal output is also appended to `~/Library/Logs/VanishCam.log`.

The same steps without the launcher:

```sh
python3.13 -m venv .venv-local
.venv-local/bin/python -m pip install -r requirements.txt
.venv-local/bin/python vanish_cam.py --mode normal
```

### Common problems

When the app does not start, the launchers of both systems refer to this table. On macOS the messages of the app are also written to `~/Library/Logs/VanishCam.log`; the messages of the launchers themselves are only in the terminal.

| Message in the terminal | What to do |
| --- | --- |
| `Python 3.12 or 3.13 is required` | Install one of the two versions. `pygame` has no packages for Python 3.14 and later |
| `Camera N is not available` | Close other apps that may use the camera, or try another index with `--camera` |
| `Could not download the MediaPipe models` | Check the internet connection and start again |
| `Virtual camera not available` | Install OBS Studio with the OBS Virtual Camera driver, or Unity Capture, and stop the virtual camera in OBS before starting the app. Unless a virtual camera was asked for explicitly, the app then goes on with the window only |
| `The camera stopped delivering frames` | The app quits. Start it again |
| `Install pygame or ffplay to play` | Install the packages of `requirements.txt`; `pygame` is among them |

### Windows

Windows was not tested with this fork. The three launchers were checked by reading, never by running. They come from the original project: this fork changed their paths to the layout of this repository and put their messages in English.

| File | What it does |
| --- | --- |
| `start_windows.bat` | Creates `.venv-local` in the folder where it lies, installs `requirements.txt` and starts `vanish_cam.py` with the given options |
| `start_windows_virtual_camera.bat` | Calls `start_windows.bat` with `--virtual-camera` |
| `start_windows_no_virtual_camera.bat` | Calls `start_windows.bat` with `--no-virtual-camera` |
| `BUILD_WINDOWS.bat` | Builds four executables with PyInstaller from `VanishCam.spec` and runs a self-test on each: `VanishCam.exe` (all modes), `Naruto Vanish Effect.exe`, `Goku Vanish Effect.exe` and `Normal and Particles Vanish.exe` |

The three launchers work from the root of this repository, where they lie together with the sources. `start_windows.bat` creates `.venv-local` there with `py -3`, or with `python` when `py` is absent, installs the requirements when `cv2`, `mediapipe`, `numpy` and `pyvirtualcam` cannot be imported and starts `vanish_cam.py` with the options given on the command line. When a step fails, it prints `Vanish Lab could not start` and refers to [Common problems](#common-problems). `BUILD_WINDOWS.bat` and `VanishCam.spec` use the files of the repository root as well.

The same steps without the launcher, from the root of the repository (not tested):

```powershell
py -3 -m venv .venv-local
.venv-local\Scripts\python -m pip install -r requirements.txt
.venv-local\Scripts\python vanish_cam.py --mode normal
```

`BUILD_WINDOWS.bat` needs a `license_config.json` next to `VanishCam.spec`; the file is ignored by Git. At every start a packaged executable checks the license online through `license_client.py`: it signs in anonymously to Firebase, uses the saved license e-mail or asks for one, binds the license to the Windows device and keeps a session in Firestore that a background thread renews every 25 seconds. `license_config.example.json` holds placeholders only; the repository has no keys. Started from source, the app makes no license request.

### First start

1. Mode menu: choose with the mouse or with keys `1` to `5` (Normal, Naruto, Particles, Goku, Itachi). `--mode` skips it.
2. Camera menu: keys `0` to `9` or `A` and `D` switch the camera; `Enter` or the button confirms once a preview is shown. `--camera N` skips it.
3. Goku only: a dialog asks how long to hold the gesture. Press `Enter` for 1 second, or type a positive number such as `0.5`; a decimal comma is accepted. The same time is used to vanish and to return.
4. The app downloads the models it needs into `models/` when they are missing, then opens the video window.
5. Step out of the frame for 4 seconds. The terminal says when the background is ready.

In the mode menu and the camera menu, `Q` or `ESC` quits. In the hold-time dialog, `ESC` or the `Cancel` button quits as well.

### Virtual camera

The app opens the virtual camera through `pyvirtualcam`. By default it tries to open one and uses it when available; the terminal prints the name of the device. When none can be opened, the app prints a message and continues with the window only.

With a virtual camera the call receives the effect, unmirrored. The app's window keeps showing the camera as a mirror, so you still see yourself while the call sees an empty room. `--no-mirror` turns the mirror off.

- `--virtual-camera`: a virtual camera is required. The app stops when none can be opened. The same happens when the camera named by `--virtual-backend` or `--virtual-device` cannot be opened.
- `--no-virtual-camera`: no virtual camera is opened, and the window shows the effect, mirrored unless `--no-mirror` is given.
- `--virtual-backend`: the driver, one of `obs`, `unitycapture` or `v4l2loopback`. `--virtual-device NAME`: the device.

If you use OBS Virtual Camera, stop it in OBS before starting the app.

## Controls

### Video window

The keys are read from the video window (`Webcam - Preview`), so it has to have the keyboard focus.

| Key | Action |
| --- | --- |
| `SPACE` | Capture a new background: step out of the frame for 4 seconds |
| `V` | Start the effect without a gesture. Off in Goku. In Itachi it works only while you are visible |
| `R` | Start the return. In Normal, Particles and Naruto it gives a 1 s dissolve while the person is invisible and cancels a running effect. In Itachi it starts the animated return once you are back in the frame. Off in Goku |
| `F` | Toggle full screen |
| `Q` or `ESC` | Quit and release the cameras. Closing the window quits as well |

`V` and `R` do nothing until the first background is captured. With a virtual camera, the keys are read every other frame while an effect renders, so a key pressed in between is handled on the next frame.

### Command-line options

`start_mac.command` passes its arguments to `vanish_cam.py`, so both take the same options.

| Option | Meaning |
| --- | --- |
| `--mode normal`, `naruto`, `particles`, `goku` or `itachi` | Skip the mode menu |
| `--camera N` | Camera index. Without it the camera menu opens |
| `--width W`, `--height H` | Video size. Default 1280x720 on macOS and 960x540 elsewhere. If only one is given, the other is 960 or 540 |
| `--fps N` | Frame rate asked from the camera and passed to the virtual camera as its frame rate. Default 30, from 1 to 120 |
| `--fullscreen` | Start in full screen |
| `--no-mirror` | Do not mirror the camera in the window |
| `--virtual-camera`, `--no-virtual-camera` | See [Virtual camera](#virtual-camera) |
| `--virtual-backend` | Driver of the virtual camera: `obs`, `unitycapture` or `v4l2loopback` |
| `--virtual-device NAME` | Name of the virtual camera to use |
| `--gesture-debug` | Print what the recognizer sees to the terminal; nothing goes on the video |

Examples (`python` is the interpreter of the environment, `.venv-local/bin/python` when the launcher created it):

```sh
./start_mac.command --mode naruto --camera 0
python vanish_cam.py --mode normal --width 960 --height 540 --fps 30
python vanish_cam.py --mode itachi --virtual-camera --gesture-debug
```

## How it works

1. `run()` in `vanish_cam.py` parses the options, shows the mode menu and the camera menu and, for Goku, the hold-time dialog.
2. `run_camera()` downloads the missing models and opens the camera. The camera is asked for the chosen size; when it reports a larger one in both dimensions, `SizedCapture` resizes every frame to the requested size.
3. `create_tasks()` builds the MediaPipe tasks: an image segmenter for the person's mask and, for the gesture, two instances of a recognizer, one in VIDEO mode that tracks and one in IMAGE mode for retries. Goku adds a face detector. The segmenter runs on the video thread. The recognizer runs on a worker thread (`AsyncRecognizer`): the video thread hands it the newest frame and goes on with the previous answer.
4. Each frame passes through a phase machine: `calibrating`, `ready`, `vanishing`, `vanished`, `appearing`, then `ready` again. Naruto has its own vanishing phase; Goku and Itachi have their own vanishing and appearing phases.
5. `calibrating` lasts 4 seconds. The background is the median of the frames captured in its last 0.55 seconds, at most the last 18 frames.
6. The gesture is handed to the recognizer no more often than every 0.14 s in Normal and Particles, 0.08 s in Naruto, 0.10 s in Goku and 0.09 s in Itachi.
7. While the person is invisible, and while the Normal, Particles and Naruto effects run, the mask is refreshed no more often than every 0.05 s (20 Hz). `erase_person()` paints the stored background over the person in the live frame; `apply_dissolve()` blends between the live person and that picture.
8. `show_output()` sends the result to the virtual camera, or to the window when none is open, as described under [Virtual camera](#virtual-camera).

Goku and Itachi render their effect from the frame captured when the gesture was accepted. The Particle effect animates the silhouette it captured at the start of the effect.

Normal and Particles accept the V sign from the geometry of the hand landmarks or from the MediaPipe classifier with a score of at least 0.62. When the tracking recognizer does not see the gesture, the IMAGE-mode instance retries on a few rotated or cropped views.

Models:

| Model | File | Used by | Source |
| --- | --- | --- | --- |
| Selfie segmenter | `models/selfie_segmenter_landscape.tflite` | Every mode: the person's mask | Downloaded when missing |
| Gesture recognizer | `models/gesture_recognizer.task` | Normal, Particles | Downloaded when missing |
| Hand landmarker | `hand_landmarker.task` | Naruto, Goku, Itachi | In the repository |
| Face detector | `models/blaze_face_short_range.tflite` | Goku: the position of the forehead | In the repository |

Downloads come from `storage.googleapis.com`. Only the models a mode uses are downloaded: the segmenter in every mode, the gesture recognizer in Normal and Particles, the face detector in Goku. A file counts as present only when it is larger than 100,000 bytes; otherwise it is downloaded again.

Source files:

| File | Role |
| --- | --- |
| `vanish_cam.py` | Entry point: options, menus, camera, MediaPipe tasks, the phase loop, keys, virtual camera |
| `goku_effect.py` | Goku pose recognition, the hold gate, the teleport effect |
| `goku_settings.py` | Dialog that asks for the Goku hold time |
| `naruto_effect.py` | Joined-hands recognition, smoke and log compositing |
| `particles_effect.py` | The particle effect |
| `itachi_effect.py` | Chest-touch recognition, the crow animation, the ink overlay |
| `vanish_sound.py` | Plays the sounds of Naruto and Goku with pygame, or with `ffplay` when pygame is missing |
| `mediapipe_log_filter.py` | When started from source, runs the app again as a child process and drops one MediaPipe warning line from the output |
| `license_client.py` | Online license check of the packaged Windows builds |
| `licensing_backend/` | Firebase configuration and a Node test suite for the Firestore rules of the license service. The rules file itself is not in the repository |
| `license_config.example.json` | Placeholder configuration for the license service |
| `start_mac.command` | macOS launcher |
| `start_windows.bat`, `start_windows_virtual_camera.bat`, `start_windows_no_virtual_camera.bat` | Windows launchers |
| `BUILD_WINDOWS.bat`, `VanishCam.spec`, `editions/` | Windows build. Each executable reads `edition.txt` and offers only its modes |
| `requirements.txt` | Python packages |
| `assets/` | Log and crow sprites, Sharingan image, two sounds, the icon of the executables |
| `marketing/` | Logo image of the original project; nothing in the code uses it |
| `test_*.py` | The test suite |

## Performance and privacy

### Performance

Measured by the maintainer on one machine, an Apple Silicon MacBook Air with macOS 26.6.2, after the first seconds of each run. These are single-machine measurements, not benchmarks.

- Normal mode at 1280x720, with OBS Virtual Camera and a person-sized silhouette: 29 to 30 frames per second in every phase.
- The other four modes at 960x540, without a virtual camera: about 30 frames per second, from the gesture to the return.

Not verified with this fork: Windows, Linux, Intel Macs, and the return gesture with a real hand. The measurements used scripted gestures.

### Privacy

- Frames are processed on the computer. They are neither recorded nor uploaded.
- The only network use is the first start: the Python packages from PyPI and up to two MediaPipe models from `storage.googleapis.com`. A model is downloaded again only when its file is missing or incomplete.
- The online license check exists only in the packaged Windows builds described under Install. Started from source, the app makes no license request.
- On macOS the launcher appends the terminal output to `~/Library/Logs/VanishCam.log`. The log holds text only: status messages, the name of the virtual camera and, with `--gesture-debug`, recognition details.

## Testing

The test modules import the package as `local_app`, so a checkout at the top level of the repository needs a link with that name. `python` must be an interpreter with the packages of `requirements.txt`, for example `.venv-local/bin/python`. Run these commands from the root of the checkout; the first line creates a folder `vanish-wrap` next to it.

```sh
mkdir -p ../vanish-wrap && ln -s "$PWD" ../vanish-wrap/local_app
PYTHONPATH=../vanish-wrap python -m unittest local_app.test_goku_effect local_app.test_goku_settings \
  local_app.test_itachi_effect local_app.test_licensing local_app.test_naruto_effect \
  local_app.test_particles_effect local_app.test_vanish_cam local_app.test_video_output
```

The suite has 107 tests. Three of them, in `test_licensing.py`, need Windows and are skipped elsewhere.

| Module | Covers |
| --- | --- |
| `test_vanish_cam.py` | The V sign (rotation, mirroring, scale, look-alikes), the immediate start, the return by the same gesture, the live-room dissolve, the multi-view retries, `SizedCapture`, `AsyncRecognizer` |
| `test_video_output.py` | The virtual camera, the camera menu, the editions and the default sizes, and the video loop with a fake camera and a scripted clock: window as a monitor, effect in the call, closing the window, the Goku holds |
| `test_naruto_effect.py` | Joined hands, smoke and log, the mode menu, the effect in the video loop |
| `test_goku_effect.py` | The forehead pose, the hold gate, the teleport animation |
| `test_goku_settings.py` | The hold-time dialog, the start sequence and a fractional hold in the video loop |
| `test_itachi_effect.py` | The chest contact, the gate, the crows, the Sharingan, the ink over a live frame |
| `test_particles_effect.py` | The particle effect |
| `test_licensing.py` | The license client and the entry point, with the network replaced by mocks |

No test opens a camera or loads a model: the camera, the MediaPipe tasks, the virtual camera, the clock and the keys are replaced by mocks. Recognition with a real hand is not covered.

## License and attribution

The original project has no license file. The original code, effects and assets are the work of their author, Nicovic17. This fork exists to run the project on macOS: builds must not be distributed, and the code must not be reused without the author's permission.

### Support the original author

The original README asked for support through [PayPal](https://paypal.me/nicovic17) and [Revolut](https://revolut.me/nicoauriemma).
