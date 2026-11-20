# Face ID: local face recognition and attendance

> **Learning project.** Built to study how face recognition works. It is not a product,
> and it is not certified for security, payroll, or disciplinary decisions.

A small program that recognises the people you have enrolled and keeps a daily
attendance list, all on your own computer. The camera watches the room, a name appears
once the program is sure, and the attendance list fills itself in. No internet, no
account, no cloud.

It runs on an ordinary laptop without a graphics card, and the same code runs on a small
board such as a Raspberry Pi.

## What it can do

- **Recognise people you enrolled.** Every face in the picture gets a box with its name
  and a confidence above it, so a second person in shot is visible instead of confusing.
  A name only appears after several frames agree.
- **Tell a real face from a photo or a screen.** The program measures small differences
  in how a face moves. A flat photo behaves differently from a real head.
- **Keep a daily attendance list.** A confirmed sighting marks the person present once
  per day, marked late if they arrive after the working hours you set. You can export a
  month to a spreadsheet file.
- **Keep personal data unreadable.** Names and face data are encrypted on disk, and the
  key file is readable by your own account only. Without that file, a copy of the
  database cannot be read.
- **Manage people.** Rename, deactivate (stops matching, keeps the history), or delete,
  plus a log of who changed what.
- **Run on weak hardware.** It adapts itself to the computer it finds, down to a small
  1 GB board.
- **Work with a normal webcam.** One USB or built-in camera is enough; 640x480 is
  plenty.

## How it works

1. You record a person once: type their name and follow five guided poses (front, left,
   right, tilt, closer). The program keeps a numerical fingerprint of the face, never a
   photo.
2. From then on the camera looks for faces. Each face is compared with the fingerprints
   of everyone you enrolled.
3. When the same person wins a few checks in a row, the name appears and the attendance
   list is updated.
4. A photo or a screen held up to the camera is labelled as such instead of being
   accepted.

Under the hood there are three separate jobs: reading the camera, recognising faces, and
drawing the window. They run independently, so a slow frame never freezes the picture and
the interface never waits for the recognition result.

Everything happens on this machine. After the face model is downloaded once during
installation, the program does not use the network at all.

## Install

You need Python 3.11 or newer, about 1.2 GB of free disk, and a webcam. Windows 10 or
newer, Linux, and macOS all work. A graphics card is optional: if your computer has one,
the program uses it and falls back to the processor when it cannot.

One command from the project folder:

```bash
python install.py
```

On Linux or a small board use `python3 install.py`. The script creates its own private
environment, installs what it needs, prepares the folders, creates the encryption key,
downloads the face model (about 120 MB, once) and finishes with a self-check. It is safe
to run again: anything already done is skipped.

Prefer to do it by hand?

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux and macOS
pip install -r requirements.txt
python main.py status
```

Either way, check the result with:

```bash
python main.py status
```

You should see your hardware, the performance profile it picked, the encrypted storage
line, and the camera numbers it could open.

## Using it

Start the window with:

```bash
python main.py
```

<img width="1465" height="942" alt="image" src="https://github.com/user-attachments/assets/7dc8dd7b-3e1e-4a2e-b3d5-0b0ebad51384" />
<img width="1473" height="987" alt="image" src="https://github.com/user-attachments/assets/fea969d9-9496-4b5b-92c1-89bea19add3d" />
<img width="1468" height="992" alt="image" src="https://github.com/user-attachments/assets/5cb5e9a8-5df9-4709-8a7d-02ec6f14ad12" />
<img width="1470" height="981" alt="image" src="https://github.com/user-attachments/assets/af9603b6-d45c-47d8-a0ee-18ef984a0d1b" />
<img width="1465" height="971" alt="image" src="https://github.com/user-attachments/assets/6047b7bd-4a4a-4cba-805f-3b336b1bfe69" />
<img width="1461" height="988" alt="image" src="https://github.com/user-attachments/assets/7ef06d3b-8db0-417c-a09d-798ba0652d9d" />


The sidebar has six pages:

| Page | What you do there |
| --- | --- |
| **Overview** | How many people are enrolled, who is present today, and shortcuts |
| **Recognition** | Start the camera and watch the live result |
| **Record a face** | Enrol a new person in five poses |
| **Attendance** | Any day's list, export to CSV, delete a wrong row |
| **Identities** | Rename, deactivate, reactivate, delete |
| **Settings** | Working hours, thresholds, camera, appearance |

The usual first day:

1. Open **Record a face**, type the person's name, press **Prepare camera**, then **Start
   capture** and follow the poses on screen. Press **Save identity** at the end.
2. Open **Recognition** and press **Start**. Leave it running during the day.
3. Look at **Attendance** in the evening, or run `python main.py attendance` from a
   terminal.

There is also a mode without any window, useful on a board next to a door:

```bash
python main.py recognize
```

## Where your files live

| Path | What it is |
| --- | --- |
| `database/face_database.db` | The enrolled people and the attendance list |
| `database/face_key.bin` | The key that unlocks the names and face data |
| `models/` | The downloaded face model |
| `settings.json` | What you changed on the Settings page |

Keep a copy of the key somewhere else, on a USB stick for example. A copy of the database
without the key cannot be read, and there is no way back: every face would have to be
recorded again.

## Limits worth knowing

- **The photo check is not certification.** It catches a flat photo or a still image on
  a screen while the face moves. It does not catch a video replay, a bent print, a mask,
  or someone who never moves at all.
- **The similarity percentage is not a probability.** It says how alike two measurements
  are, not how likely it is that the person is who the computer thinks.
- **Glasses and makeup are not detected.** Look at the person yourself.
- **Attendance trusts the computer's clock.** A wrong clock means wrong hours, and there
  is no network time source to correct it.
- **One presence per day.** Someone who leaves and returns keeps a single row with a
  later time. Multi-shift schedules need more rules than this project ships with.
- **Losing the key loses the data.**
- **One camera per program.** Two copies can share the same attendance file, but not the
  same camera.
- **A webcam may be slower than you expect,** especially in dim light, and that is what
  the preview follows.

## More detail

The technical side lives in [TECHNICAL.md](TECHNICAL.md): the realtime architecture, how
a match is decided, the enrollment poses and quality checks, the database schema, GPU
acceleration with measured numbers, every setting, troubleshooting, and the development
checks.

## Privacy

Face data and attendance are personal data. They stay in `database/face_database.db` on
this machine, excluded from version control, with names and face data encrypted under the
key in `database/face_key.bin`. Deleting a person removes their face data and their
attendance; deactivating keeps the history without matching them again.

No telemetry, no upload, and no network access once the model is on disk. You can check
that by running the program with networking turned off.

The face model itself is not part of this repository and carries its own licence from the
InsightFace model zoo; see the model section in [TECHNICAL.md](TECHNICAL.md#8-the-model).
