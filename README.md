# 🪪 Face Recognition Attendance (Streamlit)

A Streamlit app that **trains faces** and **registers attendance** (check-in /
check-out) using a machine-learning face-recognition model, with three input
options:

1. **Webcam** — computer camera through the browser (WebRTC)
2. **Video file** — mp4 / avi / mov / mkv / webm / m4v / mpg / mpeg / wmv
3. **Front camera of a mobile phone** — phone browser camera (`facingMode: user`)

## Features

- **Train faces** — upload multiple photos per person or take a live snapshot;
  builds 128-D face encodings (`dlib` / `face_recognition`), stored in
  `face_encodings.pkl`.
- **Mark attendance** — choose **check-in** or **check-out**, choose an input
  source, click **Start**:
  - Live sources show an annotated video stream (green = known, orange = unknown)
    and automatically write attendance rows as recognized faces appear.
  - Video files are processed frame-by-frame with a progress bar.
  - At most **one record per person, per day, per status** is stored.
- **Attendance records** — filter by date/status, daily metrics, CSV export.

## Run

```bash
pip install -r requirements.txt
./run.sh                    # recommended launcher
```

`run.sh`:

1. sets `LD_LIBRARY_PATH` (dlib shared libs),
2. generates a **self-signed TLS certificate** in `.certs/` (SAN includes
   `localhost`, the host name and every local IP) and starts Streamlit over
   **HTTPS** — browsers only expose the camera (`getUserMedia`) on a secure
   origin (HTTPS **or** `localhost`),
3. starts a local **coturn** TURN server in Docker (container
   `face-attendance-turn`, host networking, port 3478) so phones on a different
   network can still reach the server,
4. runs `streamlit run app.py`.

Environment flags:

| Variable        | Effect                                             |
| --------------- | -------------------------------------------------- |
| `TLS=0`         | serve plain HTTP instead of HTTPS                  |
| `TURN=0`        | skip starting the Docker coturn container          |
| `WEBRTC_DEBUG=1`| verbose `aiortc` / `aioice` logs (troubleshooting) |

Then open the printed **Local URL**. To use a phone, open the printed
**Network URL** (`https://<your-ip>:8501`), accept the self-signed certificate
warning once, select *Front camera (mobile)*, click **Start**, and then click the
**START** button inside the video panel (see below).

> **Why two clicks?** The app's **Start** begins the attendance session; the
> WebRTC video panel then shows its own **START** button. That single click
> creates exactly one peer connection. Auto-starting the component caused a new
> connection to be created on every Streamlit rerun during negotiation, mixing
> ICE generations so every connectivity check was rejected and the stream never
> connected.

### Camera troubleshooting

The **Mark attendance → Webcam / Front camera** page shows a short diagnostic
banner above the video panel:

- 🟢 secure context + camera permission granted
- 🟡 camera API ready — the browser will ask for permission on Start
- 🔴 page not secure (HTTP) / `mediaDevices` missing / permission denied

`run.sh` serves HTTPS automatically; if you start the app manually, use
`--server.sslCertFile` / `--server.sslKeyFile`.

## Workflow

1. **Train faces** → enter a name → upload photos or take a snapshot → *Train*.
2. **Mark attendance** → pick *check-in* / *check-out* → pick input source →
   **Start** → click **START** inside the video panel → **Stop** when done.
3. **Attendance records** → filter and download `attendance.csv`.

## Files

| File                    | Purpose                                           |
| ----------------------- | ------------------------------------------------- |
| `app.py`                | The Streamlit application                         |
| `run.sh`                | Launcher (TLS cert, coturn, `LD_LIBRARY_PATH`)    |
| `requirements.txt`      | Python dependencies                                |
| `test_app.py`           | Headless UI tests (`streamlit.testing.v1`)         |
| `test_e2e_webcam.py`    | Browser E2E test (Playwright + fake camera)        |
| `face_encodings.pkl`    | Trained model (created on first training)          |
| `attendance.csv`        | Attendance log (created on first record)           |

## Tests

```bash
python3 test_app.py          # headless UI tests (fast)
python3 test_e2e_webcam.py   # full browser E2E (requires the server running)
```

`test_app.py` covers: app boot on every page, empty-state warnings, training
state, the check-in/check-out + input-source + Start flow, and the records
table/metrics.

`test_e2e_webcam.py` runs real Chromium with a **fake camera** showing a
pre-trained face and verifies the whole pipeline: `getUserMedia` → WebRTC →
`aiortc` → face recognition → `attendance.csv`, plus that the video element
renders frames and no browser errors occur. Requires the app server running on
`https://localhost:8599` (override with `APP_URL`).

## Notes

- `dlib` must be installed for `face-recognition` (pip builds it from source —
  needs `cmake` and a C++ compiler; conda: `conda install -c conda-forge dlib`).
- Recognition uses the HOG detector on downscaled frames for speed; tolerance
  for accepting a match is 0.6 (default).
- TURN is only advertised to the browser when a local TURN server is actually
  reachable; otherwise only STUN is used.

