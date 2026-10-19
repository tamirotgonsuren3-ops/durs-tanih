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
./run.sh                    # or: streamlit run app.py
```

Then open the printed **Local URL** (or **Network URL** from another device).

> **Mobile use:** open the **Network URL** on the phone browser (camera access
> requires HTTPS or localhost — put the app behind a TLS proxy/`streamlit run
> --server.sslCertFile ...` for remote phones), select *Front camera (mobile)*,
> click **Start** and allow camera permission.

## Workflow

1. **Train faces** → enter a name → upload photos or take a snapshot → *Train*.
2. **Mark attendance** → pick *check-in* / *check-out* → pick input source →
   **Start** (live stream starts / video processes) → **Stop** when done.
3. **Attendance records** → filter and download `attendance.csv`.

## Files

| File                  | Purpose                                        |
| --------------------- | ---------------------------------------------- |
| `app.py`              | The Streamlit application                      |
| `run.sh`              | Launcher (sets `LD_LIBRARY_PATH` for dlib)     |
| `requirements.txt`    | Python dependencies                            |
| `test_app.py`         | Headless UI tests (`streamlit.testing.v1`)     |
| `face_encodings.pkl`  | Trained model (created on first training)      |
| `attendance.csv`      | Attendance log (created on first record)       |

## Tests

```bash
python3 test_app.py
```

Covers: app boot on every page, empty-state warnings, training state, the
check-in/check-out + input-source + Start flow, and the records table/metrics.

## Notes

- `dlib` must be installed for `face-recognition` (pip builds it from source —
  needs `cmake` and a C++ compiler; conda: `conda install -c conda-forge dlib`).
- Recognition uses the HOG detector on downscaled frames for speed; tolerance
  for accepting a match is 0.6 (default).
