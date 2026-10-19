"""
Face Recognition Attendance System
==================================
Streamlit app that:
  1. Trains (enrolls) faces from uploaded images or a camera snapshot.
  2. Marks check-in / check-out attendance using the trained model with three
     input sources:
       - Webcam (computer camera via the browser)
       - Video file (mp4, avi, mov, mkv, webm, m4v, ...)
       - Front camera of a mobile phone (browser getUserMedia, facingMode=user)
  3. Stores attendance in attendance.csv and shows / exports records.

NOTE: all data access used by the WebRTC video thread (load/save helpers below)
must stay free of Streamlit API calls — they run outside the script thread.
"""

import os
import pickle
import tempfile
import time
from datetime import timedelta

import av
import cv2
import face_recognition
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

# --------------------------------------------------------------------------- #
# Page config & constants
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Face Recognition Attendance",
    page_icon="🪪",
    layout="wide",
)

ENCODINGS_PATH = "face_encodings.pkl"
ATTENDANCE_PATH = "attendance.csv"
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm", "m4v", "mpg", "mpeg", "wmv", "wav"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]},
            {
                "urls": [
                    "turn:openrelay.metered.ca:80",
                    "turn:openrelay.metered.ca:443",
                    "turn:openrelay.metered.ca:443?transport=tcp",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

if "attendance_active" not in st.session_state:
    st.session_state.attendance_active = False


# --------------------------------------------------------------------------- #
# Persistence helpers (plain functions — safe from any thread)
# --------------------------------------------------------------------------- #
def load_encodings() -> dict:
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_encodings(encodings: dict) -> None:
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)


def load_attendance() -> pd.DataFrame:
    if os.path.exists(ATTENDANCE_PATH):
        try:
            return pd.read_csv(ATTENDANCE_PATH, dtype={"Date": str})
        except Exception:
            return pd.DataFrame(columns=["Name", "Date", "Time", "Status"])
    return pd.DataFrame(columns=["Name", "Date", "Time", "Status"])


def save_attendance(df: pd.DataFrame) -> None:
    df.to_csv(ATTENDANCE_PATH, index=False)


def mark_attendance(name: str, check_type: str) -> bool:
    """Append a record unless the same person already has this status today.
    Returns True when a new record was written."""
    from datetime import datetime

    today = str(datetime.now().date())
    df = load_attendance()
    if not df.empty:
        dup = (
            (df["Name"] == name)
            & (df["Date"].astype(str) == today)
            & (df["Status"] == check_type)
        ).any()
        if dup:
            return False
    now = datetime.now()
    new_row = pd.DataFrame(
        {
            "Name": [name],
            "Date": [today],
            "Time": [now.strftime("%H:%M:%S")],
            "Status": [check_type],
        }
    )
    df = pd.concat([df, new_row], ignore_index=True)
    save_attendance(df)
    return True


# --------------------------------------------------------------------------- #
# Core recognition: annotate frame + (optionally) mark attendance
# --------------------------------------------------------------------------- #
def recognize_frame(
    frame_bgr: np.ndarray,
    known_names: list,
    known_encodings: list,
    check_type: str | None = None,
    marked: set | None = None,
    scale: float = 0.25,
) -> tuple[np.ndarray, list[str]]:
    """Detect faces in frame_bgr, identify them against the trained model,
    draw boxes/labels, and mark attendance when check_type is given.
    Returns (annotated_bgr_frame, [recognized names in frame])."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(frame_rgb, (0, 0), fx=scale, fy=scale)
    face_locations = face_recognition.face_locations(small, model="hog")
    face_encodings = face_recognition.face_encodings(small, face_locations)

    recognized: list[str] = []
    inv = 1.0 / scale

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        color = (0, 165, 255)  # orange for unknown
        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
            distances = face_recognition.face_distance(known_encodings, encoding)
            best = int(np.argmin(distances))
            if matches[best]:
                name = known_names[best]
                color = (0, 255, 0)  # green for known
                recognized.append(name)

        t, r, b, l = int(top * inv), int(right * inv), int(bottom * inv), int(left * inv)
        cv2.rectangle(frame_bgr, (l, t), (r, b), color, 2)
        label = name if check_type is None else f"{name} ({check_type})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame_bgr, (l, b), (r, b + th + 10), color, cv2.FILLED)
        cv2.putText(
            frame_bgr,
            label,
            (l + 5, b + th + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if check_type and name != "Unknown":
            if marked is None or name not in marked:
                if mark_attendance(name, check_type):
                    if marked is not None:
                        marked.add(name)

    return frame_bgr, recognized


def encode_image(img: Image.Image) -> list[np.ndarray]:
    """Return face encodings found in a PIL image (RGB)."""
    rgb = np.array(img.convert("RGB"))
    locations = face_recognition.face_locations(rgb, model="hog")
    return face_recognition.face_encodings(rgb, locations)


# --------------------------------------------------------------------------- #
# WebRTC video processor (webcam / mobile front camera)
# --------------------------------------------------------------------------- #
class AttendanceProcessor(VideoProcessorBase):
    """Receives browser video frames, runs face recognition, and marks
    attendance for the selected check-in/check-out type."""

    def __init__(self, check_type: str):
        encodings = load_encodings()
        self.names = list(encodings.keys())
        self.encodings = [v["encoding"] for v in encodings.values()]
        self.check_type = check_type
        self.marked: set[str] = set()  # once per stream session

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        annotated, _ = recognize_frame(
            img,
            self.names,
            self.encodings,
            check_type=self.check_type,
            marked=self.marked,
        )
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_train():
    st.header("1 · Train faces")
    st.caption("Upload photos of each person (or take a snapshot) to train the model.")

    encodings = load_encodings()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.write(f"**Trained people ({len(encodings)}):**")
        if encodings:
            st.write(
                ", ".join(
                    f"{n} ({i.get('count', 1)} img)" for n, i in encodings.items()
                )
            )
        else:
            st.write("_Nobody trained yet._")
    with col_b:
        if encodings and st.button("🗑 Reset trained faces", width="stretch"):
            if os.path.exists(ENCODINGS_PATH):
                os.remove(ENCODINGS_PATH)
            st.rerun()

    st.divider()
    name = st.text_input("Person's name", placeholder="e.g. Alice Smith").strip()

    tab_upload, tab_camera = st.tabs(["📤 Upload photos", "📷 Take a snapshot"])

    with tab_upload:
        files = st.file_uploader(
            "Upload one or more clear face photos",
            type=IMAGE_TYPES,
            accept_multiple_files=True,
        )
        if st.button("Train from uploads", type="primary", disabled=not (files and name)):
            with st.spinner("Encoding faces..."):
                new_encodings: list[np.ndarray] = []
                for f in files:
                    try:
                        new_encodings.extend(encode_image(Image.open(f)))
                    except Exception as e:
                        st.warning(f"Could not read {getattr(f, 'name', 'file')}: {e}")
            if not new_encodings:
                st.error("No faces detected in the uploaded images.")
            else:
                encodings = load_encodings()
                encodings[name] = {
                    "encoding": new_encodings[0],
                    "count": len(new_encodings),
                }
                save_encodings(encodings)
                st.success(f"Trained **{name}** with {len(new_encodings)} face encoding(s).")
                st.balloons()

    with tab_camera:
        shot = st.camera_input("Take a photo", help="Uses your webcam / phone camera.")
        if st.button("Train from snapshot", type="primary", disabled=not (shot and name)):
            with st.spinner("Encoding face..."):
                found = encode_image(Image.open(shot))
            if not found:
                st.error("No face detected — try better lighting and face the camera.")
            else:
                encodings = load_encodings()
                encodings[name] = {"encoding": found[0], "count": 1}
                save_encodings(encodings)
                st.success(f"Trained **{name}**.")
                st.balloons()


def page_attendance():
    st.header("2 · Mark attendance")
    if not load_encodings():
        st.warning("No faces trained yet — go to **Train faces** first.")
        return

    # --- User selections ---------------------------------------------------- #
    col1, col2 = st.columns(2)
    with col1:
        check_type = st.radio(
            "Attendance type", ["check-in", "check-out"], horizontal=True, index=0
        )
    with col2:
        input_source = st.radio(
            "Input source",
            ["Webcam", "Video file", "Front camera (mobile)"],
            horizontal=True,
        )

    if input_source == "Video file":
        _page_video(check_type)
    else:
        _page_live(check_type, input_source)


def _page_live(check_type: str, input_source: str) -> None:
    """Webcam / mobile front-camera flow: Start & Stop buttons."""
    col_start, col_stop = st.columns([1, 1])
    with col_start:
        start = st.button(
            "🚀 Start",
            type="primary",
            width="stretch",
            disabled=st.session_state.attendance_active,
        )
    with col_stop:
        stop = st.button(
            "⏹ Stop",
            width="stretch",
            disabled=not st.session_state.attendance_active,
        )

    if start:
        st.session_state.attendance_active = True
        st.rerun()
    if stop:
        st.session_state.attendance_active = False
        st.rerun()

    if not st.session_state.attendance_active:
        st.info(
            "Select **check-in** or **check-out**, choose an input source, "
            "then click **Start**."
        )
        return

    st.success(f"Session running — **{check_type}** via **{input_source}**")

    if input_source == "Webcam":
        constraints = {"video": {"width": 640, "height": 480}, "audio": False}
        key = "webcam"
    else:  # Front camera (mobile)
        st.caption(
            "Open this app on your phone's browser (use the **Network URL** shown "
            "when the server starts; camera access requires HTTPS or localhost), "
            "allow camera permission — the front (selfie) camera will be used."
        )
        constraints = {
            "video": {"facingMode": "user", "width": 640, "height": 480},
            "audio": False,
        }
        key = "mobile"

    try:
        webrtc_ctx = webrtc_streamer(
            key=f"attendance-{key}-{check_type}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints=constraints,
            video_processor_factory=lambda: AttendanceProcessor(check_type),
            desired_playing_state=True,
        )
    except Exception as e:  # e.g. no browser session / headless context
        webrtc_ctx = None
        st.error(f"Could not start the camera stream: {e}")

    if webrtc_ctx is not None and not webrtc_ctx.state.playing:
        st.warning("Camera stream is not active — press **Start** (or allow camera access).")

    _render_today_records()


@st.fragment(run_every=timedelta(seconds=2))
def _render_today_records() -> None:
    """Auto-refreshing table of today's attendance while a session runs."""
    from datetime import datetime

    st.subheader("Today's attendance")
    df = load_attendance()
    today = str(datetime.now().date())
    todays = df[df["Date"].astype(str) == today] if not df.empty else df
    if todays.empty:
        st.caption("No records yet — recognized faces will appear here automatically.")
    else:
        st.dataframe(todays.iloc[::-1], width="stretch", hide_index=True)


def _page_video(check_type: str) -> None:
    """Video file flow: pick file, click Start to process it."""
    uploaded = st.file_uploader("Upload a video", type=VIDEO_TYPES)
    start = st.button("🚀 Start", type="primary", disabled=uploaded is None)
    if not start:
        if uploaded is not None:
            st.video(uploaded)
        else:
            st.info(
                "Select **check-in** or **check-out**, upload a video, "
                "then click **Start**."
            )
        return

    if uploaded is None:
        st.error("Upload a video first.")
        return

    data = uploaded.getvalue()
    suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Could not open the video file (audio-only or unsupported format).")
            return

        encodings = load_encodings()
        names = list(encodings.keys())
        encs = [v["encoding"] for v in encodings.values()]
        marked: set[str] = set()
        seen: dict[str, int] = {}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        progress = st.progress(0.0, text="Processing video…")
        frame_box = st.empty()
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            annotated, recognized = recognize_frame(
                frame, names, encs, check_type=check_type, marked=marked
            )
            for n in recognized:
                seen[n] = seen.get(n, 0) + 1
            frame_box.image(annotated, channels="BGR", width="stretch")
            if total:
                progress.progress(min(i / total, 1.0), text=f"Frame {i}/{total}")
            time.sleep(0.01)  # let the UI paint
        cap.release()
        progress.progress(1.0, text="Done")

        st.success(f"Processed {i} frame(s).")
        if seen:
            st.write("**Identified people:**")
            for n, count in sorted(seen.items(), key=lambda kv: -kv[1]):
                st.write(f"- {n} — seen in {count} frame(s)")
        else:
            st.info("No trained faces were identified in this video.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def page_records():
    st.header("3 · Attendance records")
    df = load_attendance()
    if df.empty:
        st.info("No attendance recorded yet.")
        return

    df["Date"] = df["Date"].astype(str)
    dates = sorted(df["Date"].unique(), reverse=True)

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        sel_dates = st.multiselect("Dates", dates, default=dates[:1])
    with col2:
        sel_status = st.multiselect(
            "Type", ["check-in", "check-out"], default=["check-in", "check-out"]
        )
    with col3:
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()

    filtered = df[df["Date"].isin(sel_dates) & df["Status"].isin(sel_status)]

    from datetime import datetime

    today = str(datetime.now().date())
    m1, m2, m3 = st.columns(3)
    m1.metric("Records (selection)", len(filtered))
    m2.metric("Check-ins today", int(((df["Date"] == today) & (df["Status"] == "check-in")).sum()))
    m3.metric("Check-outs today", int(((df["Date"] == today) & (df["Status"] == "check-out")).sum()))

    st.dataframe(filtered.iloc[::-1], width="stretch", hide_index=True)
    st.download_button(
        "⬇ Download CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="attendance.csv",
        mime="text/csv",
    )


def page_about():
    st.header("About")
    st.markdown(
        """
**Face Recognition Attendance System**

1. **Train faces** — add photos of each person; the app builds 128-D face
   encodings (`dlib` / `face_recognition`) stored in `face_encodings.pkl`.
2. **Mark attendance** — choose *check-in* or *check-out*, pick an input
   source and press **Start**:
   - *Webcam* — your computer's camera through the browser (WebRTC).
   - *Video file* — process an uploaded mp4/avi/mov/mkv/webm file frame by frame.
   - *Front camera (mobile)* — open the app on a phone browser; the selfie
     camera is used (`facingMode: user`).

   Records are appended to `attendance.csv` (max one per person, per day,
   per status).
3. **Attendance records** — filter, view and export CSV.

**Technology:** Streamlit · streamlit-webrtc (aiortc) · OpenCV · dlib/face_recognition · pandas
        """
    )


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
def main():
    st.sidebar.title("🪪 Face Attendance")
    page = st.sidebar.radio(
        "Navigation",
        ["Train faces", "Mark attendance", "Attendance records", "About"],
    )
    st.sidebar.divider()
    n = len(load_encodings())
    st.sidebar.caption(f"{n} trained {'person' if n == 1 else 'people'}")
    if st.session_state.attendance_active:
        st.sidebar.success("Session: running")

    if page == "Train faces":
        page_train()
    elif page == "Mark attendance":
        page_attendance()
    elif page == "Attendance records":
        page_records()
    else:
        page_about()


if __name__ == "__main__":
    main()
