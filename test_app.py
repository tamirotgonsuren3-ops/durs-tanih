"""Headless UI test for app.py using streamlit.testing.v1.AppTest."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Start from a clean slate (stale data from earlier runs would mask assertions)
for _p in ("face_encodings.pkl", "attendance.csv"):
    if os.path.exists(_p):
        os.remove(_p)

from streamlit.testing.v1 import AppTest

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"PASS: {msg}")
    else:
        print(f"FAIL: {msg}")
        FAILURES.append(msg)


# ---- 1. Fresh app boots ---------------------------------------------------- #
at = AppTest.from_file("app.py", default_timeout=60)
at.run()
check(not at.exception, f"app boots without exception ({at.exception})")
check(len(at.sidebar.radio) >= 1, "sidebar navigation radio exists")
check("Face Attendance" in at.sidebar.title[0].value, "sidebar title rendered")

# ---- 2. Attendance page warns when nobody is trained ----------------------- #
at.sidebar.radio[0].set_value("Mark attendance").run()
check(not at.exception, "attendance page renders (no training)")
check(
    any("No faces trained yet" in str(w.value) for w in at.warning),
    "shows warning when no faces trained",
)

# ---- 3. Records page empty state ------------------------------------------ #
at.sidebar.radio[0].set_value("Attendance records").run()
check(not at.exception, "records page renders empty")
check(any("No attendance recorded yet" in str(i.value) for i in at.info), "empty-state info shown")

# ---- 4. About page --------------------------------------------------------- #
at.sidebar.radio[0].set_value("About").run()
check(not at.exception, "about page renders")

# ---- 5. With trained face + attendance data ------------------------------- #
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

with open("face_encodings.pkl", "wb") as f:
    pickle.dump({"Test Person": {"encoding": np.zeros(128), "count": 1}}, f)
now = datetime.now()
pd.DataFrame(
    [
        {"Name": "Test Person", "Date": str(now.date()), "Time": now.strftime("%H:%M:%S"), "Status": "check-in"}
    ]
).to_csv("attendance.csv", index=False)

at2 = AppTest.from_file("app.py", default_timeout=60)
at2.run()
check(not at2.exception, "app boots with trained data")
check("1 trained person" in at2.sidebar.caption[0].value, "sidebar shows trained count")

at2.sidebar.radio[0].set_value("Mark attendance").run()
check(not at2.exception, "attendance page renders with trained data")
check(len(at2.main.radio) >= 2, "check-in/check-out + input source radios present")
radio_values = [r.value for r in at2.main.radio]
check(any(v in ("check-in", "check-out") for v in radio_values), "check-in/check-out choice present")
check(
    any(v in ("Webcam", "Video file", "Front camera (mobile)") for v in radio_values),
    "input source choice present",
)
buttons = [b.label for b in at2.main.button]
check(any("Start" in b for b in buttons), "Start button present")

# Start button -> live session state (webcam default)
at2.main.button[0].click().run()
check(not at2.exception, "Start click renders live session without exception")
check(
    any("Session running" in str(s.value) for s in at2.success),
    "session-active message shown after Start",
)

# Switch input source to Video file -> uploader appears
at2.main.radio[1].set_value("Video file").run()
check(not at2.exception, "video file source renders")
check(len(at2.file_uploader) >= 1, "video uploader present")

# Switch to mobile front camera
at2.main.radio[1].set_value("Front camera (mobile)").run()
check(not at2.exception, "mobile front camera source renders")

# Records page with data
at2.sidebar.radio[0].set_value("Attendance records").run()
check(not at2.exception, "records page renders with data")
check(len(at2.dataframe) >= 1, "attendance table rendered")
check(len(at2.metric) == 3, "three metrics rendered")

# ---- cleanup --------------------------------------------------------------- #
for p in ("face_encodings.pkl", "attendance.csv"):
    if os.path.exists(p):
        os.remove(p)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES:")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("ALL TESTS PASSED")
