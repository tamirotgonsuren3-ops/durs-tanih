"""E2E webcam test: real Chromium with a fake camera showing a trained face.

Verifies the full pipeline:
  browser getUserMedia -> WebRTC -> aiortc -> face recognition -> attendance.csv

Requires the app server running (default https://localhost:8599).
"""
import os
import re
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE_URL = os.environ.get("APP_URL", "https://localhost:8599")
FACE_JPG = "/tmp/testface.jpg"
Y4M = "/tmp/face.y4m"
TRAIN_NAME = "E2E Test Face"
FAILURES = []


def check(cond, msg):
    print(("PASS: " if cond else "FAIL: ") + msg)
    if not cond:
        FAILURES.append(msg)


# ---- 1. Prepare trained face ------------------------------------------------ #
from PIL import Image

import app as appmod

encs = appmod.encode_image(Image.open(FACE_JPG))
check(len(encs) >= 1, "face encoded from test photo")
appmod.save_encodings({TRAIN_NAME: {"encoding": encs[0], "count": 1}})
if os.path.exists("attendance.csv"):
    os.remove("attendance.csv")

# ---- 2. Fake-camera video (Chromium --use-file-for-fake-video-capture) ------ #
# Crop around the face first: the raw test photo is portrait (910x1137) and a
# blind scale to 640x480 distorts it enough that HOG face detection fails.
CROP_JPG = "/tmp/testcrop.jpg"
if not os.path.exists(CROP_JPG):
    from PIL import Image as _Img

    _im = _Img.open(FACE_JPG).convert("RGB")
    _im.crop((80, 0, 890, 640)).resize((640, 480)).save(CROP_JPG, quality=95)
if (
    not os.path.exists(Y4M)
    or os.path.getmtime(Y4M) < os.path.getmtime(CROP_JPG)
):
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", CROP_JPG, "-t", "40",
         "-vf", "scale=640:480,fps=10", "-pix_fmt", "yuv420p",
         "-f", "yuv4mpegpipe", Y4M],
        check=True, capture_output=True,
    )
check(os.path.exists(Y4M), "fake camera y4m file ready")

# ---- 3. Browser test --------------------------------------------------------- #
from playwright.sync_api import sync_playwright

console_errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(
        args=[
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={Y4M}",
            "--autoplay-policy=no-user-gesture-required",
            "--allow-running-insecure-content",
        ]
    )
    ctx = browser.new_context(ignore_https_errors=True)
    page = ctx.new_page()
    page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
    page.on(
        "console",
        lambda m: console_errors.append(f"console.{m.type}: {m.text}")
        if m.type == "error"
        else None,
    )

    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_selector("text=Face Attendance", timeout=60000)
    check(True, "app page loaded")

    # ---- diagnostics component present (secure context check) ---- #
    diag_frames = [f for f in page.frames if "camdiag" in (f.url or "") or f.url == "about:srcdoc"]
    # The html component may be srcdoc-based; try to read its text via frames
    diag_ok = False
    for f in page.frames:
        try:
            el = f.query_selector("#camdiag")
            if el:
                txt = el.inner_text()
                print(f"   diagnostics: {txt}")
                diag_ok = ("✅" in txt) or ("🟡" in txt)
        except Exception:
            pass
    # diagnostics only renders on the Mark-attendance page; navigate first
    # ---- navigate to attendance ---- #
    nav_ok = False
    for attempt in range(3):
        page.get_by_text("Mark attendance", exact=True).first.click(
            timeout=15000, force=True
        )
        try:
            page.wait_for_selector("text=2 · Mark attendance", timeout=12000)
            nav_ok = True
            break
        except Exception:
            print(f"   nav attempt {attempt + 1} failed, retrying...")
    check(nav_ok, "navigated to Mark attendance page")
    time.sleep(1.5)

    diag_ok = False
    for f in page.frames:
        try:
            el = f.query_selector("#camdiag")
            if el:
                txt = el.inner_text()
                print(f"   diagnostics: {txt}")
                diag_ok = ("✅" in txt) or ("🟡" in txt)
                break
        except Exception:
            pass
    check(diag_ok, "camera diagnostics render (secure context / camera API state)")

    # ---- click Start (app session start, then the component's own Start) ---- #
    page.get_by_role("button", name=re.compile("Start")).first.click(timeout=15000, force=True)
    print("   app Start clicked, waiting for the WebRTC component...")

    # The webrtc component renders its own START button (single-PC mode:
    # desired_playing_state is intentionally not used to avoid ICE churn).
    comp_clicked = False
    try:
        iframe_el = page.wait_for_selector(
            "iframe[src*='streamlit_webrtc']", timeout=60000
        )
        frame = iframe_el.content_frame()
        btn = frame.locator("button", has_text=re.compile("start", re.I)).first
        for _ in range(15):  # button may be disabled until devices enumerate
            if btn.count() and btn.get_attribute("disabled") is None:
                btn.click(force=True, timeout=5000)
                comp_clicked = True
                print("   component START clicked")
                break
            time.sleep(1)
    except Exception as e:
        print(f"   component start failed: {e}")
    check(comp_clicked, "webrtc component START clicked")
    print("   waiting for WebRTC stream + recognition...")

    # ---- wait for attendance row written by the WebRTC video thread ---- #
    deadline = time.time() + 75
    row_found = False
    while time.time() < deadline:
        if os.path.exists("attendance.csv"):
            txt = open("attendance.csv").read()
            if TRAIN_NAME in txt and "check-in" in txt:
                row_found = True
                break
        time.sleep(2)
    check(row_found, "attendance recorded from live webcam stream (WebRTC + recognition)")

    # ---- check the processed video element is actually rendering frames ---- #
    video_ok = False
    for f in page.frames:
        try:
            for v in f.query_selector_all("video"):
                w = v.evaluate("el => el.videoWidth")
                if w and w > 0:
                    video_ok = True
        except Exception:
            pass
    check(video_ok, "WebRTC video element rendering frames (videoWidth > 0)")

    browser.close()

# ---- 4. Console errors ------------------------------------------------------- #
real_errors = [e for e in console_errors if "favicon" not in e.lower()]
if real_errors:
    print("   browser errors:")
    for e in real_errors[:10]:
        print("    -", e)
check(not real_errors, "no browser console/page errors")

# ---- cleanup ----------------------------------------------------------------- #
for pth in ("face_encodings.pkl", "attendance.csv"):
    if os.path.exists(pth):
        os.remove(pth)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES")
    sys.exit(1)
print("E2E WEBCAM TEST PASSED")
