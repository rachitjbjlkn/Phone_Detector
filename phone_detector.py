"""
📱 Phone Detection – Streamlit App (Threaded, No-Freeze)
=========================================================
Install:
    pip install streamlit ultralytics opencv-python pygame numpy

Run:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
import pygame
from datetime import datetime
from collections import deque
from ultralytics import YOLO

st.set_page_config(page_title="📱 Phone Detection", page_icon="📱",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;700&display=swap');
html,body,[class*="css"]{font-family:'Rajdhani',sans-serif}
.stApp{background:#0a0c10;color:#e0e8f0}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid #1e3a5f}
[data-testid="stSidebar"] *{color:#c9d8e8!important}
.title{font-family:'Share Tech Mono',monospace;font-size:1.9rem;color:#00d4ff;letter-spacing:.12em;text-shadow:0 0 20px #00d4ff44}
.sub{font-size:.78rem;color:#4a7a9b;letter-spacing:.22em;text-transform:uppercase}
.card{background:linear-gradient(135deg,#0f1923,#0d1520);border:1px solid #1a3050;border-radius:8px;padding:13px 16px;text-align:center;position:relative;overflow:hidden}
.card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#00d4ff,transparent)}
.clabel{font-size:.65rem;letter-spacing:.18em;color:#4a7a9b;text-transform:uppercase}
.cval{font-family:'Share Tech Mono',monospace;font-size:1.8rem;color:#00d4ff}
.cval.r{color:#ff4444}.cval.g{color:#00ff88}.cval.o{color:#ff9900}
.alert{background:linear-gradient(90deg,#3d0000,#1a0000);border:1px solid #ff4444;border-left:4px solid #ff4444;border-radius:6px;padding:9px 16px;font-family:'Share Tech Mono',monospace;color:#ff6666;font-size:.92rem}
.ok{background:linear-gradient(90deg,#001a0d,#000f08);border:1px solid #00ff88;border-left:4px solid #00ff88;border-radius:6px;padding:9px 16px;font-family:'Share Tech Mono',monospace;color:#00cc66;font-size:.92rem}
.le{font-family:'Share Tech Mono',monospace;font-size:.72rem;padding:4px 9px;border-left:3px solid #ff4444;background:#0f1520;margin:2px 0;color:#ff8888;border-radius:0 3px 3px 0}
.le.i{border-left-color:#00d4ff;color:#7ec8e3}
.le.ok2{border-left-color:#00ff88;color:#4dcc88}
.sl{height:1px;background:linear-gradient(90deg,transparent,#00d4ff33,transparent);margin:8px 0}
</style>
""", unsafe_allow_html=True)

# ── Global camera thread state ─────────────────────────────────────────────────
if "cam_thread"    not in st.session_state: st.session_state.cam_thread    = None
if "log"           not in st.session_state: st.session_state.log           = deque(maxlen=60)
if "total"         not in st.session_state: st.session_state.total         = 0
if "start_time"    not in st.session_state: st.session_state.start_time    = time.time()
if "phone_active"  not in st.session_state: st.session_state.phone_active  = False
if "fps_val"       not in st.session_state: st.session_state.fps_val       = 0.0
if "objs"          not in st.session_state: st.session_state.objs          = []

# Shared between thread and Streamlit (use a simple dict as a mutable container)
_shared = st.session_state.setdefault("_shared", {
    "running":      False,
    "frame":        None,          # latest annotated BGR frame
    "phone":        False,
    "fps":          0.0,
    "total":        0,
    "log":          deque(maxlen=60),
    "objs":         [],
    "last_beep":    0.0,
})

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ CONFIG")
    st.markdown('<div class="sl"></div>', unsafe_allow_html=True)

    model_name   = st.selectbox("YOLOv8 Model",
                                ["yolov8n.pt","yolov8s.pt","yolov8m.pt"])
    confidence   = st.slider("Confidence", 0.10, 0.95, 0.50, 0.05)
    cam_idx      = st.number_input("Camera Index", 0, 10, 0, 1)

    st.markdown('<div class="sl"></div>', unsafe_allow_html=True)
    st.markdown("## 🔊 AUDIO")
    audio_on   = st.toggle("Enable Alert", value=True)
    audio_file = st.text_input("Audio File", value="alert.mp3")
    cooldown   = st.slider("Cooldown (s)", 0.5, 10.0, 2.0, 0.5)
    vol        = st.slider("Volume", 0.0, 1.0, 0.8, 0.05)

    st.markdown('<div class="sl"></div>', unsafe_allow_html=True)
    st.markdown("## 📦 DETECT")
    det_phone  = st.checkbox("📱 Cell Phone",  value=True)
    det_person = st.checkbox("🧍 Person",      value=False)
    det_laptop = st.checkbox("💻 Laptop",      value=False)
    det_bag    = st.checkbox("👜 Backpack",    value=False)

    st.markdown('<div class="sl"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    start_clicked = c1.button("▶ START", use_container_width=True)
    stop_clicked  = c2.button("■ STOP",  use_container_width=True)
    clear_clicked = st.button("🗑 Clear Log", use_container_width=True)

WATCH = {}
if det_phone:  WATCH[67] = ("📱 Phone",  (0,  80,255))
if det_person: WATCH[0]  = ("🧍 Person", (255,165,  0))
if det_laptop: WATCH[63] = ("💻 Laptop", (180,  0,255))
if det_bag:    WATCH[24] = ("👜 Bag",    (0, 200,180))

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    return YOLO(name)

# ── Audio ──────────────────────────────────────────────────────────────────────
def play_sound(shared, file, volume, cooldown):
    now = time.time()
    if now - shared["last_beep"] < cooldown:
        return
    shared["last_beep"] = now
    try:
        pygame.mixer.init(44100,-16,2,512)
        snd = pygame.mixer.Sound(file)
        snd.set_volume(volume)
        snd.play()
    except Exception:
        pass

# ── Camera thread ──────────────────────────────────────────────────────────────
def camera_loop(shared, model, watch, conf_thresh, cam_index,
                audio_on, audio_file, cooldown, vol):
    cap = cv2.VideoCapture(int(cam_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # ← key: prevents buffer buildup / freeze

    if not cap.isOpened():
        shared["running"] = False
        return

    prev_t = time.time()

    while shared["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        results = model(frame, verbose=False)[0]
        phone_found = False
        objs_this  = []

        for box in results.boxes:
            c      = float(box.conf[0])
            cls_id = int(box.cls[0])
            if c < conf_thresh or cls_id not in watch:
                continue

            label, color = watch[cls_id]
            x1,y1,x2,y2 = map(int, box.xyxy[0])

            # Box + corners
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            for cx,cy,dx,dy in[(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(cx,cy),(cx+dx*16,cy),color,3)
                cv2.line(frame,(cx,cy),(cx,cy+dy*16),color,3)

            # Label
            tag = f"{label} {c:.0%}"
            (tw,th),_ = cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.56,1)
            cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+10,y1),color,-1)
            cv2.putText(frame,tag,(x1+5,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.56,(255,255,255),1,cv2.LINE_AA)

            objs_this.append(f"{label} {c:.0%}")
            if cls_id == 67:
                phone_found = True
                shared["total"] += 1
                ts = datetime.now().strftime("%H:%M:%S")
                shared["log"].appendleft(f"[{ts}]  📱 Phone ({c:.0%})")

        # Alert overlay
        if phone_found:
            shared["phone"] = True
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(frame.shape[1],52),(160,0,0),-1)
            cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
            cv2.putText(frame,"! PHONE DETECTED",(14,36),
                        cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2,cv2.LINE_AA)
            if audio_on:
                threading.Thread(target=play_sound,
                                 args=(shared,audio_file,vol,cooldown),
                                 daemon=True).start()
        else:
            shared["phone"] = False

        # FPS
        now   = time.time()
        fps   = 1.0 / max(now-prev_t, 1e-6)
        prev_t = now
        shared["fps"]  = fps
        shared["objs"] = objs_this
        cv2.putText(frame,f"FPS {fps:.1f}",(10,frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,160,200),1,cv2.LINE_AA)

        shared["frame"] = frame.copy()

    cap.release()
    shared["frame"] = None

# ── Button actions ─────────────────────────────────────────────────────────────
if start_clicked and not _shared["running"]:
    _shared["running"] = True
    _shared["total"]   = 0
    _shared["log"].clear()
    st.session_state.start_time = time.time()
    mdl = load_model(model_name)
    t = threading.Thread(
        target=camera_loop,
        args=(_shared, mdl, WATCH, confidence, cam_idx,
              audio_on, audio_file, cooldown, vol),
        daemon=True
    )
    t.start()
    st.session_state.cam_thread = t

if stop_clicked:
    _shared["running"] = False
    _shared["frame"]   = None
    _shared["phone"]   = False

if clear_clicked:
    _shared["log"].clear()
    _shared["total"] = 0

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">◈ PHONE DETECTION SYSTEM</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">real-time · yolov8 · threaded opencv · no freeze</p>', unsafe_allow_html=True)
st.markdown('<div class="sl"></div>', unsafe_allow_html=True)

# ── Metrics ────────────────────────────────────────────────────────────────────
uptime = int(time.time() - st.session_state.start_time)
m1,m2,m3,m4 = st.columns(4)
m1.markdown(f'<div class="card"><div class="clabel">Detections</div><div class="cval r">{_shared["total"]}</div></div>',unsafe_allow_html=True)
m2.markdown(f'<div class="card"><div class="clabel">Uptime</div><div class="cval g">{uptime//60:02d}:{uptime%60:02d}</div></div>',unsafe_allow_html=True)
m3.markdown(f'<div class="card"><div class="clabel">Live FPS</div><div class="cval o">{_shared["fps"]:.1f}</div></div>',unsafe_allow_html=True)
m4.markdown(f'<div class="card"><div class="clabel">Confidence</div><div class="cval">{confidence:.0%}</div></div>',unsafe_allow_html=True)

st.markdown("")

# ── Banner ─────────────────────────────────────────────────────────────────────
if _shared["phone"]:
    st.markdown('<div class="alert">⚠ &nbsp; PHONE DETECTED — ALERT ACTIVE</div>',unsafe_allow_html=True)
elif _shared["running"]:
    st.markdown('<div class="ok">✔ &nbsp; MONITORING — NO PHONE IN FRAME</div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="ok">◉ &nbsp; IDLE — PRESS ▶ START TO BEGIN</div>',unsafe_allow_html=True)

st.markdown("")

# ── Layout ─────────────────────────────────────────────────────────────────────
vid_col, log_col = st.columns([2.2, 1])

# Video
with vid_col:
    st.markdown("#### 📷 Live Camera Feed")
    frame_ph = st.empty()
    if _shared["running"] and _shared["frame"] is not None:
        rgb = cv2.cvtColor(_shared["frame"], cv2.COLOR_BGR2RGB)
        frame_ph.image(rgb, channels="RGB", use_container_width=True)
    else:
        frame_ph.markdown(
            "<div style='background:#0d1117;border:1px solid #1e3a5f;border-radius:8px;"
            "padding:100px;text-align:center;color:#4a7a9b;"
            "font-family:Share Tech Mono,monospace;letter-spacing:.1em'>"
            "▶ PRESS START TO BEGIN</div>",
            unsafe_allow_html=True)

# Log
with log_col:
    st.markdown("#### 📋 Detection Log")
    log_html = "".join(
        f'<div class="le">{e}</div>' for e in list(_shared["log"])[:25]
    ) or '<div class="le i">[ Waiting… ]</div>'
    st.markdown(log_html, unsafe_allow_html=True)

    st.markdown("#### 🎯 Objects This Scan")
    if _shared["objs"]:
        for o in _shared["objs"]:
            st.markdown(f'<div class="le ok2">◉ {o}</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="le i">None detected</div>',unsafe_allow_html=True)

    st.markdown("#### 🎯 Active Classes")
    for _,(name,_c) in WATCH.items():
        st.markdown(f"- {name}")

# ── Auto-refresh while running ─────────────────────────────────────────────────
if _shared["running"]:
    time.sleep(0.04)          # ~25 UI refreshes/sec
    st.rerun()
