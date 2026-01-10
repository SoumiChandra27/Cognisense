###########################################################
# SERVER.PY - PAUSE TRACKING & PDF TABLE
###########################################################

import asyncio
import base64
import json
import os
import tempfile
import binascii
import random
from datetime import datetime
import numpy as np
import cv2
from aiohttp import web
from pydub import AudioSegment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------
ATTENTION_MODULE_PATH = "Attention_final.py"
monitor = None
att_module = None

if os.path.exists(ATTENTION_MODULE_PATH):
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("att_module", ATTENTION_MODULE_PATH)
        att_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(att_module)
        monitor = att_module.AttentionMonitor(camera_idx=0, model_path="attention_classifier_model_finetuned")
        print("[Backend] ✅ Attention_final.py loaded successfully.")
    except Exception as e:
        print(f"[Backend] ⚠️ Error loading Attention_final.py: {e}")
else:
    print("[Backend] ⚠️ Attention_final.py NOT found. Using Simulation Mode.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ffmpeg_local = os.path.join(BASE_DIR, "ffmpeg.exe")
ffprobe_local = os.path.join(BASE_DIR, "ffprobe.exe")

if os.path.exists(ffmpeg_local) and os.path.exists(ffprobe_local):
    print(f"[Backend] ✅ FFmpeg found.")
    AudioSegment.converter = ffmpeg_local
    AudioSegment.ffprobe   = ffprobe_local
else:
    print(f"[Backend] ⚠️ FFmpeg NOT found. Audio analysis may fail.")

# -----------------------------------------------
# 2. SESSION DATA STORAGE (Updated for Pauses)
# -----------------------------------------------
session_state = {
    "start_time": None,
    "stop_time": None,
    "last_pause_start": None, # Temp store for current pause start time
    "pause_logs": []          # List to store {start, end, duration}
}

session_data = {
    "focused": 0,
    "distracted": 0,
    "drowsy": 0,
    "speaking": 0,
    "silence": 0,
    "fatigue_events": 0
}

###########################################################
# PDF Generator (Charts + Pause Table)
###########################################################
def generate_pdf_report():
    """Generates a PDF with Dashboard and Pause Log Table."""
    
    # 1. Calculate Timestamps
    start_str = session_state["start_time"].strftime("%H:%M:%S") if session_state["start_time"] else "--:--:--"
    stop_str = datetime.now().strftime("%H:%M:%S")
    date_str = datetime.now().strftime("%B %d, %Y")
    
    filename = f"Cognisense_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    
    # 2. Calculate Stats
    total_frames = session_data["focused"] + session_data["distracted"] + session_data["drowsy"]
    if total_frames == 0: total_frames = 1
    
    focus_pct = (session_data["focused"] / total_frames) * 100
    distract_pct = (session_data["distracted"] / total_frames) * 100
    speak_time = session_data["speaking"] * 2.0
    
    with PdfPages(filename) as pdf:
        # ================= PAGE 1: DASHBOARD =================
        fig = plt.figure(figsize=(11, 8.5))
        
        plt.suptitle("COGNISENSE ANALYTICS REPORT", fontsize=20, fontweight='bold', color='#2c3e50', y=0.96)
        
        # Timing Card
        timing_text = (
            f"DATE: {date_str}\n"
            f"START: {start_str}   |   STOP: {stop_str}"
        )
        plt.figtext(0.5, 0.88, timing_text, ha="center", va="center", fontsize=12, fontweight='bold', color='white',
                    bbox={"boxstyle": "round,pad=0.8", "facecolor": "#007BFF", "edgecolor": "#0056b3", "linewidth": 2})

        plt.subplots_adjust(left=0.1, right=0.9, top=0.75, bottom=0.35, wspace=0.3)

        # Donut Chart
        ax1 = plt.subplot(121) 
        sizes = [session_data["focused"], session_data["distracted"], session_data["drowsy"]]
        labels = ['Focused', 'Distracted', 'Drowsy']
        colors = ['#00C853', '#FFAB00', '#D50000']
        if sum(sizes) == 0: sizes = [1, 0, 0] 

        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                           startangle=90, colors=colors, pctdistance=0.85, 
                                           wedgeprops=dict(width=0.3, edgecolor='white'))
        plt.setp(autotexts, size=9, weight="bold")
        ax1.set_title("Attention Breakdown", fontweight='bold', fontsize=12, pad=10)

        # Bar Chart
        ax2 = plt.subplot(122)
        categories = ['Speaking', 'Silence']
        values = [speak_time, session_data["silence"] * 2.0]
        bars = ax2.bar(categories, values, color=['#6200EA', '#B0BEC5'], width=0.5)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_title("Audio Activity", fontweight='bold', fontsize=12, pad=10)
        ax2.set_ylabel("Seconds")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Metrics Cards
        card_y = 0.15
        plt.figtext(0.18, card_y, f"FOCUS SCORE\n{focus_pct:.1f}%", 
                    ha="center", va="center", fontsize=11, fontweight='bold', color='#1b5e20',
                    bbox={"boxstyle": "round,pad=1", "facecolor": "#e8f5e9", "edgecolor": "#2e7d32"})
        plt.figtext(0.39, card_y, f"DISTRACTION\n{distract_pct:.1f}%", 
                    ha="center", va="center", fontsize=11, fontweight='bold', color='#e65100',
                    bbox={"boxstyle": "round,pad=1", "facecolor": "#fff3e0", "edgecolor": "#ef6c00"})
        plt.figtext(0.61, card_y, f"SPEAKING TIME\n{speak_time:.1f}s", 
                    ha="center", va="center", fontsize=11, fontweight='bold', color='#4a148c',
                    bbox={"boxstyle": "round,pad=1", "facecolor": "#f3e5f5", "edgecolor": "#7b1fa2"})
        plt.figtext(0.82, card_y, f"FATIGUE (YAWN)\n{session_data['fatigue_events']}", 
                    ha="center", va="center", fontsize=11, fontweight='bold', color='#b71c1c',
                    bbox={"boxstyle": "round,pad=1", "facecolor": "#ffebee", "edgecolor": "#c62828"})

        plt.figtext(0.98, 0.02, "Generated by Cognisense X", ha="right", fontsize=8, color="gray")
        pdf.savefig()
        plt.close()

        # ================= PAGE 2: PAUSE LOG TABLE =================
        # Only create this page if there are actually pauses to show
        if session_state["pause_logs"]:
            fig2 = plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.title("SESSION PAUSE LOGS", fontsize=16, fontweight='bold', color='#2c3e50', pad=20)
            
            # Prepare Table Data
            columns = ["Pause Start Time", "Pause End Time", "Duration (Seconds)"]
            rows = []
            for log in session_state["pause_logs"]:
                rows.append([log["start"], log["end"], f"{log['duration']:.1f}s"])
            
            # Create Table
            # loc='center' places it in the middle of the figure
            table = plt.table(cellText=rows, colLabels=columns, loc='center', cellLoc='center', colColours=['#007BFF']*3)
            
            # Styling the Table
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5) # Increase row height for readability
            
            # Header Styling (White text on Blue background)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#007BFF')
                    cell.set_edgecolor('white')
                else:
                    # Alternating Row Colors for easier reading
                    cell.set_facecolor('#f8f9fa' if row % 2 == 0 else '#ffffff')
                    cell.set_edgecolor('#dee2e6')

            plt.figtext(0.98, 0.02, "Generated by Cognisense X", ha="right", fontsize=8, color="gray")
            pdf.savefig()
            plt.close()

    return filename

###########################################################
# Helper Functions
###########################################################
def safe_b64_decode(b64_string):
    if not b64_string: return bytes()
    if isinstance(b64_string, bytes): b64_string = b64_string.decode("utf-8")
    if "," in b64_string: b64_string = b64_string.split(",")[1]
    return base64.b64decode(b64_string + '=' * (-len(b64_string) % 4))

def b64_to_image(b64string):
    try:
        data = safe_b64_decode(b64string)
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

def detect_speaking(wav_path, threshold_db=-40.0): 
    try:
        audio = AudioSegment.from_file(wav_path)
        return audio.dBFS > threshold_db
    except:
        return False

def process_frame(img):
    if monitor is None: return 2, "Simulated", 0, 0, False
    if img is None: return 1, "No Image", 0, 0, False
    
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = monitor.face_mesh.process(rgb)

    if not results.multi_face_landmarks: return 1, "No Face", 0, 0, False

    lm = results.multi_face_landmarks[0]
    pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])

    try:
        face_2d = pts[monitor.face_2d_idx].astype(np.float64)
        focal_length = w
        cam_center = (w / 2, h / 2)
        cam_mat = np.array([[focal_length, 0, cam_center[0]], [0, focal_length, cam_center[1]], [0, 0, 1]])
        dist = np.zeros((4, 1))
        _, rvec, _ = cv2.solvePnP(monitor.face_3d_model_points, face_2d, cam_mat, dist)
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, _ = angles
    except:
        yaw = pitch = 0

    try:
        vert = pts[monitor.MOUTH_VERT]
        horz = pts[monitor.MOUTH_HORZ]
        mar = np.linalg.norm(vert[0]-vert[1]) / (np.linalg.norm(horz[0]-horz[1]) + 1e-6)
    except:
        mar = 0

    yawn = mar > att_module.YAWN_THRESHOLD
    if yawn: return 0, "Yawning", yaw, pitch, True
    if abs(yaw) > att_module.YAW_THRESHOLD: return 1, "Looking Away", yaw, pitch, False
    if pitch > att_module.PITCH_UP_THRESHOLD: return 1, "Looking Up", yaw, pitch, False
    if pitch < att_module.PITCH_DOWN_THRESHOLD: return 2, "Looking Down", yaw, pitch, False
    return 2, "Focused", yaw, pitch, False

###########################################################
# WebSocket Handler (Updated)
###########################################################
clients = set()

async def broadcast(msg):
    text = json.dumps(msg)
    dead = []
    for ws in list(clients):
        try: await ws.send_str(text)
        except: dead.append(ws)
    for ws in dead: clients.discard(ws)

async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    clients.add(ws)
    
    if session_state["start_time"] is None:
        session_state["start_time"] = datetime.now()
        print(f"🕒 Session Started at {session_state['start_time'].strftime('%H:%M:%S')}")

    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT: continue
            data = json.loads(msg.data)
            typ = data.get("type")

            # --- NEW: PAUSE/RESUME HANDLER ---
            if typ == "status":
                status = data.get("status")
                
                # Case 1: PAUSE RECEIVED
                if status == "paused":
                    # Only record start if we aren't already paused
                    if session_state["last_pause_start"] is None:
                        session_state["last_pause_start"] = datetime.now()
                        print(f"⏸️ Paused at {session_state['last_pause_start'].strftime('%H:%M:%S')}")
                
                # Case 2: RESUME RECEIVED
                elif status == "resumed":
                    if session_state["last_pause_start"]:
                        now = datetime.now()
                        duration = (now - session_state["last_pause_start"]).total_seconds()
                        
                        # Save to Logs
                        session_state["pause_logs"].append({
                            "start": session_state["last_pause_start"].strftime("%H:%M:%S"),
                            "end": now.strftime("%H:%M:%S"),
                            "duration": duration
                        })
                        
                        # Reset
                        session_state["last_pause_start"] = None
                        print(f"▶️ Resumed. Duration: {duration:.1f}s")

            elif typ == "frame":
                img = b64_to_image(data["data"])
                att, status, yaw, pitch, yawn = process_frame(img)
                
                if att == 2: session_data["focused"] += 1
                elif att == 1: session_data["distracted"] += 1
                else: session_data["drowsy"] += 1
                
                if yawn: 
                    session_data["fatigue_events"] += 1
                    await broadcast({"type": "yawn", "ts": datetime.utcnow().isoformat()})

                await broadcast({"type": "attention_point", "ts": datetime.utcnow().isoformat(), "value": att, "status": status})

            elif typ == "audio":
                try:
                    raw = safe_b64_decode(data["data"])
                    if raw:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
                            f.write(raw)
                            wpath = f.name
                        
                        is_speaking = detect_speaking(wpath)
                        
                        if is_speaking: session_data["speaking"] += 1
                        else: session_data["silence"] += 1

                        if os.path.exists(wpath): os.remove(wpath)
                        await broadcast({"type": "speaking_point", "ts": datetime.utcnow().isoformat(), "value": 1 if is_speaking else 0})
                except: pass
    except:
        pass
    finally:
        clients.discard(ws)
    return ws

###########################################################
# SAVE REPORT ROUTE
###########################################################
async def save_report(request):
    print("Generating report...")
    try:
        session_state["stop_time"] = datetime.now()
        
        # --- EDGE CASE HANDLER ---
        # If user clicks "Stop" while still paused, we must close that pause interval
        if session_state["last_pause_start"]:
            now = datetime.now()
            duration = (now - session_state["last_pause_start"]).total_seconds()
            session_state["pause_logs"].append({
                "start": session_state["last_pause_start"].strftime("%H:%M:%S"),
                "end": now.strftime("%H:%M:%S"),
                "duration": duration
            })
            session_state["last_pause_start"] = None # Reset
            print(f"⚠️ Session stopped while paused. Auto-logged final pause.")

        pdf_name = generate_pdf_report()
        
        # Reset Session Data for next time
        session_state["start_time"] = None
        session_state["pause_logs"] = [] # Clear the table data
        for k in session_data: session_data[k] = 0

        if os.path.exists(pdf_name):
            with open(pdf_name, "rb") as f:
                file_content = f.read()
            return web.Response(body=file_content, content_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{pdf_name}"'})
        else:
            return web.json_response({"status": "error", "error": "PDF creation failed"})
    except Exception as e:
        print(f"Report error: {e}")
        return web.json_response({"status": "error", "error": str(e)}, status=500)

app = web.Application()
app.router.add_get("/", ws_handler)
app.router.add_post("/save_report", save_report)

print("\n------------------------------------------------")
print("✔ Backend server running at: ws://localhost:8765")
print("------------------------------------------------\n")

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8765)