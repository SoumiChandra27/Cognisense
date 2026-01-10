"""
Cognisense X - Visual Analytics Engine (Clean Report Version)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import matplotlib
matplotlib.use("Agg") # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

# Try imports
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# -------------------- Config --------------------
YAW_THRESHOLD = 30
PITCH_UP_THRESHOLD = 20
PITCH_DOWN_THRESHOLD = -15
YAWN_THRESHOLD = 0.6

# -------------------- PDF Class --------------------
class ModernReport(FPDF):
    def header(self):
        # Header Color Bar
        self.set_fill_color(30, 41, 59) # Dark Slate Blue
        self.rect(0, 0, 210, 25, 'F')
        
        # Title
        self.set_font('Arial', 'B', 18)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 8)
        self.cell(0, 10, 'COGNISENSE ANALYTICS', 0, 0, 'L')
        
        # Date
        self.set_font('Arial', '', 10)
        self.set_xy(10, 16)
        self.cell(0, 10, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'L')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Cognisense X | Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 14)
        self.set_text_color(30, 41, 59)
        self.cell(0, 10, label, 0, 1, 'L')
        self.set_draw_color(30, 41, 59)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def card(self, title, value, unit, x, y, w, h, color=(245, 245, 245)):
        self.set_xy(x, y)
        self.set_fill_color(*color)
        self.set_draw_color(220, 220, 220)
        self.rect(x, y, w, h, 'DF')
        
        # Title
        self.set_xy(x, y + 5)
        self.set_font('Arial', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(w, 5, title.upper(), 0, 1, 'C')
        
        # Value
        self.set_xy(x, y + 15)
        self.set_font('Arial', 'B', 18)
        self.set_text_color(33, 33, 33)
        self.cell(w, 10, str(value), 0, 1, 'C')
        
        # Unit
        self.set_xy(x, y + 26)
        self.set_font('Arial', '', 8)
        self.cell(w, 5, unit, 0, 1, 'C')

# -------------------- Monitor Class --------------------
class AttentionMonitor:
    def __init__(self, camera_idx=0, model_path="attention_classifier_model.h5"):
        # MediaPipe Setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Landmarks
        self.MOUTH_VERT = [13, 14]
        self.MOUTH_HORZ = [61, 291]
        self.face_2d_idx = [1, 152, 263, 33, 291, 61]
        self.face_3d_model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])

        # Load Model
        self.custom_model = None
        self._load_custom_model(model_path)
        
        # Tracking Data
        self.start_time = datetime.now()
        self.yawn_count = 0
        self.yawn_timestamps = []
        
        self.away_start_time = None
        self.distraction_periods = []
        self.total_distraction_time = 0.0
        
        self.total_speaking_time = 0.0
        
        self.focused_frames = 0
        self.away_frames = 0
        self.drowsy_frames = 0

    def _load_custom_model(self, model_path):
        if tf and os.path.exists(model_path):
            try:
                self.custom_model = tf.keras.models.load_model(model_path, compile=False)
                print(f"[Model] Loaded: {model_path}")
            except:
                self.custom_model = None

    def log_speaking(self, duration_seconds=2.0):
        self.total_speaking_time += duration_seconds

    def save_report(self, base_name=None):
        if base_name is None:
            base_name = datetime.now().strftime("Cognisense_Report_%Y%m%d_%H%M%S")
        
        csv_name = base_name + ".csv"
        json_name = base_name + ".json"
        pdf_name = base_name + ".pdf"

        # Calculate Stats
        total_frames = self.focused_frames + self.away_frames + self.drowsy_frames
        if total_frames == 0: total_frames = 1
        
        pct_focused = (self.focused_frames / total_frames) * 100
        pct_away = (self.away_frames / total_frames) * 100
        pct_drowsy = (self.drowsy_frames / total_frames) * 100
        
        # Calculate Session Duration
        session_duration = (datetime.now() - self.start_time).total_seconds()
        if session_duration < 1: session_duration = 1
        
        # Speaking %
        pct_speaking = (self.total_speaking_time / session_duration) * 100
        if pct_speaking > 100: pct_speaking = 100

        report_data = {
            "summary": {
                "session_date": self.start_time.strftime("%B %d, %Y"),
                "session_time": self.start_time.strftime("%I:%M %p"),
                "duration_min": round(session_duration / 60, 1),
                "pct_focused": round(pct_focused, 1),
                "pct_distracted": round(pct_away + pct_drowsy, 1),
                "total_yawns": self.yawn_count,
                "speaking_time": round(self.total_speaking_time, 1),
                "pct_speaking": round(pct_speaking, 1)
            },
            "charts": {
                "focused": pct_focused,
                "away": pct_away,
                "drowsy": pct_drowsy,
                "speaking": self.total_speaking_time,
                "silence": max(0, session_duration - self.total_speaking_time)
            }
        }

        # 1. Save JSON (Keep data, just don't print in PDF)
        with open(json_name, "w") as f:
            json.dump(report_data, f, indent=2)

        # 2. Save PDF
        if FPDF_AVAILABLE:
            self._generate_visual_pdf(pdf_name, report_data)

        return csv_name, json_name, pdf_name

    def _generate_visual_pdf(self, filename, data):
        pdf = ModernReport()
        pdf.add_page()
        
        summary = data["summary"]
        charts = data["charts"]

        # --- 1. Scorecards Row ---
        # Focus Score
        pdf.card("FOCUS SCORE", f"{summary['pct_focused']}%", "AVG", 10, 40, 60, 35, (232, 245, 233)) # Light Green
        # Distraction
        pdf.card("DISTRACTION", f"{summary['pct_distracted']}%", "RATE", 75, 40, 60, 35, (255, 243, 224)) # Light Orange
        # Speaking
        pdf.card("SPEAKING TIME", f"{summary['speaking_time']}", "SECONDS", 140, 40, 60, 35, (237, 231, 246)) # Light Purple
        
        pdf.ln(50)

        # --- 2. Visual Charts ---
        pdf.chapter_title("Session Analytics")
        
        # Generate Charts using Matplotlib
        chart_filename = "temp_analysis.png"
        self._create_analysis_charts(charts, chart_filename)
        
        # Embed Image
        if os.path.exists(chart_filename):
            pdf.image(chart_filename, x=10, y=pdf.get_y(), w=190)
            os.remove(chart_filename) # Cleanup
        
        pdf.ln(10)

        # --- 3. Footer Stats ---
        pdf.set_y(-40) # Position at bottom
        pdf.set_font("Arial", "I", 10)
        pdf.set_text_color(200, 50, 50) # Red
        pdf.cell(0, 10, f"Total Fatigue Events (Yawns Detected): {summary['total_yawns']}", 0, 1, 'C')

        pdf.output(filename)

    def _create_analysis_charts(self, data, filename):
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # --- Chart 1: Attention Donut ---
        labels_att = ['Focused', 'Distracted', 'Drowsy']
        sizes_att = [data['focused'], data['away'], data['drowsy']]
        colors_att = ['#00C853', '#FFAB00', '#D50000'] # Material Colors
        
        if sum(sizes_att) == 0: sizes_att = [100, 0, 0]

        wedges, texts, autotexts = ax1.pie(sizes_att, labels=labels_att, colors=colors_att, 
                                           autopct='%1.1f%%', startangle=90, pctdistance=0.85,
                                           textprops=dict(color="black"))
        
        # Draw circle for Donut effect
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        ax1.add_artist(centre_circle)
        ax1.set_title("Attention Breakdown", fontsize=12, fontweight='bold')

        # --- Chart 2: Speaking Bar ---
        labels_spk = ['Speaking', 'Silence']
        values_spk = [data['speaking'], data['silence']]
        colors_spk = ['#6200EA', '#B0BEC5']
        
        bars = ax2.bar(labels_spk, values_spk, color=colors_spk, width=0.6)
        ax2.set_title("Audio Activity", fontsize=12, fontweight='bold')
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}s', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(filename, dpi=100)
        plt.close()