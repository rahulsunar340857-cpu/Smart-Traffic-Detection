"""
Smart Traffic Detection and Monitoring System
---------------------------------------------
Flask backend:
  * reads traffic_detection.csv with Pandas (dashboard, analytics, data table)
  * detects vehicles in uploaded images / videos with Ultralytics YOLO
  * calculates the traffic level and reads traffic-light colour with OpenCV
"""
import logging
import json
import os
import shutil
import subprocess
import uuid
import threading
from datetime import datetime, timezone
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# ----------------------------------------------------------------------------
# 1. SETTINGS  (easy to change)
# ----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(BASE_DIR, "traffic_detection.csv")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.40       # vehicles below this confidence are ignored
LIGHT_CONFIDENCE = 0.25           # traffic lights are small, so allow lower confidence
VIDEO_STRIDE = 2                  # run YOLO on every 2nd video frame (faster)
MAX_VIDEO_FRAMES = 900            # safety limit so a huge video cannot freeze the PC
MAX_UPLOAD_MB = 200

IMAGE_EXT = {"jpg", "jpeg", "png"}
VIDEO_EXT = {"mp4", "avi", "mov"}
ALLOWED_EXT = IMAGE_EXT | VIDEO_EXT

# COCO class ids we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRAFFIC_LIGHT_CLASS = 9

# BGR colours used when drawing boxes
BOX_COLORS = {
    "car": (80, 200, 60), "motorcycle": (0, 190, 255),
    "bus": (255, 140, 40), "truck": (60, 60, 240),
    "traffic light": (255, 255, 255),
}

REQUIRED_COLUMNS = [
    "Record_ID", "Timestamp", "Location", "Zone", "Total_Vehicles", "Cars", "Buses",
    "Trucks", "Motorcycles", "Pedestrians", "Traffic_Light",
    "Estimated_Delay_Seconds", "Traffic_Level",
]
LEVEL_ORDER = ["Normal", "Moderate", "Heavy", "Critical"]
LIGHT_ORDER = ["Red", "Yellow", "Green"]

# ----------------------------------------------------------------------------
# 2. FLASK SETUP
# ----------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key")
logging.basicConfig(level=logging.INFO)


def ensure_folders():
    """Create uploads/ and results/ if they do not exist."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


ensure_folders()


class DataError(Exception):
    """Problem with the CSV file (shown to the user as a friendly message)."""


class DetectionError(Exception):
    """Problem while running detection (shown to the user as a friendly message)."""


# ----------------------------------------------------------------------------
# 3. CSV DATA (Pandas)
# ----------------------------------------------------------------------------
def load_csv():
    """Read the CSV fresh each time so edits to the file show up immediately."""
    if not os.path.exists(CSV_PATH):
        raise DataError("traffic_detection.csv was not found. Place it in the project folder "
                        "(next to app.py) and refresh the page.")
    try:
        df = pd.read_csv(CSV_PATH)
    except pd.errors.EmptyDataError:
        raise DataError("traffic_detection.csv is empty. Add traffic records and refresh.")
    except Exception:
        raise DataError("traffic_detection.csv could not be read. Check that it is a valid CSV file.")

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise DataError("traffic_detection.csv has headers but no records.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError("The CSV is missing these columns: " + ", ".join(missing))

    for col in ["Total_Vehicles", "Cars", "Buses", "Trucks", "Motorcycles",
                "Pedestrians", "Estimated_Delay_Seconds"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for col in ["Location", "Zone", "Traffic_Light", "Traffic_Level", "Record_ID"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df["Traffic_Light"] = df["Traffic_Light"].str.title()
    df["Traffic_Level"] = df["Traffic_Level"].str.title()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Date"] = df["Timestamp"].dt.strftime("%Y-%m-%d").fillna("")
    return df


def csv_problem():
    """Return an error message if the CSV can't be loaded, otherwise None."""
    try:
        load_csv()
        return None
    except DataError as err:
        return str(err)


def apply_filters(df, args):
    """Filter the DataFrame using ?location=&zone=&level=&light=&date="""
    mapping = {"location": "Location", "zone": "Zone", "level": "Traffic_Level",
               "light": "Traffic_Light", "date": "Date"}
    for key, col in mapping.items():
        value = (args.get(key) or "").strip()
        if value:
            df = df[df[col].str.lower() == value.lower()] if col != "Date" else df[df[col] == value]
    return df


def ordered_counts(series, order):
    counts = series.value_counts().to_dict()
    labels = [x for x in order if x in counts] + [x for x in counts if x not in order and x]
    return {"labels": labels, "values": [int(counts[x]) for x in labels]}


@app.route("/api/stats")
def api_stats():
    """Everything the dashboard and analytics pages need, calculated from the CSV."""
    try:
        full = load_csv()
    except DataError as err:
        return jsonify(error=str(err)), 500

    options = {
        "locations": sorted(x for x in full["Location"].unique() if x),
        "zones": sorted(x for x in full["Zone"].unique() if x),
        "levels": [x for x in LEVEL_ORDER if x in set(full["Traffic_Level"])],
        "lights": [x for x in LIGHT_ORDER if x in set(full["Traffic_Light"])],
    }
    df = apply_filters(full, request.args)

    if df.empty:
        return jsonify(empty=True, options=options,
                       message="No records match the selected filters.")

    latest = df.sort_values("Timestamp").iloc[-1]
    by_loc = df.groupby("Location")
    loc_vehicles = by_loc["Total_Vehicles"].sum().sort_values(ascending=False)
    loc_delay = by_loc["Estimated_Delay_Seconds"].mean().round(1).sort_values(ascending=False)

    summary = {
        "total_vehicles": int(df["Total_Vehicles"].sum()),
        "cars": int(df["Cars"].sum()),
        "buses": int(df["Buses"].sum()),
        "trucks": int(df["Trucks"].sum()),
        "motorcycles": int(df["Motorcycles"].sum()),
        "pedestrians": int(df["Pedestrians"].sum()),
        "avg_delay": round(float(df["Estimated_Delay_Seconds"].mean()), 1),
        "current_level": latest["Traffic_Level"],
        "current_location": latest["Location"],
        "records": int(len(df)),
    }
    charts = {
        "vehicles_by_location": {"labels": list(loc_vehicles.index),
                                 "values": [int(v) for v in loc_vehicles.values]},
        "vehicle_types": {"labels": ["Cars", "Buses", "Trucks", "Motorcycles"],
                          "values": [summary["cars"], summary["buses"],
                                     summary["trucks"], summary["motorcycles"]]},
        "level_distribution": ordered_counts(df["Traffic_Level"], LEVEL_ORDER),
        "light_distribution": ordered_counts(df["Traffic_Light"], LIGHT_ORDER),
        "delay_by_location": {"labels": list(loc_delay.index),
                              "values": [float(v) for v in loc_delay.values]},
    }
    return jsonify(empty=False, options=options, summary=summary, charts=charts)


@app.route("/api/records")
def api_records():
    """All CSV rows for the searchable table."""
    try:
        df = load_csv()
    except DataError as err:
        return jsonify(error=str(err)), 500
    df = df.sort_values("Timestamp", ascending=False)
    df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M").fillna("")
    df = df[REQUIRED_COLUMNS]
    for col in ["Total_Vehicles", "Cars", "Buses", "Trucks", "Motorcycles",
                "Pedestrians", "Estimated_Delay_Seconds"]:
        df[col] = df[col].astype(int)
    return jsonify(records=df.to_dict(orient="records"))


# ----------------------------------------------------------------------------
# 4. YOLO DETECTION
# ----------------------------------------------------------------------------
_model = None


def get_model():
    """Load YOLO once and reuse it. Uses yolov8n.pt next to app.py."""
    global _model
    if _model is not None:
        return _model
    try:
        from ultralytics import YOLO
    except ImportError:
        raise DetectionError("The 'ultralytics' package is not installed. "
                             "Run: pip install -r requirements.txt")
    try:
        # If the file is missing, Ultralytics tries to download "yolov8n.pt" automatically.
        _model = YOLO(MODEL_PATH if os.path.exists(MODEL_PATH) else "yolov8n.pt")
    except Exception:
        raise DetectionError("The YOLO model (yolov8n.pt) could not be loaded. Place yolov8n.pt "
                             "in the project folder, or connect to the internet for the first run.")
    return _model


def traffic_level(total):
    """0-10 Normal, 11-25 Moderate, 26-40 Heavy, 41+ Critical."""
    if total <= 10:
        return "Normal"
    if total <= 25:
        return "Moderate"
    if total <= 40:
        return "Heavy"
    return "Critical"


def classify_light(frame, box):
    """Decide RED / YELLOW / GREEN from the pixels inside a traffic-light box.
    Returns None when the colour is not clear (we never guess)."""
    x1, y1, x2, y2 = box
    crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
    if crop.shape[0] < 8 or crop.shape[1] < 4:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lit = (s > 90) & (v > 150)                     # bright, saturated pixels = a lit lamp
    scores = {
        "RED": int(np.sum(lit & ((h <= 10) | (h >= 170)))),
        "YELLOW": int(np.sum(lit & (h >= 18) & (h <= 35))),
        "GREEN": int(np.sum(lit & (h >= 40) & (h <= 90))),
    }
    best = max(scores, key=scores.get)
    others = sorted(scores.values())[-2]
    if scores[best] < 6 or (others > 0 and scores[best] < 1.5 * others):
        return None
    return best


def detect(frame):
    """Run YOLO on one frame and return a list of detections."""
    model = get_model()
    try:
        result = model.predict(frame, conf=LIGHT_CONFIDENCE, verbose=False,
                               classes=list(VEHICLE_CLASSES) + [TRAFFIC_LIGHT_CLASS])[0]
    except Exception:
        raise DetectionError("Detection failed while running the YOLO model.")
    detections = []
    for b in result.boxes:
        cls, conf = int(b.cls[0]), float(b.conf[0])
        box = [int(v) for v in b.xyxy[0].tolist()]
        if cls in VEHICLE_CLASSES and conf >= CONFIDENCE_THRESHOLD:
            detections.append({"label": VEHICLE_CLASSES[cls], "confidence": round(conf, 2),
                               "box": box, "state": None})
        elif cls == TRAFFIC_LIGHT_CLASS:
            detections.append({"label": "traffic light", "confidence": round(conf, 2),
                               "box": box, "state": classify_light(frame, box)})
    return detections


def count_vehicles(detections):
    counts = {"cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0}
    names = {"car": "cars", "motorcycle": "motorcycles", "bus": "buses", "truck": "trucks"}
    for d in detections:
        if d["label"] in names:
            counts[names[d["label"]]] += 1
    counts["total"] = sum(counts.values())
    return counts


def light_from(detections):
    """Traffic-light state of the most confident light with a clear colour."""
    lights = [d for d in detections if d["label"] == "traffic light" and d["state"]]
    return max(lights, key=lambda d: d["confidence"])["state"] if lights else None


def draw(frame, detections):
    """Return a copy of the frame with boxes and labels drawn on it."""
    out = frame.copy()
    thick = max(2, int(min(out.shape[:2]) / 400))
    scale = max(0.5, min(out.shape[:2]) / 900)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = BOX_COLORS.get(d["label"], (200, 200, 200))
        text = f'{d["label"]} {d["confidence"]:.2f}'
        if d["state"]:
            text = f'traffic light: {d["state"]}'
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
        cv2.putText(out, text, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def detection_table(detections):
    """Rows for the 'Object / Confidence / Bounding Box' table (vehicles only)."""
    return [{"object": d["label"], "confidence": d["confidence"], "box": d["box"]}
            for d in detections if d["label"] != "traffic light"][:150]


def build_summary(kind, filename, url, detections, **extra):
    counts = count_vehicles(detections)
    return {"ok": True, "type": kind, "filename": filename, "url": url, "counts": counts,
            "level": traffic_level(counts["total"]).upper(),
            "light": light_from(detections),
            "detections": detection_table(detections), **extra}


def process_image(src_path, result_name):
    frame = cv2.imread(src_path)
    if frame is None:
        raise DetectionError("This image is corrupt or not a valid image file.")
    detections = detect(frame)
    cv2.imwrite(os.path.join(RESULT_DIR, result_name), draw(frame, detections))
    return build_summary("image", result_name, f"/results/{result_name}", detections,
                         note="Counts are for the whole image.")


def process_video(src_path, result_name):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise DetectionError("This video is corrupt or its format is not supported.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    if width == 0 or height == 0:
        cap.release()
        raise DetectionError("This video is corrupt: no readable frames were found.")

    # Try browser-friendly H.264 first, fall back to mp4v (still downloadable).
    raw_name = "raw_" + result_name
    raw_path = os.path.join(RESULT_DIR, raw_name)
    writer = None
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if writer.isOpened():
            break
    if writer is None or not writer.isOpened():
        cap.release()
        raise DetectionError("Could not create the output video on this computer.")

    peak, last, frames, truncated = [], [], 0, False
    peak_total, light_votes = -1, Counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frames >= MAX_VIDEO_FRAMES:
                truncated = True
                break
            if frames % VIDEO_STRIDE == 0:
                last = detect(frame)
                state = light_from(last)
                if state:
                    light_votes[state] += 1
                total = count_vehicles(last)["total"]
                if total > peak_total:           # remember the busiest moment
                    peak_total, peak = total, last
            writer.write(draw(frame, last))
            frames += 1
    finally:
        cap.release()
        writer.release()

    if frames == 0:
        raise DetectionError("This video is corrupt: no readable frames were found.")

    # If ffmpeg is installed, convert to H.264 so the video plays in the browser.
    final_path = os.path.join(RESULT_DIR, result_name)
    playable = False
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", final_path, "-loglevel", "error"],
                           check=True, timeout=600)
            os.remove(raw_path)
            playable = True
        except Exception:
            pass
    if not playable:
        os.replace(raw_path, final_path)
        playable = (codec == "avc1")   # H.264 plays in browsers; mp4v does not

    note = "Counts show the busiest moment of the video (most vehicles seen in one frame)."
    if truncated:
        note += f" Only the first {MAX_VIDEO_FRAMES} frames were processed."
    summary = build_summary("video", result_name, f"/results/{result_name}", peak,
                            note=note, frames=frames, playable=playable)
    summary["light"] = light_votes.most_common(1)[0][0] if light_votes else None
    return summary


# ----------------------------------------------------------------------------
# 5. PAGE ROUTES
# ----------------------------------------------------------------------------
def page(template, **kwargs):
    return render_template(template, csv_error=csv_problem(), **kwargs)


@app.route("/")
def index():
    return page("index.html")


@app.route("/dashboard")
def dashboard():
    return page("dashboard.html")


@app.route("/detection")
def detection():
    return page("detection.html", max_mb=MAX_UPLOAD_MB, threshold=CONFIDENCE_THRESHOLD)


@app.route("/traffic-data")
def traffic_data():
    return page("traffic_data.html")


@app.route("/analytics")
def analytics():
    return page("analytics.html")


@app.route("/about")
def about():
    return page("about.html")


# ----------------------------------------------------------------------------
# 6. UPLOAD + RESULT FILES
# ----------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    ensure_folders()
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify(error="No file selected. Please choose an image or video first."), 400

    original = file.filename
    ext = original.rsplit(".", 1)[-1].lower() if "." in original else ""
    if ext not in ALLOWED_EXT:
        return jsonify(error="Invalid file type. Allowed: JPG, JPEG, PNG, MP4, AVI, MOV."), 400

    safe = secure_filename(original) or f"upload.{ext}"      # secure file name
    uid = uuid.uuid4().hex[:10]
    src_path = os.path.join(UPLOAD_DIR, f"{uid}_{safe}")
    file.save(src_path)
    if os.path.getsize(src_path) == 0:
        os.remove(src_path)
        return jsonify(error="The uploaded file is empty."), 400

    try:
        if ext in IMAGE_EXT:
            return jsonify(process_image(src_path, f"result_{uid}.jpg"))
        return jsonify(process_video(src_path, f"result_{uid}.mp4"))
    except DetectionError as err:
        return jsonify(error=str(err)), 422
    except Exception:
        app.logger.exception("Unexpected detection error")
        return jsonify(error="Something went wrong while processing the file. "
                             "Try a different file."), 500


@app.route("/results/<path:filename>")
def results(filename):
    """Serve processed images/videos (only from the results folder)."""
    return send_from_directory(RESULT_DIR, filename,
                               as_attachment=request.args.get("download") == "1")




# ----------------------------------------------------------------------------
# 8. INDIA LIVE MONITORING (add-on; existing CSV and upload flow unchanged)
# ----------------------------------------------------------------------------
LIVE_HISTORY_PATH = os.path.join(BASE_DIR, "results", "live_detection_history.jsonl")
LIVE_CLASSES = {"car": "cars", "motorcycle": "motorcycles", "bus": "buses", "truck": "trucks"}

# State/UT list with capitals and selected major-city map points. Users can also
# type any additional city; a marker is shown only where coordinates are known.
INDIA_LOCATIONS = [
 {"state":"Andhra Pradesh","cities":[["Visakhapatnam",17.6868,83.2185],["Vijayawada",16.5062,80.6480],["Amaravati",16.5745,80.3598],["Tirupati",13.6288,79.4192]]},
 {"state":"Arunachal Pradesh","cities":[["Itanagar",27.0844,93.6053]]},
 {"state":"Assam","cities":[["Guwahati",26.1445,91.7362],["Dibrugarh",27.4728,94.9120],["Silchar",24.8333,92.7789]]},
 {"state":"Bihar","cities":[["Patna",25.5941,85.1376],["Gaya",24.7914,85.0002],["Muzaffarpur",26.1209,85.3647]]},
 {"state":"Chhattisgarh","cities":[["Raipur",21.2514,81.6296],["Bhilai",21.2094,81.4285],["Bilaspur",22.0797,82.1409]]},
 {"state":"Goa","cities":[["Panaji",15.4909,73.8278],["Margao",15.2832,73.9862]]},
 {"state":"Gujarat","cities":[["Ahmedabad",23.0225,72.5714],["Surat",21.1702,72.8311],["Vadodara",22.3072,73.1812],["Rajkot",22.3039,70.8022],["Gandhinagar",23.2156,72.6369]]},
 {"state":"Haryana","cities":[["Gurugram",28.4595,77.0266],["Faridabad",28.4089,77.3178],["Ambala",30.3782,76.7767],["Panipat",29.3909,76.9635],["Karnal",29.6857,76.9905],["Hisar",29.1492,75.7217],["Chandigarh (shared capital)",30.7333,76.7794]]},
 {"state":"Himachal Pradesh","cities":[["Shimla",31.1048,77.1734],["Dharamshala",32.2190,76.3234],["Mandi",31.7080,76.9320]]},
 {"state":"Jharkhand","cities":[["Ranchi",23.3441,85.3096],["Jamshedpur",22.8046,86.2029],["Dhanbad",23.7957,86.4304]]},
 {"state":"Karnataka","cities":[["Bengaluru",12.9716,77.5946],["Mysuru",12.2958,76.6394],["Mangaluru",12.9141,74.8560],["Hubballi",15.3647,75.1240]]},
 {"state":"Kerala","cities":[["Thiruvananthapuram",8.5241,76.9366],["Kochi",9.9312,76.2673],["Kozhikode",11.2588,75.7804],["Thrissur",10.5276,76.2144]]},
 {"state":"Madhya Pradesh","cities":[["Bhopal",23.2599,77.4126],["Indore",22.7196,75.8577],["Jabalpur",23.1815,79.9864],["Gwalior",26.2183,78.1828]]},
 {"state":"Maharashtra","cities":[["Mumbai",19.0760,72.8777],["Pune",18.5204,73.8567],["Nagpur",21.1458,79.0882],["Nashik",19.9975,73.7898],["Chhatrapati Sambhajinagar",19.8762,75.3433]]},
 {"state":"Manipur","cities":[["Imphal",24.8170,93.9368]]},
 {"state":"Meghalaya","cities":[["Shillong",25.5788,91.8933]]},
 {"state":"Mizoram","cities":[["Aizawl",23.7271,92.7176]]},
 {"state":"Nagaland","cities":[["Kohima",25.6751,94.1086],["Dimapur",25.9091,93.7266]]},
 {"state":"Odisha","cities":[["Bhubaneswar",20.2961,85.8245],["Cuttack",20.4625,85.8830],["Rourkela",22.2604,84.8536]]},
 {"state":"Punjab","cities":[["Ludhiana",30.9010,75.8573],["Amritsar",31.6340,74.8723],["Jalandhar",31.3260,75.5762],["Patiala",30.3398,76.3869]]},
 {"state":"Rajasthan","cities":[["Jaipur",26.9124,75.7873],["Jodhpur",26.2389,73.0243],["Udaipur",24.5854,73.7125],["Kota",25.2138,75.8648],["Ajmer",26.4499,74.6399]]},
 {"state":"Sikkim","cities":[["Gangtok",27.3389,88.6065]]},
 {"state":"Tamil Nadu","cities":[["Chennai",13.0827,80.2707],["Coimbatore",11.0168,76.9558],["Madurai",9.9252,78.1198],["Tiruchirappalli",10.7905,78.7047],["Salem",11.6643,78.1460]]},
 {"state":"Telangana","cities":[["Hyderabad",17.3850,78.4867],["Warangal",17.9689,79.5941],["Nizamabad",18.6725,78.0941]]},
 {"state":"Tripura","cities":[["Agartala",23.8315,91.2868]]},
 {"state":"Uttar Pradesh","cities":[["Lucknow",26.8467,80.9462],["Noida",28.5355,77.3910],["Agra",27.1767,78.0081],["Kanpur",26.4499,80.3319],["Varanasi",25.3176,82.9739],["Prayagraj",25.4358,81.8463],["Ghaziabad",28.6692,77.4538]]},
 {"state":"Uttarakhand","cities":[["Dehradun",30.3165,78.0322],["Haridwar",29.9457,78.1642],["Rishikesh",30.0869,78.2676]]},
 {"state":"West Bengal","cities":[["Kolkata",22.5726,88.3639],["Howrah",22.5958,88.2636],["Siliguri",26.7271,88.3953],["Durgapur",23.5204,87.3119]]},
 {"state":"Andaman and Nicobar Islands","cities":[["Port Blair",11.6234,92.7265]]},
 {"state":"Chandigarh","cities":[["Chandigarh",30.7333,76.7794]]},
 {"state":"Dadra and Nagar Haveli and Daman and Diu","cities":[["Daman",20.3974,72.8328],["Silvassa",20.2766,73.0083]]},
 {"state":"Delhi","cities":[["New Delhi",28.6139,77.2090],["Dwarka",28.5921,77.0460],["Rohini",28.7495,77.0565]]},
 {"state":"Jammu and Kashmir","cities":[["Srinagar",34.0837,74.7973],["Jammu",32.7266,74.8570]]},
 {"state":"Ladakh","cities":[["Leh",34.1526,77.5771],["Kargil",34.5539,76.1349]]},
 {"state":"Lakshadweep","cities":[["Kavaratti",10.5667,72.6417]]},
 {"state":"Puducherry","cities":[["Puducherry",11.9416,79.8083],["Karaikal",10.9254,79.8380]]}
]

@app.route("/india-monitor")
def india_monitor():
    return page("india_monitor.html")

@app.route("/api/india/locations")
def api_india_locations():
    return jsonify(states=INDIA_LOCATIONS, state_count=len(INDIA_LOCATIONS),
                   note="City list includes mapped major cities and administrative centres, not every settlement in India.")

@app.route("/api/live/detect", methods=["POST"])
def api_live_detect():
    """Accept one browser-camera frame, run the existing YOLO model, return annotated JPEG."""
    if "frame" not in request.files:
        return jsonify(error="No camera frame received."), 400
    raw = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="Could not decode camera frame."), 400
    try:
        detections = detect(frame)
        counts = count_vehicles(detections)
        annotated = draw(frame, detections)
        ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            raise DetectionError("Could not encode the processed camera frame.")
        state = (request.form.get("state") or "Unknown").strip()[:100]
        city = (request.form.get("city") or "Unknown").strip()[:100]
        camera_name = (request.form.get("camera_name") or "Browser webcam").strip()[:100]
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "state": state,
                 "city": city, "camera": camera_name, **counts}
        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(LIVE_HISTORY_PATH, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry) + "\n")
        return jsonify(ok=True, counts=counts, level=traffic_level(counts["total"]),
                       state=state, city=city, timestamp=entry["timestamp"],
                       image="data:image/jpeg;base64," + __import__("base64").b64encode(encoded).decode("ascii"))
    except DetectionError as err:
        return jsonify(error=str(err)), 422
    except Exception:
        app.logger.exception("Live frame detection failed")
        return jsonify(error="Live detection failed. Check YOLO model and server log."), 500

@app.route("/api/live/history")
def api_live_history():
    rows = []
    if os.path.exists(LIVE_HISTORY_PATH):
        try:
            with open(LIVE_HISTORY_PATH, encoding="utf-8") as fp:
                for line in fp.readlines()[-100:]:
                    try: rows.append(json.loads(line))
                    except ValueError: pass
        except OSError:
            pass
    return jsonify(records=list(reversed(rows)))


# ----------------------------------------------------------------------------
# 7. ERROR HANDLERS (friendly messages instead of tracebacks)
# ----------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File too large. The limit is {MAX_UPLOAD_MB} MB."), 413


@app.errorhandler(404)
def not_found(_):
    return render_template("index.html", csv_error=None, page_error="Page not found."), 404


@app.errorhandler(500)
def server_error(_):
    if request.path.startswith(("/api", "/upload")):
        return jsonify(error="Unexpected server error. Please try again."), 500
    return render_template("index.html", csv_error=None,
                           page_error="Unexpected server error. Please try again."), 500


if __name__ == "__main__":
    app.run(debug=True)


# Remote road camera feeds are configured ONLY by the project operator in cameras.json.
# Never accept arbitrary camera URLs from a browser request.
CAMERAS_PATH = os.path.join(BASE_DIR, "cameras.json")

def configured_cameras():
    try:
        with open(CAMERAS_PATH, encoding="utf-8") as fh:
            records = json.load(fh)
        return [c for c in records if isinstance(c, dict) and c.get("id") and c.get("name")]
    except (OSError, ValueError, TypeError):
        return []

@app.route("/api/remote/cameras")
def remote_cameras():
    # Do not expose stream URLs or embedded credentials to browsers.
    cameras = [{"id":c["id"],"name":c["name"],"city":c.get("city",""),
                "state":c.get("state",""),"configured":bool(c.get("enabled") and c.get("url"))}
               for c in configured_cameras()]
    return jsonify(cameras=cameras)

@app.route("/api/remote/snapshot/<camera_id>")
def remote_snapshot(camera_id):
    from urllib.parse import urlsplit
    cameras = [c for c in configured_cameras() if c["id"] == camera_id]
    if not cameras:
        return jsonify(error="Camera is not registered."), 404
    camera = cameras[0]
    url = camera.get("url", "")
    if not camera.get("enabled") or not url:
        return jsonify(error="Feed unavailable: authorised camera URL not configured."), 503
    # Explicit operator-configured network URL, not user supplied. No browser URL proxying.
    if urlsplit(url).scheme.lower() not in ("rtsp", "rtsps", "http", "https"):
        return jsonify(error="Unsupported camera URL scheme."), 400
    cap = cv2.VideoCapture(url)
    try:
        if not cap.isOpened():
            return jsonify(error="Feed unavailable: could not open authorised camera."), 503
        ok, frame = cap.read()
        if not ok or frame is None:
            return jsonify(error="Feed unavailable: no video frame received."), 503
    finally:
        cap.release()
    try:
        detections = detect(frame)
        counts = count_vehicles(detections)
        annotated = draw(frame, detections)
        success, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        if not success:
            raise DetectionError("Could not encode frame.")
        timestamp = datetime.now(timezone.utc).isoformat()
        return jsonify(ok=True, counts=counts, level=traffic_level(counts["total"]),
                       timestamp=timestamp, city=camera.get("city", ""),
                       state=camera.get("state", ""),
                       image="data:image/jpeg;base64," + __import__("base64").b64encode(jpg).decode("ascii"))
    except DetectionError as exc:
        return jsonify(error=str(exc)), 422
    except Exception:
        app.logger.exception("Remote detection error")
        return jsonify(error="Remote detection failed; check server logs."), 500
