# Smart Traffic Detection and Monitoring System

A Flask website that detects vehicles in images and videos with YOLO, rates the traffic level,
and shows traffic records from a CSV file in a dashboard, charts and a searchable table.

---

## What is inside the folder

```text
Traffic_Detection_Project/
├── app.py                  <- the main program
├── requirements.txt        <- list of packages to install
├── traffic_detection.csv   <- your traffic data (a sample is included)
├── yolov8n.pt              <- the YOLO model file (see "Placing yolov8n.pt")
├── README.md
├── uploads/                <- uploaded files are saved here
├── results/                <- processed images/videos are saved here
├── templates/              <- the HTML pages
└── static/                 <- CSS, JavaScript, images
```

---

## How to run it on Windows (step by step)

### Step 1: Install Python
1. Go to https://www.python.org/downloads/ and download Python 3.13 (3.11 or 3.12 also work).
2. Run the installer. **Tick the box "Add python.exe to PATH"** on the first screen, then click *Install Now*.
3. To check, open Command Prompt and type `python --version`. You should see a version number.

### Step 2: Open Command Prompt in the project folder
1. Open the `Traffic_Detection_Project` folder in File Explorer.
2. Click the address bar at the top, type `cmd` and press **Enter**.
3. A black window opens already inside the project folder. (In VS Code you can use Terminal > New Terminal instead.)

### Step 3: Create a virtual environment
A virtual environment is a private box for this project's packages, so nothing else on your computer is affected.
```bash
python -m venv venv
```

### Step 4: Activate the virtual environment
```bash
venv\Scripts\activate
```
You should now see `(venv)` at the start of the line. In PowerShell, if you get a red "running scripts is disabled" error, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try again, or just use Command Prompt.

### Step 5: Install the packages
```bash
pip install -r requirements.txt
```
This downloads Flask, OpenCV, YOLO (and PyTorch, which is large). It can take 5-15 minutes. Wait until it finishes.

### Step 6: Start the website
```bash
python app.py
```
Leave this window open while you use the website. The first detection may take a few extra seconds while the model loads.

### Step 7: Open the website
Open your browser and go to:
```text
http://127.0.0.1:5000
```
To stop the website, click the Command Prompt window and press **Ctrl + C**.

---

## Placing traffic_detection.csv
Put `traffic_detection.csv` **in the same folder as `app.py`**. A sample file with 540 records is already included; you can replace it with your own.
The first row must contain these exact column names:

```text
Record_ID, Timestamp, Location, Zone, Total_Vehicles, Cars, Buses, Trucks,
Motorcycles, Pedestrians, Traffic_Light, Estimated_Delay_Seconds, Traffic_Level
```
`Timestamp` should look like `2026-10-12 08:30:00`. `Traffic_Light` should be Red, Yellow or Green.
The website reads the file every time a page loads, so you can edit the CSV and just refresh the browser.

## Placing yolov8n.pt
1. Download `yolov8n.pt` from the Ultralytics releases page:
   https://github.com/ultralytics/assets/releases (search the page for `yolov8n.pt`).
2. Copy it **into the same folder as `app.py`**.

If you skip this and you are online, Ultralytics downloads `yolov8n.pt` automatically the first time you run a detection.

---

## Changing settings
Open `app.py`. The settings are at the top:

| Setting | Meaning | Default |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | Minimum confidence for a vehicle to be counted | `0.40` |
| `VIDEO_STRIDE` | Run YOLO on every Nth frame of a video (higher = faster) | `2` |
| `MAX_VIDEO_FRAMES` | Maximum frames processed per video | `900` |
| `MAX_UPLOAD_MB` | Largest allowed upload | `200` |

## Good to know (useful for your viva)
- **Traffic level:** 0-10 Normal, 11-25 Moderate, 26-40 Heavy, 41+ Critical.
- **Videos:** the counts show the busiest moment (the most vehicles in a single frame), not a sum over all frames, so the same car is not counted again and again.
- **Traffic light:** the colour is read only when YOLO finds a traffic light and one colour clearly dominates. Otherwise the page shows *Traffic Light: Not Detected*. Nothing is guessed.
- **Video playback:** OpenCV often cannot write browser-friendly H.264. If you install **ffmpeg** (https://ffmpeg.org) and add it to PATH, videos play inside the page. Without it, use the *Download result* button.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick **Add python.exe to PATH**. Then open a new Command Prompt. |
| `'venv\Scripts\activate' cannot be loaded` (PowerShell) | Use Command Prompt, or run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. |
| `pip install` fails on `torch` | Make sure you use a 64-bit Python 3.10-3.13. Upgrade pip first: `python -m pip install --upgrade pip`. |
| `ModuleNotFoundError: No module named 'flask'` (or cv2, ultralytics) | The virtual environment is not active. Run `venv\Scripts\activate`, then `pip install -r requirements.txt`. |
| Red message: *traffic_detection.csv was not found* | Put the CSV in the same folder as `app.py`. |
| Red message: *CSV is missing these columns* | Rename the CSV header row to match the column names above. |
| Red message: *YOLO model could not be loaded* | Place `yolov8n.pt` next to `app.py`, or connect to the internet for the first run. |
| `Address already in use` / port 5000 busy | Close the other program, or change the last line of `app.py` to `app.run(debug=True, port=5001)` and open `http://127.0.0.1:5001`. |
| Charts or table icons do not appear | The page loads Bootstrap, Chart.js and DataTables from the internet. Check your connection. |
| Upload says *File too large* | Use a smaller file or raise `MAX_UPLOAD_MB` in `app.py`. |
| Video is slow | Use a short clip (10-30 seconds), or raise `VIDEO_STRIDE` to 3 or 4. |
| Video will not play in the page | Install ffmpeg, or use the *Download result* button. |
| Page looks unstyled | Hard refresh with **Ctrl + F5**. |


## India Live Traffic Monitor (new)

The upgraded project adds `/india-monitor` without replacing the existing dashboard,
CSV, upload detection, analytics, templates, or results.

### Run on Windows

1. Open Command Prompt in this project folder.
2. Install dependencies (first setup only):

   ```bat
   python -m pip install -r requirements.txt
   ```

3. Start Flask:

   ```bat
   python app.py
   ```

4. Open `http://127.0.0.1:5000/india-monitor` in Chrome or Edge.
5. Select a State/UT and mapped city, click **Start webcam**, and allow camera access.
6. The existing YOLO model (`yolov8n.pt`) will process webcam frames. On first use,
   Ultralytics may need internet access to download the model if it is not already present.

### Important live-feed limitations

- The webcam feature is a real live camera source and works on localhost.
- No government/public CCTV streams are included. Select a city to label the camera
  session; it does not connect to that city's public roads.
- The map contains mapped major cities and administrative centres for all 28 States
  and 8 Union Territories. It is not an exhaustive gazetteer of every Indian town/village.
- Map markers are geographic reference points, not proof of CCTV coverage or live traffic.
- To connect a real road CCTV camera, obtain an authorized RTSP/ONVIF feed and permission.
  Publicly guessing camera URLs or bypassing access controls is not supported.
- Browser webcam access generally requires `localhost` or HTTPS. Keep the Flask terminal open.
- Each processed frame is appended to `results/live_detection_history.jsonl`.
  The existing `traffic_detection.csv` is not modified by this feature.
- Live frame inference uses your computer's CPU/GPU and can be slow on lower-end systems.
  Stop the camera when you finish.

### Existing project data

The original `traffic_detection.csv` and existing uploads/results are retained. The new
live detection history is stored in a separate JSONL file under `results/`.
