# Real-Time Multi-Stream Hybrid Self-Driving Simulation

> Digital Image Processing — End Term Project (CLO4 → PLO5)  
> Computer Engineering, 6th Semester

A hybrid classical + deep learning self-driving pipeline that processes real campus road footage and outputs real-time driving decisions — **FORWARD, TURN LEFT, TURN RIGHT, STOP** — entirely on a PC.

---

## Demo Output

| Scenario | Decision |
|---|---|
| Clear road, centered | FORWARD ✅ |
| Lane drifting left | TURN RIGHT 🟠 |
| Lane drifting right | TURN LEFT 🟠 |
| Boom barrier ahead | STOP — BARRIER 🔴 |
| Person/car too close | STOP — OBSTACLE 🔴 |

---

## System Architecture

```
Input Video (.mp4)
       │
       ▼
┌──────────────────────────────────────────┐
│              Frame Processor             │
│                                          │
│  ┌─────────────┐  ┌─────────────────┐   │
│  │  Thread 1   │  │    Thread 2     │   │
│  │    YOLO     │  │  Classical CV   │   │
│  │  (Objects)  │  │   (Barrier)     │   │
│  └──────┬──────┘  └───────┬─────────┘   │
│         │                 │             │
│         └────────┬────────┘             │
│                  │                      │
│          ┌───────▼──────┐               │
│          │   Thread 3   │               │
│          │     Lane     │               │
│          │  Detection   │               │
│          └───────┬──────┘               │
└──────────────────┼───────────────────── ┘
                   │
                   ▼
           Decision Engine
     (Barrier > Obstacle > Lane > Forward)
                   │
                   ▼
           Display Overlay
     (FPS / Decision / Arrow / Offset)
                   │
                   ▼
         Output Window + Saved MP4
```

---

## Features

- **Hybrid Vision System** — Classical image processing and deep learning running concurrently on every frame
- **3-Thread Parallel Pipeline** — YOLO, barrier detection, and lane detection run simultaneously using `ThreadPoolExecutor`
- **Real-Time Display** — Live OpenCV window with FPS counter, decision indicator, direction arrow, and lane offset
- **Adaptive Exposure Correction** — Automatically darkens overexposed frames every 15 frames
- **Robust Lane Filtering** — Slope rejection + statistical outlier removal to handle kerbs and paving noise
- **Distance Estimation** — Classifies detected objects as CLOSE or FAR based on bounding box height ratio
- **No Hardcoded Values** — All coordinates and thresholds expressed as frame fractions, works on any resolution
- **Output Saving** — Processed videos automatically saved to `output_videos/`

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) |
| Edge Detection | Canny (OpenCV) |
| Lane Fitting | Polynomial fitting via NumPy |
| Barrier Detection | HSV masking + Hough lines |
| Concurrency | Python ThreadPoolExecutor |
| Display | OpenCV (cv2.imshow) |
| Language | Python 3.x |

---

## Project Structure

```
├── integration.py          # Main pipeline — run this
├── backup_lane_detection.py  # Backup lane module (cited sources)
├── DIP Project Videos/     # Input .mp4 files go here
└── output_videos/          # Processed output videos saved here
```

---

## Installation

```bash
pip install opencv-python numpy ultralytics
```

> YOLOv8n weights (`yolov8n.pt`) will be downloaded automatically on first run.

---

## Usage

1. Place your `.mp4` video files inside the `DIP Project Videos/` folder
2. Run the pipeline:

```bash
python integration.py
```

3. Press `Q` to stop processing at any time
4. Processed videos are saved to `output_videos/`

---

## Pipeline Details

### Thread 1 — YOLO Object Detection
Runs YOLOv8n on each frame to detect `person`, `car`, `motorcycle`, `bus`, `truck`. Each detection is classified as:
- **CLOSE** — bounding box height > 45% of frame height → triggers STOP
- **FAR** — bounding box height ≤ 45% of frame height → noted but no action

Includes an aspect ratio filter to reject lamp posts and poles misclassified as people.

### Thread 2 — Classical Barrier Detection
Detects boom barriers using a multi-stage pipeline:
1. Dynamic exposure correction on overexposed frames
2. HSV red color masking (two hue ranges — red wraps around hue wheel)
3. Horizontal dilation to bridge red/white barrier stripes
4. Contour geometry filtering — keeps only wide, horizontal shapes (aspect ratio > 3.0)
5. Hough line detection — confirms horizontal line geometry
6. Y-position filter — rejects ground shadows misidentified as barriers

### Thread 3 — Lane Detection
1. Grayscale → Gaussian Blur → Canny edge detection
2. Trapezoidal ROI mask focusing on the road ahead
3. Left/right lane point separation at frame midpoint
4. Hybrid noise filter: slope rejection + statistical outlier removal (1.5σ)
5. Adaptive polynomial fitting — degree 1 for sparse points, degree 2 for dense
6. Lane center offset calculation → drives steering decisions
7. Extreme offset clamping — offsets > 40% of frame width discarded as detection errors

### Decision Engine
Priority order (highest to lowest):

```
1. Barrier detected          → STOP - BARRIER
2. CLOSE object detected     → STOP - OBSTACLE  
3. Lane offset > threshold   → TURN LEFT / TURN RIGHT
4. Default                   → FORWARD
```

---

## Performance

| Metric | Value |
|---|---|
| FPS (CPU) | ~4–5 FPS |
| Threads | 3 concurrent |
| Resolution | 1280 × 720 |
| YOLO confidence threshold | 0.4 |
| Lane offset threshold | 40px |

> FPS is limited by running three concurrent heavy threads on CPU. GPU inference would significantly improve this.

---

## Known Limitations

- Lane detection returns N/A on plain concrete roads with no markings or kerbing — expected behavior, system defaults to FORWARD
- Intersection scenes can produce noisy lane overlays due to changing road geometry — clamped to prevent wrong steering decisions
- FPS is CPU-bound — not suitable for real-time deployment without GPU acceleration

---

## References

- [1] ksakmann, *Canny-Edge-Lane-Line-Detector*, GitHub, 2017. https://github.com/ksakmann/Canny-Edge-Lane-Line-Detector
- [2] mbshbn, *Lane-Line-Detection-using-color-transform-and-gradient*, GitHub. https://github.com/mbshbn/Lane-Line-Detection-using-color-transform-and-gradient
- [3] Ultralytics, *YOLOv8*, https://github.com/ultralytics/ultralytics
