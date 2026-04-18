import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# CONFIGURATION
VIDEO_FOLDER = "DIP Project Videos"          # <-- Put your .mp4 files in this folder
OUTPUT_FOLDER = "output_videos"  # <-- Processed videos saved here
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLOSE_THRESHOLD_HEIGHT_RATIO = 0.45   # Object taller than 45% of frame = CLOSE
LANE_OFFSET_THRESHOLD = 80            # Pixels offset before suggesting a turn


# YOLO PIPELINE (Thread 1)
# Detects: cars, people, motorcycles, buses, trucks
# Returns: list of detection dicts with coords, color, label

model = YOLO('yolov8n.pt')

def run_yolo_pipeline(frame):
    img = frame.copy()
    frame_height = img.shape[0]
    results = model(img, conf=0.4, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            if class_name in ['person', 'car', 'motorcycle', 'bus', 'truck']:
                obj_width = x2 - x1
                obj_height = y2 - y1

                # Reject pole-shaped false positives labeled as "person"
                if class_name == 'person' and (obj_width / (obj_height + 1e-5)) < 0.20:
                    continue

                height_ratio = obj_height / frame_height
                if height_ratio > CLOSE_THRESHOLD_HEIGHT_RATIO:
                    distance_estimate = "CLOSE"
                    color = (0, 0, 255)       # Red
                else:
                    distance_estimate = "FAR"
                    color = (0, 255, 0)       # Green

                label = f"{class_name} {conf:.2f} [{distance_estimate}]"
                detections.append({
                    'coords': (x1, y1, x2, y2),
                    'color': color,
                    'label': label,
                    'distance': distance_estimate
                })

    return detections


# CLASSICAL CV PIPELINE (Thread 2)
# Detects: boom barriers using HSV masking + contour filtering + Hough lines
# Returns: (barrier_detected: bool, barrier_line, debug_mask)

def run_classical_pipeline(frame):
    img = frame.copy()
    height, width = img.shape[:2]

    # --- Dynamic Exposure Check ---
    tiny = cv2.resize(img, (64, 64))
    gray_tiny = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
    if np.mean(gray_tiny) > 180:
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=-80)

    # --- ROI (center band where barrier would appear) ---
    roi_mask_img = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(roi_mask_img,
                  (int(width * 0.25), int(height * 0.35)),
                  (int(width * 0.70), int(height * 0.70)),
                  255, -1)

    # --- HSV Red Mask (red wraps around hue wheel, needs two ranges) ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 120, 100]),   np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 120, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_and(cv2.bitwise_or(mask1, mask2), cv2.bitwise_or(mask1, mask2), mask=roi_mask_img)

    # --- Dilation: bridge gaps between red/white barrier stripes ---
    red_mask = cv2.dilate(red_mask, np.ones((1, 100), np.uint8), iterations=1)

    # --- Contour filtering: keep only wide horizontal shapes ---
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(red_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        if area > 500 and w > 200 and (float(w) / h) > 3.0:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # --- Hough lines: confirm horizontal line geometry ---
    edges = cv2.Canny(clean_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=150, maxLineGap=100)

    barrier_detected = False
    barrier_line = None

    if lines is not None:
        lines = lines[:, 0, :]
        angles = np.abs(np.arctan2(lines[:, 3] - lines[:, 1],
                                   lines[:, 2] - lines[:, 0]) * 180.0 / np.pi)
        valid = lines[(angles < 25) | (angles > 155)]
        if len(valid) > 0:
            barrier_detected = True
            barrier_line = valid[0]

    return barrier_detected, barrier_line, clean_mask

# AYESHA'S MODULE — LANE DETECTION PIPELINE (Thread 3)
# Detects road lane lines using Canny + polynomial fitting
# Returns: (annotated_frame, lane_center_offset)
#   lane_center_offset > 0  → lane center is to the RIGHT of frame center
#   lane_center_offset < 0  → lane center is to the LEFT of frame center
#   None                    → lanes not detected

def ROI_lane(edge_frame):
    height, width = edge_frame.shape[:2]
    mask = np.zeros_like(edge_frame)
    bl = (int(width * 0.05), height)
    br = (int(width * 0.95), height)
    tr = (int(width * 0.65), int(height * 0.40))
    tl = (int(width * 0.35), int(height * 0.40))
    points = np.array([[bl, br, tr, tl]], dtype=np.int32)
    cv2.fillPoly(mask, points, 255)
    return cv2.bitwise_and(edge_frame, edge_frame, mask=mask)


def run_lane_pipeline(frame):
    img = frame.copy()
    height, width = img.shape[:2]
    midpoint = width // 2

    # --- Preprocessing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blur, 70, 100)
    roi = ROI_lane(edges)

    # --- Use only bottom half for stability ---
    roi_bottom = roi[int(height * 0.5):, :]
    y_left, x_left = np.where(roi_bottom[:, :midpoint] > 0)
    y_right, x_right = np.where(roi_bottom[:, midpoint:] > 0)

    # Shift y coordinates back to full frame space
    y_left  = y_left  + int(height * 0.5)
    y_right = y_right + int(height * 0.5)

    # --- Noise filter: remove near-horizontal points ---
    def filter_points(x, y):
        fx, fy = [], []
        for xi, yi in zip(x, y):
            if abs(yi - height) / (abs(xi - midpoint) + 1e-5) > 0.3:
                fx.append(xi)
                fy.append(yi)
        return np.array(fx), np.array(fy)

    x_left,  y_left  = filter_points(x_left,  y_left)
    x_right, y_right = filter_points(x_right, y_right)

    line_image = np.zeros_like(img)
    plot_y = np.linspace(int(height * 0.5), height - 1, 50)

    left_x_bottom  = None
    right_x_bottom = None

    # --- Fit and draw LEFT lane ---
    if len(x_left) > 50:
        left_func = np.poly1d(np.polyfit(y_left, x_left, 2))
        pts = np.array([np.transpose(np.vstack([left_func(plot_y), plot_y]))], np.int32)
        cv2.polylines(line_image, pts, False, (0, 255, 0), 6)
        left_x_bottom = int(left_func(height - 1))

    # --- Fit and draw RIGHT lane ---
    if len(x_right) > 50:
        right_func = np.poly1d(np.polyfit(y_right, x_right, 2))
        pts = np.array([np.transpose(np.vstack([right_func(plot_y) + midpoint, plot_y]))], np.int32)
        cv2.polylines(line_image, pts, False, (0, 255, 0), 6)
        right_x_bottom = int(right_func(height - 1) + midpoint)

    # --- Calculate lane center offset ---
    lane_center_offset = None
    if left_x_bottom is not None and right_x_bottom is not None:
        lane_center = (left_x_bottom + right_x_bottom) // 2
        lane_center_offset = lane_center - midpoint
        # Draw lane center indicator
        cv2.line(line_image, (lane_center, height - 20), (lane_center, height - 80), (0, 255, 255), 3)
        cv2.line(line_image, (midpoint,     height - 20), (midpoint,     height - 80), (255, 255, 0), 3)

    annotated = cv2.addWeighted(img, 1, line_image, 1, 0)
    return annotated, lane_center_offset

# DECISION ENGINE
# Priority: Barrier > Close Object > Lane Offset > Forward
# Returns: (decision_text, decision_color)

def make_decision(barrier_detected, yolo_detections, lane_center_offset):

    # Priority 1: Barrier in path
    if barrier_detected:
        return "STOP — BARRIER", (0, 0, 255)

    # Priority 2: Any close object
    close_objects = [d for d in yolo_detections if d['distance'] == 'CLOSE']
    if close_objects:
        return "STOP — OBSTACLE", (0, 0, 255)

    # Priority 3: Lane offset steering
    if lane_center_offset is not None:
        if lane_center_offset > LANE_OFFSET_THRESHOLD:
            return "TURN RIGHT", (0, 165, 255)
        elif lane_center_offset < -LANE_OFFSET_THRESHOLD:
            return "TURN LEFT", (0, 165, 255)

    # Default
    return "FORWARD", (0, 255, 0)

# DISPLAY OVERLAY — For Aleeza to expand on
# Draws decision, FPS, and detection count onto the frame

def draw_overlay(frame, decision, decision_color, fps, yolo_detections, lane_center_offset):
    h, w = frame.shape[:2]

    # --- Top bar background ---
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)

    # FPS — top left
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Decision indicator dot — top center
    dot_x = w // 2 - 120
    cv2.circle(frame, (dot_x, 25), 10, decision_color, -1)
    cv2.putText(frame, decision, (dot_x + 20, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, decision_color, 2)

    # Object count — top right
    cv2.putText(frame, f"Objects: {len(yolo_detections)}", (w - 200, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # --- Bottom bar background ---
    cv2.rectangle(frame, (0, h - 40), (w, h), (30, 30, 30), -1)

    # Lane offset info — bottom left
    if lane_center_offset is not None:
        offset_text = f"Lane Offset: {lane_center_offset:+d}px"
    else:
        offset_text = "Lane Offset: N/A"
    cv2.putText(frame, offset_text, (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    return frame

# MAIN PIPELINE
# Reads each .mp4 in VIDEO_FOLDER, runs all three threads per frame,
# makes a decision, overlays results, saves output video

def process_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open {filename}, skipping.")
        return

    fps_source = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out_path = os.path.join(OUTPUT_FOLDER, f"processed_{filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps_source, (1280, 720))

    frame_count = 0
    fps_display = 0.0
    prev_time = time.time()

    print(f"  Processing: {filename}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            frame_count += 1

            # --- Submit all three pipelines concurrently ---
            future_yolo     = executor.submit(run_yolo_pipeline,      frame)
            future_barrier  = executor.submit(run_classical_pipeline,  frame)
            future_lane     = executor.submit(run_lane_pipeline,       frame)

            # --- Collect results ---
            yolo_detections                    = future_yolo.result()
            barrier_detected, barrier_line, _  = future_barrier.result()
            lane_frame, lane_center_offset     = future_lane.result()

            # --- FPS calculation ---
            curr_time = time.time()
            fps_display = 1.0 / (curr_time - prev_time + 1e-5)
            prev_time = curr_time

            # --- Draw lane overlay on top of base frame ---
            output_frame = lane_frame.copy()

            # --- Draw YOLO bounding boxes ---
            for det in yolo_detections:
                x1, y1, x2, y2 = det['coords']
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), det['color'], 2)
                cv2.putText(output_frame, det['label'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, det['color'], 2)

            # --- Draw barrier line if detected ---
            if barrier_detected and barrier_line is not None:
                vx1, vy1, vx2, vy2 = barrier_line
                cv2.line(output_frame, (vx1, vy1), (vx2, vy2), (255, 0, 0), 4)

            # --- Decision ---
            decision, decision_color = make_decision(
                barrier_detected, yolo_detections, lane_center_offset
            )

            # --- Draw overlay (Aleeza will expand this function) ---
            output_frame = draw_overlay(
                output_frame, decision, decision_color,
                fps_display, yolo_detections, lane_center_offset
            )

            # --- Show live window ---
            cv2.imshow("Self-Driving Pipeline", output_frame)
            out.write(output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("  Stopped by user.")
                break

    cap.release()
    out.release()
    print(f"  Saved to: {out_path}")


def main():
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]

    if not video_files:
        print(f"No .mp4 files found in '{VIDEO_FOLDER}/' folder.")
        return

    print(f"Found {len(video_files)} video(s). Starting pipeline...\n")

    for filename in video_files:
        process_video(os.path.join(VIDEO_FOLDER, filename), filename)

    cv2.destroyAllWindows()
    print("\nAll videos processed.")


if __name__ == "__main__":
    main()
