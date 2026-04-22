import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# CONFIGURATION

VIDEO_FOLDER  = "DIP Project Videos"
OUTPUT_FOLDER = "output_videos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CLOSE_THRESHOLD_HEIGHT_RATIO = 0.45
LANE_OFFSET_THRESHOLD        = 40


# YOLO PIPELINE (Thread 1)

model = YOLO('yolov8n.pt')

def run_yolo_pipeline(frame):
    img          = frame.copy()
    frame_height = img.shape[0]
    results      = model(img, conf=0.4, verbose=False)
    detections   = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id          = int(box.cls[0])
            conf            = float(box.conf[0])
            class_name      = model.names[cls_id]

            if class_name in ['person', 'car', 'motorcycle', 'bus', 'truck']:
                obj_width  = x2 - x1
                obj_height = y2 - y1

                # FIX 2: Pole person filter restored — rejects lamp posts / thin verticals
                if class_name == 'person' and (obj_width / (obj_height + 1e-5)) < 0.20:
                    continue

                height_ratio      = obj_height / frame_height
                distance_estimate = "CLOSE" if height_ratio > CLOSE_THRESHOLD_HEIGHT_RATIO else "FAR"
                color             = (0, 0, 255) if distance_estimate == "CLOSE" else (0, 255, 0)

                detections.append({
                    'coords':   (x1, y1, x2, y2),
                    'color':    color,
                    'label':    f"{class_name} {conf:.2f} [{distance_estimate}]",
                    'distance': distance_estimate
                })

    return detections


# CLASSICAL CV PIPELINE (Thread 2)

def run_classical_pipeline(frame, needs_darkening):
    # FIX 1: exposure flag passed in from main loop (checked every 15 frames there)
    img    = frame.copy()
    height, width = img.shape[:2]

    if needs_darkening:
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=-80)

    roi_mask_img = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(roi_mask_img,
                  (int(width * 0.25), int(height * 0.35)),
                  (int(width * 0.70), int(height * 0.70)),
                  255, -1)

    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,   120, 100]), np.array([10,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 120, 100]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_and(
        cv2.bitwise_or(mask1, mask2),
        cv2.bitwise_or(mask1, mask2),
        mask=roi_mask_img
    )
    red_mask = cv2.dilate(red_mask, np.ones((1, 100), np.uint8), iterations=1)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask  = np.zeros_like(red_mask)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 0 and cv2.contourArea(cnt) > 500 and w > 200 and (float(w) / h) > 3.0:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    edges = cv2.Canny(clean_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=150, maxLineGap=100)

    barrier_detected, barrier_line = False, None
    if lines is not None:
        lines  = lines[:, 0, :]
        angles = np.abs(np.arctan2(lines[:, 3] - lines[:, 1],
                                   lines[:, 2] - lines[:, 0]) * 180.0 / np.pi)
        valid  = lines[(angles < 25) | (angles > 155)]
        if len(valid) > 0:
            barrier_detected, barrier_line = True, valid[0]

    return barrier_detected, barrier_line, clean_mask

# LANE DETECTION PIPELINE (Thread 3)

def run_lane_pipeline(frame):
    img      = frame.copy()
    height, width = img.shape[:2]
    midpoint = width // 2

    edges = cv2.Canny(
        cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0),
        70, 100
    )

    mask    = np.zeros_like(edges)
    pts_roi = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.95), height),
        (int(width * 0.65), int(height * 0.40)),
        (int(width * 0.35), int(height * 0.40))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts_roi, 255)
    roi = cv2.bitwise_and(edges, edges, mask=mask)

    roi_bottom = roi[int(height * 0.5):, :]
    y_left,  x_left  = np.where(roi_bottom[:, :midpoint] > 0)
    y_right, x_right = np.where(roi_bottom[:, midpoint:] > 0)

    y_left  = y_left  + int(height * 0.5)
    y_right = y_right + int(height * 0.5)

    def filter_points(x, y, midpoint, height):
        if len(x) == 0:
            return np.array([]), np.array([])
        slope_mask = np.array([
            abs(yi - height) / (abs(xi - midpoint) + 1e-5) > 0.4
            for xi, yi in zip(x, y)
        ])
        x, y = x[slope_mask], y[slope_mask]
        if len(x) == 0:
            return np.array([]), np.array([])
        median_x, std_x = np.median(x), np.std(x)
        stat_mask = np.abs(x - median_x) < 1.5 * std_x
        return x[stat_mask], y[stat_mask]

    x_left,  y_left  = filter_points(x_left,  y_left,  midpoint, height)
    x_right, y_right = filter_points(x_right, y_right, midpoint, height)

    line_image     = np.zeros_like(img)
    left_x_bottom  = None
    right_x_bottom = None
    plot_y         = np.linspace(height // 2, height - 1, 50)

    if len(x_left) > 50:
        degree    = 2 if len(x_left) > 200 else 1
        left_func = np.poly1d(np.polyfit(y_left, x_left, degree))
        left_x_bottom = int(left_func(height - 1))
        pts = np.array([np.transpose(np.vstack([left_func(plot_y), plot_y]))], np.int32)
        cv2.polylines(line_image, pts, False, (0, 255, 0), 6)

    if len(x_right) > 50:
        degree     = 2 if len(x_right) > 200 else 1
        right_func = np.poly1d(np.polyfit(y_right, x_right, degree))
        right_x_bottom = int(right_func(height - 1) + midpoint)
        pts = np.array([np.transpose(np.vstack([right_func(plot_y) + midpoint, plot_y]))], np.int32)
        cv2.polylines(line_image, pts, False, (0, 255, 0), 6)

    lane_center_offset = (
        (left_x_bottom + right_x_bottom) // 2 - midpoint
        if left_x_bottom is not None and right_x_bottom is not None
        else None
    )

    return cv2.addWeighted(img, 1, line_image, 1, 0), lane_center_offset

# DECISION ENGINE

def make_decision(barrier_detected, yolo_detections, lane_center_offset):
    if barrier_detected:
        return "STOP - BARRIER", (0, 0, 255)
    if any(d['distance'] == 'CLOSE' for d in yolo_detections):
        return "STOP - OBSTACLE", (0, 0, 255)
    if lane_center_offset is not None:
        if lane_center_offset > LANE_OFFSET_THRESHOLD:
            return "TURN RIGHT", (0, 165, 255)
        if lane_center_offset < -LANE_OFFSET_THRESHOLD:
            return "TURN LEFT", (0, 165, 255)
    return "FORWARD", (0, 255, 0)


# DISPLAY OVERLAY

def draw_overlay(frame, decision, decision_color, fps, yolo_detections, lane_center_offset):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text_size = cv2.getTextSize(decision, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    start_x = (w - (text_size[0] + 35)) // 2
    cv2.circle(frame, (start_x + 12, 25), 10, decision_color, -1)
    cv2.putText(frame, decision, (start_x + 37, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, decision, (start_x + 35, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, decision_color, 2)

    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
    offset_text = f"OFFSET: {lane_center_offset:+d}px" if lane_center_offset is not None else "LANE: N/A"
    cv2.putText(frame, offset_text, (15, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    obj_text = f"OBJECTS: {len(yolo_detections)}"
    cv2.putText(frame, obj_text, (w - 180, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cx, cy = w // 2, h - 25
    if "FORWARD" in decision:
        pts = np.array([[cx, cy-15], [cx-12, cy+10], [cx+12, cy+10]], np.int32)
        cv2.fillPoly(frame, [pts], decision_color)
    elif "RIGHT" in decision:
        pts = np.array([[cx+15, cy], [cx-10, cy-12], [cx-10, cy+12]], np.int32)
        cv2.fillPoly(frame, [pts], decision_color)
    elif "LEFT" in decision:
        pts = np.array([[cx-15, cy], [cx+10, cy-12], [cx+10, cy+12]], np.int32)
        cv2.fillPoly(frame, [pts], decision_color)
    elif "STOP" in decision:
        cv2.rectangle(frame, (cx-12, cy-12), (cx+12, cy+12), decision_color, -1)

    if "BARRIER" in decision:
        cv2.rectangle(frame, (w//2 - 220, 80), (w//2 + 220, 140), (0, 0, 255), -1)
        cv2.putText(frame, "!!! BARRIER DETECTED !!!", (w//2 - 195, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


# MAIN PIPELINE

def process_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open {filename}, skipping.")
        return

    # FIX 4: use source FPS, not hardcoded 30
    fps_source = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = cv2.VideoWriter(
        os.path.join(OUTPUT_FOLDER, f"processed_{filename}"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_source,
        (1280, 720)
    )

    frame_count     = 0
    prev_time       = time.time()
    needs_darkening = False   # FIX 1: exposure state lives here

    print(f"  Processing: {filename}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.resize(frame, (1280, 720))
            frame_count += 1

            # FIX 1: brightness check every 15 frames — cheap, matches Shaheer's original
            if frame_count % 15 == 0:
                tiny_frame      = cv2.resize(frame, (64, 64))
                gray_tiny       = cv2.cvtColor(tiny_frame, cv2.COLOR_BGR2GRAY)
                needs_darkening = (np.mean(gray_tiny) > 180)

            f_yolo = executor.submit(run_yolo_pipeline, frame)
            f_bar  = executor.submit(run_classical_pipeline, frame, needs_darkening)
            f_lane = executor.submit(run_lane_pipeline, frame)

            yolo_dets            = f_yolo.result()
            bar_det, bar_line, _ = f_bar.result()
            lane_frame, offset   = f_lane.result()

            curr_time   = time.time()
            fps_display = 1.0 / (curr_time - prev_time + 1e-5)
            prev_time   = curr_time

            output_frame = lane_frame.copy()

            for det in yolo_dets:
                x1, y1, x2, y2 = det['coords']
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), det['color'], 2)
                cv2.putText(output_frame, det['label'], (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, det['color'], 2)

            if bar_det and bar_line is not None:
                cv2.line(output_frame,
                         (bar_line[0], bar_line[1]),
                         (bar_line[2], bar_line[3]),
                         (255, 0, 0), 4)

            decision, d_color = make_decision(bar_det, yolo_dets, offset)
            output_frame = draw_overlay(
                output_frame, decision, d_color, fps_display, yolo_dets, offset)

            cv2.imshow("Self-Driving Pipeline", output_frame)
            out.write(output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    print(f"  Saved: processed_{filename}")


if __name__ == "__main__":
    v_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")]
    if not v_files:
        print(f"No .mp4 files found in '{VIDEO_FOLDER}/'")
    else:
        print(f"Found {len(v_files)} video(s).\n")
        for vid in v_files:
            process_video(os.path.join(VIDEO_FOLDER, vid), vid)
        # FIX 5: destroyAllWindows after ALL videos done, not inside each one
        cv2.destroyAllWindows()