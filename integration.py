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

                # Reject pole-shaped false positives labelled as "person"
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
        valid = valid[valid[:, 1] < int(height * 0.65)]
        if len(valid) > 0:
            barrier_detected, barrier_line = True, valid[0]

    return barrier_detected, barrier_line, clean_mask

# LANE DETECTION PIPELINE (Thread 3)

# --- Global state for lane smoothing and adaptive ROI ---
# Kept as module-level variables as per Ayesha's design.
# _reset_lane_state() is called between videos to prevent bleed-over.

_prev_left_fit_avg  = None
_prev_right_fit_avg = None
_roi_state          = None


def _reset_lane_state():
    """Call between videos to clear smoothing and ROI state."""
    global _prev_left_fit_avg, _prev_right_fit_avg, _roi_state
    _prev_left_fit_avg  = None
    _prev_right_fit_avg = None
    _roi_state          = None


def _canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 70, 170)


def _adaptive_roi_cca(edge_image, original_shape):
    """
    Ayesha's CCA-based adaptive ROI.
    Finds edge components on left/right sides and builds a trapezoid around them.
    Downscales 4x for performance, then smooths coordinates over time.
    """
    global _roi_state
    h, w = original_shape[:2]

    scale       = 4
    small_edge  = cv2.resize(edge_image, (w // scale, h // scale))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(small_edge, connectivity=8)

    left_coords  = []
    right_coords = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < (50 // scale):
            continue
        x     = stats[i, cv2.CC_STAT_LEFT]  * scale
        width = stats[i, cv2.CC_STAT_WIDTH]  * scale
        if x < w // 2:
            left_coords.append(x)
        else:
            right_coords.append(x + width)

    target_left  = min(left_coords)  if left_coords  else int(w * 0.1)
    target_right = max(right_coords) if right_coords else int(w * 0.9)

    target_state = [
        target_left,
        target_right,
        target_left  + int(w * 0.15),
        target_right - int(w * 0.15)
    ]

    if _roi_state is None:
        _roi_state = target_state
    else:
        _roi_state = [int(0.9 * c + 0.1 * t) for c, t in zip(_roi_state, target_state)]

    top_y = int(h * 0.6)
    pts   = np.array([[
        (_roi_state[0], h),
        (_roi_state[1], h),
        (_roi_state[3], top_y),
        (_roi_state[2], top_y)
    ]], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask


def _make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    if abs(slope) < 1e-3:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.45)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def _average_slope_intercept(image, lines):
    """
    Averages Hough lines into one left and one right lane line.
    Filters near-horizontal lines (slope < 0.5).
    Applies temporal smoothing: 20% new + 80% previous.
    """
    global _prev_left_fit_avg, _prev_right_fit_avg

    left_fit  = []
    right_fit = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope      = parameters[0]
            intercept  = parameters[1]

            if abs(slope) < 0.5:
                continue

            x_at_bottom = int((image.shape[0] - intercept) / slope)

            if slope < 0 and x_at_bottom < image.shape[1] // 2:
                left_fit.append((slope, intercept))
            elif slope > 0 and x_at_bottom > image.shape[1] // 2:
                right_fit.append((slope, intercept))

    left_fit_average  = _prev_left_fit_avg
    right_fit_average = _prev_right_fit_avg

    if len(left_fit) > 0:
        current = np.average(left_fit, axis=0)
        left_fit_average = (
            0.2 * current + 0.8 * _prev_left_fit_avg
            if _prev_left_fit_avg is not None else current
        )

    if len(right_fit) > 0:
        current = np.average(right_fit, axis=0)
        right_fit_average = (
            0.2 * current + 0.8 * _prev_right_fit_avg
            if _prev_right_fit_avg is not None else current
        )

    left_line  = None
    right_line = None

    if left_fit_average is not None:
        left_line = _make_coordinates(image, left_fit_average)
        _prev_left_fit_avg = left_fit_average

    if right_fit_average is not None:
        right_line = _make_coordinates(image, right_fit_average)
        _prev_right_fit_avg = right_fit_average

    return left_line, right_line


def _display_lanes(image, left_line, right_line):
    """Fills the polygon between left and right lane lines."""
    line_image = np.zeros_like(image)

    if left_line is not None and right_line is not None:
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        pts = np.array([[
            (int(lx1), int(ly1)),
            (int(lx2), int(ly2)),
            (int(rx2), int(ry2)),
            (int(rx1), int(ry1))
        ]], dtype=np.int32)
        cv2.fillPoly(line_image, pts, (255, 0, 100))
    elif left_line is not None:
        lx1, ly1, lx2, ly2 = left_line
        cv2.line(line_image, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 0, 100), 8)
    elif right_line is not None:
        rx1, ry1, rx2, ry2 = right_line
        cv2.line(line_image, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255, 0, 100), 8)

    return line_image


def run_lane_pipeline(frame):
    """
    Wrapper integrating Ayesha's adaptive CCA ROI lane detection.
    Returns: (annotated_frame, lane_center_offset)
        offset > 0  → lane center is RIGHT of frame center → steer right
        offset < 0  → lane center is LEFT of frame center  → steer left
        offset None → lanes not detected this frame
    """
    img    = frame.copy()
    height, width = img.shape[:2]
    midpoint = width // 2

    # Canny edges
    cannied = _canny(img)

    # Adaptive CCA-based ROI
    adaptive_mask = _adaptive_roi_cca(cannied, img.shape)
    roi_image     = cv2.bitwise_and(cannied, adaptive_mask)

    # Hough lines
    lines = cv2.HoughLinesP(roi_image, 2, np.pi / 180, 50,
                            np.array([]), minLineLength=20, maxLineGap=100)

    # Average and smooth lines
    left_line, right_line = _average_slope_intercept(img, lines)

    # Draw lane fill
    line_image = _display_lanes(img, left_line, right_line)
    annotated  = cv2.addWeighted(img, 0.9, line_image, 1, 1)

    # Calculate lane center offset for steering
    lane_center_offset = None
    if left_line is not None and right_line is not None:
        # x at bottom of frame for each line
        left_x_bottom  = left_line[0]
        right_x_bottom = right_line[0]
        lane_center        = (left_x_bottom + right_x_bottom) // 2
        lane_center_offset = lane_center - midpoint

        # Draw center indicators
        cv2.line(annotated, (lane_center, height - 20), (lane_center, height - 80), (0, 255, 255), 3)
        cv2.line(annotated, (midpoint,    height - 20), (midpoint,    height - 80), (255, 255,  0), 3)

    if lane_center_offset is not None and abs(lane_center_offset) > int(width * 0.40):
        lane_center_offset = None  # too extreme, likely a detection error

    return annotated, lane_center_offset

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
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    text_size = cv2.getTextSize(decision, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    start_x   = (w - (text_size[0] + 35)) // 2
    cv2.circle(frame, (start_x + 12, 25), 10, decision_color, -1)
    cv2.putText(frame, decision, (start_x + 37, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(frame, decision, (start_x + 35, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, decision_color, 2)

    cv2.rectangle(frame, (0, h - 50), (w, h), (20, 20, 20), -1)
    offset_text = f"OFFSET: {lane_center_offset:+d}px" if lane_center_offset is not None else "LANE: N/A"
    cv2.putText(frame, offset_text, (15, h - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, f"OBJECTS: {len(yolo_detections)}", (w - 180, h - 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cx, cy = w // 2, h - 25
    if "FORWARD" in decision:
        pts = np.array([[cx, cy - 15], [cx - 12, cy + 10], [cx + 12, cy + 10]], np.int32)
        cv2.fillPoly(frame, [pts], decision_color)
    elif "RIGHT" in decision:
        pts = np.array([[cx + 15, cy], [cx - 10, cy - 12], [cx - 10, cy + 12]], np.int32)
        cv2.fillPoly(frame, [pts], decision_color)
    elif "LEFT" in decision:
        pts = np.array([[cx - 15, cy], [cx + 10, cy - 12], [cx + 10, cy + 12]], np.int32)
        cv2.fillPoly(frame, [pts], decision_color)
    elif "STOP" in decision:
        cv2.rectangle(frame, (cx - 12, cy - 12), (cx + 12, cy + 12), decision_color, -1)

    if "BARRIER" in decision:
        cv2.rectangle(frame, (w // 2 - 220, 80), (w // 2 + 220, 140), (0, 0, 255), -1)
        cv2.putText(frame, "!!! BARRIER DETECTED !!!", (w // 2 - 195, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame

# MAIN PIPELINE
def process_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Could not open {filename}, skipping.")
        return

    fps_source = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    out = cv2.VideoWriter(
        os.path.join(OUTPUT_FOLDER, f"processed_{filename}"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_source,
        (1280, 720)
    )

    frame_count     = 0
    prev_time       = time.time()
    needs_darkening = False

    # Reset Ayesha's lane state for each new video
    _reset_lane_state()

    print(f"  Processing: {filename}")

    with ThreadPoolExecutor(max_workers=3) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame        = cv2.resize(frame, (1280, 720))
            frame_count += 1

            # Brightness check every 15 frames
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
        cv2.destroyAllWindows()
        print("\nAll done.")