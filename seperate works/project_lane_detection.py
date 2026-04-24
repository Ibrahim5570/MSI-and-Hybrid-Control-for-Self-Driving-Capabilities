import cv2
import numpy as np

prev_left_fit_avg = None
prev_right_fit_avg = None

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 70, 170)
    return canny

roi_state = None 
smoothing_factor = 0.1 # Adjust between 0.05 (very stable) and 0.3 (fast adapting)

def adaptive_roi_cca(edge_image, original_shape):
    global roi_state
    h, w = original_shape[:2]
    
    # --- Optimization: Only process a smaller version for ROI logic ---
    # We shrink the image by 4x to speed up CCA significantly
    scale = 4
    small_edge = cv2.resize(edge_image, (w // scale, h // scale))
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(small_edge, connectivity=8)
    
    left_coords = []
    right_coords = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Skip tiny noise immediately
        if area < (50 // scale): 
            continue
            
        x = stats[i, cv2.CC_STAT_LEFT] * scale
        width = stats[i, cv2.CC_STAT_WIDTH] * scale
        
        # Sort components into left and right sides of the screen
        if x < w // 2:
            left_coords.append(x)
        else:
            right_coords.append(x + width)

    # Calculate target based on the most extreme left/right components found
    # Fallback to defaults if nothing is detected
    target_left = min(left_coords) if left_coords else int(w * 0.1)
    target_right = max(right_coords) if right_coords else int(w * 0.9)

    target_state = [
        target_left,                       # bottom_left
        target_right,                      # bottom_right
        target_left + int(w * 0.15),       # top_left (tapered)
        target_right - int(w * 0.15)       # top_right (tapered)
    ]

    # --- Temporal Smoothing (Same as before) ---
    if roi_state is None:
        roi_state = target_state
    else:
        roi_state = [int(0.9 * c + 0.1 * t) for c, t in zip(roi_state, target_state)]

    # Create mask on the FULL scale
    top_y = int(h * 0.6)
    pts = np.array([[(roi_state[0], h), (roi_state[1], h), 
                     (roi_state[3], top_y), (roi_state[2], top_y)]], dtype=np.int32)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, pts, 255)
    return mask

# # --- New Global Variables for ROI Stability ---
# # Format: [bottom_left_x, bottom_right_x, top_left_x, top_right_x]


# def adaptive_roi_cca(edge_image, original_shape):
#     global roi_state
#     h, w = original_shape[:2]
    
#     # 1. Define a "Base" Trapezoid as a fallback
#     base_bottom_left = int(w * 0.1)
#     base_bottom_right = int(w * 0.9)
#     base_top_left = int(w * 0.4)
#     base_top_right = int(w * 0.6)
#     top_y = int(h * 0.6)

#     # 2. CCA to find lane-like features
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_image, connectivity=8)
    
#     candidate_coords = []
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
#         width = stats[i, cv2.CC_STAT_WIDTH]
#         height = stats[i, cv2.CC_STAT_HEIGHT]
        
#         # Filter for elongated shapes (lane markings)
#         if area > 50 and height > width * 1.2:
#             # Get all pixel coordinates for this component
#             component_points = np.argwhere(labels == i)
#             candidate_coords.append(component_points)

#     # 3. Calculate "Target" boundaries based on detections
#     if len(candidate_coords) > 0:
#         all_pts = np.vstack(candidate_coords)
#         # We only care about the X-coordinates (columns)
#         new_left_x = np.min(all_pts[:, 1])
#         new_right_x = np.max(all_pts[:, 1])
        
#         # Constrain the movement so it doesn't shrink to nothing
#         target_state = [
#             max(0, new_left_x - 20),           # bottom_left
#             min(w, new_right_x + 20),          # bottom_right
#             max(int(w*0.2), new_left_x + 50),  # top_left (narrower than bottom)
#             min(int(w*0.8), new_right_x - 50)  # top_right
#         ]
#     else:
#         target_state = [base_bottom_left, base_bottom_right, base_top_left, base_top_right]

#     # 4. Temporal Smoothing (The "Fix")
#     if roi_state is None:
#         roi_state = target_state
#     else:
#         # Move the current ROI state slowly towards the target
#         roi_state = [
#             int((1 - smoothing_factor) * curr + smoothing_factor * targ)
#             for curr, targ in zip(roi_state, target_state)
#         ]

#     # 5. Create the Mask from the smoothed coordinates
#     smoothed_trapezoid = np.array([[
#         (roi_state[0], h), 
#         (roi_state[1], h), 
#         (roi_state[3], top_y), 
#         (roi_state[2], top_y)
#     ]], dtype=np.int32)
    
#     mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillPoly(mask, smoothed_trapezoid, 255)
    
#     return mask

# # ---------- Friend's CCA-based adaptive ROI (replaces old adaptive_roi) ----------
# def adaptive_roi_cca(edge_image, original_shape):
#     """
#     edge_image: output of Canny (binary)
#     original_shape: (height, width) of the original frame
#     Returns a binary mask (same size as edge_image) that highlights probable lane regions.
#     """
#     h, w = original_shape[:2]
    
#     # Step 1: Initial trapezoid ROI (road area at bottom, narrowing towards top)
#     top_width = int(w * 0.35)
#     top_x = (w - top_width) // 2
#     top_y = int(h * 0.6)#0.8
#     trapezoid = np.array([[(int(w*0.1), h), (int(w*0.9), h), (top_x + top_width, top_y), (top_x, top_y)]], dtype=np.int32)
    
#     roi_mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillPoly(roi_mask, trapezoid, 255)
    
#     # Apply trapezoid mask to edge image
#     edges_roi = cv2.bitwise_and(edge_image, roi_mask)
    
#     # Step 2: Connected Components Analysis
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges_roi, connectivity=8)
    
#     if num_labels <= 1:
#         return roi_mask  # fallback
    
#     # Step 3: Filter components (keep large, elongated ones)
#     valid_components = []
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
#         width = stats[i, cv2.CC_STAT_WIDTH]
#         height = stats[i, cv2.CC_STAT_HEIGHT]
#         # Lane markings are typically elongated (height > width*1.5) and not tiny
#         if area > 70 and height > width * 1.3:
#             valid_components.append(i)
    
#     if not valid_components:
#         return roi_mask
    
#     # Step 4: Create mask from valid components
#     lane_mask = np.zeros((h, w), dtype=np.uint8)
#     for comp in valid_components:
#         lane_mask[labels == comp] = 255
    
#     # Step 5: Morphological closing to fill holes and connect gaps
#     kernel = np.ones((5,5), np.uint8)
#     lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel)
#     # lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel)  # remove small noise
    
#     return lane_mask
# # ---------------------------------------------------------------------------------

def make_coordinates(image, line_parameters): 
    slope, intercept = line_parameters
    if abs(slope) < 1e-3:
        return np.array([0,0,0,0])
    y1 = image.shape[0]
    y2 = int(y1 * 0.45)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    global prev_left_fit_avg, prev_right_fit_avg

    left_fit = []
    right_fit = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if abs(slope) < 0.5 :  # Filter out near-horizontal and extremely steep lines
                continue

            x_at_bottom = int((image.shape[0] - intercept) / slope)

            if slope < 0 and x_at_bottom < image.shape[1] // 2:
                left_fit.append((slope, intercept))
            elif slope > 0 and x_at_bottom > image.shape[1] // 2:
                right_fit.append((slope, intercept))


    left_fit_average = prev_left_fit_avg
    right_fit_average = prev_right_fit_avg

    if len(left_fit) > 0:
        current_left_avg = np.average(left_fit, axis=0)
        if prev_left_fit_avg is not None:
            left_fit_average = 0.2 * current_left_avg + 0.8 * prev_left_fit_avg
        else:
            left_fit_average = current_left_avg

    if len(right_fit) > 0:
        current_right_avg = np.average(right_fit, axis=0)
        if prev_right_fit_avg is not None:
            right_fit_average = 0.2 * current_right_avg + 0.8 * prev_right_fit_avg
        else:
            right_fit_average = current_right_avg

    print('left', left_fit_average)
    print('right', right_fit_average)

    left_line = None
    right_line = None

    if left_fit_average is not None:
        left_line = make_coordinates(image, left_fit_average)
        prev_left_fit_avg = left_fit_average

    if right_fit_average is not None:
        right_line = make_coordinates(image, right_fit_average)
        prev_right_fit_avg = right_fit_average

    return np.array([left_line, right_line], dtype=object)


# def average_slope_intercept(image, lines):
#     global prev_left_fit_avg, prev_right_fit_avg
#     left_fit = []
#     right_fit = []
#     if lines is None:
#         # If no lines, reuse previous values (optional, keep as original)
#         pass
#     else:
#         for l in lines:
#             x1, y1, x2, y2 = l.reshape(4)
#             parameters = np.polyfit((x1, x2), (y1, y2), 1)
#             slope = parameters[0]
#             intercept = parameters[1]
#             if abs(slope) < 0.5 or abs (slope) >2.5:  # Filter out near-horizontal and extremely steep lines
#                 continue
#             # Calculate x at bottom (y = image height)
#             x_at_bottom = int((image.shape[0] - intercept) / slope)
#             # Classify by position: left half vs right half
#             if x_at_bottom < image.shape[1] // 2:
#                 left_fit.append((slope, intercept))
#             else:
#                 right_fit.append((slope, intercept))
#             # if slope < 0:
#             #     left_fit.append((slope, intercept))
#             # else:
#             #     right_fit.append((slope, intercept))

#         if len(left_fit) > 0:
#             current_left_avg = np.average(left_fit, axis=0)
#             if prev_left_fit_avg is not None:
#                 left_fit_average = 0.2 * current_left_avg + 0.8 * prev_left_fit_avg
#             else:
#                 left_fit_average = current_left_avg
#         else:
#             left_fit_average = prev_left_fit_avg

#         if len(right_fit) > 0:
#             current_right_avg = np.average(right_fit, axis=0)
#             if prev_right_fit_avg is not None:
#                 right_fit_average = 0.2 * current_right_avg + 0.8 * prev_right_fit_avg
#             else:
#                 right_fit_average = current_right_avg
#         else:
#             right_fit_average = prev_right_fit_avg

#     # if len(left_fit) > 0:
#     #     current_left_avg = np.average(left_fit, axis=0)
#     #     left_fit_average = 0.2 * current_left_avg + 0.8 * prev_left_fit_avg
#     # else:
#     #     left_fit_average = prev_left_fit_avg
    
#     # if len(right_fit) > 0:
#     #     current_right_avg = np.average(right_fit, axis=0)
#     #     right_fit_average = 0.2 * current_right_avg + 0.8 * prev_right_fit_avg
#     # else:
#     #     right_fit_average = prev_right_fit_avg

#     print('left', left_fit_average)
#     print('right', right_fit_average)

#     left_line = None
#     right_line = None


#     if left_fit_average is not None:
#         left_line = make_coordinates(image, left_fit_average)
#         prev_left_fit_avg = left_fit_average
#     elif prev_left_fit_avg is not None:
#         left_line = make_coordinates(image, prev_left_fit_avg)
        
#     if right_fit_average is not None:
#         right_line = make_coordinates(image, right_fit_average)
#         prev_right_fit_avg = right_fit_average
#     elif prev_right_fit_avg is not None:
#         right_line = make_coordinates(image, prev_right_fit_avg)

#     return np.array([left_line, right_line], dtype=object)

# def display(image, lines):
#     line_image = np.zeros_like(image)
#     if lines is not None:
#         for line in lines:
#             if line is not None:
#                 x1, y1, x2, y2 = line
#                 cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,100), 12)
#     return line_image

def display(image, lines):
    line_image = np.zeros_like(image)

    if lines is not None and len(lines) == 2:
        left_line = lines[0]
        right_line = lines[1]

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

    return line_image

cap = cv2.VideoCapture("PXL_20250325_043922504.TS.mp4")

while cap.isOpened():
    a, frame = cap.read()
    if not a or frame is None:
        break
    
    cannied = canny(frame)
    
    # ---------- Use CCA-based adaptive mask (on edge image) ----------
    adaptive_mask = adaptive_roi_cca(cannied, frame.shape)
    roi_image = cv2.bitwise_and(cannied, adaptive_mask)
    # ----------------------------------------------------------------
    
    lines = cv2.HoughLinesP(roi_image, 2, np.pi/180, 50, np.array([]), minLineLength=20, maxLineGap=100)
    avg_lines = average_slope_intercept(frame, lines)
    line_image = display(frame, avg_lines)
    combine_image = cv2.addWeighted(frame, 0.9, line_image, 1, 1)
    
    cv2.imshow('ROI', roi_image)
    cv2.imshow('result', combine_image)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
