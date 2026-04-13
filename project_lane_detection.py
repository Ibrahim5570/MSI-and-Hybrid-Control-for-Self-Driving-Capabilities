
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

input_folder="C:\\Users\\hp\\Downloads\\DIP Project Videos"

def ROI(frame, filename):
    height, width = frame.shape[:2]
    mask = np.zeros_like(frame)

    # These coordinates are wider and taller to catch those side lanes
    bl = (int(width * 0.05), height)           # Bottom Left (very wide)
    br = (int(width * 0.95), height)           # Bottom Right (very wide)
    tr = (int(width * 0.65), int(height * 0.40)) # Top Right (higher up)
    tl = (int(width * 0.35), int(height * 0.40)) # Top Left (higher up)

    points = np.array([[bl, br, tr, tl]], dtype=np.int32)
    
    cv2.fillPoly(mask, points, 255)
    
    # This bitwise_and will now actually have edges to catch!
    return cv2.bitwise_and(frame, frame, mask=mask)
# def ROI(frame,filename):
#                 height=frame.shape[0]
#                 print(frame.shape)
#                 triangle=np.array([[(int(0.1*frame.shape[1]),height),(int(0.9*frame.shape[1]),height),(int(0.5*frame.shape[1]),int(0.6*frame.shape[0]))]])
                
                
#                 if "PXL_20250325_043754655.TS" in filename:
#                     triangle=np.array([[(250,height),(1700,height),(900,350)]]) 
#                 elif "PXL_20250325_044603023.TS" in filename: 
#                     triangle=np.array([[(250,height),(1700,height),(350,1000)]])
#                 elif "PXL_20250325_044746327.TS" in filename:
#                     triangle=np.array([[(250,height),(1700,height),(350,1000)]])
#                 elif "PXL_20250325_045117252.TS" in filename:
#                      triangle=np.array([[(250,height),(1700,height),(350,1000)]]) 
#                 elif "PXL_20250325_044505516.TS" in filename: 
#                     triangle=np.array([[(250,height),(1700,height),(350,1000)]]) 
#                 elif "PXL_20250325_043922504.TS" in filename: 
#                     triangle=np.array([[(250,height),(1700,height),(350,1000)]]) 
                
                     
#                 mask=np.zeros(frame.shape[:2],dtype=np.uint8)
#                 cv2.fillPoly(mask, triangle, 255)
#                 return cv2.bitwise_and(frame,frame,mask=mask)

def draw_lines(frame, roi_lines):
    height, width = roi_lines.shape
    midpoint = width // 2
    line_image = np.zeros_like(frame)

    # 🔥 NEW: Use only bottom half (more stable)
    roi_bottom = roi_lines[int(height*0.5):, :]

    # 1. Split into left and right
    left_roi = roi_bottom[:, :midpoint]
    right_roi = roi_bottom[:, midpoint:]

    # 2. Get points
    y_left, x_left = np.where(left_roi > 0)
    y_right, x_right = np.where(right_roi > 0)

    # 🔥 NEW: shift y because we cropped image
    y_left = y_left + int(height*0.5)
    y_right = y_right + int(height*0.5)

    # 🔥 NEW: simple noise filtering (remove weak slopes)
    def filter_points(x, y):
        filtered_x = []
        filtered_y = []

        for xi, yi in zip(x, y):
            if abs(yi - height) / (abs(xi - midpoint) + 1e-5) > 0.3:
                filtered_x.append(xi)
                filtered_y.append(yi)

        return np.array(filtered_x), np.array(filtered_y)

    x_left, y_left = filter_points(x_left, y_left)
    x_right, y_right = filter_points(x_right, y_right)

    # 3. Fit LEFT curve
    if len(x_left) > 50:
        left_curve = np.polyfit(y_left, x_left, 2)
        left_func = np.poly1d(left_curve)

        plot_y = np.linspace(int(height * 0.5), height - 1, 50)
        plot_x_left = left_func(plot_y)

        pts_left = np.array([np.transpose(np.vstack([plot_x_left, plot_y]))], np.int32)
        cv2.polylines(line_image, pts_left, False, (0, 255, 0), 6)

    # 4. Fit RIGHT curve
    if len(x_right) > 50:
        right_curve = np.polyfit(y_right, x_right, 2)
        right_func = np.poly1d(right_curve)

        plot_y = np.linspace(int(height * 0.5), height - 1, 50)
        plot_x_right = right_func(plot_y) + midpoint

        pts_right = np.array([np.transpose(np.vstack([plot_x_right, plot_y]))], np.int32)
        cv2.polylines(line_image, pts_right, False, (0, 255, 0), 6)

    return cv2.addWeighted(frame, 1, line_image, 1, 0)             

def pre_process(input_folder):

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_folder, filename)
            cap = cv2.VideoCapture(video_path)

            print(f"Processing video: {filename}")
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Skipping {filename} (cannot read frame)")
                break
           
            img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            img_blur=cv2.GaussianBlur(img,(9,9),0)
            img_edges=cv2.Canny(img_blur,70,100)

            ROI_img = ROI(img_edges, filename)
            result_frame = draw_lines(frame, ROI_img)
            resized_frame=cv2.resize(result_frame,(960,540))
            cv2.imshow("road Boundary", resized_frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

            
            # plt.figure(figsize=(10,5))
            # plt.subplot(2,1,1)
            # plt.title("Canny")
            # plt.imshow(img_edges,cmap='gray')
            # plt.axis('off')
            # plt.subplot(2,1,2)
            # plt.title("ROI")
            # plt.imshow(ROI_img,cmap='gray')
            # plt.axis('off')
            # plt.show()

            cap.release()

    cv2.destroyAllWindows()
    print("Done pre-processing!")

pre_process(input_folder)



    




