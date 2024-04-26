import cv2
import numpy as np

def process_image(frame):
   # Convert to grayscale
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   # Apply Gaussian blur
   blurred = cv2.GaussianBlur(gray, (5, 5), 0)

   # Apply Canny edge detection
   edges = cv2.Canny(blurred, 50, 150)

   # Define region of interest (ROI)
   height, width = edges.shape
   roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
   mask = np.zeros_like(edges)
   cv2.fillPoly(mask, roi_vertices, 255)
   roi_edges = cv2.bitwise_and(edges, mask)

   return roi_edges

def calculate_lane_centers(lines, height, width):
   left_x, right_x = [], []

   for line in lines:
       x1, y1, x2, y2 = line[0]
       slope = (y2 - y1) / (x2 - x1)

       # Filter out lines with extreme slopes (almost vertical)
       if abs(slope) > 0.5:
           continue

       center_x = (x1 + x2) // 2

       if slope < 0:
           left_x.append(center_x)
       else:
           right_x.append(center_x)

   # Average the lane centers
   left_lane_center = int(np.mean(left_x)) if left_x else width // 4
   right_lane_center = int(np.mean(right_x)) if right_x else 3 * width // 4

   return left_lane_center, right_lane_center

# Open video capture
cap = cv2.VideoCapture(0)

while True:
   ret, frame = cap.read()

   # Process the frame for lane detection
   processed_frame = process_image(frame)

   # Apply Hough transform to detect lines
   lines = cv2.HoughLinesP(processed_frame, rho=1, theta=np.pi/180, threshold=50,
                           minLineLength=100, maxLineGap=50)

   if lines is not None:
       # Calculate lane centers
       left_center, right_center = calculate_lane_centers(lines, frame.shape[0], frame.shape[1])

       # Calculate lane deviation
       lane_deviation = frame.shape[1] // 2 - (left_center + right_center) // 2

       # Adjust steering based on lane deviation
       if lane_deviation > 50:
           print("Turn left")
       elif lane_deviation < -50:
           print("Turn right")
       else:
           print("Drive straight")

   # Display the frame with detected lines
   cv2.imshow('Lane Detection', frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()