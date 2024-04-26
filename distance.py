import cv2 as cv
import numpy as np

# Function to filter contours based on size and aspect ratio
def filter_contours(contours, min_area=100, min_aspect_ratio=0.2):
   filtered_contours = []
   for contour in contours:
       area = cv.contourArea(contour)
       if area > min_area:
           x, y, w, h = cv.boundingRect(contour)
           aspect_ratio = float(w) / h if h != 0 else 0
           if aspect_ratio > min_aspect_ratio:
               filtered_contours.append(contour)
   return filtered_contours

# Function to calculate the centroid of a contour
def calculate_centroid(contour):
   M = cv.moments(contour)
   if M["m00"] != 0:
       cx = int(M["m10"] / M["m00"])
       cy = int(M["m01"] / M["m00"])
       return cx, cy
   return None

cap = cv.VideoCapture(0)

while True:
   ret, frame = cap.read()
   frame_copy = frame.copy()

   hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
   hsv = cv.blur(hsv, (5, 5))

   # Fine-tune these values for your specific case
   lower_bound = np.array([45, 116, 180])
   upper_bound = np.array([255, 255, 255])
   mask = cv.inRange(hsv, lower_bound, upper_bound)

   # Apply morphological transformations
   mask = cv.erode(mask, None)
   mask = cv.dilate(mask, None)

   # Find contours and filter based on size and aspect ratio
   contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
   filtered_contours = filter_contours(contours)

   # Inside the loop where you process contours
   for contour in filtered_contours:
       cv.drawContours(frame, [contour], -1, (255, 0, 255), 3)

       # Calculate centroid and distance estimation
       centroid = calculate_centroid(contour)
       if centroid is not None:
           cx, cy = centroid

           # Calculate the area of the bounding box
           _, _, w, h = cv.boundingRect(contour)
           bounding_box_area = w * h

           # Assume a constant factor for distance estimation (adjust as needed)
           distance = 1.0 / np.sqrt(bounding_box_area)

           # Adjust the threshold distance value as needed
           if distance < 100:  # Placeholder value, adjust as needed
               cv.putText(frame, f"Distance: {distance}", (cx, cy),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               # Send signal to Raspberry Pi for the detected sign being near
               # Replace the comment above with your code to send signals to Raspberry Pi

   cv.imshow("Frame", frame)

   if cv.waitKey(1) == ord("q"):
       break

cap.release()
cv.destroyAllWindows()
