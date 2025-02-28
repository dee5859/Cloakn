import cv2
import numpy as np

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize background subtractor (to capture the background and remove it later)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Convert the frame to HSV color space (for easier color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the range for green color detection (adjust as needed)
    lower_green = np.array([40, 50, 50])  # Lower bound of green in HSV
    upper_green = np.array([90, 255, 255])  # Upper bound of green in HSV
    
    # Create a mask for the green areas (representing the cloak)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Optional: Get the background using background subtraction (helps remove static background)
    background = fgbg.apply(frame)

    # Inverse of the mask to get everything except the green areas
    result = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    # Show the original frame, the mask, and the final result (invisible cloak effect)
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Invisible Cloak', result)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
