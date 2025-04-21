import cv2
import numpy as np

# Load the image you want to zoom
image_path = 'your_image.jpg'  # Change this to your image file path
image = cv2.imread(image_path)

# Define the skin color range in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Start the camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to HSV (for skin color detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to detect skin color
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply the mask to the frame to extract the skin-colored regions
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)

    # Convert the skin region to grayscale for contour detection
    gray_skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_skin, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the detected skin regions (this is where the hand will be)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are detected, find the largest one (most likely the hand)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # Create a bounding box around the hand (largest contour)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box around the hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get the leftmost and rightmost points of the hand (thumb and index fingers)
        leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
        rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])

        # Calculate the distance between the leftmost and rightmost points (thumb and index)
        distance = np.linalg.norm(np.array(leftmost) - np.array(rightmost))

        # Zoom in or out based on the distance between the thumb and index finger tips
        zoom_factor = distance / 100  # Adjust the factor for desired sensitivity
        scale = 1 + zoom_factor

        # Resize the image based on the zoom scale
        zoomed_image = cv2.resize(image, None, fx=scale, fy=scale)

        # Crop the zoomed image to match the original frame size
        center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
        start_x = max(center_x - frame.shape[1] // 2, 0)
        start_y = max(center_y - frame.shape[0] // 2, 0)
        end_x = min(center_x + frame.shape[1] // 2, zoomed_image.shape[1])
        end_y = min(center_y + frame.shape[0] // 2, zoomed_image.shape[0])

        zoomed_cropped_image = zoomed_image[start_y:end_y, start_x:end_x]

        # Display the zoomed image
        cv2.imshow("Zoomed Image", zoomed_cropped_image)

    # Show the original camera feed with the skin color detection
    cv2.imshow('Hand Gesture Zoom - Camera Feed', frame)

    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
