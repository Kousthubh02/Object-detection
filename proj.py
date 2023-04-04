import cv2
import numpy as np

# Load the input image
# image = cv2.imread('image1-1.png')
# image = cv2.imread('cars.jpg')
image = cv2.imread('pen.jpeg')

# Convert the input image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to reduce noise
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)

# Find contours of the edges
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and find the square shape
for contour in contours:
    # Calculate the perimeter of the contour
    perimeter = cv2.arcLength(contour, True)
    
    # Approximate the shape of the contour with a polygon
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    
    # Check if the polygon has four sides (i.e. a square)
    if len(approx) == 4:
        # Draw a rectangle around the square
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the input and output images
cv2.imshow('Input', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
