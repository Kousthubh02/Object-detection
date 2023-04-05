import cv2

# Load the image
img = cv2.imread('cars.jpeg')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('carx.xml')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars in the image
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

# Draw rectangles around the detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image
cv2.imshow('Cars', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
