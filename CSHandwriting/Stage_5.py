
#Stage 5 
#labeled features of specific characters 

import cv2
import numpy as np


characters_image = cv2.imread('OOP.MT.170317.H051_p1_bin.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the characters image
_, characters_binary = cv2.threshold(characters_image, 125, 255, cv2.THRESH_BINARY)

# Find contours of characters
contours, _ = cv2.findContours(characters_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Function to classify and label characters
def labelling(contour):
    # Get bounding box coordinates
  x, y, z, h = cv2.boundingRect(contour)

  if h != 0:
    aspect_ratio = z / h 
  else:
    aspect_ratio = 0
   
# Calculate circularity
  if cv2.arcLength(contour, True) != 0: 
    circularity = (4 * np.pi * cv2.contourArea(contour)) / (cv2.arcLength(contour, True) ** 2)
  else:
    circularity = 0
    

# Label and draw bounding boxes for characters
for contour in contours:
    label = labelling(contour)

    # Draw bounding box and label on the original image
    x, y, z, h = cv2.boundingRect(contour)
    cv2.rectangle(characters_image, (x, y), (x + z, y + h), (0, 255, 0), 2)
    cv2.putText(characters_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the labeled characters image
cv2.imshow('Labeled Characters Here', characters_image)

