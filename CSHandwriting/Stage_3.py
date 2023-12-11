#Stage 3
#straight lines in handwriting samples

import cv2
import numpy as np

# Assuming 'bin_img' is the binary image of the handwriting
#example is the first image I have

bin_img = cv2.imread('OOP.MT.170317.H051_p1_bin.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Hough Transform to detect the lines
lines = cv2.HoughLines(bin_img, 1, np.pi / 180, threshold = 105)

for i in lines:
  theta = i[0]
  a = np.cos(theta)
  b = np.sin(theta)
  x_0 = a * theta
  y_0 = b * theta
  x_1 = int(x_0 + 1000 * (-b))
  y1 = int(y_0 + 1000 * (a))
  x2 = int(x_0 - 1000 * (-b))
  y2 = int(y_0 - 1000 * (a))
  cv2.line(bin_img.copy(), (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display the image with detected lines
cv2.imshow('Handwriting with Lines', bin_img.copy())
