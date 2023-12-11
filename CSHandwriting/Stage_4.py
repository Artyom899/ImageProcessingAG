
#Stage 4 
#binary regions in handwriting samples 

import cv2
import numpy as np

# Load the cropped handwriting image to be hand
hand = cv2.imread('OOP.MT.170317.H051_p1_bin.jpg', cv2.IMREAD_GRAYSCALE)

# Threshold the image to create a binary image
_, binary_image = cv2.threshold(hand, 125, 255, cv2.THRESH_BINARY_INV)

# Apply morphological operations to enhance regions
kernel = np.ones((3, 3), np.uint8)

morph_image_first = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=5)

morph_image = cv2.morphologyEx(morph_image_first, cv2.MORPH_OPEN, kernel, iterations=5)

# Separate characters using connected component analysis
_, labels, stats, centroids = cv2.connectedComponentsWithStats(morph_image, connectivity=8)



# Example: Separate characters based on horizontal distance
avg_hor_dist = np.mean(np.diff(np.sort(centroids[:, 0])))
character_mask = (centroids[:, 0] < avg_hor_dist)

chars = morph_image.copy()
chars[labels == 0] = 0
chars[labels != character_mask + 1] = 0

# Perform skeletonization to detect baselines
skel = cv2.ximgproc.thinning(morph_image)

# Displaying results
cv2.imshow('Binary Image is', binary_image)
cv2.imshow('Morphological Operations :', morph_image)
cv2.imshow('Separated Characters :', chars)
cv2.imshow('Skeletonized Image :', skel)
