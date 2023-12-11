#Stage 1
#eliminate the printed text

#1. Color Distribution Analysis:
#Let us use Python with libraries like OpenCV, NumPy, and Matplotlib to analyze color distribution in each image.

#!pip install opencv-contrib-python-headless
#!pip install numpy==1.20.0

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image : example is the first image
image = cv2.imread('/Users/artyomghazaryan/Desktop/OOP.MT.170317.H051_p1.jpg')

# RGB Histogram
histogram_rgb = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256],
                             [0, 256, 0, 256, 0, 256])

# Grayscale Histogram
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
histogram_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Plot histograms
plt.subplot(121), plt.imshow(cv2.cvtColor(
    image, cv2.COLOR_BGR2RGB)), plt.title('Original Image Plotting')
plt.subplot(122), plt.plot(histogram_gray), plt.title(
    'Grayscale Histogram Plotting')
plt.show()

#2. Automated Removal of Printed Text:
#For text removal, you can use techniques like image thresholding, morphological operations, or content-aware filling.
# Example using thresholding

_, thresh = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
result = cv2.bitwise_and(image, image, mask=thresh)

cv2.imshow('Text Removed', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

#3. Automated Removal of Instructor’s Marks:
#Removing specific colors (red shades) can be done using color thresholding.

lower_red = np.array([0, 0, 100])
upper_red = np.array([100, 100, 255])
mask = cv2.inRange(image, lower_red, upper_red)
result_no_marks = cv2.bitwise_and(image, image, mask=~mask)

cv2.imshow('No Marks', result_no_marks)
cv2.waitKey(0)
cv2.destroyAllWindows()

#4. Handwriting Extraction and Cropping:
#Use OCR (Optical Character Recognition) libraries like Tesseract to extract text and then crop the region.

import pytesseract

# Extract text using Tesseract OCR
text = pytesseract.image_to_string(gray_image)

# Get the bounding box of the text
box = pytesseract.image_to_boxes(gray_image)

# Crop the region containing text
x, y, w, h = map(int, box.split()[1:5])
cropped_text = image[y:h, x:w]

cv2.imshow('Cropped Text', cropped_text)
cv2.waitKey(0)
cv2.destroyAllWindows()

#5. Convert to Binary Format, Adjust Brightness/Contrast:
#Use OpenCV for conversion and adjustment.

# Convert to binary format
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Adjust brightness and contrast (example using equalizeHist)
equ = cv2.equalizeHist(gray_image)

#Just for this one case we would write to save the image like this
cv2.imwrite('OOP.MT.170317.H051_p1_bin.png', binary_image)
cv2.imwrite('OOP.MT.170317.H051_p1_eq.png', equ)

#But as we have 4 folders with several images, we would save the images For H051 below
for i in range(1, 2):
  for j in range(1, 4):
    cv2.imwrite('OOP.MT.170317.H05' + str(i) + '_p' + str(j) + '_bin.png',
                binary_image)
    cv2.imwrite('OOP.MT.170317.H05' + str(i) + '_p' + str(j) + '_bin.png', equ)

# And for H052, H053, H054 below
for i in range(2, 5):
  for j in range(1, 5):
    cv2.imwrite('OOP.MT.170317.H05' + str(i) + '_p' + str(j) + '_bin.png',
                binary_image)
    cv2.imwrite('OOP.MT.170317.H05' + str(i) + '_p' + str(j) + '_bin.png', equ)
