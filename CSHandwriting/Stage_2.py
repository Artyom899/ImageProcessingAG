#Stage 2
#evaluate page features

import cv2
import numpy as np

#Image Preprocessing

# Assume we have the 'OOP.MT.170317.H051_p1_bin.jpg' image from the previous steps
texti = cv2.imread('OOP.MT.170317.H051_p1_bin.jpg', cv2.IMREAD_GRAYSCALE)

# Edge detection
edges = cv2.Canny(cv2.GaussianBlur(texti, (5, 5), 0), 50, 150)

# Dilate edges for better detection
dilatedEdges = cv2.dilate(edges, None, iterations=3)

# Find contours in the dilated image
contours, _ = cv2.findContours(dilatedEdges.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour (assumed to be the page boundary)
largest_contour = max(contours, key=cv2.contourArea)

# Find the orientation of the page
angle = cv2.minAreaRect(largest_contour)[-1]

# Rotate the image to correct the orientation
(h, w) = texti.shape[:2]
center = (w // 2, h // 2)
rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rot_image = cv2.warpAffine(texti,
                           rot_matrix, (w, h),
                           flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)

# Display the rotated image
cv2.imshow('Rotated Image', rot_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#2. Size and Orientation

# Get the size and orientation of the page
page_size = cv2.minAreaRect(largest_contour)[1]
page_orient = angle if page_size[0] < page_size[1] else angle + 90

print('Page Size is : ', page_size)
print('Page Orientation is ', page_orient, 'degree')

#3. Font Size or Average Letter Width

# Assuming our 'texti' is the binary image with removed text
char_widths = []
for contour in contours:
  x, y, w, h = cv2.boundingRect(contour)
  char_widths.append(w)

avg_character_width = np.mean(char_widths)
print('Average Character Width Is', avg_character_width, 'pixels')
