import cv2
import numpy as np
import time


# a) Load the image as a grayscale image

IMAGE_FILE = "./data/raw/New_york_times_square-terabass_(cropped).jpg"
#IMAGE_FILE = "/Users/ravi/gcp_project/data/raw/By_The_River_Thames_at_Vauxhall,_London_-_geograph.org.uk_-_5726285.jpg"
image = cv2.imread(IMAGE_FILE, cv2.IMREAD_GRAYSCALE)
# b) Apply Gaussian Blur to the image
# Kernel size of 5x5 is used for the Gaussian Blur
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# c) Calculate the Sobel kernel for x and y directions
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel X
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel Y

# d) Combine the output to calculate the gradient of the image
gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Optional: Convert the gradient magnitude to an 8-bit image (for visualization)
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

# e) Display and/or save the gradient image
#cv2.imshow('Gradient Magnitude', gradient_magnitude)
cv2.imwrite('./src/visualization/gradient_image.jpg', gradient_magnitude)
time.sleep(1)  # 1-second delay
#cv2.waitKey(0)
#cv2.destroyAllWindows()

