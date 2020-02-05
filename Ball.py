'''
The code works in the following way:
    - Load image and create enough of a defined shape on the ball to allow the HoughCircles function to find it.
    - The kernel'd image is then converted to gray scale and testing for circles of a min. and max. size.
    - This information is then converted to a usable int which gives the x and y co-ords. as well as the radius size.
    - Using the x and y, a rectangle just larger than the radius is cut from an area near the original xy co-ords.
    - This is then cropped into a circle .png image using the mask_circle_transparent() function.
    - Finally, the cropped circle is placed on top of the original image using the x and y gathered earlier.
    - The code works well for "spottheball.jpg" and "golf.jpg", however due to the snooker balls shadow, it does not
    - work well for "snooker.jpg"
'''


import cv2
import numpy as np
from PIL import Image, ImageDraw


# Function that displays the image
def viewImage(image):
    cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
    cv2.imshow("Display", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function that blurs the image
def kernel(image):
    kernel = np.array([
        [(1 / 9), (1 / 9), (1 / 9)],
        [(1 / 9), (1 / 9), (1 / 9)],
        [(1 / 9), (1 / 9), (1 / 9)]
    ])
    output = cv2.filter2D(image, -1, kernel)
    return output


# The function below code came from [1]
# It creates a grayscale image with a circle the size of the radius
# An then adds a mask as alpha channel and save as a png which keeps the alpha channel
def mask_circle_transparent(pil_img, x, y, r):
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((x, y, x + r, y + r), fill = 255)
    result = pil_img.copy()
    result.putalpha(mask)
    return result


# Combines 2 images by pasting image2 onto image1
def combine(image1, image2, name):
    background = Image.open(image1)
    foreground = Image.open(image2)
    background.paste(foreground, (x - r, y - r), foreground)
    background.save(name)


# Load image, the only place that the name needs to be changed, however the commmented out function can also work
# * Note the image must be in the same file as this script *
original = input("Enter the image file name (eg. 'snooker.jpg'): ")
# original = "spottheball.jpg"
img = cv2.imread(original)


# Create enough of an outline on ball to get a defined circle on the ball
img = kernel(img)
img = kernel(img)


# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))
# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
    cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
    param2 = 30, minRadius = 1, maxRadius = 60)
detected_circles = np.uint16(np.around(detected_circles))  # Converting Hough Circles to usable ints
x, y, r = detected_circles[0][0]  # Get x, y co - ordinates and radius
# [2]


# This set of code crops the image into a square the size of the radius of the ball
# And then crops it into a circle a tiny bit larger than the ball and then pastes it onto the ball
crop = cv2.imread(original, cv2.IMREAD_UNCHANGED)
crop_img = crop[y + r: y + 3 * r + 6, x + r: x + 3 * r + 6]
cv2.imwrite("sq_crop.png", crop_img)
im = Image.open("sq_crop.png")  #
im_thumb = mask_circle_transparent(im, 0, 0, 2 * r + 3)
im_thumb.save("cir_crop.png")
combine(original, "cir_crop.png", "Final.jpg")

# Final image loaded and viewable
Final = cv2.imread("Final.jpg")
viewImage(Final)

## REFERENCES ##
#[1]S. Meschke, "What's the most simple way to crop a circle thumbnail from an image?", Stack Overflow, 2019.
# [Online]. Available: https://stackoverflow.com/questions/58543750/whats-the-most-simple-way-to-crop-a-circle-thumbnail-from-an-image. [Accessed: 14- Nov- 2019]
#[2]A. Uberoi, "Circle Detection using OpenCV | Python", GeeksForGeeks, 2019.
#[Online]. Available: https://www.geeksforgeeks.org/circle-detection-using-opencv-python/. [Accessed: 14- Nov- 2019]
