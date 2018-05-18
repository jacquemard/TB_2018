import cv2 as cv
import numpy as np

# loading image
image = cv.imread(r"C:\Users\Remi\Desktop\2013-03-13_08_50_03.jpg")

# gray scale
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


'''
# getting the channels and gray image
imgs = cv.split(image)
#imgs.append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
#imgs.append(image)

# computing sobel edge detection for each chanel
sobels = []
for img in imgs:
    sobels.append(cv.Sobel(img, cv.CV_8U, 1, 1, ksize=3))


for i, sobel in enumerate(imgs):
    cv.imshow('image' + str(i), sobel)
 
'''


cv.waitKey(0)
cv.destroyAllWindows()

