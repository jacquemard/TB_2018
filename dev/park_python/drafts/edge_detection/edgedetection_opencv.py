##################################################################
#        MESURE DU TAUX D'OCCUPATION DE PARKINGS A L'AIDE        #
#                       DE CAMERAS VIDEOS                        #
# -------------------------------------------------------------- #
#               Rémi Jacquemard - TB 2018 - HEIG-VD              #
#                   remi.jacquemard@heig-vd.ch                   #
#                           July 2018                            #
# -------------------------------------------------------------- #
# Draft: used to detect edges from an image with OpenCV          #
##################################################################

import cv2 as cv
import numpy as np

# loading image
image = cv.imread(r"park.jpg")

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

