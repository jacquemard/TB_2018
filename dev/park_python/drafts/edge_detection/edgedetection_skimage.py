#import skimage as sk
from skimage import io, filters, color, exposure, util, morphology
from skimage.viewer.viewers import ImageViewer, CollectionViewer
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
import numpy as np
#from scipy import ndimage as ndi

print("Reading")
image = io.imread(r"park.jpg")
print("Read")

# done for each chanel (http://scikit-image.org/docs/0.12.x/auto_examples/color_exposure/plot_adapt_rgb.html)
@adapt_rgb(each_channel)
def sobel_each(image):
    return filters.sobel(image)

@adapt_rgb(hsv_value)
def sobel_hsv(image):
    return filters.sobel(image)

@adapt_rgb(each_channel)
def scharr_each(image):
    return filters.scharr(image)

@adapt_rgb(hsv_value)
def scharr_hsv(image):
    return filters.scharr(image)

# We use 1 - sobel_each(image)
# but this will not work if image is not normalized
image = exposure.rescale_intensity(1 - scharr_each(image))

# negating the image
image = util.invert(image)
#image = color.rgb2gray(image)

#print(image)

'''
image = ndi.binary_fill_holes(image)
'''
# markers

# exploring markers
'''
m = []
for i in np.arange(0.0, 1.0, 0.05):
    for j in np.arange(i + 0.05, 1.0, 0.05):
        markers = np.zeros_like(image)
        markers[image < i] = 1
        markers[image > j] = 2

        m.append(morphology.watershed(image, markers))

image = m
print(len(image))
'''
'''
markers = np.zeros_like(image)
markers[image < 0.2] = 1
markers[image > 0.8] = 2

image = morphology.watershed(image, markers)
'''
#image = filters.scharr(color.rgb2gray(image))

# Thresholding
'''
thresh =  filters.threshold_otsu(color.rgb2gray(image))
image = color.rgb2gray(image)
binary = image <  0.97
image *= binary
'''


# Displaying

print("Displaying")
if isinstance(image, list):
    CollectionViewer(image).show()
else:
    ImageViewer(image).show()