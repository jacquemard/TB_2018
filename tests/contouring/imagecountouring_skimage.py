#import skimage as sk
from skimage import io, filters, color, exposure
from skimage.viewer import ImageViewer
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value

print("Reading")
image = io.imread(r"C:\Users\Remi\Desktop\2013-03-13_08_50_03.jpg")
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

image = exposure.rescale_intensity(1 - scharr_each(image))
#image = filters.scharr(color.rgb2gray(image))

# Thresholding
'''
thresh =  filters.threshold_otsu(color.rgb2gray(image))
image = color.rgb2gray(image)
binary = image <  0.97
image *= binary
'''

# We use 1 - sobel_each(image)
# but this will not work if image is not normalized
ImageViewer(image).show()
#ImageViewer(filters.scharr(color.rgb2gray(image))).show()

#ImageViewer(image).show()