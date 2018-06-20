import skimage
from skimage import io, transform, exposure, filters
from skimage.color.adapt_rgb import adapt_rgb, hsv_value

IMAGE_OUTPUT_SIZE = (270, 480)

'''
We are using a scharr filter to detect edges. 
The @adapt_rgb is a skimage annotation which adapts any grayscale filters to a per 
channel basis
The 'image' will be separated by channel (using the hsv color space here)
and the filter will be applied to each of them. The final image results of a concatenation
of these channels.
'''
@adapt_rgb(hsv_value)
def _scharr_hsv(image):
    return filters.scharr(image)

# Defining what to do when an image is received
def process_image(image_stream):
    # Converting to a skimage/opencv image (simply a [x, y, 3] numpy array)
    image = io.imread(image_stream)

    # Firstly, downsampling the image
    image = transform.resize(image, IMAGE_OUTPUT_SIZE, mode='reflect', anti_aliasing=True)

    # Secondly, detecting the edges
    image = _scharr_hsv(image)

    # return the processed image
    return image
