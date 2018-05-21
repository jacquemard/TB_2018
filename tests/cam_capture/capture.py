import requests
from requests.auth import HTTPBasicAuth
from PIL import Image
from io import BytesIO

class CameraClient:
    SNAP_ENDPOINT = "/web/tmpfs/snap.jpg"
    WEB_PORT = 80

    def __init__(self, host, login="admin", password="admin"):
        self.host = host
        self.login = login
        self.password = password

        self.auth = HTTPBasicAuth(self.login, self.password)

        self.url = "http://{}:{}{}".format(host, self.WEB_PORT, self.SNAP_ENDPOINT)
        print(self.url)

    def capture_raw(self):
        response = requests.get(self.url, auth=self.auth)
        content = response.content
        response.close()
        return BytesIO(content)

if __name__ == "__main__":
    # if not used as a module: testing
    from skimage import io
    from skimage.viewer import ImageViewer

    camera = CameraClient("10.0.0.29")
    image = camera.capture_raw()
    
    # converting to a skimage/opencv image (simply a [x, y, 3] numpy array)
    image = io.imread(image)

    ImageViewer(image).show()