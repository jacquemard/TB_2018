import requests
from requests.auth import HTTPBasicAuth
from io import BytesIO

class CameraClient:
    SNAP_ENDPOINT = "/web/tmpfs/snap.jpg"
    WEB_PORT = 80

    def __init__(self, host, username="admin", password="admin"):
        """
        Creates a CameraClient which could be used to capture frame from the camera
        
        Arguments:
            host {str} -- The ip or hostname of the camera  
        
        Keyword Arguments:
            username {str} -- The username to use to connect (default: {"admin"})
            password {str} -- The password to use to connect (default: {"admin"})
        """
        self.host = host
        self.username = username

        self.password = password

        # Creating a basic authentification from the arguments
        self.auth = HTTPBasicAuth(self.username, self.password)

        # Defining the http request url which can be used to request an image to the camera
        self.url = "http://{}:{}{}".format(host, self.WEB_PORT, self.SNAP_ENDPOINT)

    def capture_raw(self): 
        """
        Could be used to get a single image from the camera. It is described by bytes.
        A skimage/opencv image (a simple [x, y, 3] numpy array) can be easily created with io.imread(camera.capture_raw()).
        
        Returns:
            BytesIO -- the byte stream describing the image

        Raises:
            RequestException -- if the connection was not succesfully completed, for ex. if the host can not be reached.
            
        """
        response = requests.get(self.url, auth=self.auth)
        if response.status_code == 401: # bad credentials
            raise self.BadCredentialsError()
        elif not str(response.status_code).startswith("2"): # bad response code
            raise self.BadResponseFormat()

        content = response.content
        response.close()
        return BytesIO(content)

    # ------- ERRORS ------- #
    class Error(Exception):
        pass

    class BadCredentialsError(Error):
        def __init__(self):
            super().__init__("The credentials cannot be used to connect to the camera")

    class BadResponseFormat(Error):
        def __init__(self):
            super().__init__("Something was wrong with the camera response")


if __name__ == "__main__":
    # here are some tests when not used as a module
    from skimage import io
    from skimage.viewer import ImageViewer

    camera = CameraClient("10.0.0.29")
    image = camera.capture_raw()
    
    # converting to a skimage/opencv image (simply a [x, y, 3] numpy array)
    image = io.imread(image)

    ImageViewer(image).show()