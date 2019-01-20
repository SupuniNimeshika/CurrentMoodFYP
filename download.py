# import urllib.request
#
#
# urllib.request.urlretrieve('https://scontent.xx.fbcdn.net/v/t1.0-9/48383744_107190060343281_6545851727600418816_n.jpg?_nc_cat=106&_nc_ht=scontent.xx&oh=5983ab467204a0e8d1b3e8ab229372c6&oe=5CD4088F','image.jpg')
# import the necessary packages
import numpy as np
import urllib.request
import cv2


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # cv2.imshow("Image", image)
    # cv2.waitKey();
    # return the image
    return image