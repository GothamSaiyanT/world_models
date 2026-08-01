import cv2
import numpy as np

class Preprocessor:

    def __init__(self,image_size = 64):
        self.image_size = image_size

    def process(self,frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )
        resized = cv2.resize(
            gray,
            (self.image_size,self.image_size),
            interpolation = cv2.INTER_AREA
        )

        return(
            resized.astype(np.float32)/255.0
        )