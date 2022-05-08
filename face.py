import cv2 as cv
import os
import utils
import numpy as np
from net.mtcnn import mtcnn

class face():

    # initialize the models
    def __init__(self):
        # initialize the dataset 
        self.dataset = []

        # set up the mtcnn
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5,0.6,0.8]

    # detect and process human face from the input image
    def face_detect(self,image):
        # convert to RGB type
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # detect all the faces in the image
        faces = self.mtcnn_model.detectFace(img, self.threshold)

        # process the face
        if len(faces) == 1:
            img = utils.face_process(faces[0],img)
        else:
            exit("Please make sure there is (only) one person on the screen!\n")
    
        return img


if __name__ == "__main__":
    celery = face()

    a = cv.imread("mask.jpg")
    a = celery.face_detect(a)
    a = cv.cvtColor(a, cv.COLOR_BGR2HSV)
    cv.imwrite("mask_output.jpg",a[:,:,0])

    b = cv.imread("no_mask.jpg")
    b = celery.face_detect(b)
    b = cv.cvtColor(b, cv.COLOR_BGR2HSV)
    cv.imwrite("no_mask_output.jpg",b[:,:,0])