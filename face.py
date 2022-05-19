import cv2 as cv
import os
import utils
import numpy as np
from net.mtcnn import mtcnn

class face():

    # initialize the models
    def __init__(self):

        # set up the mtcnn
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5,0.6,0.8]

    # detect and process human face from the input image
    def face_detect(self,path):
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # detect all the faces in the image
        faces = self.mtcnn_model.detectFace(img, self.threshold)
        #print(path.split("/")[-1])
        if len(faces) == 1:
            print(1)
            face = utils.face_process(faces[0],img)
            face = cv.cvtColor(face, cv.COLOR_RGB2BGR)
            cv.imwrite("./data/processed/Test/Mask/"+path.split("/")[-1],face)


if __name__ == "__main__":
    celery = face()
    after_generate1 = os.listdir("./data/raw/Test/Mask")
    for image in after_generate1:
        celery.face_detect("./data/raw/Test/Mask/"+image)