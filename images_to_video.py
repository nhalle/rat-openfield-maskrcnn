import cv2
import numpy as np
import os
#from os.path import isfile, join

def video_to_images(pathIn, pathOut, maxFrames=300, start=0):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    i = 0
    while success:
        if(i == maxFrames):
            success = False

        else:
            if(i < 10):
                save_image_path = pathOut+ '/0000' + str(i)+ '.jpg'

            elif(i<100):
                save_image_path = pathOut+ '/000' + str(i) + '.jpg'

            elif(i<1000):
                save_image_path = pathOut+ '/00' + str(i) + '.jpg'

            elif(i<10000):
                save_image_path = pathOut+ '/0' + str(i) + '.jpg'


            cv2.imwrite(save_image_path, image)     # save frame as JPEG file
            success,image = vidcap.read()
            print('Read a new frame: ', success)
            i += 1

    return i

# function converts images to video
def images_to_video(pathIn, pathOut, fps):

    frame_array = []

    files = os.listdir(pathIn)
    files.sort()
    for i in range(len(files)):
        filename=pathIn + files[i]

        if files[i] == ".DS_Store":
            print("no thanks")

        #reading each files
        else:
            img = cv2.imread(filename)
            print(filename)
            height, width, layers = img.shape
            size = (width,height)
            #inserting the frames into an image array
            frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
