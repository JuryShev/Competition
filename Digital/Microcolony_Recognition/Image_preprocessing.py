import cv2
import glob
import os

def color_to_gray(filename_list , path_s):
    png = 'png'
    cowl=1000
    rowl=1000
    for openimj in filename_list:
        # print(openimj)
        name_imj = base = os.path.basename(openimj)
        k = len(name_imj)
        k = k - 3
        name_n_imj = name_imj[:k] + png
        image = cv2.imread(openimj)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if(gray_image.shape[0]<cowl):
            cowl=gray_image.shape[0]
        if (gray_image.shape[1]<rowl):
            rowl = gray_image.shape[1]
        print ("minimal cowl=", cowl)
        print("minimal rowl=", rowl)

    gray_image = cv2.resize(gray_image, (rowl, cowl))
    cv2.imwrite(os.path.join(path_s, name_n_imj), gray_image)
    print("color to gray saccesfull")

