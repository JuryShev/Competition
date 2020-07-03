import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def color_to_gray(path_image_c , path_image_g):
    png = 'png'
    cowl=512
    rowl=640
    filename_list = glob.glob(path_image_c)
    for openimj in filename_list:
        # print(openimj)
        name_imj = os.path.basename(openimj)
        k = len(name_imj)
        k = k - 3
        name_n_imj = name_imj[:k] + png
        image = cv2.imread(openimj)
        gray_image_orig=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image=CutStickImage(gray_image_orig)

        flag_dark=0
        for i in range(4):
            print("***********************************")
            print("name=",name_n_imj, "square=", i)
            gray_image[i] = comp_mean_light(gray_image[i],
                                            w=gray_image[i].shape[1],
                                            h=gray_image[i].shape[0])




        gray_image_stick=CutStickImage(gray_image, 0)
        # if flag_dark==1:
        #     viewTwoImage(gray_image_orig, gray_image_stick, name=name_n_imj)


        # if(gray_image.shape[0]<cowl):
        #     cowl=gray_image.shape[0]
        # if (gray_image.shape[1]<rowl):
        #     rowl = gray_image.shape[1]
        #gray_image = cv2.resize(gray_image, (rowl, cowl))


        cv2.imwrite(os.path.join(path_image_g, name_n_imj),gray_image_stick)

    print("minimal cowl=", cowl)
    print("minimal rowl=", rowl)
    print("color to gray saccesfull")

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey()


def viewTwoImage(img_O, img_A, name="img" ):

    numpy_horizontal_concat = np.concatenate((img_O, img_A), axis=1)
    cv2.imshow(name, numpy_horizontal_concat)
    cv2.waitKey()

def dark_image(img):
    kernel_0=np.array([[-0.1, 0.1, -0.1, 0.1,0.5,0.1,-0.1,0.1,-0.1]])
    kernel = np.ones((3, 3), np.float32)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # kernel[0][0]=-0.1
    # kernel[0][1]=0.2
    # kernel[0][2]=-0.1
    #
    # kernel[1][0] = 0.1
    # kernel[1][1] = 1.2
    # kernel[1][2] = 0.1
    #
    # kernel[2][0] = -0.1
    # kernel[2][1] = 0.2
    # kernel[2][2] = -0.1




    dark_img=img
    #viewImage(img, "image_l")
    #cv2.filter2D(src=img,ddepth=dark_img,kernel=kernel,anchor=(-1, -1) )
    dark_img=cv2.filter2D(img, -1, kernel)
    viewTwoImage(img, dark_img)
    #viewImage(dark_img, "image_l2")

    return dark_img
# import the necessary packages


def adjust_gamma(image, gray_image_est):

        gamma = 150 * 0.709 / gray_image_est
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        image=cv2.LUT(image, table)
        image_d=comp_mean_light(img_O=image,
                        w=image.shape[1],
                        h=image.shape[0])
        # apply gamma correction using the lookup table
        return image_d




def CutStickImage(img, flag_choise=1):
    """

    :param img: список или изображение
    :param flag_choise: 1-режим 0 склеиваем
    :return:  1 список фрашментов 0 целое изображение


    """
    list_img=[]





    if flag_choise==1:
        width_0 = int(img.shape[0] / 2)
        height_0 = int(img.shape[1] / 2)
        list_img.append(img[:width_0,:height_0])
        list_img.append(img[:width_0, height_0:])
        list_img.append(img[width_0:, :height_0])
        list_img.append(img[width_0:, height_0:])

        # viewImage(list_img[0], "1")
        # viewImage(list_img[1], "2")
        # viewImage(list_img[2], "3")
        # viewImage(list_img[3], "4")

        return list_img
    else:
        horizontal_concat_H = np.concatenate((img[0], img[1]), axis=1)
        horizontal_concat_L = np.concatenate((img[2], img[3]), axis=1)
        img_stick=np.concatenate((horizontal_concat_H,horizontal_concat_L), axis=0)
       # viewImage(img_stick, "final_stick")
        return img_stick



def comp_mean_light(img_O, w, h, flag_choise=1):

    if flag_choise==1:
        gray_image_est = img_O.astype(np.float32).reshape(-1, h * w)
        gray_image_est = np.mean(gray_image_est)
        print("          ------                   ")
        print("gray_image_est=", gray_image_est)
        if gray_image_est > 160:

            img_A=adjust_gamma(img_O, gray_image_est)

            print("           *******                 ")
            print("gray_image_est=", gray_image_est)
            print("           *******                 ")
            return img_A
        else:

            print("          ------                   ")
            return img_O





    # else:
    #     gray_image_est = img_O.astype(np.float32).reshape(-1, h * w)
    #     gray_image_dark_est = img_A.astype(np.float32).reshape(-1, h * w)
    #
    #     print("mean_L=", np.mean(gray_image_est))
    #     print("mean_D=", np.mean(gray_image_dark_est))



