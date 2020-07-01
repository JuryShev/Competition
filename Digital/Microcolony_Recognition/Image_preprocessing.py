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
        name_imj = base = os.path.basename(openimj)
        k = len(name_imj)
        k = k - 3
        name_n_imj = name_imj[:k] + png
        image = cv2.imread(openimj)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image_est = gray_image.astype(np.float32).reshape(-1, 512 * 640)
        gray_image_est=np.mean(gray_image_est)


        # if(gray_image.shape[0]<cowl):
        #     cowl=gray_image.shape[0]
        # if (gray_image.shape[1]<rowl):
        #     rowl = gray_image.shape[1]
        #gray_image = cv2.resize(gray_image, (rowl, cowl))
        if gray_image_est>158:
            #gray_image=dark_image(img=gray_image)
            gamma = 150 * 0.309 / gray_image_est

            gray_image=adjust_gamma(image=gray_image, gamma=gamma)

           # gray_image_est=gray_image.astype(np.float32).reshape(-1, 512 * 640)
            #gray_image_dark_est=gray_image_dark.astype(np.float32).reshape(-1, 512 * 640)

            #print("mean_L=",np.mean(gray_image_est))
            #print("mean_D=", np.mean(gray_image_dark_est))



            #viewTwoImage(gray_image,gray_image_dark)

        cv2.imwrite(os.path.join(path_image_g, name_n_imj), gray_image)

    print("minimal cowl=", cowl)
    print("minimal rowl=", rowl)
    print("color to gray saccesfull")

def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def viewTwoImage(img_O, img_A ):

    numpy_horizontal_concat = np.concatenate((img_O, img_A), axis=1)
    cv2.imshow('Numpy Vertical Concat', numpy_horizontal_concat)
    cv2.waitKey()

def dark_image(img):
    kernel_0=np.array([[-0.1, 0.1, -0.1, 0.1,0.5,0.1,-0.1,0.1,-0.1]])
    kernel = np.ones((3, 3), np.float32)

    kernel[0][0]=-0.1
    kernel[0][1]=0.1
    kernel[0][2]=-0.1

    kernel[1][0] = 0.1
    kernel[1][1] = 0.5
    kernel[1][2] = 0.1

    kernel[2][0] = -0.1
    kernel[2][1] = 0.1
    kernel[2][2] = -0.1




    dark_img=img
    #viewImage(img, "image_l")
    #cv2.filter2D(src=img,ddepth=dark_img,kernel=kernel,anchor=(-1, -1) )
    dark_img=cv2.filter2D(img, -1, kernel)
    viewTwoImage(img, dark_img)
    #viewImage(dark_img, "image_l2")

    return dark_img
# import the necessary packages


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def winVar(img, wlen):
  wmean, wsqrmean = (cv2.boxFilter(x, -1, (wlen, wlen),borderType=cv2.BORDER_REFLECT) for x in (img, img*img))
  return wsqrmean - wmean*wmean
