import glob
import Image_preprocessing

if __name__ == '__main__':
    path_image_c='./Data_bacteria/train_c/*.png'
    path_image_g='./Data_bacteria/train_g'
    Image_preprocessing.color_to_gray(path_image_c,path_image_g)



    print("finish")
    print("finish")

