import os
import sys
import random
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Image_preprocessing
from PIL import Image
from PIL import ImageFilter
import tensorflow as tf
from tensorflow import keras
## Seeding
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, height=640, width=512):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.height = height
        self.width=width
        self.on_epoch_end()

    def viewImage(self, image, name_of_window):
        cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
        cv2.imshow(name_of_window, image)
        cv2.waitKey()

    def __load__(self, id_name, id_mask ):
        ## Path
        # image_path = './Data_bacteria/train_g/*.png'
        mask_path = './Data_bacteria/train_neg/'
        # filename_image_list = glob.glob(image_path)
        # filename_mask_list = glob.glob(mask_path)
        name_mask = os.path.basename(id_name)

        ## Reading Image
        _mask_path=mask_path+name_mask
        image = cv2.imread(id_name, 1)
        #self.viewImage(image,'orig')
        """resize off_______________________________________________
        image = cv2.resize(image, ( self.height, self.width))
        ____________________________________________________________"""
        #self.viewImage(image, 'res')


        mask=cv2.imread(_mask_path, -1)
        #self.viewImage(mask, 'orig_m')
        """resize off_______________________________________________
        mask = cv2.resize(mask, (self.height, self.width))
        ____________________________________________________________"""
        #self.viewImage(mask, 'res_m')
        mask=np.expand_dims(mask, axis=-1)

        """Reading Masks OFF____________________________________________________
        for name in all_masks:
            _mask_path = mask_path + name
            _mask_image = cv2.imread(_mask_path, -1)
            _mask_image = cv2.resize(_mask_image, (self.image_size, self.image_size))  # 128x128
            _mask_image = np.expand_dims(_mask_image, axis=-1)
            mask = np.maximum(mask, _mask_image)
            mask_view=mask[:,:,0]
        _____________________________________________________________"""

        ## Normalizaing
        image = image / 255.0
        mask = mask / 255.0

        return image, mask

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size

        files_batch = self.ids[index * self.batch_size: (index + 1) * self.batch_size]

        image = []
        mask = []
        id_mask=0
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name, id_mask)
            _img_cut=Image_preprocessing.CutStickImage(_img)
            _mask_cut = Image_preprocessing.CutStickImage(_mask)
            for i in range(4):
                image.append(_img_cut[i])
                mask.append(_mask_cut[i])
                id_mask=id_mask+1

        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

height=320
width=256
train_path = './Data_bacteria/train_g/*.png'
epochs = 30
batch_size = 5

## Training Ids
train_ids = glob.glob(train_path)

## Validation Data Size
val_data_size = 20
select_data=train_ids[0:69]
random.shuffle(select_data)
valid_ids = select_data[:val_data_size]
train_ids = select_data[val_data_size:]

gen = DataGen(train_ids, train_path, batch_size=batch_size, height=height, width=width)
# x, y = gen.__getitem__(0)
# print(x.shape, y.shape)

def viewTwoImage(img_O, img_A, name="img" ):

    numpy_horizontal_concat = np.concatenate((img_O, img_A), axis=1)
    cv2.imshow(name, numpy_horizontal_concat)
    cv2.waitKey()

def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

#########################################################

def UNet():
    f = [8, 16, 32, 64, 256]
    inputs = keras.layers.Input(( width, height, 3))

    p0 = inputs
    c1, p1 = down_block(p0, f[0])  # 128 -> 64
    c2, p2 = down_block(p1, f[1])  # 64 -> 32
    c3, p3 = down_block(p2, f[2])  # 32 -> 16
    c4, p4 = down_block(p3, f[3])  # 16->8

    bn = bottleneck(p4, f[4])

    u1 = up_block(bn, c4, f[3])  # 8 -> 16
    u2 = up_block(u1, c3, f[2])  # 16 -> 32
    u3 = up_block(u2, c2, f[1])  # 32 -> 64
    u4 = up_block(u3, c1, f[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

def predict_processing(img_predict, img_ans_neg, img_ans_orig, name=0 ):

    ###################################################
    y = Image_preprocessing.CutStickImage(img_ans_neg, 0)
    x = Image_preprocessing.CutStickImage(img_ans_orig, 0)
    result = Image_preprocessing.CutStickImage(img_predict, 0)
    ####################################################


    result_float = result * 255
    result_v1 = result_float.astype('uint8')

    ########################################################################################
    #cv2.imwrite(os.path.join('./Data_bacteria/CNN_pred', 'result_float.png'), result_float)
    ########################################################################################

    result_v2 = cv2.morphologyEx(result_v1, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))
    #pil_im = Image.fromarray(result_v2)
    pil_im = Image.fromarray(result_v1[:, :, 0])
    sharp_pil = pil_im.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=15))
    sharp_img = np.array(sharp_pil)
    sharp_img_mean = sharp_img[sharp_img.shape[0] - 20:sharp_img.shape[0] - 5, 5:20]
    print(np.mean(sharp_img_mean))
    result_bool = (sharp_img > np.mean(sharp_img_mean) + 120) | (sharp_img < np.mean(sharp_img_mean) - 5)
    result_255 = result_bool * 255
    result_255 = result_255.astype('uint8')


    # image = cv2.imread('./Data_bacteria/CNN_pred/result_v1.png')
    # Image_preprocessing.viewImage(result_v1, "result_v1")
    # Image_preprocessing.viewImage(result_v2, "result_v2")
    # Image_preprocessing.viewImage(sharp_img, "sharp_img")


    result_open = cv2.morphologyEx(result_255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    image_morpho = cv2.morphologyEx(result_open, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    result_v2 = cv2.morphologyEx(image_morpho, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    y_v1 = y * 255

    Image_preprocessing.viewImage(x, "x")
    Image_preprocessing.viewImage(y_v1, "y_v0")
    Image_preprocessing.viewImage(result_255, "result_255")
    Image_preprocessing.viewImage(image_morpho, "morpho")

    Image_preprocessing.viewImage(result_v2, "result_v2")
################################################################

action=0
model = UNet()
if action==-1:

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    train_gen = DataGen(train_ids, train_path,  height=height, width=width, batch_size=batch_size)
    valid_gen = DataGen(valid_ids, train_path,  height=height, width=width, batch_size=batch_size)

    train_steps = len(train_ids)//batch_size
    valid_steps = len(valid_ids)//batch_size

    model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                        epochs=epochs)
    model.save_weights("c_kefir.h5")


elif action==0:
    model.load_weights('c_kefir.h5')
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    result = model.predict(x)

    predict_processing(result, y, x)
elif action==1:
    model.load_weights('ent_cloacae.h5')
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    result = model.predict(x)

    predict_processing(result, y, x)

elif action==2:
    model.load_weights('klebsiella_pneumoniae.h5')
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    result = model.predict(x)

    predict_processing(result, y, x)

elif action==3:
    model.load_weights('moraxella_catarrhalis.h5')
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    result = model.predict(x)

    predict_processing(result, y, x)
elif action==4:
    model.load_weights('all.h5')
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    result = model.predict(x)

    predict_processing(result, y, x)
elif action==5:
    model.load_weights('all.h5')
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    result = model.predict(x)

    predict_processing(result, y, x)







else:
    ## Dataset for prediction
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    x_v=x[0]

    #y_v0 = np.reshape(y[0] * 255, (height, width))




    result = model.predict(x)
    result_v = result[0]
    print(np.mean(result))
    result = result > 0.5
    result_255=np.reshape(result * 255, (height, width))
    result_255=result_255.astype('uint8')
    Image_preprocessing.viewImage(x_v, "X_V")
    Image_preprocessing.viewImage(y[0], "y")
    Image_preprocessing.viewImage(result_255, "result_255")
    # print(np.mean(result[0]))
    # result = result[0] > np.mean(result[0])+0.001
    # result_v0 = result
    # result_v0 = np.reshape(result_v0 * 255, (height, width))
    # #result_v0 = cv2.cvtColor(result_v0, cv2.COLOR_BGR2GRAY)
    # #y_v0 = cv2.resize(y_v0[0], (640, 580))
    # result_v0=np.float32(result_v0)
    # #result_v0=cv2.resize(result_v0[0], (640, 580))



    #viewTwoImage(y_v0,result_v0)

    print("finish")
