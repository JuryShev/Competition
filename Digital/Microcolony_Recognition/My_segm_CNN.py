import os
import sys
import random
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Image_preprocessing

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
        """resize off_______________________________________________"""
        image = cv2.resize(image, ( self.height, self.width))
        #self.viewImage(image, 'res')


        mask=cv2.imread(_mask_path, -1)
        #self.viewImage(mask, 'orig_m')
        mask = cv2.resize(mask, (self.height, self.width))
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
            image.append(_img)
            mask.append(_mask)
            id_mask=id_mask+1
        image = np.array(image)
        mask = np.array(mask)

        return image, mask

    def on_epoch_end(self):
        pass

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

height=160
width=128
train_path = './Data_bacteria/train_g/*.png'
epochs = 40
batch_size = 5

## Training Ids
train_ids = glob.glob(train_path)

## Validation Data Size
val_data_size = 6
select_data=train_ids[164:177]
random.shuffle(select_data)
valid_ids = select_data[:val_data_size]
train_ids = select_data[val_data_size:]

gen = DataGen(train_ids, train_path, batch_size=batch_size, height=height, width=width)
#x, y = gen.__getitem__(0)
#print(x.shape, y.shape)

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
    f = [16, 32, 64, 128, 256]
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
################################################################

action=2
model = UNet()
if action==1:

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()

    train_gen = DataGen(train_ids, train_path,  height=height, width=width, batch_size=batch_size)
    valid_gen = DataGen(valid_ids, train_path,  height=height, width=width, batch_size=batch_size)

    train_steps = len(train_ids)//batch_size
    valid_steps = len(valid_ids)//batch_size

    model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps,
                        epochs=epochs)
    model.save_weights("ent_cloacae.h5")
elif action==2:
    a=5
    valid_gen = DataGen(valid_ids, train_path, height=height, width=width, batch_size=1)
    x, y = valid_gen.__getitem__(1)
    result = model.predict(x)
    print(np.mean(result))
    result = result > np.mean(result)+0.0042
    result_v0 = np.reshape(result * 255, (width,height))
    y_v0 = np.reshape(y[0] * 255, (width,height))
    result_v0=result_v0.astype('uint8')
    result_v1 = np.reshape(result * 255, (width,height))
    y_v1 = np.reshape(y * 255, (width,height))

    Image_preprocessing.viewImage(x[0], "y[0]")
    Image_preprocessing.viewImage(y_v0, "y_v0")
    Image_preprocessing.viewImage(result_v0, "result_v0")

    print("finish")
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
