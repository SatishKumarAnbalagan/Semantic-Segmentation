import keras
from keras.models import load_model
import numpy as np
import cv2
import os
import PIL
import torch
import numpy as np
from tqdm import tqdm
import pickle
from statistics import mean
import extcolors

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard


from keras.layers import Conv2D, Dense, Flatten, Input, Activation, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate, Lambda, SeparableConv2D, GlobalAveragePooling2D, Multiply
from keras.layers.merge import add
from keras.regularizers import l2
from keras.optimizers import Adam
import pickle
import tensorflow as tf

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3
num_classes = 29
batch_size = 4
size = (512, 912)

img_folder_path = ""
mask_folder_path = ""
model_file = "./model/DLSE_SF.h5"




def _shortcut(inp, residual):
    input_shape = K.int_shape(inp)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = inp
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(inp)

    z = add([shortcut, residual])
    return z




def se_block(in_block, ch, ratio=16):
    scale = 1.5 
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch // ratio, activation='relu')(x)
    x = Dense(ch, activation='softmax')(x)
    x = Lambda(lambda z: tf.where(z == K.max(x, axis=-1), z*scale, z))(x)
    return Multiply()([in_block, x])


def cnct(net):
    return K.concatenate(net, axis=-1)


def ASPP(network):
    image_features = network

    image_features = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001)(image_features)

    at_pool1x1 = Conv2D(256, (1, 1), padding='same')(network)

    at_pool3x3_1 = Conv2D(256, (3, 3), dilation_rate=6, padding='same')(network)

    at_pool3x3_2 = Conv2D(256, (3, 3), dilation_rate=12, padding='same')(network)

    at_pool3x3_3 = Conv2D(256, (3, 3), dilation_rate=18, padding='same')(network)
    net = [image_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3]

    network = Lambda(cnct)(net)
    network = SeparableConv2D(256, (1, 1), padding='same')(network)
    network = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001)(network)

    return network



model_inp = Input(shape=(128, 224, 3), name='original')

# BLOCK_1

x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1a')(model_inp)

x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn_conv1')(x)
x = Activation('relu')(x)
y_down1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='max_pooling2d_1')(x)


# BLOCK_2

# branch1
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2a_branch2a')(y_down1)
x = Activation('relu')(x)
x_down2 = Conv2D(64, (1, 1), strides=(2, 2), padding='same', name='res2a_branch2a')(x)
# print(x_down2.shape)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2a_branch2b')(x_down2)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='res2a_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2a_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res2a_branch2c')(x)
x = se_block(x, 256, ratio=16)
y = _shortcut(y_down1, x)

# branch2
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2b_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='res2b_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2b_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='res2b_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2b_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res2b_branch2c')(x)
x = se_block(x, 256, ratio=16)
y = _shortcut(y, x)

# branch3
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2c_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='res2c_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2c_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='res2c_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2c_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res2c_branch2c')(x)
x = se_block(x, 256, ratio=16)
y = _shortcut(y, x)

# BLOCK_3

# branch1
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3a_branch2a_new')(y)
x = Activation('relu')(x)
x_down3 = Conv2D(128, (1, 1), strides=(2, 2), padding='same', name='res3a_branch2a')(x)
# print(x_down3.shape)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3a_branch2b')(x_down3)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='res3a_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3a_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='res3a_branch2c')(x)
x = se_block(x, 512, ratio=16)
y = _shortcut(y, x)

# branch2
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3b_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='res3b_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3b_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='res3b_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3b_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='res3b_branch2c')(x)
x = se_block(x, 512, ratio=16)
y = _shortcut(y, x)
# branch3
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3c_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='res3c_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3c_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='res3c_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3c_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='res3c_branch2c')(x)
x = se_block(x, 512, ratio=16)
y = _shortcut(y, x)
# branch4
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3d_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='res3d_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3d_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='res3d_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn3d_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='res3d_branch2c')(x)
x = se_block(x, 512, ratio=16)
y = _shortcut(y, x)

# BLOCK_4

# branch1
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4a_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res4a_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4a_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='res4a_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4a_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='res4a_branch2c')(x)
x = se_block(x, 1024, ratio=16)
y = _shortcut(y, x)
# branch2
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4b_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res4b_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4b_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='res4b_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4b_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='res4b_branch2c')(x)
x = se_block(x, 1024, ratio=16)
y = _shortcut(y, x)

# branch3
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4c_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res4c_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4c_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='res4c_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4c_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='res4c_branch2c')(x)
x = se_block(x, 1024, ratio=16)
y = _shortcut(y, x)
# branch4
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4d_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res4d_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4d_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='res4d_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4d_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='res4d_branch2c')(x)
x = se_block(x, 1024, ratio=16)
y = _shortcut(y, x)
# branch5
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4e_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res4e_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4e_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='res4e_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4e_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='res4e_branch2c')(x)
x = se_block(x, 1024, ratio=16)
y = _shortcut(y, x)
# branch6
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4f_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='res4f_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4f_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='res4f_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn4f_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='res4f_branch2c')(x)
x = se_block(x, 1024, ratio=16)
y = _shortcut(y, x)

# BLOCK_5_ATROUS

# branch1
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5a_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='res5a_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5a_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, name='res5a_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5a_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(2048, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='res5a_branch2c')(x)
y = _shortcut(y, x)

# branch2
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5b_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='res5b_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5b_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, name='res5b_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5b_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(2048, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='res5b_branch2c')(x)
y = _shortcut(y, x)

# branch3
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5c_branch2a_new')(y)
x = Activation('relu')(x)
x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='res5c_branch2a')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5c_branch2b')(x)
x = Activation('relu')(x)
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', dilation_rate=4, name='res5c_branch2b')(x)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn5c_branch2c_new')(x)
x = Activation('relu')(x)
x = Conv2D(2048, (1, 1), strides=(1, 1), padding='same', dilation_rate=2, name='res5c_branch2c')(x)
y = _shortcut(y, x)


Resnet = y
network = ASPP(Resnet)


network = UpSampling2D(size=(2, 2), name='US1')(network)
network = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1), name='up1')(network)
Low_f8x = SeparableConv2D(48, (1, 1), padding='same')(x_down2)


def concat(listt):
    return K.concatenate(listt, axis=-1)


combine_1 = [Low_f8x, network]

network = Lambda(concat)(combine_1)

network = UpSampling2D(size=(2, 2), name='US2')(network)
network = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(1, 1), name='up2')(network)

Low_f4x = SeparableConv2D(48, (1, 1), padding='same')(y_down1)

combine_2 = [Low_f4x, network]

network = Lambda(concat)(combine_2)

network = UpSampling2D(size=(2, 2), name='US3')(network)
network = Conv2D(32, (3, 3), activation='relu', padding='same', strides=(1, 1), name='up3')(network)

network = UpSampling2D(size=(2, 2), name='US4')(network)
network = Conv2D(16, (3, 3), activation='relu', padding='same', strides=(1, 1), name='up4')(network)


network = Conv2D(8, (3, 3), strides=(1, 1), padding='same', name='before_output')(network)
network = Conv2D(num_classes, (3, 3), activation='softmax', strides=(1, 1), padding='same', name='Final_Output')(network)


Optimizer = Adam(lr=0.00001)

model = Model(inputs=model_inp, outputs=network)
model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights(model_file, by_name=True)


miou = []

with open('one_hot_dict.pickle', 'rb') as handle:
    onehot = pickle.load(handle)

for key in onehot:
    curr = onehot[key]
    value = np.argmax(np.array(curr))
    onehot[key] = value


for file in tqdm(os.listdir(img_folder_path)):
    if not file.lower().endswith(('.jpg', '.png')):
        continue

    if not os.path.exists('results'):
        os.makedirs('results')

    imgname = file.split('.')[0][:-12]
    img_path = os.path.join(img_folder_path, file)
    label_path = os.path.join(mask_folder_path, imgname + '_gtFine_color.png')
    img = cv2.resize(cv2.imread(img_path) / 255, (912, 512))
    label = cv2.resize(cv2.imread(label_path), (912, 512))

    model_inp = np.expand_dims(img, axis=0)

    out = model.predict(model_inp)
    out = np.argmax(out, axis=-1).transpose(1,2,0)
    new = np.zeros((out.shape[0], out.shape[1], 3))
    ious = []

    for color in onehot:
        mask1 = np.all(out == onehot[color], axis=-1)
        mask2 = np.all(label == color, axis=-1)

        intersection = np.logical_and(mask2, mask1)
        union = np.logical_or(mask2, mask1)
        if np.sum(union) <= 0 or np.sum(intersection) <= 0:
            continue
        iou = np.sum(intersection) / np.sum(union)
        ious.append(iou)
        new[mask1] = color

    curr_iou = mean(ious)
    miou.append(curr_iou)


    cv2.imwrite('results/{}'.format(file), new)

print(mean(miou))
