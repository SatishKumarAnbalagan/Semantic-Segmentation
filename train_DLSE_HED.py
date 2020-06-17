'''

Project - Sematic segmentation

DeepLabs v3 with Squeeze and Excitation and HED Preprocessing (DLSE-HED)

by -
Satish Kumar Anbalagan

'''

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from generator import BatchGenerator

from keras.layers import Conv2D, Dense, Flatten, Input, Activation, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate, Lambda, SeparableConv2D, GlobalAveragePooling2D, Multiply
from keras.layers.merge import add
from keras.regularizers import l2
from keras.optimizers import Adam

import pickle

ROW_AXIS = 1
COL_AXIS = 2
CHANNEL_AXIS = 3
num_classes = 29
batch_size = 16
#size = (128, 224)
#size = (256, 448)
size = (512, 912)

with open('one_hot_dict.pickle', 'rb') as handle:
    onehot = pickle.load(handle)

train_gen = BatchGenerator(onehot, 'train.txt', size, num_classes, batch_size)
val_gen = BatchGenerator(onehot, 'val.txt', size, num_classes, batch_size)


#connection of residual block in resnet
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
    x = GlobalAveragePooling2D()(in_block)
    x = Dense(ch // ratio, activation='relu')(x)
    x = Dense(ch, activation='sigmoid')(x)
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


model_inp = Input(shape=(512, 912, 3), name='original')
hed_inp = Input(shape=(512, 912, 1), name='hed')

inp = Lambda(cnct)([model_inp, hed_inp])
print(inp.shape)


# BLOCK_1

x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1a')(inp)
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn_conv1')(x)
x = Activation('relu')(x)
y_down1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same', name='max_pooling2d_1')(x)


# BLOCK_2

# branch1
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.00001, name='bn2a_branch2a')(y_down1)
x = Activation('relu')(x)
x_down2 = Conv2D(64, (1, 1), strides=(2, 2), padding='same', name='res2a_branch2a')(x)
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
network = Lambda(concat)([network, hed_inp])

network = Conv2D(8, (3, 3), strides=(1, 1), padding='same', name='before_output')(network)
network = Conv2D(num_classes, (3, 3), activation='softmax', strides=(1, 1), padding='same', name='Final_Output')(network)

Optimizer = Adam(lr=0.00001)

resnet_encoder = Model(inputs=[model_inp, hed_inp], outputs=network)
resnet_encoder.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_loss = ModelCheckpoint('model/se_hed_512x912/Segmentation_val_loss.{val_loss:.4f}-{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpoint_acc = ModelCheckpoint('model/se_hed_512x912/Segmentation_val_acc.{val_acc:.4f}-{epoch:03d}.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

#checkpoint_loss = ModelCheckpoint('model/se_hed_512x912/resnet50_val_loss.{val_loss:.4f}-{epoch:03d}.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#checkpoint_acc = ModelCheckpoint('model/se_hed_512x912/resnet50_val_acc.{val_acc:.4f}-{epoch:03d}.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

photo = TensorBoard(log_dir='logs')

resnet_encoder.load_weights('model/se_hed_512x912/Segmentation_val_acc.0.8870-001.h5', by_name=True)
#resnet_encoder.load_weights('model/se_hed_256x448/Segmentation_val_acc.0.8152-010.h5', by_name=True)
#resnet_encoder.load_weights('model/se_hed_128x224/Segmentation_val_acc.0.6995-010.h5', by_name=True)
#resnet_encoder.load_weights('model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

resnet_encoder.fit_generator(train_gen.get_batch(),
                             steps_per_epoch=train_gen.get_size() // batch_size,
                             epochs=100,
                             verbose=1,
                             callbacks=[checkpoint_loss, checkpoint_acc, photo],
                             validation_data=val_gen.get_batch(),
                             validation_steps=val_gen.get_size() // batch_size)
