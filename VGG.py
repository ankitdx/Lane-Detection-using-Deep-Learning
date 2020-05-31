from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input, concatenate, \
    Dropout, UpSampling2D

from tensorflow.keras.models import Model


def vgg16_unet(height, width, depth, weights=None):
    classes = 2
    input_ = Input((height, width, depth), name='input')
    dp = 0.0

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_1')(input_)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1_2')(conv1_1)
    mx_pool_1 = MaxPooling2D((2, 2))(conv1_2)
    mx_pool_1 = Dropout(dp)(mx_pool_1)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2_1')(mx_pool_1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2_2')(conv2_1)
    mx_pool_2 = MaxPooling2D((2, 2))(conv2_2)
    mx_pool_2 = Dropout(dp)(mx_pool_2)

    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3_1')(mx_pool_2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3_3')(conv3_2)
    mx_pool_3 = MaxPooling2D((2, 2))(conv3_3)
    mx_pool_3 = Dropout(dp)(mx_pool_3)

    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_4_1')(mx_pool_3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_4_3')(conv4_2)
    mx_pool_4 = MaxPooling2D((2, 2))(conv4_3)
    mx_pool_4 = Dropout(dp)(mx_pool_4)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5_1')(mx_pool_4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5_3')(conv5_2)
    mx_pool_5 = MaxPooling2D((2, 2))(conv5_3)
    mx_pool_5 = Dropout(dp)(mx_pool_5)

    # layer 72-74 in keras
    upsize = (2, 2)
    up = (UpSampling2D(size=upsize, name='upsample_1'))(mx_pool_5)
    up = concatenate([up, conv5_3], axis=-1)
    c = (Conv2D(256, (3, 3), activation='relu', padding='same'))(up)
    c = (Conv2D(256, (3, 3), activation='relu', padding='same'))(c)
    c = Dropout(dp)(c)

    up = (UpSampling2D(size=upsize, name='upsample_2'))(c)
    up = concatenate([up, conv4_3], axis=-1)
    c = (Conv2D(128, (3, 3), activation='relu', padding='same'))(up)
    c = (Conv2D(128, (3, 3), activation='relu', padding='same'))(c)
    c = Dropout(dp)(c)

    up = (UpSampling2D(size=upsize, name='upsample_3'))(c)
    up = concatenate([up, conv3_3], axis=-1)
    c = (Conv2D(64, (3, 3), activation='relu', padding='same'))(up)
    c = (Conv2D(64, (3, 3), activation='relu', padding='same'))(c)
    c = Dropout(dp)(c)

    up = (UpSampling2D(size=upsize, name='upsample_4'))(c)
    up = concatenate([up, conv2_2], axis=-1)
    c = (Conv2D(32, (3, 3), activation='relu', padding='same'))(up)
    c = (Conv2D(32, (3, 3), activation='relu', padding='same'))(c)
    c = Dropout(dp)(c)

    up = (UpSampling2D(size=upsize, name='upsample_5'))(c)
    up = concatenate([up, conv1_2], axis=-1)
    c = (Conv2D(32, (3, 3), activation='relu', padding='same'))(up)
    c = (Conv2D(32, (3, 3), activation='relu', padding='same'))(c)
    c = Dropout(dp)(c)

    c = (Conv2D(1, 1, 1, activation='sigmoid', padding='same'))(c)
    model = Model([input_], [c])

    if weights is not None:
        model.load_weights(weights)

    return model
