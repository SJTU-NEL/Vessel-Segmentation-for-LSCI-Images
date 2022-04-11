import numpy as np
from keras.models import *
from keras.layers import *


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def get_unet(input_img, n_filters, dropout_rate, batchnorm):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout_rate)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout_rate)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout_rate)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout_rate)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)
    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout_rate)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout_rate)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout_rate)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout_rate)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.summary()
    return model


def get_unet_lstm(img_height, img_width, n_filters, dropout_rate, batchnorm=True, num_class = 2):
    # contracting path
    input_img = Input((img_height, img_width, 1), name='img')
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout_rate * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout_rate)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout_rate)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout_rate)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    # reshaping:
    x1 = Reshape(target_shape=(1, np.int32(img_height/8), np.int32(img_width/8), n_filters * 8))(c4)
    x2 = Reshape(target_shape=(1, np.int32(img_height/8), np.int32(img_width/8), n_filters * 8))(u6)
    # concatenation:
    u6 = concatenate([x2, x1], axis=1)
    # LSTM:
    u6 = ConvLSTM2D(n_filters * 4, (3, 3), padding='same', return_sequences=False, go_backwards=True)(u6)
    u6 = Dropout(dropout_rate)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    # reshaping:
    x1 = Reshape(target_shape=(1, np.int32(img_height/4), np.int32(img_width/4), n_filters * 4))(c3)
    x2 = Reshape(target_shape=(1, np.int32(img_height/4), np.int32(img_width/4), n_filters * 4))(u7)
    # concatenation:
    u7 = concatenate([x2, x1], axis=1)
    # LSTM:
    u7 = ConvLSTM2D(n_filters * 2, (3, 3), padding='same', return_sequences=False, go_backwards=True)(u7)
    u7 = Dropout(dropout_rate)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    # reshaping:
    x1 = Reshape(target_shape=(1, np.int32(img_height/2), np.int32(img_width/2), n_filters * 2))(c2)
    x2 = Reshape(target_shape=(1, np.int32(img_height/2), np.int32(img_width/2), n_filters * 2))(u8)
    # concatenation:
    u8 = concatenate([x2, x1], axis=1)
    # LSTM:
    u8 = ConvLSTM2D(n_filters * 1, (3, 3), padding='same', return_sequences=False, go_backwards=True)(u8)
    u8 = Dropout(dropout_rate)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    # reshaping:
    x1 = Reshape(target_shape=(1, np.int32(img_height/1), np.int32(img_width/1), n_filters * 1))(c1)
    x2 = Reshape(target_shape=(1, np.int32(img_height/1), np.int32(img_width/1), n_filters * 1))(u9)
    # concatenation:
    u9 = concatenate([x2, x1], axis=1)
    # LSTM:
    u9 = ConvLSTM2D(n_filters * 1, (3, 3), padding='same', return_sequences=False, go_backwards=True)(u9)
    u9 = Dropout(dropout_rate)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    if num_class == 2:
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    else:
        outputs = Conv2D(num_class, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    # model.summary()
    return model


# parameter for loss function
smooth = 1.


#  metric function and loss function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())


def weighted_loss(weight_map, weight_strength):
    def weighted_dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        weight_f = K.flatten(weight_map)
        y_pred_f = K.flatten(y_pred)
        weight_f = weight_f * weight_strength + 1
        wy_true_f = y_true_f * weight_f
        wy_pred_f = y_pred_f * weight_f
        return 1 - dice_coef(wy_true_f, wy_pred_f)
    return weighted_dice_loss


