import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
import tensorflow as tf




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
    true_positives = K.sum(K.round(K.clip(tf.convert_to_tensor(y_true * y_pred), 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(tf.convert_to_tensor(y_pred), 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(tf.convert_to_tensor(y_true * y_pred), 0, 1)))
    possible_positives = K.sum(K.round(K.clip(tf.convert_to_tensor(y_true), 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip(tf.convert_to_tensor((1-y_true) * (1-y_pred)), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(tf.convert_to_tensor(1-y_true), 0, 1)))
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


def ResNet(name=None):
    # Specify input
    input_img = Input(shape=(256,256,1))
    # Layer 1
    x = ReflectionPadding2D((3, 3))(input_img)
    x = c7Ak(x, 32)
    # Layer 2
    x = dk(x, 64)
    # Layer 3
    x = dk(x, 128)

    # Layer 4-12: Residual layer
    for _ in range(4, 13):
        x = Rk(x)

    # Layer 13
    x = uk(x, 64)
    # Layer 14
    x = uk(x, 32)
    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(1, kernel_size=7, strides=1)(x)
    x = Activation('tanh')(x)  # They say they use Relu but really they do not
    return Model(inputs=input_img, outputs=x, name=name)


def c7Ak(x, k):
    x = Conv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
    x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def dk(x, k):
    x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
    x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def uk(x, k):
    # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
    x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractinoally stided with stride 1/2
    x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    return x


def Rk(x0):
    k = int(x0.shape[-1])
    # first layer
    x = ReflectionPadding2D((1, 1))(x0)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    x = Activation('relu')(x)
    # second layer
    x = ReflectionPadding2D((1, 1))(x)
    x = Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
    x = InstanceNormalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
    # merge
    x = add([x, x0])
    return x


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')