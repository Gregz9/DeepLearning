import tensorflow as tf
from tensorflow.keras import layers, models



def simple_model(input_shape):

    height, width, channels = input_shape
    image = layers.Input(input_shape)
    x = layers.Conv2D(32, 5, strides=(2, 2), padding='same', activation='relu')(image)
    x = layers.Conv2D(64, 5, strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(1, 1, padding='same', activation=None)(x)
    # resize back into same size as regularization mask
    x = tf.image.resize(x, [height, width])
    x = tf.keras.activations.sigmoid(x)

    model = models.Model(inputs=image, outputs=x)

    return model


def conv2d_3x3(filters):
    conv = layers.Conv2D(
        filters, kernel_size=(3, 3), activation='relu', padding='same'
    )
    return conv

def up_conv2d_2x2(filters): 
    up_conv = layers.Conv2DTranspose(
            filters, kernel_size=(2, 2), strides=(2,2), activation='relu', padding='same')
    return up_conv

def max_pool():
    return layers.MaxPooling2D((2, 2), strides=2, padding='same')


def unet(input_shape):

    image = layers.Input(shape=input_shape)

    c1 = conv2d_3x3(8)(image)
    c1 = conv2d_3x3(8)(c1)
    print(c1.shape)
    p1 = max_pool()(c1)
    
    c2 = conv2d_3x3(16)(p1)
    c2 = conv2d_3x3(16)(c2)
    print(c2.shape)
    p2 = max_pool()(c2)

    c3 = conv2d_3x3(32)(p2)
    c3 = conv2d_3x3(32)(c3)
    print(c3.shape)
    p3 = max_pool()(c3)

    c4 = conv2d_3x3(64)(p3)
    c4 = conv2d_3x3(64)(c4)
    print(c4.shape)
    p4 = max_pool()(c4)

    c5 = conv2d_3x3(128)(p4)
    c5 = conv2d_3x3(128)(c5)
    print(c5.shape)

    u6 = up_conv2d_2x2(64)(c5)
    print("u6 shape", u6.shape)
    u6 = layers.concatenate([u6, c4])
    c6 = conv2d_3x3(64)(u6)
    c6 = conv2d_3x3(64)(u6)
    print(c6.shape)

    u7 = up_conv2d_2x2(32)(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = conv2d_3x3(32)(u7)
    c7 = conv2d_3x3(32)(c7)
    print(c7.shape)

    u8 = up_conv2d_2x2(16)(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = conv2d_3x3(16)(u8)
    c8 = conv2d_3x3(16)(u8)
    print(c8.shape)

    u9 = up_conv2d_2x2(8)(c8)
    print(u9.shape)
    u9 = layers.concatenate([u9, c1])
    c9 = conv2d_3x3(8)(u9)
    c9 = conv2d_3x3(8)(c9)

    #raise NotImplementedError("You have some work to do here!")

    # Fill the layers from 2 to 9.
    # .........................
    probs = layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=image, outputs=probs)

    return model
