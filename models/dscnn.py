import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dscnn(input_shape=(40, 49, 1), num_classes=3):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        16,
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        32,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        64,
        kernel_size=(1, 1),
        padding="same",
        use_bias=False,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="dscnn")


if __name__ == "__main__":
    model = build_dscnn()
    model.summary()
    print("TOTAL PARAMETERS:", model.count_params())
