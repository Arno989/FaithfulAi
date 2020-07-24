import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import urllib
import platform
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt

# define root
PROJECT_ROOT = os.path.dirname(os.path.abspath("./"))

# define necessary paths for different os'es
opSys = platform.system()
if opSys == "Windows":
    images_Processed_F = f"{PROJECT_ROOT}\\Data\\Processed-images\\FaithfulBlocks"
    images_Processed_V = f"{PROJECT_ROOT}\\Data\\Processed-images\\VanillaBlocks"
    checkpoint_path = f"{PROJECT_ROOT}\\ML\\Checkpoints\\training_test_2.2"
elif opSys == "Linux":
    images_Processed_F = f"{PROJECT_ROOT}/FaithfulAi/Data/Processed-images/FaithfulBlocks"
    images_Processed_V = f"{PROJECT_ROOT}/FaithfulAi/Data/Processed-images/VanillaBlocks"
    checkpoint_path = f"{PROJECT_ROOT}/ML/Checkpoints/training_test_2.2"

# checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_path, f"cp-{epoch:02d}.ckpt"), verbose=1, save_weights_only=True, period=5)

# color channels RGB(a)  (3 or 4)
res = (32, 32)
channels = 3
image_index = 500  # test image to view
l2_alpha = 10e-10

# conv == size of convolution window
def define_model(conv=2):
    input_img = tf.keras.layers.Input(shape=(32, 32, channels))

    l1 = tf.keras.layers.Conv2D(8, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(input_img)
    l2 = tf.keras.layers.Conv2D(8, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l1)
    l3 = tf.keras.layers.MaxPool2D(padding="same")(l2)

    l4 = tf.keras.layers.Conv2D(16, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l3)
    l5 = tf.keras.layers.Conv2D(16, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l4)
    l6 = tf.keras.layers.MaxPool2D(padding="same")(l5)

    l7 = tf.keras.layers.Conv2D(32, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l6)

    l8 = tf.keras.layers.UpSampling2D()(l7)
    l9 = tf.keras.layers.Conv2D(16, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l8)
    l10 = tf.keras.layers.Conv2D(16, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l9)

    l11 = tf.keras.layers.add([l10, l5])

    l12 = tf.keras.layers.UpSampling2D()(l11)
    l13 = tf.keras.layers.Conv2D(8, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l12)
    l14 = tf.keras.layers.Conv2D(8, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha))(l13)

    l15 = tf.keras.layers.add([l14, l2])

    decoded_image = tf.keras.layers.Conv2D(3, conv, padding="same", kernel_initializer="he_uniform", activation="relu", activity_regularizer=tf.keras.regularizers.l2(l2_alpha),)(l15)

    return tf.keras.models.Model(inputs=(input_img), outputs=decoded_image)


model = define_model()
model.save_weights(os.path.join(checkpoint_path, "cp-00.ckpt"))
model.compile(optimizer="adam", loss="mean_squared_error")


def get_training_data():
    feature_images = []
    target_images = []
    images_alphas = []

    for img in os.listdir(images_Processed_F):
        try:
            image = cv2.cvtColor(cv2.imread(f"{images_Processed_F}/{img}", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
            resized_image = cv2.resize(image, (256, 256))
            downscaled_image = cv2.resize(cv2.resize(resized_image, (100, 100)), (256, 256))

            if channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            resized_image = cv2.resize(image, (256, 256))

            feature_images.append(cv2.resize(image, (256, 256)))
            target_images.append(resized_image)
            images_alphas.append(resized_image[:, :, 3])

        except Exception as e:
            print(e)

    return (
        np.array(feature_images),
        np.array(target_images),
        np.array(images_alphas),
    )


downsized_images, real_images, image_alphas = get_training_data()


print(downsized_images.shape)
print(real_images.shape)
print(image_alphas.shape)
plt.imshow(image_alphas[image_index])


model.fit(downsized_images, real_images, epochs=1, batch_size=32, shuffle=True, validation_split=0.15, callbacks=[cp_callback])
# model.fit(downsized_images, real_images, epochs=5, batch_size=32, shuffle=True, validation_split=0.15, callbacks=[cp_callback])

model.save("my_model")


sr1 = np.clip(model.predict(downsized_images), 0.0, 1.0)
# sr1 = cv2.cvtColor(sr1, cv2.COLOR_BGR2BGRA)

sr1[:, :, 3] = image_alphas


plt.figure(figsize=(256, 256))

plt.subplot(10, 10, 1)
plt.imshow(downsized_images[image_index])

plt.subplot(10, 10, 2)
plt.imshow(real_images[image_index])

plt.subplot(10, 10, 3)
plt.imshow(sr1[image_index])

plt.savefig("./")

