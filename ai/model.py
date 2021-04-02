from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import Model
from keras import layers
from PIL import Image
import numpy as np


def get_model_mobilenetv2_224(weights_path):
    model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    x = model.layers[-1].output
    # x = Flatten()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    output = layers.Dense(10, activation='softmax')(x)
    model = Model(inputs=model.inputs, outputs=output)
    model.load_weights(weights_path)
    return model


def load_image_to_array(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path, 'r')
        img = img.resize((224, 224))
        img = np.array(img)
        images.append(img)
    return np.array(images)


def prediction(model, image_paths):
    images = load_image_to_array(image_paths)
    results = model.predict(images)
    sorted_indices = np.argsort(-results)
    return sorted_indices, results
    # for r in results:


if __name__ == '__main__':
    model = get_model_mobilenetv2_224('./plants_mobilenetv2_224.h5')
    # model.summary()
    indices, result = prediction(model, ['/Volumes/MacOS Data/Project/Python/plants-identification/plants-identification-flask/app/api/image.jpg'])
    print(indices, result)


# model v1
# model = VGG16(include_top=False, input_shape=(256, 256, 3))
# for layer in model.layers:
#     layer.trainable=False
# flat1 = Flatten()(model.layers[-1].output)
# class1 = Dense(512, activation='relu')(flat1)
# class1 = Dense(256, activation='relu')(class1)
# output = Dense(10, activation='softmax')(class1)
# model = Model(inputs=model.inputs, outputs=output)


# model v2
# # https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# model = VGG16(include_top=False, input_shape=(256, 256, 3))
# for layer in model.layers:
#     layer.trainable=False

# x = model.layers[-1].output
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# output = Dense(10, activation='softmax')(x)
# model = Model(inputs=model.inputs, outputs=output)


# model mobilenetv2 224x224x3
# model = MobileNetV2(include_top=False, input_shape=(224, 224, 3))
# for layer in model.layers:
#     layer.trainable = False
# x = model.layers[-1].output
# # x = Flatten()(x)
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# x = layers.Dense(128, activation='relu')(x)
# output = layers.Dense(10, activation='softmax')(x)
# model = Model(inputs=model.inputs, outputs=output)
