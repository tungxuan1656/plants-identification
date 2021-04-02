from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from PIL import Image
import numpy as np
import os


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
        img = preprocess_input(img)
        images.append(img)
    return np.array(images)


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES_PATH = os.path.join(CURRENT_PATH, '../../../ai/classes.txt')
MODEL_PATH = os.path.join(CURRENT_PATH, '../../../ai/plants_mobilenetv2_224.h5')
model = get_model_mobilenetv2_224(MODEL_PATH)
arr_class_names = None


def predict(image_paths):
    assert isinstance(image_paths, list), 'image_paths require a array of image'
    global model
    images = load_image_to_array(image_paths)
    results = model.predict(images)
    sorted_indices = np.argsort(-results).tolist()
    # {'SortedIndices': x, 'ClassesPrecision': y}
    list_results = results.tolist()
    outputs = [[x, y] for x, y in zip(sorted_indices, list_results)]
    return outputs
    # for r in results:


def get_class_names():
    global arr_class_names
    if arr_class_names is None:
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            arr_class_names = list(map(lambda x: x.replace('\n', ''), lines))
    return arr_class_names


if __name__ == '__main__':
    # model = get_model_mobilenetv2_224('./plants_mobilenetv2_224.h5')
    # # model.summary()
    result = predict(['/Volumes/MacOS Data/Project/Python/plants-identification/plants-identification-flask/app/api/image.jpg'])
    print(result)
    pass


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
