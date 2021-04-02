import os
import time
from PIL import Image
from flask import request
from app.api import bp
from app.ai import plants_mobilenetv2
from app.main.utils import make_response, request_save_image


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(CURRENT_PATH, 'plants_mobilenetv2_224.h5')
IMAGE_UPLOAD_FOLDER = os.path.join(CURRENT_PATH, '../../logs/image_upload')
IMAGE_PATH = os.path.join(CURRENT_PATH, 'image.jpg')
FILENAME = ''


if not os.path.exists(IMAGE_UPLOAD_FOLDER):
    os.makedirs(IMAGE_UPLOAD_FOLDER)


@bp.route('/flower/identification', methods=['GET', 'POST'])
def flowers_identification():
    if request.method == 'GET':
        return make_response(False, description='The get method is not available')

    receive_image = request_save_image(request, IMAGE_PATH)
    if receive_image['Result'] != 'Success':
        return make_response(False, description=receive_image['Description'])
    else:
        FILENAME = receive_image['FileName']

    start = time.time()
    result = plants_mobilenetv2.predict([IMAGE_PATH])[0]
    class_id = result[0][0]
    accuracy = round(float(result[1][class_id]), 2)
    # Prediction image
    params_response = {'ClassID': class_id + 1, 'ClassName': plants_mobilenetv2.get_class_names()[class_id], 'Accuracy': accuracy}

    basename, ext = os.path.splitext(FILENAME)
    basename += time.strftime("_%Y%m%d_%H%M%S")
    basename += f'_classid_{class_id}'
    FILENAME = basename + ext

    im = Image.open(IMAGE_PATH)
    im_resize = im.resize((int(im.width * 256 / im.height), 256))
    im_resize.save(os.path.join(IMAGE_UPLOAD_FOLDER, FILENAME))
    print(f'Model classify in {time.time() - start} seconds. Result: {result}. File: {FILENAME}')

    return make_response(True, params_response, '')
