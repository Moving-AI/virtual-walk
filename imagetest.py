import cv2 as cv
import numpy as np
import os
import tensorflow as tf
import numpy as np
import json
from person import Person


def save_model_details(m):
    with open('model_details.json', 'w') as outfile:
        info = dict(list(enumerate(m.get_tensor_details())))
        s = json.dumps(str(info))
        outfile.write(s)


def draw(person, img):
    radius = 1
    color = (0, 0, 255)  # BGR
    thickness = 3
    for p in person.get_coords():
        cv.circle(img, p, radius, color, thickness)
    for p1, p2 in person.get_limbs():
        cv.line(img, p1, p2, color, thickness)
    return img


model = tf.lite.Interpreter(
    r'models\posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite')
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()
floating_model = input_details[0]['dtype'] == np.float32

for file in os.scandir('photos/football'):
    img = cv.imread(file.path)
    print(file.path, "original shape: ", img.shape)
    img2 = cv.resize(img, (257, 257), interpolation=cv.INTER_LINEAR)
    img = tf.reshape(tf.image.resize(img, [257, 257]), [1, 257, 257, 3])
    if floating_model:
        img = (np.float32(img) - 127.5) / 127.5
    # print('input shape: {}, img shape: {}'.format(input_details[0]['shape'], img.shape))
    model.set_tensor(input_details[0]['index'], img)
    model.invoke()

    output_data = np.squeeze(
        model.get_tensor(
            output_details[0]['index']))  # o()
    offset_data = np.squeeze(model.get_tensor(output_details[1]['index']))
    p = Person(output_data, offset_data)

    img = draw(p, img2)
    cv.imshow("{}".format(p.confidence()), img2)
    cv.waitKey(0)
    cv.destroyAllWindows()
# print(p.to_string())
# results = np.squeeze(output_data)
# offsets_results = np.squeeze(offset_data)
# print("output shape: {}".format(output_data.shape))
# np.savez('sample3.npz', results, offsets_results)
