from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from tqdm import tqdm

img_width, img_height = 300, 300

def testDog(filename):

    model = load_model("weights/inception/weights-improvement-04-0.19.hdf5")
    img = load_img(filename)
    img = img.resize((img_width, img_height), Image.ANTIALIAS)
    arr_img = image.img_to_array(img)
    im = np.expand_dims(arr_img, axis=0)
    im = preprocess_input(im)
    preds = model.predict(im)
    return preds[0]

def test_result(test_path):
    onlyfiles = [join(test_path,f) for f in listdir(test_path) if isfile(join(test_path, f))]
    result = np.array([])
    for file in tqdm(listdir(test_path)):
        if isfile(join(test_path, f)):
            file_path = join(test_path, f)
            prediction = testDog(file_path)
            result = np.append(result, prediction)

    print(result.shape)

test_result("test/")