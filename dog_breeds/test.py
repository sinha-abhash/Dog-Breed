from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from PIL import Image
import numpy as np

img_width, img_height = 300, 300

def testDog():
        model = load_model("weights/xception/weights-improvement-03-1.00.hdf5")
        path = "09836.png"
        img = load_img(path)
        img = img.resize((img_width, img_height), Image.ANTIALIAS)
        arr_img = image.img_to_array(img)
        im = np.expand_dims(arr_img, axis=0)
        im = preprocess_input(im)
        preds = model.predict(im)
        print(np.argmax(preds[0]))

testDog()
