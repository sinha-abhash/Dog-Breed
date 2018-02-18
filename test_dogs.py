from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import threading
from keras.applications.inception_v3 import preprocess_input, InceptionV3
import argparse
from keras.models import Model
from keras.optimizers import SGD
from os import listdir, walk
from os.path import isfile, join
from prompt_toolkit import prompt
import time



dogs_maping = {0:"chihuahua",1:"german shepherd",2:"golden retriever",3: "labrador_retriever", 4: "muffin", 5: "pug", 6: "siberian husky"}
queue = []
results = {}
class CVModel(threading.Thread):
    def run(self):
        model = load_model("dog_breeds_6.hdf5")
        while True:
            if len(queue)>0:
                fid,image_path = queue.pop()
                img = load_img(image_path)
                img = img.resize((300, 300), Image.ANTIALIAS)
                arr_img = image.img_to_array(img)
                im = np.expand_dims(arr_img, axis=0)
                im = preprocess_input(im)

                preds = model.predict(im)
                print("preds: ", preds[0])
                print(np.argmax(preds, axis=1))
                if results.get(fid,None):
                    results[fid].append((dogs_maping[np.argmax(preds, axis=1)[0]],preds[0][np.argmax(preds, axis=1)[0]]*100))
                else:
                    results[fid] = [(dogs_maping[np.argmax(preds, axis=1)[0]],preds[0][np.argmax(preds, axis=1)[0]]*100)]
            else:
                time.sleep(2)

def main(fid,image_path):
    queue.append((fid,image_path))


def get_result():
    print(results)

if __name__ == '__main__':
    obj = CVModel()
    obj.start()
