from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
#from model import getModel
from keras.utils import np_utils
from keras import applications, optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD


NB_IV3_LAYERS_TO_FREEZE = 172


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  #for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
   #  layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
	layer.trainable = True
  model.compile(optimizer=SGD(lr=0.01, momentum=0.99), loss='categorical_crossentropy', metrics=['accuracy'])


vgg_weights = 'vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'

inputs = Input(shape=(150,150,3))
#base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(150,150,3))
print('Model loaded.')
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
#top_model.add(Flatten(input_shape=(150,150,3)))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(7, activation='sigmoid'))
#top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
#model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in base_model.layers:
    layer.trainable = False
for layer in base_model.layers:
    layer.trainable = True

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
lr = 1e-2
nb_epochs = 20
dr = lr / nb_epochs
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=dr),
              metrics=['accuracy'])
'''

model = getModel()
'''
batch_size = 16
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'dataset/val',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')

print(model.summary())

filepath="weights/inception/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=25,
	    callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
'''
setup_to_finetune(model)

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=25,
	    callbacks=callbacks_list,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
#model.save_weights('inception_25_0.0004.h5')
'''