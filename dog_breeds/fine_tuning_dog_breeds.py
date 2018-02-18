import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import itertools
import numpy as np
#%matplotlib inline

#from resnext import ResNextImageNet
from keras.applications.xception import Xception
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout,BatchNormalization
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,RMSprop,Adam,Adamax,Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

IM_WIDTH, IM_HEIGHT = 300, 300 #fixed size for InceptionV3
NB_EPOCHS = 20
BAT_SIZE = 10
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172 # train only block 14 Xception


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    (This function is copied from the scikit docs.)
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    nadam=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    x = base_model.output
    print("Shape of output from Base Model",x._keras_shape)
    x = GlobalAveragePooling2D()(x)
    #x = Flatten(input_shape=base_model.output_shape[1:])(x)
    print("Shape after flattening",x._keras_shape)
    #x = Dropout(0.5)(x)
    #x = BatchNormalization()(x)
    #x = Dense(FC_SIZE)(x) #new FC layer, random init
    #x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    #x = Activation('relu')(x)
    #x = Dense(FC_SIZE//2)(x)
    #x = BatchNormalization()(x)
    #x = Activation('relu')
    #x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
    model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    print("Number of classes found:", nb_classes)
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # data prep
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # rotation_range=30,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # zoom_range=0.2,
        # channel_shift_range=10,
        # horizontal_flip=True

    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
        # shuffle=False
    )

    # setup model

    # base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(IM_HEIGHT,IM_WIDTH,3)) #include_top=False excludes final FC layer
    base_model=Xception(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IM_HEIGHT,IM_WIDTH,3), pooling=None, classes=1000)
    # base_model=InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=(IM_HEIGHT,IM_WIDTH,3), pooling=None, classes=1000)
    # base_model = ResNextImageNet(input_shape=(IM_HEIGHT, IM_WIDTH, 3), weights='imagenet')
    model = add_new_last_layer(base_model, nb_classes)

    # model = keras.models.load_model('saved_models/inceptionv2-ftv2.model')

    # transfer learning
    setup_to_transfer_learn(model, base_model)
    # setup_to_transfer_learn(model, model)
    # checkpoint
    filepath = "weights/resnext/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')
    reduceLR = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001,
                                 cooldown=0, min_lr=0)
    callbacks_list = [checkpoint, reduceLR]

    history_tl = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto',
        callbacks=callbacks_list)

    model.save("inception_transferv1.model")

    # model = keras.models.load_model('saved_models/inceptionv3_transferv5.model')
    # fine-tuning
    setup_to_finetune(model)
    # checkpoint
    filepath = "weights/xception/weights-ft126-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
    #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001,
                                 cooldown=0, min_lr=0)
    callbacks_list = [checkpoint, reduceLR]

    history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=nb_val_samples // batch_size,
        class_weight='auto',
        callbacks=callbacks_list
    )

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_ft)


class Args:
    def __init__(self):
        self.train_dir = "/home/kl_team/Documents/Aby/dog_breed/dataset/train"
        self.val_dir = "/home/kl_team/Documents/Aby/dog_breed/dataset/val"
        self.test_dir = "/home/kl_team/Documents/Aby/dog_breed/dataset/test/"
        self.nb_epoch = NB_EPOCHS
        self.batch_size = BAT_SIZE
        self.output_model_file = "resnext_ft_v1.model"
        self.plot=True


if __name__ == '__main__':
    args = Args()
    train(args=args)