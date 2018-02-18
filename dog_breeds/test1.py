from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import numpy as np


def test_model1(model_path):
    img_width, img_height = 300, 300

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory('dataset/test', target_size=(img_width, img_height), batch_size=10, shuffle=False)
    model = load_model(model_path)

    scores = model.evaluate_generator(test_generator, 10)

    correct = 0

    print("Correct:", correct, " Total: ", len(test_generator.filenames))
    print("Loss: ", scores[0], "Accuracy: ", scores[1])

    predictor = model.predict_generator(test_generator, steps=10)
    print(predictor.shape)
    y_pred = np.argmax(predictor, axis=1)
    print(y_pred)
    class_dict = test_generator.class_indices

    #for i,n in enumerate(test_generator.filenames):
        #print(n, class_dict.keys()[class_dict.values().index(np.argmax(predictor[i]))])
    '''
    print(len(test_generator.filenames))
    
    for i in range(len(predictor)):
        print(predictor[i])'''


def test_model(model_path):
    img_width, img_height = 300, 300

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        'dataset/train',  # this is the target directory
        target_size=(img_width, img_height),  # all images will be resized to 150x150
        batch_size=10,
        class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory('dataset/test', target_size=(img_width, img_height),
                                                      batch_size=10, shuffle=False)
    model = load_model(model_path)
    lr = 1e-2
    nb_epochs = 20
    dr = lr / nb_epochs
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, decay=dr),
                  metrics=['accuracy'])
    '''
    filepath = "weights/xception/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    '''

    model.fit_generator(
        test_generator,
        steps_per_epoch=len(test_generator.filenames) // 10,
        epochs=1,
        #callbacks=callbacks_list,
        validation_data=test_generator,
        validation_steps=100 // 10)


test_model(model_path="weights/xception/kaggle/weights-improvement-02-0.98.hdf5")