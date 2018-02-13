# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 22:15:04 2017

@author: FPTShop
"""
# update keras
# Convolutional Neural Network

# Part 1: Build CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

from keras.preprocessing import image
single_pred = image.load_image('dataset/single_prediction/cat_or_dog_1.jpq', grey_scale = False, target_size = (64, 64))
single_pred = image.img_to_array(single_pred)
result = classifier.predict(single_pred, batch_size = 1)
if result[0][0]==1:
    pred = 'cat'
else:
    pred = 'dog'

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import DropOut

def build_classifier(optimizer, drop):
    classifier = Sequential()
    classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 128, activation = 'relu'))
    classifier.add(DropOut(drop))
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {  'batch_size': [20, 40]
                'epochs': [20, 30]
                'optimizer': ['adam', 'rmsprop', 'adagrad']
                'drop': [0.1, 0.2, 0.3]}
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search.fit_generator((training_set,
                           steps_per_epoch=8000,
                           validation_data=test_set,
                           validation_steps=2000))
best_param = grid_search.best_params_
best_score = grid_search.best_score_