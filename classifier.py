# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:25:23 2020

@author: Vishwa
"""

import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras.models import Model

#Load data

main_dir = os.path.join(os.path.dirname('.\dataset'), 'dataset')
CATEGORIES = ['cardboard', 'paper', 'plastic', 'metal' , 'glass']

train_data = []
x_train = []
y_train = []

def load_train_data():
    for category in CATEGORIES:
        path = os.path.join(main_dir, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            img_array1 = cv2.resize(img_array, (50, 50))
            train_data.append([img_array1, class_num])
            
load_train_data()
random.shuffle(train_data)

for features, label in train_data:
    x_train.append(features)
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)

#Model

i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3,3), strides=2, activation='relu')(i)
x = Conv2D(64, (3,3), strides=2, activation='relu')(x)
x = Conv2D(128, (3,3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(4, activation='softmax')(x)

model = Model(i, x)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(x_train, y_train, validation_split=0.1, epochs=60)

# Plot accuracy

plt.plot(history.history['accuracy'], label='acc')
plt.ylim(bottom=0)
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()

model.save('my_model.h5')
