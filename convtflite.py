# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 03:04:14 2020

@author: Vishwa
"""

import tensorflow as tf

model = tf.keras.models.load_model('./saved_model')
model.summary()
#model.save('saved_model', save_format='tf')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_model_name = "classifier.tflite"
open(tflite_model_name, "wb").write(tflite_model)
print("converted")