# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 03:50:59 2020

@author: Vishwa
"""

import tensorflow as tf


model = tf.keras.models.load_model('model50501.h5')
model.summary()
model.save('saved_model', save_format='tf')