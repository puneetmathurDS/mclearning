# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:28:35 2017

@author: PUNEETMATHUR
"""

import tensorflow as tf




hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))