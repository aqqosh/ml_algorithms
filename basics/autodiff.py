import numpy as np
import tensorflow as tf
from tensorflow.python.eager.backprop import GradientTape

# ex 1

x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x**2

dy_dx = tape.gradient(y, x)
dy_dx.numpy()


# ex 2

w = tf.Variable(tf.random.normal((3, 2)), name="w")
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name="b")
x = [[1., 2., 3.]]

with tf.GradientTape() as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)
    
[dl_w, dl_b] = tape.gradient(loss, [w, b])

# ex 3

layer = tf.keras.layers.Dense(2, activation="relu")
x = tf.constant([[1., 2., 3.]])

with tf.GradientTape() as tape:
    y = layer(x)
    loss = tf.reduce_mean(y**2)

grad = tape.gradient(loss, layer.trainable_variables)

# ex 4

x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x**2

dx = tape.gradient(y, x)