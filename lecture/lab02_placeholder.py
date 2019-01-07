import tensorflow as tf

x_data = [1, 2, 3, 4]
y_data = [2, 4, 6, 8]

W = tf.Variable(tf.random_uniform([1], -100., 100.))
b = tf.Variable(tf.random_uniform([1], -100., 100.))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis