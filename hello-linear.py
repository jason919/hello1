import tensorflow as tf
import numpy as np

W = tf.Variable(tf.zeros([1, 1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")


@tf.function
def calHousePrice(houseSize):
    return W * houseSize + b


@tf.function
def calCost(housePrice, actualHousePrice_):
    return tf.reduce_sum(tf.pow(actualHousePrice_ - housePrice), 2)


for i in range(100):
    houseSizes = np.array([[i]])
    housePrices = np.array([[2 * i]])

train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
