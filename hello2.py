import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# W = tf.Variable(tf.ones(shape=(2, 2)), name="W")
W = tf.Variable([[1, 3, 2], [4, 0, 1]], name="W")
b = tf.Variable(tf.zeros(shape=3), name="b")


@tf.function
def mul(xx):
    return W * xx


out_a = mul([1, 0, 5])
print(out_a)

tf.print(out_a)

# x = tf.constant([35, 40, 45], name='x')
# y = tf.Variable(x + 5, name='y')
# print(y)

# print(tf.zeros([1, 1]))

x, y = 3, 5
print("x", x, "y", y)

p1 = tf.reshape(3, [1, 1])
print("p1", p1)
