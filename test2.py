import tensorflow as tf

x = tf.constant([[[1, 2], [3, 4]], [[5, 6, [7, 8]], ])
print(x.shape)

r = tf.math.reduce_max(x, axis=1)
print(r)
