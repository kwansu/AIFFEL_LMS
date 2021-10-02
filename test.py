import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import tensorflow_datasets as tfds

mnist, info =  tfds.load(
    "mnist", split="train", with_info=True
)

BATCH_SIZE = 128

def gan_preprocessing(data):
    image = data["image"]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def cgan_preprocessing(data):
    image = data["image"]
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    
    label = tf.one_hot(data["label"], 10)
    return image, label

gan_datasets = mnist.map(gan_preprocessing).shuffle(1000).batch(BATCH_SIZE)
cgan_datasets = mnist.map(cgan_preprocessing).shuffle(100).batch(BATCH_SIZE)
print("✅")


class Maxout(layers.Layer):
    def __init__(self, units, pieces):
        super(Maxout, self).__init__()
        self.dense = layers.Dense(units*pieces, activation="relu")
        self.dropout = layers.Dropout(.5)    
        self.reshape = layers.Reshape((-1, pieces, units))
    
    def call(self, x):
        print(x.shape)
        x = self.dense(x)
        print(x.shape)
        x = self.dropout(x)
        x = self.reshape(x)
        print(x.shape)
        x = tf.math.reduce_max(x, axis=1)
        print(f'----- x.shape : {x.shape}')
        return x

class DiscriminatorCGAN(Model):
    def __init__(self):
        super(DiscriminatorCGAN, self).__init__()
        self.flatten = layers.Flatten()
        
        self.image_block = Maxout(240, 5)
        self.label_block = Maxout(50, 5)
        self.combine_block = Maxout(240, 4)
        
        self.dense = layers.Dense(1, activation=None)
    
    def call(self, image, label):
        print(tf.shape(image))
        image = self.flatten(image)
        print(tf.shape(image))
        image = self.image_block(image)
        print(tf.shape(image))
        label = self.label_block(label)
        print(tf.shape(label))
        x = layers.Concatenate()([image, label])
        print(tf.shape(x))
        x = self.combine_block(x)
        print(tf.shape(x))
        return self.dense(x)


test_model = DiscriminatorCGAN()

for images, labels in cgan_datasets : break

print(images.shape)
print(labels.shape)

temp = test_model(images, labels)
print(type(temp))
#print(temp)