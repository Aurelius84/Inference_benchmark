
import tensorflow as tf
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from tensorflow.python.eager import tape


class FakeData(object):
    def __init__(self, length):
        super(FakeData, self).__init__()
        self.length = length
        self.X_train = np.random.random((224, 224, 3)).astype('float32')
        self.Y_train = np.array([np.random.randint(1000)]).astype('int32')

    def __iter__(self):
        for _ in range(self.length):
            yield self.X_train, self.Y_train

    def __len__(self):
        return self.length

    def output_shapes(self):
        return (self.X_train.shape, self.Y_train.shape)

    def output_types(self):
        return (tf.float32, tf.int32)

def get_data(df, batch_size):
    tdf = tf.data.Dataset.from_generator(
        generator=df.__iter__,
        output_types=df.output_types(),
        output_shapes=df.output_shapes())
    tdf = tdf.batch(batch_size)
    tdf = tdf.prefetch(tf.data.experimental.AUTOTUNE)
    return tdf


def train_keras_model_by_fit(defun=False):
    # warm up by first batch_size = 1
    for batch_size in [1, 1, 4, 16, 32]:
        df = FakeData(batch_size * 100)

        model = tf.keras.applications.resnet.ResNet50(
            input_shape=df.output_shapes()[0],
            include_top=True,
            weights=None)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        if defun:
            model.call = tf.function(model.call)

        start = time.time()
        model.fit(get_data(df, batch_size), epochs=1)
        # model.call(get_data(df, batch_size))
        end = time.time()
        print("batch_size: {}, cost: {} ms.".format(batch_size, (end - start)*10))


def compute_gradients(model, images, labels, num_replicas=1):
    with tf.GradientTape() as grad_tape:
        logits = model(images, training=True)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        if num_replicas != 1:
            loss /= num_replicas

    with tape.stop_recording():
        grads = grad_tape.gradient(loss, model.variables)
    return grads


def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.variables))


def random_batch(batch_size, data_format='channels_first'):
    shape = (3, 224, 224) if data_format == 'channels_first' else (224, 224, 3)
    shape = (batch_size,) + shape

    num_classes = 1000
    images = tf.random.uniform(shape)
    labels = tf.random.uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32)
    one_hot = tf.one_hot(labels, num_classes)

    return images, one_hot


def train_eager_with_tf_function(defun=True):
    from resnet50 import ResNet50

    model = ResNet50(data_format='channels_first', classes=1000)
    if defun:
        model.call = tf.function(model.call)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)

    for batch_size in [1, 1, 4, 16, 32]:

        images, labels = random_batch(batch_size)

        for i in range(105):
            if i==5:
                start = time.time()
            apply_gradients(model, optimizer, compute_gradients(model, images, labels))
        
        end = time.time()
        print("batch_size: {}, cost: {} ms.".format(batch_size, (end - start)*10))


if __name__ == '__main__':
    defun = True
    # train_keras_model_by_fit(defun)
    train_eager_with_tf_function(defun)


