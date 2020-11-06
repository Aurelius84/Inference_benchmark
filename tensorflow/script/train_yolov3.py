import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import time
import tensorflow as tf
import numpy as np

from yolov3 import cfg, YOLOv3, decode, compute_loss


def fake_data(batch_size):
    input_size = 224
    train_output_sizes = input_size // np.array(cfg.strides)
    max_bbox_per_scale = 150
    image = np.random.uniform(size=(batch_size, input_size, input_size,
                                    3)).astype(np.float32)

    label_sbbox = np.random.uniform(
        size=(batch_size, train_output_sizes[0], train_output_sizes[0],
              cfg.anchor_per_scale, 5 + cfg.class_num)).astype(np.float32)
    label_mbbox = np.random.uniform(
        size=(batch_size, train_output_sizes[1], train_output_sizes[1],
              cfg.anchor_per_scale, 5 + cfg.class_num)).astype(np.float32)
    label_lbbox = np.random.uniform(
        size=(batch_size, train_output_sizes[2], train_output_sizes[2],
              cfg.anchor_per_scale, 5 + cfg.class_num)).astype(np.float32)

    sbboxes = np.random.uniform(size=(batch_size, max_bbox_per_scale,
                                      4)).astype(np.float32)
    mbboxes = np.random.uniform(size=(batch_size, max_bbox_per_scale,
                                      4)).astype(np.float32)
    lbboxes = np.random.uniform(size=(batch_size, max_bbox_per_scale,
                                      4)).astype(np.float32)

    return image, ((label_sbbox, sbboxes), (label_mbbox, mbboxes),
                   (label_lbbox, lbboxes))


def train():
    input_size = 224
    input_tensor = tf.keras.layers.Input([input_size, input_size, 3])
    conv_tensors = YOLOv3(input_tensor)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, i)
        output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    model = tf.keras.Model(input_tensor, output_tensors)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, *target[i], i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))

    for batch_size in [1, 1, 4, 16, 32, 64, 128]:
        image_data, target = fake_data(batch_size)

        start = time.time()
        for i in range(100):
            train_step(image_data, target)
        end = time.time()
        print("batch_size: {}, cost: {} ms.".format(batch_size, (end - start) *
                                                    10))


if __name__ == '__main__':
    train()
