#!/usr/bin/env python

import cupy
import chainer
chainer.cuda.get_device_from_id(0).use()
import chainer.functions as F
from chainer.cuda import to_cpu

from chainer_addons.links.pooling import PoolingType

import tensorflow as tf
slim = tf.contrib.slim

import numpy as np
import unittest

import matplotlib.pyplot as plt

class AveragePoolingTest(unittest.TestCase):

	def test_it(self):
		size, channels = 7, 3
		ksize, stride, pad = 3, 1, 1

		data = np.random.randn(4, channels, size, size).astype(np.float32)
		data = np.ones((4, channels, size, size), dtype=np.float32)

		tf_data = data.transpose(0, 2, 3, 1)
		tf_inputs = tf.placeholder(tf.float32, [None, size, size, channels])
		tf_avg_pool = slim.avg_pool2d(tf_inputs, [ksize, ksize],
			stride=stride, padding='SAME')

		config_sess = tf.ConfigProto(allow_soft_placement=True)
		config_sess.gpu_options.allow_growth = True
		with tf.Session(config=config_sess) as sess:
			tf_res = sess.run(tf_avg_pool, {tf_inputs: tf_data})

		tf_res = tf_res.transpose(0, 3, 1, 2)

		ch_data = cupy.array(data)
		ch_avg_pool = PoolingType.new(PoolingType.TF_AVG,
			ksize=ksize, stride=stride, pad=pad)
		ch_res = ch_avg_pool(ch_data)
		ch_res = to_cpu(ch_res.array)

		self.assertEqual(tf_res.shape, ch_res.shape)
		diff = np.abs(tf_res - ch_res)
		# print(diff.mean(axis=(-2, -1)))

		fig, axs = plt.subplots(1, 3)

		vmin = min(ch_res[0, 0].min(), tf_res[0, 0].min(), diff[0, 0].min())
		vmax = max(ch_res[0, 0].max(), tf_res[0, 0].max(), diff[0, 0].max())

		axs[0].imshow(ch_res[0, 0], vmin=vmin, vmax=vmax)
		axs[0].set_title("Chainer")
		axs[0].axis("off")

		axs[1].imshow(tf_res[0, 0], vmin=vmin, vmax=vmax)
		axs[1].set_title("Tensorflow")
		axs[1].axis("off")

		axs[2].imshow(diff[0, 0], vmin=vmin, vmax=vmax)
		axs[2].set_title("Diff")
		axs[2].axis("off")

		plt.show()
		plt.close()

		self.assertTrue(np.allclose(tf_res, ch_res))



if __name__ == '__main__':
	with chainer.using_config("train", False), chainer.no_backprop_mode(), tf.Graph().as_default():

		unittest.main()


