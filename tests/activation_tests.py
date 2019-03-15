import unittest

import numpy as np
import chainer
import chainer.functions as F
import tensorflow as tf

from feature_extract.core.models import InceptionV3
from feature_extract.core.models.tf_inception import InceptionV3 as InceptionV3_TF, inception_arg_scope
from .configs import configs

from chainer.cuda import to_cpu

from functools import reduce


class ActivationTests(unittest.TestCase):

	def run(self, *args, **kwargs):
		with chainer.using_config("train", False), \
			chainer.no_backprop_mode():
			return super(ActivationTests, self).run(*args, **kwargs)

	@property
	def xp(self):
		return self.chainer_model.xp

	def setUp(self):

		self.chainer_model = InceptionV3()
		self.chainer_model.load_for_finetune(
			weights=configs.CHAINER_WEIGHTS,
			n_classes=1000
		)

		if configs.DEVICE >= 0:
			chainer.cuda.get_device_from_id(configs.DEVICE).use()
			self.chainer_model.to_gpu(configs.DEVICE)

		arg_scope = inception_arg_scope(batch_norm_epsilon=2e-5)
		self.graph = tf.Graph()
		self.tf_model = InceptionV3_TF.new(
			configs.TF_WEIGHTS,
			train=False, arg_scope=arg_scope,
			graph=self.graph
		)

		self.rnd = np.random.RandomState(configs.SEED)

	def check_activation(self, check_title, size, chainer_layer, tf_layer):
		data = self.rnd.randn(6, 3, size, size).astype(np.float32)
		# data = np.ones((6, 3, size, size), dtype=np.float32)

		tf_data = data.transpose(0, 2, 3, 1)
		ch_data = self.xp.array(data)

		ch_res = chainer_layer(ch_data)
		ch_res = to_cpu(ch_res.array)

		tf_res = self.tf_model.sess.run(tf_layer,
			feed_dict={self.tf_model.inputs: tf_data}
		)

		if tf_res.ndim == 4:
			tf_res = tf_res.transpose(0, 3, 1, 2)

		self.assertEqual(ch_res.shape, tf_res.shape)
		diff = np.abs(ch_res - tf_res)
		axis = (1,2,3) if diff.ndim == 4 else 1
		msg = diff_message(diff, check_title, axis=axis)

		# print(msg)

		# import pdb; pdb.set_trace()
		self.assertTrue(np.allclose(ch_res, tf_res, atol=configs.TOL), msg)


	def test_first_conv(self):

		for size in [7, 299]:

			self.check_activation(
				"Conv0 Activation (size {})".format(size),
				size=size,
				chainer_layer=self.chainer_model.head.conv1,
				tf_layer=self.tf_model.head_conv0
			)

	def test_head(self):
		for size in [27, 299]:
			self.check_activation(
				"Head Activation (size {})".format(size),
				size=size,
				chainer_layer=self.chainer_model.head,
				tf_layer=self.tf_model.head
			)

	def test_mixed00(self):
		size = 299
		data = self.rnd.randn(3, 3, size, size).astype(np.float32)
		# data = np.ones((6, 3, size, size), dtype=np.float32)

		tf_data = data.transpose(0, 2, 3, 1)
		ch_data = self.xp.array(data)


		def call_block(block, x):
			y0 = block.conv1x1(x)
			y1 = block.conv5x5_2(block.conv5x5_1(x))
			y2 = block.conv3x3_3(block.conv3x3_2(block.conv3x3_1(x)))
			y3 = block.pool_conv(block.pool(x))

			values = [y0, y1, y2, y3]
			for v in values: v.to_cpu()
			return [v.array for v in values]

		ch_head = self.chainer_model.head(ch_data)
		ch_values = call_block(self.chainer_model.mixed00, ch_head)

		sess = self.tf_model.sess
		tf_values = [sess.run(p, {self.tf_model.inputs: tf_data})
			for p in self.tf_model.mixed_5b_values]

		tf_values = [val.transpose(0, 3, 1, 2) for val in tf_values]

		for i, (ch_val, tf_val) in enumerate(zip(ch_values, tf_values)):
			check = np.all(ch_val == tf_val)
			self.assertTrue(check, "[Mixed00] Values in branch #{} do not match!".format(i))


	def test_mixed04(self):
		size = 299
		data = self.rnd.randn(3, 3, size, size).astype(np.float32)
		# data = np.ones((6, 3, size, size), dtype=np.float32)

		tf_data = data.transpose(0, 2, 3, 1)
		ch_data = self.xp.array(data)

		def call_block(block, x):
			y0 = block.conv1x1(x)
			y1 = block.conv7x7_3(block.conv7x7_2(block.conv7x7_1(x)))
			y2 = block.conv7x7x2_5(block.conv7x7x2_4(block.conv7x7x2_3(block.conv7x7x2_2(block.conv7x7x2_1(x)))))
			y3 = block.pool_conv(block.pool(x))

			values = [y0, y1, y2, y3]
			for v in values: v.to_cpu()
			return [v.array for v in values]

		y = self.chainer_model.head(ch_data)
		y = self.chainer_model.mixed00(y)
		y = self.chainer_model.mixed01(y)
		y = self.chainer_model.mixed02(y)
		y = self.chainer_model.mixed03(y)
		ch_values = call_block(self.chainer_model.mixed04, y)

		sess = self.tf_model.sess
		tf_values = [sess.run(p, {self.tf_model.inputs: tf_data})
			for p in self.tf_model.mixed_6b_values]

		tf_values = [val.transpose(0, 3, 1, 2) for val in tf_values]

		for i, (ch_val, tf_val) in enumerate(zip(ch_values, tf_values)):
			check = np.all(ch_val == tf_val)
			self.assertTrue(check, "[Mixed04] Values in branch #{} do not match!".format(i))


	def test_mixed04_branch2(self):
		size = 299
		data = self.rnd.randn(3, 3, size, size).astype(np.float32)
		# data = np.ones((6, 3, size, size), dtype=np.float32)

		tf_data = data.transpose(0, 2, 3, 1)
		ch_data = self.xp.array(data)

		getter = self.graph.get_tensor_by_name
		prefix = "InceptionV3/InceptionV3/Mixed_6b/Branch_2"
		tf_parts = [getter("{}/{}/Relu:0".format(prefix, name)) for name in [
			"Conv2d_0a_1x1",
			"Conv2d_0b_7x1",
			"Conv2d_0c_1x7",
			"Conv2d_0d_7x1",
			"Conv2d_0e_1x7",
		]]

		sess = self.tf_model.sess
		tf_values = [sess.run(p, {self.tf_model.inputs: tf_data})
			for p in tf_parts]

		tf_values = [v.transpose(0, 3, 1, 2) for v in tf_values]


		block = self.chainer_model.mixed04
		ch_parts = [block.conv7x7x2_5, block.conv7x7x2_4, block.conv7x7x2_3, block.conv7x7x2_2, block.conv7x7x2_1][::-1]

		y = self.chainer_model.head(ch_data)
		y = self.chainer_model.mixed00(y)
		y = self.chainer_model.mixed01(y)
		y = self.chainer_model.mixed02(y)
		y = self.chainer_model.mixed03(y)

		def _caller(x, i):
			return reduce(lambda x, f: f(x), ch_parts[:i+1], x)

		ch_values = [_caller(y, i) for i in np.arange(len(ch_parts))]
		ch_values = [to_cpu(v.array) for v in ch_values]
		self.assertEqual(len(ch_values), len(tf_values))
		for i, (ch_val, tf_val) in enumerate(zip(ch_values, tf_values)):
			self.assertEqual(ch_val.shape, tf_val.shape,
				"[Mixed04] Shapes in branch #2, part #{} do not match!".format(i))
			check = np.all(ch_val == tf_val)
			self.assertTrue(check,
				"[Mixed04] Values in branch #2, part #{} do not match!".format(i))

	def test_mixed09(self):
		size = 299
		data = self.rnd.randn(3, 3, size, size).astype(np.float32)
		# data = np.ones((6, 3, size, size), dtype=np.float32)

		tf_data = data.transpose(0, 2, 3, 1)
		ch_data = self.xp.array(data)

		def call_block(block, x):
			y0 = block.conv1x1(x)

			y1 = block.conv3x3_1(x)
			y1 = F.concat([block.conv3x3_2(y1), block.conv3x3_3(y1)])

			y2 = block.conv3x3x2_2(block.conv3x3x2_1(x))
			y2 = F.concat([block.conv3x3x2_3(y2), block.conv3x3x2_4(y2)])

			y3 = block.pool_conv(block.pool(x))

			values = [y0, y1, y2, y3]
			for v in values: v.to_cpu()
			return [v.array for v in values]

		y = self.chainer_model.head(ch_data)
		y = self.chainer_model.mixed00(y)
		y = self.chainer_model.mixed01(y)
		y = self.chainer_model.mixed02(y)
		y = self.chainer_model.mixed03(y)
		y = self.chainer_model.mixed04(y)
		y = self.chainer_model.mixed05(y)
		y = self.chainer_model.mixed06(y)
		y = self.chainer_model.mixed07(y)
		y = self.chainer_model.mixed08(y)

		ch_values = call_block(self.chainer_model.mixed09, y)

		sess = self.tf_model.sess
		tf_values = [sess.run(p, {self.tf_model.inputs: tf_data})
			for p in self.tf_model.mixed_7b_values]

		tf_values = [val.transpose(0, 3, 1, 2) for val in tf_values]

		for i, (ch_val, tf_val) in enumerate(zip(ch_values, tf_values)):
			check = np.allclose(ch_val, tf_val, atol=configs.TOL)
			# if not check:
			# 	print(msg)
			# 	import matplotlib.pyplot as plt
			# 	diff = np.abs(ch_val - tf_val)

			# 	print(diff.shape)
			# 	import pdb; pdb.set_trace()
			self.assertTrue(check,
				"[Mixed09] Values in branch #{} do not match!".format(i))

	def test_mixed09_branch2(self):
		size = 299
		data = self.rnd.randn(3, 3, size, size).astype(np.float32)
		# data = np.ones((6, 3, size, size), dtype=np.float32)

		tf_data = data.transpose(0, 2, 3, 1)
		ch_data = self.xp.array(data)

		getter = self.graph.get_tensor_by_name
		prefix = "InceptionV3/InceptionV3/Mixed_7b/Branch_2"
		tf_parts = [getter("{}/{}/Relu:0".format(prefix, name)) for name in [
			"Conv2d_0a_1x1",
			"Conv2d_0b_3x3",
			"Conv2d_0c_1x3",
			"Conv2d_0d_3x1",
		]]

		sess = self.tf_model.sess
		tf_values = [sess.run(p, {self.tf_model.inputs: tf_data})
			for p in tf_parts]

		tf_values = [v.transpose(0, 3, 1, 2) for v in tf_values]


		def call_block(block, x):
			y0 = block.conv3x3x2_1(x)
			y1 = block.conv3x3x2_2(y0)
			y2 = block.conv3x3x2_3(y1)
			y3 = block.conv3x3x2_4(y1)

			values = [y0, y1, y2, y3]
			return [to_cpu(v.array) for v in values]


		y = self.chainer_model.head(ch_data)
		y = self.chainer_model.mixed00(y)
		y = self.chainer_model.mixed01(y)
		y = self.chainer_model.mixed02(y)
		y = self.chainer_model.mixed03(y)
		y = self.chainer_model.mixed04(y)
		y = self.chainer_model.mixed05(y)
		y = self.chainer_model.mixed06(y)
		y = self.chainer_model.mixed07(y)
		y = self.chainer_model.mixed08(y)


		ch_values = call_block(self.chainer_model.mixed09, y)

		self.assertEqual(len(ch_values), len(tf_values))
		for i, (ch_val, tf_val) in enumerate(zip(ch_values, tf_values)):
			self.assertEqual(ch_val.shape, tf_val.shape,
				"[Mixed04] Shapes in branch #2, part #{} do not match!".format(i))
			check = np.allclose(ch_val, tf_val, atol=configs.TOL)
			# if not check:
			# 	import matplotlib.pyplot as plt
			# 	diff = np.abs(ch_val - tf_val)
			# 	print(diff.shape)
			# 	import pdb; pdb.set_trace()

			self.assertTrue(check,
				"[Mixed09] Values in branch #2, part #{} do not match!".format(i))


	def test_blocks(self):
		size = 299

		def block_caller(name):
			def inner(X):
				res, _ = self.chainer_model(X, name)
				return res
			return inner

		blocks = [
			("head"   , self.tf_model.head),
			("mixed00", self.tf_model.mixed_5b),
			("mixed01", self.tf_model.mixed_5c),
			("mixed02", self.tf_model.mixed_5d),

			("mixed03", self.tf_model.mixed_6a),

			("mixed04", self.tf_model.mixed_6b),
			("mixed05", self.tf_model.mixed_6c),
			("mixed06", self.tf_model.mixed_6d),
			("mixed07", self.tf_model.mixed_6e),

			("mixed08", self.tf_model.mixed_7a),

			("mixed09", self.tf_model.mixed_7b),
			("mixed10", self.tf_model.mixed_7c),
		]

		for i, (ch_block, tf_block) in enumerate(blocks):

			self.check_activation(
				"Block Activation [{}] (size {})".format(ch_block, size),
				size=size,
				chainer_layer=block_caller(ch_block),
				tf_layer=tf_block
		)


	def test_features(self):
		for size in [299]:
			self.check_activation(
				"Features (size {})".format(size),
				size=size,
				chainer_layer=self.chainer_model.extract,
				tf_layer=self.tf_model.feature
			)





def diff_message(diff, title, axis):
	return "\n\t".join([
		title,
		"MAE per sample: {}".format(diff.mean(axis=axis)),
		"max error: {}".format(diff.max(axis=axis))
	])
