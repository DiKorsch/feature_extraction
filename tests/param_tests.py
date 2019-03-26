import unittest

import numpy as np
import chainer

from chainer_addons.models import InceptionV3
from feature_extract.core.models.tf_inception import InceptionV3 as InceptionV3_TF, inception_arg_scope
from .configs import configs


class ParamTests(unittest.TestCase):

	def run(self, *args, **kwargs):
		with chainer.using_config("train", False):
			return super(ParamTests, self).run(*args, **kwargs)

	def setUp(self):

		self.chainer_model = InceptionV3()
		self.chainer_model.load_for_finetune(
			weights=configs.CHAINER_WEIGHTS,
			n_classes=1000
		)

		arg_scope = inception_arg_scope(batch_norm_epsilon=2e-5)
		self.tf_model = InceptionV3_TF.new(
			configs.TF_WEIGHTS, train=False, arg_scope=arg_scope)
		self.rnd = np.random.RandomState(configs.SEED)

	def compare_layer(self, ch_conv, tf_conv_name):
		chainer_conv = ch_conv.conv.W.array
		chainer_bn_beta = ch_conv.bn.beta.array
		chainer_bn_mean = ch_conv.bn.avg_mean
		chainer_bn_var = ch_conv.bn.avg_var

		params = self.tf_model.params

		tf_conv_param = params["{}/weights".format(tf_conv_name)]
		tf_bn_beta_param = params["{}/BatchNorm/beta".format(tf_conv_name)]
		tf_bn_mean_param = params["{}/BatchNorm/moving_mean".format(tf_conv_name)]
		tf_bn_var_param = params["{}/BatchNorm/moving_variance".format(tf_conv_name)]

		sess = self.tf_model.sess
		tf_conv = sess.run(tf_conv_param).transpose(3, 2, 0, 1)
		tf_bn_beta = sess.run(tf_bn_beta_param)
		tf_bn_mean = sess.run(tf_bn_mean_param)
		tf_bn_var = sess.run(tf_bn_var_param)

		self.assertTrue((chainer_conv == tf_conv).all(),
			"[{}] Weights does not match!".format(tf_conv_name))
		self.assertTrue((chainer_bn_beta == tf_bn_beta).all(),
			"[{}] BN beta does not match!".format(tf_conv_name))
		self.assertTrue((chainer_bn_mean == tf_bn_mean).all(),
			"[{}] BN mean does not match!".format(tf_conv_name))
		self.assertTrue((chainer_bn_var == tf_bn_var).all(),
			"[{}] BN var does not match!".format(tf_conv_name))

	def test_head_weights(self):

		layers = [
			(self.chainer_model.head.conv1, "Conv2d_1a_3x3"),
			(self.chainer_model.head.conv2, "Conv2d_2a_3x3"),
			(self.chainer_model.head.conv3, "Conv2d_2b_3x3"),
			(self.chainer_model.head.conv5, "Conv2d_3b_1x1"),
			(self.chainer_model.head.conv6, "Conv2d_4a_3x3"),
		]

		for ch_conv, tf_conv in layers:
			self.compare_layer(ch_conv, tf_conv)

	def test_block00_weights(self):

		layers = [
			(
				self.chainer_model.mixed00.conv1x1,
				"Mixed_5b/Branch_0/Conv2d_0a_1x1"
			),
			(
				self.chainer_model.mixed00.conv5x5_1,
				"Mixed_5b/Branch_1/Conv2d_0a_1x1"
			),
			(
				self.chainer_model.mixed00.conv5x5_2,
				"Mixed_5b/Branch_1/Conv2d_0b_5x5"
			),
			(
				self.chainer_model.mixed00.conv3x3_1,
				"Mixed_5b/Branch_2/Conv2d_0a_1x1"
			),
			(
				self.chainer_model.mixed00.conv3x3_2,
				"Mixed_5b/Branch_2/Conv2d_0b_3x3"
			),
			(
				self.chainer_model.mixed00.conv3x3_3,
				"Mixed_5b/Branch_2/Conv2d_0c_3x3"
			),
			(
				self.chainer_model.mixed00.pool_conv,
				"Mixed_5b/Branch_3/Conv2d_0b_1x1"
			),
		]

		for ch_conv, tf_conv in layers:
			self.compare_layer(ch_conv, tf_conv)


	def test_bn_params_should_not_change_after_call(self):

		self.compare_layer(
			self.chainer_model.head.conv1,
			"Conv2d_1a_3x3"
		)

		data = self.rnd.randn(6, 3, 7, 7).astype(np.float32)
		tf_data = data.transpose(0, 2, 3, 1)

		ch_res = self.chainer_model.head.conv1(data).array
		tf_res = self.tf_model.sess.run(
			self.tf_model.head_conv0,
			feed_dict={self.tf_model.inputs: tf_data}
		).transpose(0, 3, 1, 2)

		self.compare_layer(
			self.chainer_model.head.conv1,
			"Conv2d_1a_3x3"
		)
