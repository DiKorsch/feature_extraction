import unittest
import abc
import os

from os.path import isdir, isfile, join

class configs(abc.ABC):

	BASE_DIR=os.environ["BASE_DIR"]

	MODEL_DIR=join(BASE_DIR, "MODELS")

	CHAINER_WEIGHTS=join(MODEL_DIR, "inception", "ft_inat", "model.ckpt.npz")
	TF_WEIGHTS=join(MODEL_DIR, "inception_tf", "ft_inat", "inception_v3_iNat_299.ckpt")

	TOL = 1e-5
	DEVICE = int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))

	SEED = 952965810


class CheckConfigs(unittest.TestCase):

	def test_data_folder(self):
		self.assertTrue(isdir(configs.MODEL_DIR),
			"Model directory is missing: {}".format(configs.MODEL_DIR)
		)

	def test_weights(self):
		self.assertTrue(isfile(configs.CHAINER_WEIGHTS),
			"Weights missing: {}".format(configs.CHAINER_WEIGHTS)
		)
		self.assertTrue(isfile(configs.TF_WEIGHTS),
			"Weights missing: {}".format(configs.TF_WEIGHTS)
		)
