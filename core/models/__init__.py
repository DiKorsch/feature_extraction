import numpy as np

from chainer_addons.models import ResnetLayers, VGG19Layers, InceptionV3
from cvargparse.utils.enumerations import BaseChoiceType
from functools import partial
from utils.preprocessing import PrepareType, custom_prepare, prepare_tf

from .tf_inception import InceptionV3 as InceptionV3_TF

# from abc import ABC, abstractmethod
# class BaseModelWrapper(ABC):

class ChainerModelWrapper(object):

	def __init__(self, model_cls, *args, **kwargs):
		self.model_cls = model_cls
		self.args = args
		self.kwargs = kwargs

	def __call__(self, opts, device=-1, **kwargs):
		model = self.new(**kwargs)
		model.load_for_inference(
			weights=opts.weights,
			n_classes=opts.n_classes
		)
		if device >= 0:
			model.to_gpu(device)

		return model, self.prepare_func(opts.prepare_type)

	def prepare_func(self, prepare_type):
		prepare_type = PrepareType.get(prepare_type)

		if prepare_type == PrepareType.MODEL:
			return self.model_cls.meta.prepare_func

		elif prepare_type == PrepareType.CUSTOM:
			return partial(custom_prepare, size=self.model_cls.meta.input_size)

		elif prepare_type == PrepareType.CUSTOM:
			return prepare_tf(size=self.model_cls.meta.input_size, from_path=False)

		else:
			raise ValueError("Unknown preprocessing type: {}".format(prepare_type))


	def new(self, **kwargs):
		kw = dict(self.kwargs)
		kw.update(kwargs)
		return self.model_cls(*self.args, **kw)

	def extract_features(self, model, batch):

		X = np.array(batch)
		n, n_parts, c, h, w = X.shape
		X = X.reshape((-1, c, h, w))

		var = model.xp.array(X)
		batch_feats = model(var, model.meta.feature_layer)
		if isinstance(batch_feats, tuple):
			batch_feats = batch_feats[0]
		batch_feats.to_cpu()
		return batch_feats.array.reshape((n, -1, model.meta.feature_size))


class TFModelWrapper(object):

	def __init__(self, model_cls, *args, **kwargs):
		self.model_cls = model_cls
		self.args = args
		self.kwargs = kwargs

	def __call__(self, opts, device=-1):
		model = self.model_cls.new(opts.weights, train=False)
		return model, self.prepare_func(opts.prepare_type)

	def prepare_func(self, prepare_type):
		prepare_type = PrepareType.get(prepare_type)

		if prepare_type == PrepareType.MODEL:
			raise NotImplementedError("FIX ME!")
			return self.model_cls.meta.prepare_func

		elif prepare_type == PrepareType.CUSTOM:
			return partial(custom_prepare, size=self.model_cls.meta.input_size)

		elif prepare_type == PrepareType.CUSTOM:
			return prepare_tf(size=self.model_cls.meta.input_size, from_path=False)

		else:
			raise ValueError("Unknown preprocessing type: {}".format(prepare_type))

	def extract_features(self, model, batch):

		X = np.array(batch)
		n, n_parts, c, h, w = X.shape
		X = X.reshape((-1, c, h, w))
		# transpose from [n, c, h, w] to [n, h, w, c]
		batch_feats = model(X.transpose(0, 2, 3, 1))
		return batch_feats.reshape((n, -1, model.meta.feature_size))


class ModelType(BaseChoiceType):
	ResNet = ChainerModelWrapper(ResnetLayers)
	VGG19 = ChainerModelWrapper(VGG19Layers)
	Inception = ChainerModelWrapper(InceptionV3)
	Inception_TF = TFModelWrapper(InceptionV3_TF)

	Default = ChainerModelWrapper(ResnetLayers)

