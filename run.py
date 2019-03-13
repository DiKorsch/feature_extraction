#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer
import tensorflow as tf

import logging
import numpy as np

from core.dataset import Dataset#, TFDataset
from utils.arguments import extract_args, ModelType
from chainer.iterators import SerialIterator, MultiprocessIterator
from chainercv.transforms import center_crop, resize
from nabirds import CUB_Annotations

import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

def _prepare(im, size, crop_fraction=0.875):
	crop_size = None
	h, w, c = im.shape

	if isinstance(size, int):
		size = (size, size)

	_im = im.transpose(2, 0, 1)

	if crop_fraction:
		crop_size = (np.array([h, w]) * crop_fraction).astype(np.int32)
		_im = center_crop(_im, crop_size)

	_im = resize(_im, size, interpolation=2)
	if _im.dtype == np.uint8:
		# rescale [0 .. 255] -> [0 .. 1]
		_im = (_im / 255).astype(np.float32)

	# rescale [0 .. 1] -> [-1 .. 1]
	return _im * 2 - 1

def _prepare_tf(size, crop_fraction=0.875, from_path=False):

	config_sess = tf.ConfigProto(allow_soft_placement=True)
	config_sess.gpu_options.allow_growth = True
	sess = tf.Session(config=config_sess)

	if from_path:
		im_input = tf.placeholder(tf.string)
		image = tf.image.decode_jpeg(tf.read_file(im_input), channels=3)
		image = tf.image.convert_image_dtype(image, tf.float32)
	else:
		image = im_input = tf.placeholder(tf.float32, shape=(None, None, 3))


	image = tf.image.central_crop(image, central_fraction=crop_fraction)
	image = tf.expand_dims(image, 0)
	image = tf.image.resize_bilinear(image, [size, size], align_corners=False)
	image = tf.squeeze(image, [0])
	image = tf.subtract(image, 0.5)
	output = tf.multiply(image, 2)

	def inner(im):
		if not from_path and im.dtype == np.uint8:
			im = im / 255

		res = sess.run(output, feed_dict={im_input: im})
		return res.transpose(2, 0, 1)

	return inner

def main(args):
	if args.debug:
		chainer.set_debug(args.debug)
		logging.warning("DEBUG MODE ON!")
	GPU = args.gpu[0]

	model, prepare = ModelType.new(args.model_type)
	model.load_for_inference(weights=args.weights, n_classes=args.n_classes)


	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()
		model.to_gpu(GPU)

	annot = CUB_Annotations(args.data)

	# model's prepare
	# prepare_func = prepare
	# custom prepare
	prepare_func = partial(_prepare, size=model.meta.input_size)
	# TensorFlow's prepare without image reading
	# prepare_func = _prepare_tf(model.meta.input_size)
	data = Dataset(args, annot, prepare=prepare_func)
	# TensorFlow's prepare with image reading
	# prepare_func = _prepare_tf(model.meta.input_size, from_path=True)
	# data = TFDataset(args, annot, prepare=prepare_func)

	logging.info("Loaded dataset with {} samples from \"{}\"".format(
		len(data), args.data))

	logging.info("There will be {} crops (on {} scales) from {} parts".format(
		data.n_crops, data.n_scales, data.n_parts
	))


	feats = np.zeros((len(data), data.n_crops, model.meta.feature_size), dtype=np.float32)
	logging.info("Features ({}, {:.3f} GiB) will be saved to \"{}\" and \"{}\"".format(
		feats.shape,
		feats.nbytes / 1024**3,
		*args.output
	))
	if args.n_jobs > 0:
		it = MultiprocessIterator(data,
			n_processes=args.n_jobs,
			n_prefetch=2,
			batch_size=args.batch_size,
			repeat=False, shuffle=False)
	else:
		it = SerialIterator(data,
			batch_size=args.batch_size,
			repeat=False, shuffle=False)
	logging.info("Using {it.__class__.__name__} with batch size {it.batch_size}".format(it=it))
	n_batches = int(np.ceil(len(data) / it.batch_size))
	for batch_i, batch in tqdm(enumerate(it), total=n_batches):

		in_shape = batch.shape[-3:]

		batch = batch.reshape((-1, ) + in_shape)

		var = model.xp.array(batch)
		batch_feats = model(var, model.meta.feature_layer)
		if isinstance(batch_feats, tuple):
			batch_feats = batch_feats[0]
		batch_feats.to_cpu()
		batch_feats = batch_feats.array.reshape((len(batch), -1, model.meta.feature_size))

		i = batch_i * it.batch_size
		feats[i : i + len(batch)] = batch_feats


	train_feats, val_feats = feats[annot.train_split], feats[annot.test_split]
	train_labs, val_labs = annot.labels[annot.train_split], annot.labels[annot.test_split]

	logging.info("Saving features ({}compressed) according to the given split".format(
		"" if args.compress_output else "un"))
	save = np.savez_compressed if args.compress_output else np.savez

	save(
		args.output[0],
		features=train_feats,
		labels=train_labs + args.label_shift)

	save(
		args.output[1],
		features=val_feats,
		labels=val_labs + args.label_shift)



with chainer.using_config("train", False), chainer.no_backprop_mode():
	main(extract_args())
