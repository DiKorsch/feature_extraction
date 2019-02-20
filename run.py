#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer

import logging
import numpy as np

from core.dataset import Dataset
from utils.arguments import extract_args, ModelType
from chainer.iterators import SerialIterator, MultiprocessIterator

from nabirds import CUB_Annotations

import matplotlib.pyplot as plt

def main(args):
	if args.debug:
		chainer.set_debug(args.debug)
		logging.warning("DEBUG MODE ON!")
	GPU = args.gpu[0]

	model, prepare = ModelType.new(args.model_type)
	model.load_for_inference(weights=args.weights, n_classes=201)

	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()
		model.to_gpu(GPU)

	annot = CUB_Annotations(args.data)

	data = Dataset(args, annot, prepare=prepare)
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
			n_prefetch=32,
			batch_size=args.batch_size,
			repeat=False, shuffle=False)
	else:
		it = SerialIterator(data,
			batch_size=args.batch_size,
			repeat=False, shuffle=False)
	logging.info("Using {it.__class__.__name__} with batch size {it.batch_size}".format(it=it))
	for batch_i, batch in enumerate(it):

		var = model.xp.array(batch)
		in_shape = var.shape[-3:]

		var = var.reshape((-1, ) + in_shape)
		batch_feats = model(var, model.meta.feature_layer)
		batch_feats.to_cpu()
		batch_feats = batch_feats.array.reshape((len(batch), -1, model.meta.feature_size))

		i = batch_i * it.batch_size
		feats[i : i + len(batch)] = batch_feats
		samples_ready = min(len(data), (batch_i + 1) * it.batch_size)
		print("{} / {} ready".format(samples_ready, len(data)), end="\r")

	print()

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
