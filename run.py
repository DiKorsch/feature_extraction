#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import chainer

import logging
import numpy as np

from core.dataset import Dataset
from utils.arguments import extract_args, ModelType

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


	for i, crops in enumerate(data):
		var = model.xp.array(crops)

		feat = model(var, model.meta.feature_layer)
		feat.to_cpu()
		feats[i] = feat.array

		print("{} / {} ready".format(i+1, len(data)), end="\r")

	print()

	train_feats, val_feats = feats[annot.train_split], feats[annot.test_split]
	train_labs, val_labs = annot.labels[annot.train_split], annot.labels[annot.test_split]

	logging.info("Saving features according to the given split")
	np.savez_compressed(
		args.output[0],
		features=train_feats,
		labels=train_labs)

	np.savez_compressed(
		args.output[1],
		features=val_feats,
		labels=val_labs)



with chainer.using_config("train", False), chainer.no_backprop_mode():
	main(extract_args())
