#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import numpy as np
import matplotlib.pyplot as plt

import chainer
import tensorflow as tf

from functools import partial
from tqdm import tqdm

from feature_extract.core.dataset import Dataset
from feature_extract.utils.arguments import extract_args, ModelType

from nabirds import CUB_Annotations

def main(args):
	if args.debug:
		chainer.set_debug(args.debug)
		logging.warning("DEBUG MODE ON!")
	GPU = args.gpu[0]

	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()

	model_wrapper = ModelType.get(args.model_type).value
	model, prepare_func = model_wrapper(opts=args, device=GPU)

	annot = CUB_Annotations(args.data)
	data = Dataset(args, annot, prepare=prepare_func)
	it, n_batches = data.new_iterator(args.n_jobs, args.batch_size)

	feats = np.zeros((len(data), data.n_crops, model.meta.feature_size), dtype=np.float32)
	logging.info("Features ({}, {:.3f} GiB) will be saved to \"{}\" and \"{}\"".format(
		feats.shape,
		feats.nbytes / 1024**3,
		*args.output
	))

	for batch_i, batch in tqdm(enumerate(it), total=n_batches):
		batch_feats = model_wrapper.extract_features(model, batch)
		i = batch_i * it.batch_size
		n = batch_feats.shape[0]
		feats[i : i + n] = batch_feats

	logging.info("Splitting features ...")
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
