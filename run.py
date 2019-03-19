#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import numpy as np
import matplotlib.pyplot as plt

import chainer
import tensorflow as tf

from os.path import join
from functools import partial
from tqdm import tqdm

from feature_extract.core.dataset import Dataset
from feature_extract.utils.arguments import extract_args, ModelType

from nabirds.annotations import AnnotationType

def main(args):
	if args.debug:
		chainer.set_debug(args.debug)
		logging.warning("DEBUG MODE ON!")
	GPU = args.gpu[0]

	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()

	annot = AnnotationType.CUB.value(args.data, args.parts)

	model_wrapper = ModelType.get(args.model_type).value
	cls_name = model_wrapper.model_cls.__name__

	data_info = annot.info
	model_info = data_info.MODELS[cls_name]
	part_info = data_info.PARTS[args.parts]


	args.weights = join(
		data_info.BASE_DIR,
		data_info.MODEL_DIR,
		model_info.folder,
		model_info.weights
	)
	model, prepare_func = model_wrapper(opts=args, device=GPU)

	data = annot.new_dataset(
		subset=None,
		dataset_cls=Dataset,

		prepare=prepare_func,
		augment_positions=args.augment_positions,
	)
	n_samples = len(data)
	logging.info("Loaded {} parts dataset with {} samples from \"{}\"".format(
		args.parts, n_samples, annot.root))
	it, n_batches = data.new_iterator(args.n_jobs, args.batch_size)

	feats = np.zeros((n_samples, data.n_crops, model.meta.feature_size), dtype=np.float32)
	output = [join(args.output, "{}_{}.{}.npz".format(
		subset, part_info.feature_suffix, model_info.folder)) for subset in ["train", "test"]]
	logging.info("Features ({}, {:.3f} GiB) will be saved to \"{}\" and \"{}\"".format(
		feats.shape,
		feats.nbytes / 1024**3,
		*output
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
		output[0],
		features=train_feats,
		labels=train_labs + args.label_shift)

	save(
		output[1],
		features=val_feats,
		labels=val_labs + args.label_shift)



with chainer.using_config("train", False), chainer.no_backprop_mode():
	main(extract_args())
