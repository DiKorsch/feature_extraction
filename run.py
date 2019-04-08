#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import numpy as np
import matplotlib.pyplot as plt

import chainer

from os.path import join, isfile
from functools import partial
from tqdm import tqdm

from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType
from chainer_addons.links.pooling import PoolingType

from feature_extract.core.dataset import Dataset
from feature_extract.core.models import ModelWrapper
from feature_extract.utils.arguments import extract_args

from nabirds.annotations import AnnotationType

def main(args):
	if args.debug:
		chainer.set_debug(args.debug)
		logging.warning("DEBUG MODE ON!")
	GPU = args.gpu[0]

	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()

	annot_cls = AnnotationType.get(args.dataset).value
	annot = annot_cls(args.data, args.parts)

	data_info = annot.info
	model_info = data_info.MODELS[args.model_type]
	part_info = data_info.PARTS[args.parts]

	if model_info.class_key == "inception_tf":
		import pdb; pdb.set_trace()
		raise ValueError("FIX ME!")

	model = ModelType.new(
		model_type=model_info.class_key,
		input_size=args.input_size,
		pooling=args.pooling,
		aux_logits=False
	)
	size = model.meta.input_size

	prepare = PrepareType[args.prepare_type](model)

	logging.info("Created {} model with \"{}\" prepare function. Image input size: {}"\
		.format(
			model.__class__.__name__,
			args.prepare_type,
			size
		)
	)


	if args.weights:
		weights_file = join("ft_{}".format(args.dataset), args.weights)
	else:
		weights_file = model_info.weights

	weights = join(
		data_info.BASE_DIR,
		data_info.MODEL_DIR,
		model_info.folder,
		weights_file
	)

	assert isfile(weights), "Could not find weights \"{}\"".format(weights)
	logging.info("Loading weights from \"{}\"".format(weights))

	wrapped_model = ModelWrapper(model,
		weights=weights,
		n_classes=part_info.n_classes + args.label_shift,
		device=GPU)

	data = annot.new_dataset(
		subset=args.subset,
		dataset_cls=Dataset,

		preprocess=prepare,
		size=size,
		augment_positions=args.augment_positions,
	)
	n_samples = len(data)
	logging.info("Loaded \"{}\"-parts dataset with {} samples from \"{}\"".format(
		args.parts, n_samples, annot.root))

	it, n_batches = data.new_iterator(
		args.n_jobs, args.batch_size,
		repeat=False, shuffle=False)

	feats = np.zeros((n_samples, data.n_crops, model.meta.feature_size), dtype=np.float32)
	output = [join(args.output, "{}{}.{}.npz".format(
		subset, part_info.feature_suffix, model_info.folder)) for subset in ["train", "test"]]
	logging.info("Features ({}, {:.3f} GiB) will be saved to \"{}\" and \"{}\"".format(
		feats.shape,
		feats.nbytes / 1024**3,
		*output
	))

	preds = np.zeros((n_samples, data.n_crops), dtype=np.int32)

	for batch_i, batch in tqdm(enumerate(it), total=n_batches):
		batch_feats, pred = wrapped_model(batch)
		i = batch_i * it.batch_size
		n = batch_feats.shape[0]
		feats[i : i + n] = batch_feats
		preds[i : i + n] = pred

	logging.info("Saving features ({}compressed) according to the given split".format(
		"" if args.compress_output else "un"))
	save = np.savez_compressed if args.compress_output else np.savez

	if args.subset is None:
		logging.info("Splitting features ...")
		train_feats, test_feats = feats[annot.train_split], feats[annot.test_split]
		train_labs, test_labs = annot.labels[annot.train_split], annot.labels[annot.test_split]
		train_preds = preds[annot.train_split]
		test_preds = preds[annot.test_split]

		train_acc = (train_preds == (train_labs+args.label_shift)[:, None]).mean(axis=0)
		test_acc = (test_preds == (test_labs+args.label_shift)[:, None]).mean(axis=0)
		train_part_accs = ["{:.3%}".format(acc) for acc in train_acc]
		test_part_accs = ["{:.3%}".format(acc) for acc in test_acc]
		logging.info("Train Accuracy: {}".format(" | ".join(train_part_accs)))
		logging.info("Test Accuracy : {}".format(" | ".join(test_part_accs)))

		save(
			output[0],
			features=train_feats,
			labels=train_labs + args.label_shift)

		save(
			output[1],
			features=test_feats,
			labels=test_labs + args.label_shift)

	elif args.subset == "train":
		save(
			output[0],
			features=feats,
			labels=data.labels + args.label_shift)

	elif args.subset == "test":
		save(
			output[1],
			features=feats,
			labels=data.labels + args.label_shift)
	else:
		raise ValueError("Unknown subset: {}".format(args.subset))








with chainer.using_config("train", False), chainer.no_backprop_mode():
	main(extract_args())
