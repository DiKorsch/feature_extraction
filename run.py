#!/usr/bin/env python
if __name__ != '__main__': raise Exception("Do not import me!")

import logging
import matplotlib.pyplot as plt
import numpy as np

import chainer

from functools import partial
from os.path import isfile
from os.path import join
from tqdm import tqdm

from chainer_addons.links.pooling import PoolingType
from chainer_addons.models import ModelType
from chainer_addons.models import PrepareType

from feature_extract.core.dataset import Dataset
from feature_extract.core.models import ModelWrapper
from feature_extract.utils.arguments import extract_args

from cvdatasets import AnnotationType
from cvdatasets.dataset.image import Size
from cvdatasets.utils import feature_file_name
from cvdatasets.utils import new_iterator

def main(args):
	if args.debug:
		chainer.set_debug(args.debug)
		logging.warning("DEBUG MODE ON!")
	GPU = args.gpu[0]

	if GPU >= 0:
		chainer.cuda.get_device(GPU).use()

	annot = AnnotationType.new_annotation(args, load_strict=False)

	data_info = annot.info
	model_info = data_info.MODELS[args.model_type]
	ds_info = annot.dataset_info

	if model_info.class_key == "inception_tf":
		import pdb; pdb.set_trace()
		raise ValueError("FIX ME!")

	assert args.input_size != 0, "input size is set to 0!"
	model = ModelType.new(
		model_type=model_info.class_key,
		input_size=Size(args.input_size),
		pooling=args.pooling,
		aux_logits=False
	)

	size = model.meta.input_size


	prepare = partial(PrepareType[args.prepare_type](model),
		swap_channels=args.swap_channels,
		keep_ratio=not args.no_center_crop_on_val,
	)

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

	# is absolute path
	if weights_file.startswith("/"):
		weights = weights_file
	else:
		weights = join(
			data_info.BASE_DIR,
			data_info.MODEL_DIR,
			model_info.folder,
			weights_file
		)

	assert isfile(weights), "Could not find weights \"{}\"".format(weights)
	logging.info("Loading \"{}\" weights from \"{}\"".format(
		model_info.class_key, weights))
	n_classes = ds_info.n_classes + args.label_shift
	wrapped_model = ModelWrapper(model,
		weights=weights,
		n_classes=n_classes,
		device=GPU)

	data = annot.new_dataset(
		subset=args.subset,
		dataset_cls=Dataset,

		center_cropped=not args.no_center_crop_on_val,
		size=size,
		augment_positions=args.augment_positions,
		prepare=prepare,
	)

	with data.enable_img_profiler():
		data[np.random.randint(len(data))]

	n_samples = len(data)
	logging.info("Loaded \"{}\"-parts dataset with {} samples from \"{}\"".format(
		args.parts, n_samples, annot.root))

	it, n_batches = new_iterator(data,
		args.n_jobs, args.batch_size,
		repeat=False, shuffle=False)

	feats = np.zeros((n_samples, data.n_crops, model.meta.feature_size), dtype=np.float32)

	_output = lambda subset: join(args.output, feature_file_name(subset, ds_info, model_info))

	output = [_output(subset) for subset in ["train", "test"]]
	logging.info("Features ({}, {:.3f} GiB) will be saved to \"{}\" and \"{}\"".format(
		feats.shape,
		feats.nbytes / 1024**3,
		*output
	))

	preds = np.zeros((n_samples, data.n_crops), dtype=np.int32)

	bar = tqdm(enumerate(it), total=n_batches, desc="Extracting features")

	labs = np.expand_dims(data.labels, axis=1).repeat(data.n_crops, axis=1)

	for batch_i, batch in bar:
		batch_feats, pred = wrapped_model(batch)
		i = batch_i * it.batch_size
		n = batch_feats.shape[0]
		feats[i : i + n] = batch_feats
		preds[i : i + n] = pred

		curr_accu = (preds[:i+n] == labs[:i+n]).mean()
		bar.set_description(f"Extracting features (Accuracy: {curr_accu:.2%})")


	logging.info("Saving features ({}compressed) according to the given split".format(
		"" if args.compress_output else "un"))
	save = np.savez_compressed if args.compress_output else np.savez

	if args.subset is None:
		logging.info("Splitting features ...")
		train_feats, test_feats = feats[annot.train_split], feats[annot.test_split]
		train_labs, test_labs = annot.labels[annot.train_split], annot.labels[annot.test_split]

		# train_labs += args.label_shift
		# test_labs += args.label_shift

		if train_labs.max() > n_classes:
			_, train_labs = np.unique(train_labs, return_inverse=True)
			_, test_labs = np.unique(test_labs, return_inverse=True)


		train_preds = preds[annot.train_split]
		test_preds = preds[annot.test_split]

		train_acc = (train_preds == train_labs[:, None]).mean(axis=0)
		test_acc = (test_preds == test_labs[:, None]).mean(axis=0)
		train_part_accs = ["{:.3%}".format(acc) for acc in train_acc]
		test_part_accs = ["{:.3%}".format(acc) for acc in test_acc]
		logging.info("Train Accuracy: {}".format(" | ".join(train_part_accs)))
		logging.info("Test Accuracy : {}".format(" | ".join(test_part_accs)))

		save(
			output[0],
			features=train_feats,
			labels=train_labs)

		save(
			output[1],
			features=test_feats,
			labels=test_labs)

	elif args.subset == "train":
		save(
			output[0],
			features=feats,
			labels=data.labels #+ args.label_shift
		)

	elif args.subset == "test":
		save(
			output[1],
			features=feats,
			labels=data.labels #+ args.label_shift
		)
	else:
		raise ValueError("Unknown subset: {}".format(args.subset))







chainer.global_config.cv_resize_backend = "cv2"
with chainer.using_config("train", False), chainer.no_backprop_mode():
	main(extract_args())
