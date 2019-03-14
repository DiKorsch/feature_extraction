from nabirds.dataset import AnnotationsReadMixin
from chainer.iterators import SerialIterator, MultiprocessIterator

import numpy as np
import logging


def augmented_positions(im_obj, scale):
	im, parts, labels = im_obj.as_tuple()
	h, w = im.shape[:2]

	crop_h, crop_w = h * scale, w * scale

	d_h, d_w = int(crop_h / 4), int(crop_w / 4)

	diff = np.array([
		[-d_w,-d_h],
		[ d_w,-d_h],
		[-d_w, d_h],
		[ d_w, d_h],
	])

	for d in diff:
		aug_im_obj = im_obj.copy()

		xy = aug_im_obj.parts[:, 1:3]
		xy += d
		xy = np.maximum(xy, 0)
		xy = np.minimum(xy, np.array([w, h]) - 1)

		aug_im_obj.parts[:, 1:3] = xy
		yield aug_im_obj

class Dataset(AnnotationsReadMixin):

	def __init__(self, opts, annot, prepare, **foo):
		assert callable(prepare), "prepare must be callable!"
		super(Dataset, self).__init__(
			annotations=annot,
			uuids=annot.uuids,
			part_rescale_size=opts.part_rescale
		)
		logging.info("Loaded dataset with {} samples from \"{}\"".format(
			len(self), opts.data))

		self._crop_scales = opts.scales
		self._augment_positions = opts.augment_positions
		logging.info("There will be {} crops (on {} scales) from {} parts".format(
			self.n_crops, self.n_scales, self.n_parts
		))

		if opts.augment_positions and opts.is_bbox_parts:
			raise ValueError("Either position augmentation or bbox parts should be enabled!")

		self.prepare = prepare

	def new_iterator(self, n_jobs, batch_size, n_prefetch=2):

		if n_jobs > 0:
			it = MultiprocessIterator(self,
				n_processes=n_jobs,
				n_prefetch=n_prefetch,
				batch_size=batch_size,
				repeat=False, shuffle=False)
		else:
			it = SerialIterator(self,
				batch_size=batch_size,
				repeat=False, shuffle=False)
		logging.info("Using {it.__class__.__name__} with batch size {it.batch_size}".format(it=it))
		n_batches = int(np.ceil(len(self) / it.batch_size))
		return it, n_batches


	@property
	def n_parts(self):
		return len(self._annot.part_names)

	@property
	def n_scales(self):
		return len(self._crop_scales)

	@property
	def n_positions(self):
		return 1 if not self._augment_positions else 4

	@property
	def n_crops(self):
		return self.n_parts * self.n_scales * self.n_positions + 1


	def generate_crops(self, im_obj):
		for scale in self._crop_scales:
			if self._augment_positions:
				for aug_im_obj in augmented_positions(im_obj, scale):
					for crop in aug_im_obj.visible_crops(scale):
						yield crop
			else:
				if scale > 0:
					for crop in im_obj.visible_crops(scale):
						yield crop
				else:
					for i, x,y,w,h in im_obj.parts:
						yield im_obj.im_array[y:y+h, x:x+w]

		yield im_obj.im_array

	def get_example(self, i):
		im_obj = super(Dataset, self).get_example(i)
		crops = list(map(self.prepare, self.generate_crops(im_obj)))
		return np.stack(crops)#, im_obj.label


from chainer.dataset import DatasetMixin
class TFDataset(DatasetMixin):

	def __init__(self, opts, annot, prepare, **foo):
		super(TFDataset, self).__init__()
		assert callable(prepare), "prepare must be callable!"
		self.uuids = annot.uuids
		self._annot = annot
		self._crop_scales = opts.scales
		self.prepare = prepare

	@property
	def n_parts(self):
		return len(self._annot.part_names)

	@property
	def n_scales(self):
		return len(self._crop_scales)

	@property
	def n_crops(self):
		return self.n_parts * self.n_scales + 1


	def __len__(self):
		return len(self.uuids)


	def _get(self, method, i):
		return getattr(self._annot, method)(self.uuids[i])

	def get_example(self, i):
		methods = ["image", "parts", "label"]
		im_path, parts, label = [self._get(m, i) for m in methods]

		im = self.prepare(im_path)
		part_crops = [np.zeros_like(im) for _ in range(len(parts))]
		return np.array(part_crops + [im])#, label