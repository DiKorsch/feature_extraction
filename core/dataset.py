from nabirds.dataset import Dataset as BaseDataset
from nabirds.dataset import AnnotationsReadMixin
import numpy as np


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

	def __init__(self, opts, annot, prepare, rescale_size=227, **foo):
		assert callable(prepare), "prepare must be callable!"
		super(Dataset, self).__init__(
			annotations=annot,
			uuids=annot.uuids
		)

		self._rescale_size = rescale_size
		self._crop_scales = opts.scales
		self._augment_positions = opts.augment_positions

		self.prepare = prepare

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
				for crop in im_obj.visible_crops(scale):
					yield crop

		yield im_obj.im_array

	def get_example(self, i):
		im_obj = super(Dataset, self).get_example(i)
		im = im_obj.im_array

		hw = np.array(im.shape[:-1])
		xy = im_obj.parts[:, 1:3]
		xy = xy / self._rescale_size * hw[::-1]
		im_obj.parts[:, 1:3] = xy

		crops = list(map(self.prepare, self.generate_crops(im_obj)))

		return np.stack(crops)#, im_obj.label
