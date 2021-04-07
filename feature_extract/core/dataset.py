import logging
import numpy as np

from abc import ABC

from chainercv import transforms as tr
from cvdatasets.dataset import AnnotationsReadMixin
from cvdatasets.dataset import ImageProfilerMixin
from cvdatasets.dataset import IteratorMixin
from cvdatasets.dataset import TransformMixin

from feature_extract.utils.preprocessing import augmented_positions


# class _part_crop_mixin(ABC):
# 	label_shift = 1

# 	def __init__(self, augment_positions, **kwargs):
# 		super(_part_crop_mixin, self).__init__(**kwargs)

# 		self._crop_scales = self._annot.dataset_info.scales or []

# 		self._augment_positions = augment_positions
# 		logging.info("There will be {} crops (on {} scales) from {} parts".format(
# 			self.n_crops, self.n_scales, self.n_parts
# 		))

# 	@property
# 	def n_parts(self):
# 		return len(self._annot.part_names)

# 	@property
# 	def n_scales(self):
# 		return len(self._crop_scales)

# 	@property
# 	def n_positions(self):
# 		return 1 if not self._augment_positions else 4

# 	@property
# 	def n_crops(self):
# 		return self.n_parts * self.n_scales * self.n_positions + 1

# 	def generate_crops(self, im_obj):
# 		for scale in self._crop_scales:
# 			if self._augment_positions:
# 				for aug_im_obj in augmented_positions(im_obj, scale):
# 					for crop in aug_im_obj.visible_crops(scale):
# 						yield crop
# 			else:
# 				for crop in im_obj.visible_crops(scale):
# 					yield crop

# 		yield im_obj.im_array


# 	def get_example(self, i):
# 		im_obj = super(_part_crop_mixin, self).get_example(i)
# 		crops = list(self.generate_crops(im_obj))
# 		return crops, im_obj.label + self.label_shift

class Dataset(
	ImageProfilerMixin,
	TransformMixin,
	IteratorMixin,
	AnnotationsReadMixin):

	def __init__(self, *args, prepare, augment_positions, **kwargs):
		super(Dataset, self).__init__(*args, **kwargs)

		self._crop_scales = self._annot.dataset_info.scales or []
		self.prepare = prepare

		self._augment_positions = augment_positions
		logging.info("There will be {} crops (on {} scales) from {} parts".format(
			self.n_crops, self.n_scales, self.n_parts
		))

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


	def transform(self, im_obj):

		crops = []
		for i, crop in enumerate(self.generate_crops(im_obj)):
			self._profile_img(crop, f"[Crop #{i}] Initial")

			crop = self.prepare(crop, size=self.size)
			self._profile_img(crop, f"[Crop #{i}] After prepare")

			crop = tr.center_crop(crop, size=self._size)
			self._profile_img(crop, f"[Crop #{i}] After center crop")

			crop = crop * 2 - 1
			self._profile_img(crop, f"[Crop #{i}] Final")

			crops.append(crop)

		return np.array(crops)


from chainer.dataset import DatasetMixin
class TFDataset(DatasetMixin):

	def __init__(self, opts, annot, prepare, **foo):
		raise RuntimeError("FIX ME!")
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
