from cvargparse import ArgFactory, Arg, GPUParser
from cvargparse.utils.enumerations import BaseChoiceType

from chainer_addons.models import ResnetLayers, VGG19Layers, InceptionV3


class ModelType(BaseChoiceType):
	ResNet = ResnetLayers
	VGG19 = VGG19Layers
	Inception = InceptionV3
	Default = ResnetLayers

	@classmethod
	def new(cls, model_type, *args, **kwargs):
		model_type = cls.get(model_type)
		model_cls = model_type.value
		prepare = model_cls.meta.prepare_func

		model = model_cls(*args, **kwargs)

		return model, prepare

def extract_args():
	parser = GPUParser(ArgFactory([
		Arg("data"),
		Arg("weights"),
		Arg("--output", "-o", nargs=2, required=True, type=str),
		Arg("--n_classes", type=int, default=201),

		Arg("--scales", "-s", nargs="+", type=float, default=[0.31, 0.45]),
		Arg("--n_jobs", "-j", type=int, default=0,
			help="number of loading processes. If 0, then images are loaded in the same process"),
		Arg("--label_shift", type=int, default=1),
		Arg("--part_rescale", type=int, default=-1),

		Arg("--augment_positions", action="store_true"),
		Arg("--compress_output", action="store_true"),

		Arg("--is_bbox_parts", action="store_true"),

		ModelType.as_arg("model_type", "mt",
			help_text="type of the model"),

	])\
	.batch_size()\
	.debug()\
	.seed()\
	)

	parser.init_logger()

	return parser.parse_args()
