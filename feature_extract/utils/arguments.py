from cvargparse import ArgFactory, Arg, GPUParser

from chainer_addons.models import PrepareType
from chainer_addons.links import PoolingType

from nabirds.utils import read_info_file

DEFAULT_INFO_FILE="/home/korsch/Data/info.yml"

info_file = read_info_file(DEFAULT_INFO_FILE)

def extract_args():
	parser = GPUParser(ArgFactory([
		Arg("data", default=DEFAULT_INFO_FILE),
		Arg("dataset", choices=info_file.DATASETS.keys()),
		Arg("parts", choices=info_file.PARTS.keys()),

		Arg("output", help="output folder for the extracted features"),

		Arg("--model_type", "-mt",
			default="resnet", choices=info_file.MODELS.keys(),
			help="type of the model"),

		Arg("--input_size", type=int, default=0,
			help="overrides default input size of the model, if greater than 0"),

		PrepareType.as_arg("prepare_type",
			help_text="type of image preprocessing"),

		Arg("--weights", type=str,
			help="network weights used for feature extraction"),

		PoolingType.as_arg("pooling",
			help_text="type of pre-classification pooling"),

		Arg("--subset", choices=["train", "test"], default=None),

		Arg("--n_jobs", "-j", type=int, default=0,
			help="number of loading processes. If 0, then images are loaded in the same process"),

		Arg("--label_shift", type=int, default=1),

		Arg("--augment_positions", action="store_true"),
		Arg("--compress_output", action="store_true"),

		Arg("--is_bbox_parts", action="store_true"),


	])\
	.batch_size()\
	.debug()\
	.seed()\
	)

	parser.init_logger()

	return parser.parse_args()
