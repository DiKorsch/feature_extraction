from cvargparse import ArgFactory, Arg, GPUParser

from feature_extract.core.models import ModelType
from feature_extract.utils.preprocessing import PrepareType

from nabirds.utils import read_info_file

DEFAULT_INFO_FILE="/home/korsch/Data/info.yml"

info_file = read_info_file(DEFAULT_INFO_FILE)

def extract_args():
	parser = GPUParser(ArgFactory([
		Arg("data", default=DEFAULT_INFO_FILE),
		Arg("dataset", choices=info_file.DATASETS.keys()),
		Arg("parts", choices=info_file.PARTS.keys()),

		Arg("output", help="output folder for the extracted features"),

		Arg("--n_classes", type=int, default=201),

		Arg("--n_jobs", "-j", type=int, default=0,
			help="number of loading processes. If 0, then images are loaded in the same process"),
		Arg("--label_shift", type=int, default=1),

		Arg("--augment_positions", action="store_true"),
		Arg("--compress_output", action="store_true"),

		Arg("--is_bbox_parts", action="store_true"),

		ModelType.as_arg("model_type", "mt",
			help_text="type of the model"),

		PrepareType.as_arg("prepare_type",
			help_text="type of image preprocessing"),

	])\
	.batch_size()\
	.debug()\
	.seed()\
	)

	parser.init_logger()

	return parser.parse_args()
