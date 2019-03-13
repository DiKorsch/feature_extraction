
def prepare(im, size, crop_fraction=0.875):
	crop_size = None
	h, w, c = im.shape

	if isinstance(size, int):
		size = (size, size)

	_im = im.transpose(2, 0, 1)

	if crop_fraction:
		crop_size = (np.array([h, w]) * crop_fraction).astype(np.int32)
		_im = center_crop(_im, crop_size)

	_im = resize(_im, size, interpolation=2)
	if _im.dtype == np.uint8:
		# rescale [0 .. 255] -> [0 .. 1]
		_im = (_im / 255).astype(np.float32)

	# rescale [0 .. 1] -> [-1 .. 1]
	return _im * 2 - 1

def prepare_tf(size, crop_fraction=0.875, from_path=False):

	config_sess = tf.ConfigProto(allow_soft_placement=True)
	config_sess.gpu_options.allow_growth = True
	sess = tf.Session(config=config_sess)

	if from_path:
		im_input = tf.placeholder(tf.string)
		image = tf.image.decode_jpeg(tf.read_file(im_input), channels=3)
		image = tf.image.convert_image_dtype(image, tf.float32)
	else:
		image = im_input = tf.placeholder(tf.float32, shape=(None, None, 3))


	image = tf.image.central_crop(image, central_fraction=crop_fraction)
	image = tf.expand_dims(image, 0)
	image = tf.image.resize_bilinear(image, [size, size], align_corners=False)
	image = tf.squeeze(image, [0])
	image = tf.subtract(image, 0.5)
	output = tf.multiply(image, 2)

	def inner(im):
		if not from_path and im.dtype == np.uint8:
			im = im / 255

		res = sess.run(output, feed_dict={im_input: im})
		return res.transpose(2, 0, 1)

	return inner
