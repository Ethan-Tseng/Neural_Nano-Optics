import tensorflow as tf

def load(image_width, image_width_padded, augment):
    # image_width = Width for image content
    # image_width_padded = Width including padding to accomodate PSF
    def load_fn(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = image / 255.

        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
        image = tf.image.resize_with_crop_or_pad(image, image_width_padded, image_width_padded) 
        return (image, image) # Input and GT
    return load_fn

def train_dataset_sim(image_width, image_width_padded, args):
    load_fn = load(image_width, image_width_padded, augment=True)
    ds = tf.data.Dataset.list_files(args.train_dir+'*.jpg')
    ds = ds.map(load_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(20)
    ds = ds.repeat() # Repeat forever
    ds = ds.batch(1) # Batch size = 1
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def test_dataset_sim(image_width, image_width_padded, args):
    load_fn = load(image_width, image_width_padded, augment=False)
    ds = tf.data.Dataset.list_files(args.test_dir+'*.jpg', shuffle=False)
    ds = ds.map(load_fn, num_parallel_calls=None)
    ds = ds.batch(1) # Batch size = 1
    return ds
