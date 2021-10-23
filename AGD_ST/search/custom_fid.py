# Doesn't work since tensorflow can't load properly.

import tensorflow_gan as tfgan
import tensorflow as tf


def compute_fid(path1, path2, verbose=False, batch_size=60):
    def _make_img_ds(path):
        ds = tf.data.Dataset.list_files([str(path / "*.jpg"), str(path / "*.png")], shuffle=False)

        def _map(e):
            img = tf.io.read_file(e)
            img = tf.io.decode_image(img, channels=3)
            return img

        ds = ds.map(_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    ds = tf.data.Dataset.zip((_make_img_ds(path1), _make_img_ds(path2)))
    model = tfgan.eval.inception_metrics.frechet_inception_distance_streaming
    for a, b in ds:
        fid, model = model(a, b)
    return fid
