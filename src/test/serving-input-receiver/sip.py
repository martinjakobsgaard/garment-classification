from typing import Callable, Tuple
import tensorflow as tf

# if using tf >=2.0, disable eager execution to use tf.placeholder
tf.compat.v1.disable_eager_execution()


class ServingInputReceiver:
    def __init__(self, img_size: Tuple[int], input_name: str = "lambda_input"):
        self.img_size = img_size
        self.input_name = input_name

    def decode_img_bytes(self, img_b64: str) -> tf.Tensor:
        img = tf.io.decode_image(
            img_b64,
            channels=3,
            dtype=tf.uint8,
            expand_animations=False
        )
        img = tf.image.resize(img, size=self.img_size)
        img = tf.ensure_shape(img, (*self.img_size, 3))
        img = tf.cast(img, tf.float32)
        return img

    def __call__(self) -> tf.estimator.export.ServingInputReceiver:
        imgs_b64 = tf.compat.v1.placeholder(
            shape=(None,),
            dtype=tf.string,
            name="image_bytes")

        imgs = tf.map_fn(
            self.decode_img_bytes,
            imgs_b64,
            dtype=tf.float32)

        return tf.estimator.export.ServingInputReceiver(
            features={self.input_name: imgs},
            receiver_tensors={"image_bytes": imgs_b64}
        )
