import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
from keras.models import load_model
tf.disable_v2_behavior()

sess = tf.Session() # get the tensorflow session to reuse it in keras

K.set_session(sess) # set it
K.set_learning_phase(0) # make sure we disable dropout and other training specific layers

string_inp = tf.placeholder(tf.string, shape=(None,)) #string input for the base64 encoded image
imgs_map = tf.map_fn(
    tf.image.decode_image,
    string_inp,
    dtype=tf.uint8
) # decode the jpeg
imgs_map.set_shape((None, None, None, 3))
imgs = tf.image.resize_images(imgs_map, [224, 224]) # resize images
imgs = tf.reshape(imgs, (-1, 224, 224, 3)) # reshape them
#img_float = tf.cast(imgs, dtype=tf.float32) / 255 # - 0.5 # and convert them to floats
img_float = tf.cast(imgs, dtype=tf.float32)
img_float = tf.keras.applications.resnet50.preprocess_input(img_float)

to_res = (224, 224)
model = load_model('../models/RESNET50-NO-LINEN_res224_batch64_epoch50_frozen0--1_class4_size7029.h5', compile=False) # load the keras model

w = model.get_weights() # save weights to be sure that they are not messed up by the global and local initialization later on

output = model(img_float) # Stack the keras model on top of the tensorflow graph -> the efficient net model is accepting base64 encoded images as a string

builder = tf.saved_model.builder.SavedModelBuilder('resnet50-server/1')

tensor_info_input = tf.saved_model.utils.build_tensor_info(string_inp)
tensor_info_output = tf.saved_model.utils.build_tensor_info(output)

# we need to init all missing placeholders
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

#set the weights to make sure they are not somehow changed by the init we did before
#single_gpu.set_weights(w)
model.set_weights(w)

# define the signature
signature = tf.saved_model.signature_def_utils.predict_signature_def(
    inputs={'b64': string_inp}, outputs={'predictions': output})

#finally save the model
builder.add_meta_graph_and_variables(
    sess=K.get_session(),
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature
    })
builder.save()

