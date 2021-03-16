[1mdiff --git a/src/build-server.py b/src/build-server.py[m
[1mold mode 100644[m
[1mnew mode 100755[m
[1mindex e39e26c..bb8b926[m
[1m--- a/src/build-server.py[m
[1m+++ b/src/build-server.py[m
[36m@@ -15,12 +15,14 @@[m [mimgs_map = tf.map_fn([m
     dtype=tf.uint8[m
 ) # decode the jpeg[m
 imgs_map.set_shape((None, None, None, 3))[m
[31m-imgs = tf.image.resize_images(imgs_map, [100, 100]) # resize images[m
[31m-imgs = tf.reshape(imgs, (-1, 100, 100, 3)) # reshape them[m
[31m-img_float = tf.cast(imgs, dtype=tf.float32) / 255 # - 0.5 # and convert them to floats[m
[31m-[m
[31m-to_res = (100, 100)[m
[31m-model = load_model('../models/resnet50.h5', compile=False) # load the keras model[m
[32m+[m[32mimgs = tf.image.resize_images(imgs_map, [224, 224]) # resize images[m
[32m+[m[32mimgs = tf.reshape(imgs, (-1, 224, 224, 3)) # reshape them[m
[32m+[m[32m#img_float = tf.cast(imgs, dtype=tf.float32) / 255 # - 0.5 # and convert them to floats[m
[32m+[m[32mimg_float = tf.cast(imgs, dtype=tf.float32)[m
[32m+[m[32mimg_float = tf.keras.applications.resnet50.preprocess_input(img_float)[m
[32m+[m
[32m+[m[32mto_res = (224, 224)[m
[32m+[m[32mmodel = load_model('../models/RESNET50-NO-LINEN_res224_batch64_epoch50_frozen0--1_class4_size7029.h5', compile=False) # load the keras model[m
 [m
 w = model.get_weights() # save weights to be sure that they are not messed up by the global and local initialization later on[m
 [m
