# Wiener deconvolution

import tensorflow as tf
from metasurface.conv import deconvolve_wnr

def Wiener(params, args):
    print('Loading Wiener model')
    h = params['network_width']
    w = params['network_width']
    inputs = tf.keras.layers.Input(shape=[h   ,w   ,3])
    snr    = tf.keras.layers.Input(shape=[])
    otf_1x = tf.keras.layers.Input(shape=[3, h  , w  ], dtype=tf.complex64)
    ew_1x  = tf.keras.layers.Input(shape=[h   ,w   ,3])
    otf_2x = tf.keras.layers.Input(shape=[3,h//2,w//2], dtype=tf.complex64)
    ew_2x  = tf.keras.layers.Input(shape=[h//2,w//2,3])
    otf_4x = tf.keras.layers.Input(shape=[3,h//4,w//4], dtype=tf.complex64)
    ew_4x  = tf.keras.layers.Input(shape=[h//4,w//4,3])

    snr10 = tf.math.pow(10.0, snr)
    outputs, blur = deconvolve_wnr(inputs, snr10, otf_1x, ew_1x, do_taper=args.do_taper)

    outputs = tf.clip_by_value(outputs, 0.0, 1.0)
    outputs = tf.image.resize_with_crop_or_pad(outputs, params['out_width'], params['out_width'])
    return tf.keras.Model(inputs=[inputs,snr,otf_1x,ew_1x,otf_2x,ew_2x,otf_4x,ew_4x], outputs=[outputs, blur, ew_1x])
