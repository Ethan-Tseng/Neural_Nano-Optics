# Neural feature propagator network

import tensorflow as tf
import tensorflow_addons as tfa
from metasurface.conv import deconvolve_wnr

def conv(filters, size, stride, activation, apply_instnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, stride, padding='same', use_bias=True))
    if apply_instnorm:
        result.add(tfa.layers.InstanceNormalization())
    if not activation == None:
        result.add(activation())
    return result

def conv_transp(filters, size, stride, activation, apply_instnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=True))
    if not activation == None:
        result.add(activation())
    return result

def feat_extract(img, snr, otf_1x, ew_1x, otf_2x, ew_2x, otf_4x, ew_4x, params, args):
    LReLU = tf.keras.layers.LeakyReLU
    ReLU  = tf.keras.layers.ReLU
    
    down_l0 = conv(15, 7, 1, LReLU, apply_instnorm=False)(img)
    down_l0 = conv(15, 7, 1, LReLU, apply_instnorm=False)(down_l0)

    down_l1 = conv(30, 5, 2, LReLU, apply_instnorm=False)(down_l0)
    down_l1 = conv(30, 3, 1, LReLU, apply_instnorm=False)(down_l1)
    down_l1 = conv(30, 3, 1, LReLU, apply_instnorm=False)(down_l1)

    down_l2 = conv(60, 5, 2, LReLU, apply_instnorm=False)(down_l1)
    down_l2 = conv(60, 3, 1, LReLU, apply_instnorm=False)(down_l2)
    down_l2 = conv(60, 3, 1, LReLU, apply_instnorm=False)(down_l2)

    # 4x
    conv_l2_k0 = conv(60, 3, 1, LReLU, apply_instnorm=False)(down_l2)
    conv_l2_k1 = conv(60, 3, 1, LReLU, apply_instnorm=False)(conv_l2_k0)

    conv_l2_k2 = conv(60, 3, 1, LReLU, apply_instnorm=False)(tf.concat([down_l2, conv_l2_k1], axis=3))
    conv_l2_k3 = conv(60, 3, 1, LReLU, apply_instnorm=False)(conv_l2_k2)

    conv_l2_k4 = conv(60, 3, 1, LReLU, apply_instnorm=False)(conv_l2_k3)
    conv_l2_k5 = conv(60, 3, 1, LReLU, apply_instnorm=False)(conv_l2_k4)

    wien_l2_b, _, = deconvolve_wnr(conv_l2_k5, snr, tf.tile(otf_4x, [1, 20, 1, 1]), tf.tile(ew_4x, [1, 1, 1, 20]), do_taper=(args.do_taper))

    # 2x
    conv_l1_k0 = conv(30, 3, 1, LReLU, apply_instnorm=False)(down_l1)
    conv_l1_k1 = conv(30, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k0)

    conv_l1_k2 = conv(30, 3, 1, LReLU, apply_instnorm=False)(tf.concat([down_l1, conv_l1_k1], axis=3))
    conv_l1_k3 = conv(30, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k2)

    conv_l1_k4 = conv(30, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k3)
    conv_l1_k5 = conv(30, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k4)

    up_l2 = conv_transp(30, 2, 2, LReLU, apply_instnorm=False)(conv_l2_k5)
    conv_l1_k6 = conv(30, 3, 1, LReLU, apply_instnorm=False)(tf.concat([up_l2, conv_l1_k5], axis=3))
    conv_l1_k7 = conv(30, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k6)

    wien_l1_b, _ = deconvolve_wnr(conv_l1_k7, snr, tf.tile(otf_2x, [1, 10, 1, 1]), tf.tile(ew_2x, [1, 1, 1, 10]), do_taper=(args.do_taper))

    # 1x
    conv_l0_k0 = conv(15, 5, 1, LReLU, apply_instnorm=False)(down_l0)
    conv_l0_k1 = conv(15, 5, 1, LReLU, apply_instnorm=False)(conv_l0_k0)

    conv_l0_k2 = conv(15, 5, 1, LReLU, apply_instnorm=False)(tf.concat([down_l0, conv_l0_k1], axis=3))
    conv_l0_k3 = conv(15, 5, 1, LReLU, apply_instnorm=False)(conv_l0_k2)

    conv_l0_k4 = conv(15, 5, 1, LReLU, apply_instnorm=False)(conv_l0_k3)
    conv_l0_k5 = conv(15, 5, 1, LReLU, apply_instnorm=False)(conv_l0_k4)

    up_l1 = conv_transp(15, 2, 2, LReLU, apply_instnorm=False)(conv_l1_k5)
    conv_l0_k6 = conv(15, 5, 1, LReLU, apply_instnorm=False)(tf.concat([up_l1, conv_l0_k5], axis=3))
    conv_l0_k7 = conv(15, 5, 1, LReLU, apply_instnorm=False)(conv_l0_k6)

    wiener_1x = tf.math.conj(otf_1x) / (tf.cast(tf.abs(otf_1x) ** 2, tf.complex64) + tf.cast(1 / tf.abs(snr), tf.complex64))
    wiener_1x = tf.tile(wiener_1x, [1, 5, 1, 1])
    wien_l0_b, _ = deconvolve_wnr(conv_l0_k7, snr, tf.tile(otf_1x, [1, 5, 1, 1]), tf.tile(ew_1x, [1, 1, 1, 5]), do_taper=(args.do_taper))

    return wien_l0_b, wien_l1_b, wien_l2_b, ew_1x, ew_2x, ew_4x
 
def FP(params, args):
    LReLU = tf.keras.layers.LeakyReLU
    ReLU  = tf.keras.layers.ReLU

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

    ## Feature Extractor
    deconv0, deconv1, deconv2, edge0, edge1, edge2 = \
        feat_extract(inputs, tf.math.pow(10.0, snr), otf_1x, ew_1x, otf_2x, ew_2x, otf_4x, ew_4x, params, args)
    side = (h - params['out_width']) // 2
    deconv0 = deconv0[:,side:-side,side:-side,:]
    deconv1 = deconv1[:,side//2:-side//2,side//2:-side//2,:]
    deconv2 = deconv2[:,side//4:-side//4,side//4:-side//4,:]

    ## Decoder
    conv_l0_k0 = conv(30, 5, 1, LReLU, apply_instnorm=False)(deconv0)
    conv_l0_k1 = conv(30, 5, 1, LReLU, apply_instnorm=False)(conv_l0_k0)
    down_l0 = conv(30, 5, 2, LReLU, apply_instnorm=False)(conv_l0_k1)

    conv_l1_k0 = tf.concat([deconv1, down_l0], axis=3)
    conv_l1_k1 = conv(60, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k0)
    conv_l1_k2 = conv(60, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k1)
    down_l1 = conv(60, 3, 2, LReLU, apply_instnorm=False)(conv_l1_k2)

    conv_l2_k0 = tf.concat([deconv2, down_l1], axis=3)
    conv_l2_k1 = conv(120, 3, 1, LReLU, apply_instnorm=False)(conv_l2_k0)
    conv_l2_k2 = conv(120, 3, 1, LReLU, apply_instnorm=False)(conv_l2_k1)

    conv_l2_k3 = conv(120, 3, 1, LReLU, apply_instnorm=False)(tf.concat([conv_l2_k0, conv_l2_k2], axis=3))
    conv_l2_k4 = conv(120, 3, 1, LReLU, apply_instnorm=False)(conv_l2_k3)

    up_l2 = conv_transp(60, 2, 2, LReLU, apply_instnorm=False)(conv_l2_k4)
    conv_l1_k3 = conv(60, 3, 1, LReLU, apply_instnorm=False)(tf.concat([conv_l1_k2, up_l2], axis=3))
    conv_l1_k4 = conv(60, 3, 1, LReLU, apply_instnorm=False)(conv_l1_k3)

    up_l1 = conv_transp(30, 2, 2, LReLU, apply_instnorm=False)(conv_l1_k4)
    conv_l0_k2 = conv(30, 5, 1, LReLU, apply_instnorm=False)(tf.concat([conv_l0_k1, up_l1], axis=3))
    conv_l0_k3 = conv(30, 5, 1, LReLU, apply_instnorm=False)(conv_l0_k2)

    out = conv(3, 1, 1, None, apply_instnorm=False)(conv_l0_k3)
    out = tf.clip_by_value(out, 0.0, 1.0)

    return tf.keras.Model(inputs=[inputs,snr,otf_1x,ew_1x,otf_2x,ew_2x,otf_4x,ew_4x], 
                          outputs=[out, out])
