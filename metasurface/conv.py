# Convolution and Fourier operations

import tensorflow as tf
import time

def fft(img):
    img = tf.transpose(tf.cast(img, dtype = tf.complex64), perm = [0, 3, 1, 2])
    Fimg = tf.signal.fft2d(img)
    return Fimg

def ifft(Fimg):
    img = tf.cast(tf.abs(tf.signal.ifft2d(Fimg)), dtype=tf.float32)
    img = tf.transpose(img, perm = [0, 2, 3, 1])
    return img

def psf2otf(psf, h, w):
    psf = tf.image.resize_with_crop_or_pad(psf, h, w)
    psf = tf.transpose(tf.cast(psf, dtype = tf.complex64), perm = [0, 3, 1, 2])
    psf = tf.signal.fftshift(psf, axes=(2,3))
    otf = tf.signal.fft2d(psf)
    return otf

# Assume non-padded PSF as input
def get_edgetaper_weight(psf, im_h, im_w, mode='autocorrelation'):
    assert(im_h // 2 >= psf.shape[1])
    assert(im_w // 2 >= psf.shape[2])

    if mode == 'autocorrelation':
        padding_h = im_h - 1 - psf.shape[1]
        padding_w = im_w - 1 - psf.shape[2]
    
        psf_prj_h = tf.reduce_sum(psf, axis=1, keepdims=True)
        psf_prj_h = tf.transpose(psf_prj_h, perm=[0,3,1,2]) # Move dimension to inner-most location
        psf_prj_h = tf.pad(psf_prj_h, paddings=((0,0),(0,0),(0,0),(padding_h,0)),mode='constant')
        psf_prj_h = tf.square(tf.abs(tf.signal.fft(tf.cast(psf_prj_h[:,:,:,:], dtype=tf.complex64))))
        psf_prj_h = tf.math.real(tf.signal.ifft(tf.cast(psf_prj_h, dtype=tf.complex64)))
        psf_prj_h = tf.concat([psf_prj_h, psf_prj_h[:,:,:,0:1]],axis=-1)
        psf_prj_h = psf_prj_h / tf.reduce_max(psf_prj_h, axis=-1, keepdims=True)
    
        psf_prj_w = tf.reduce_sum(psf, axis=2, keepdims=True)
        psf_prj_w = tf.transpose(psf_prj_w, perm=[0,3,2,1]) # Move dimension to inner-most location
        psf_prj_w = tf.pad(psf_prj_w, paddings=((0,0),(0,0),(0,0),(padding_w,0)),mode='constant')
        psf_prj_w = tf.square(tf.abs(tf.signal.fft(tf.cast(psf_prj_w[:,:,:,:], dtype=tf.complex64))))
        psf_prj_w = tf.math.real(tf.signal.ifft(tf.cast(psf_prj_w, dtype=tf.complex64)))
        psf_prj_w = tf.concat([psf_prj_w, psf_prj_w[:,:,:,0:1]],axis=-1)
        psf_prj_w = psf_prj_w / tf.reduce_max(psf_prj_w, axis=-1, keepdims=True)
    
        psf_prj_h = tf.transpose(psf_prj_h, perm=[0,3,2,1])
        psf_prj_w = tf.transpose(psf_prj_w, perm=[0,2,3,1])
        
        weight = (1 - psf_prj_h) * (1 - psf_prj_w)
        return weight
    elif mode == 'bilinear':
        im_h = int(im.shape[1])
        im_w = int(im.shape[2])
        channels = int(im.shape[3])
        psf_h = int(psf.shape[1])
        psf_w = int(psf.shape[2])

        window_vec_h = tf.ones_like(im[0,0:im_h - psf_h//2,0,0])
        window_vec_w = tf.ones_like(im[0,0,0:im_w - psf_w//2,0])

        window_vec_h = tf.concat([tf.zeros(psf_h//4, dtype=tf.float32),
                                  tf.linspace(0.0,1.0, num=psf_h//4),
                                  window_vec_h,
                                  tf.linspace(1.0,0.0, num=psf_h//4),
                                  tf.zeros(psf_h//4, dtype=tf.float32)], axis=0)
        window_vec_w = tf.concat([tf.zeros(psf_w//4, dtype=tf.float32),
                                  tf.linspace(0.0,1.0, num=psf_w//4),
                                  window_vec_w,
                                  tf.linspace(1.0,0.0, num=psf_w//4),
                                  tf.zeros(psf_w//4, dtype=tf.float32)], axis=0)
        window_vec_h = tf.cast(window_vec_h, dtype=tf.float32)
        window_vec_w = tf.cast(window_vec_w, dtype=tf.float32)
        window_vec_h = window_vec_h[tf.newaxis,:,tf.newaxis,tf.newaxis]
        window_vec_w = window_vec_w[tf.newaxis,tf.newaxis,:,tf.newaxis]
        edgetaper_weight = window_vec_h * window_vec_w
        edgetaper_weight = tf.concat(channels * [edgetaper_weight], axis=3)
        edgetaper_weight = tf.image.resize_with_crop_or_pad(edgetaper_weight, im_h, im_w)
        return edgetaper_weight
    else:
        assert 0


# Wiener filter deconvolution with optional edgetapering
# otf - Precomputed Optical Transfer Function
# ew  - Precomputed edgetaper weight
def deconvolve_wnr(blur, snr, otf, ew, do_taper=False):
    if do_taper:
       blur_tapered = ifft(fft(blur) * otf)
       blur = ew * blur + (1 - ew) * blur_tapered
    blur_debug = blur

    wiener_filter = tf.math.conj(otf) / (tf.cast(tf.abs(otf) ** 2, tf.complex64) + tf.cast(1 / tf.abs(snr), tf.complex64))
    output = tf.cast(tf.abs(ifft(wiener_filter * fft(blur))), tf.float32)
    return output, blur_debug


# Forward pass
def convolution_tf(params, args):
    def conv_fn(image, psf):
        if args.conv_mode == 'REAL':
            return image
        assert((image.shape[1]) == params['load_width'])
        assert((image.shape[2]) == params['load_width'])
        otf  = psf2otf(psf, params['load_width'], params['load_width'])
        blur = ifft(fft(image) * otf)
        blur = tf.image.resize_with_crop_or_pad(blur, params['network_width'], params['network_width'])
        return blur
    return conv_fn


# Backwards pass
def deconvolution_tf(params, args):
    def deconv_fn(blur, psf, snr, G, training):
        h = blur.shape[1]
        w = blur.shape[2]

        # Pre-compute optical transfer function and edgetaper weights at different scales
        psf_1x = psf
        otf_1x = psf2otf(psf_1x, h   , w   )
        ew_1x  = get_edgetaper_weight(psf_1x, h   , w   )

        psf_2x = tf.image.resize(psf, [tf.constant(psf.shape[1]//2, dtype=tf.int32),
                                       tf.constant(psf.shape[2]//2, dtype=tf.int32)],
                                       method='bilinear', preserve_aspect_ratio=True)
        psf_2x = psf_2x / tf.reduce_sum(psf_2x, axis=[1,2], keepdims=True)
        otf_2x = psf2otf(psf_2x, h//2, w//2)
        ew_2x  = get_edgetaper_weight(psf_2x, h//2, w//2)
 
        psf_4x = tf.image.resize(psf, [tf.constant(psf.shape[1]//4, dtype=tf.int32),
                                       tf.constant(psf.shape[2]//4, dtype=tf.int32)],
                                       method='bilinear', preserve_aspect_ratio=True)
        psf_4x = psf_4x / tf.reduce_sum(psf_4x, axis=[1,2], keepdims=True)
        otf_4x = psf2otf(psf_4x, h//4, w//4)
        ew_4x  = get_edgetaper_weight(psf_4x, h//4, w//4)

        # Apply deconvolution algorithm and return time spent
        start = time.time()
        G_img, *G_debug = G([blur, tf.expand_dims(snr, 0), otf_1x, ew_1x, otf_2x, ew_2x, otf_4x, ew_4x], training=training)
        end = time.time()
        t = end - start
        return t, G_img, G_debug
    return deconv_fn
