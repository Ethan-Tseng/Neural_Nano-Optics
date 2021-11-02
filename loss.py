# Loss functions

import tensorflow as tf

# Per-Pixel loss
def Norm_loss(G_img, gt_img, args):
    if args.loss_mode == 'L1': metric = tf.abs
    elif args.loss_mode == 'L2': metric = tf.square
    else: assert False, ("Mode needs to be L1 or L2")

    loss = 0.0
    for i, weight in enumerate(args.batch_weights):
        loss = loss + weight * tf.reduce_mean(metric(G_img[i,:,:,:] - gt_img[0,:,:,:]))
    return loss

# Perceptual loss (VGG19)
def P_loss(G_img, gt_img, vgg_model, args):
    if args.loss_mode == 'L1': metric = tf.abs
    elif args.loss_mode == 'L2': metric = tf.square
    else: assert False, ("Mode needs to be L1 or L2")

    preprocessed_G_img  = tf.keras.applications.vgg19.preprocess_input(G_img*255.0)
    preprocessed_gt_img = tf.keras.applications.vgg19.preprocess_input(gt_img*255.0)

    G_layer_outs = vgg_model(preprocessed_G_img)
    gt_layer_outs = vgg_model(preprocessed_gt_img)

    loss = 0.0
    for i, weight in enumerate(args.batch_weights):
        loss = loss + weight * tf.add_n([tf.reduce_mean(metric( (G_layer_out[i,:,:,:] - gt_layer_out[0,:,:,:]) / 255. ))
                                for G_layer_out, gt_layer_out in zip(G_layer_outs, gt_layer_outs)])
    return loss

# Spatial gradient loss
def Spatial_loss(output_img, GT_img, args):
    if args.loss_mode == 'L1': metric = tf.abs
    elif args.loss_mode == 'L2': metric = tf.square
    else: assert False, ("Mode needs to be L1 or L2")

    def spatial_gradient(x):
        diag_down = x[:, 1:, 1:, :] - x[:, :-1, :-1, :]
        dv = x[:, 1:, :, :] - x[:, :-1, :, :]
        dh = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_up = x[:, :-1, 1:, :] - x[:, 1:, :-1, :]

        return [dh, dv, diag_down, diag_up]

    total_loss = 0.0
    for i, weight in enumerate(args.batch_weights):
        gx = spatial_gradient(output_img[i:i+1,:,:,:])
        gy = spatial_gradient(GT_img)
        loss = 0
        for xx, yy in zip(gx, gy):
            loss = loss + tf.reduce_mean(metric(xx - yy))
        total_loss = total_loss + weight * loss
    return total_loss

# Loss for the entire end-to-end imaging pipeline
def G_loss(G_img, gt_img, vgg_model, args):
    # Compute metrics
    PSNR = tf.reduce_mean(tf.image.psnr(G_img, gt_img, max_val=1.0))
    SSIM = tf.reduce_mean(tf.image.ssim(G_img, gt_img, max_val=1.0))
    metrics = {'PSNR':PSNR, 'SSIM':SSIM}

    # Compute losses
    Norm_loss_val = 0.0
    P_loss_val = 0.0
    Spatial_loss_val = 0.0
    if not args.Norm_loss_weight == 0.0:
        Norm_loss_val = args.Norm_loss_weight * Norm_loss(G_img, gt_img, args)
    if not args.P_loss_weight == 0.0:
        P_loss_val = args.P_loss_weight * P_loss(G_img, gt_img, vgg_model, args)
    if not args.Spatial_loss_weight == 0.0:
        Spatial_loss_val = args.Spatial_loss_weight * Spatial_loss(G_img, gt_img, args)
    Content_loss_val = Norm_loss_val + P_loss_val + Spatial_loss_val
    loss_components = {'Norm':Norm_loss_val, 'P':P_loss_val, 'Spatial':Spatial_loss_val}

    return Content_loss_val, loss_components, metrics
