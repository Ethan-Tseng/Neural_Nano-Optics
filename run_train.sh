# Data loading
TRAIN_DIR=./training/data/train/
TEST_DIR=./training/data/test/

# Saving and logging
SAVE_DIR=./training/ckpt/
LOG_FREQ=20
SAVE_FREQ=50
CKPT_DIR=None
MAX_TO_KEEP=2

# Loss
LOSS_MODE=L1
BATCH_WEIGHTS=1.0,1.0,1.0
NORM_LOSS_WEIGHT=1.0
P_LOSS_WEIGHT=0.1
VGG_LAYERS=block2_conv2,block3_conv2
SPATIAL_LOSS_WEIGHT=0.0

# Training
AUG_ROTATE=True

# Convolution
REAL_PSF=./experimental/data/psf/psf.npy
PSF_MODE=SIM_PSF
CONV_MODE=SIM
CONV=patch_size
DO_TAPER=True
OFFSET=True
NORMALIZE_PSF=True
THETA_BASE=0.0,5.0,10.0,15.0

# Metasurface
NUM_COEFFS=8
USE_GENERAL_PHASE=False
METASURFACE=log_asphere #zeros
S1=0.9e-3
S2=1.4e-3
ALPHA=270.176968209
TARGET_WAVELENGTH=511.0e-9
BOUND_VAL=1000.0

# Sensor
A_POISSON=0.00004
B_SQRT=0.00001
MAG=8.1 # Set so that image size is 720 x 720

# Optimization
PHASE_LR=0.005
PHASE_ITERS=0
G_LR=0.0001
G_ITERS=10
G_NETWORK=FP
SNR_OPT=False #True
SNR_INIT=3.0

python train.py --train_dir $TRAIN_DIR --test_dir $TEST_DIR --save_dir $SAVE_DIR --log_freq $LOG_FREQ --save_freq $SAVE_FREQ --ckpt_dir $CKPT_DIR --max_to_keep $MAX_TO_KEEP --loss_mode $LOSS_MODE --batch_weights $BATCH_WEIGHTS --Norm_loss_weight $NORM_LOSS_WEIGHT --P_loss_weight $P_LOSS_WEIGHT --vgg_layers $VGG_LAYERS --Spatial_loss_weight $SPATIAL_LOSS_WEIGHT --aug_rotate $AUG_ROTATE --real_psf $REAL_PSF --psf_mode $PSF_MODE --conv_mode $CONV_MODE --conv $CONV --do_taper $DO_TAPER --offset $OFFSET --normalize_psf $NORMALIZE_PSF --theta_base $THETA_BASE --num_coeffs $NUM_COEFFS --use_general_phase $USE_GENERAL_PHASE --metasurface $METASURFACE --s1 $S1 --s2 $S2 --alpha $ALPHA --target_wavelength $TARGET_WAVELENGTH --bound_val $BOUND_VAL --a_poisson $A_POISSON --b_sqrt $B_SQRT --mag $MAG --Phase_lr $PHASE_LR --Phase_iters $PHASE_ITERS --G_lr $G_LR --G_iters $G_ITERS --G_network $G_NETWORK --snr_opt $SNR_OPT --snr_init $SNR_INIT 
