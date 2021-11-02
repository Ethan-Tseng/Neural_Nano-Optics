import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np

# Make the phase function for even asphere polynomials.
#num_coeffs = 8
#def phase_func(x, a2, a4, a6, a8, a10, a12, a14, a16):
#  return a2 * x ** 2 + a4 * x ** 4 + a6 * x ** 6 + a8 * x ** 8 + a10 * x ** 10 + a12 * x ** 12 + a14 * x ** 14 + a16 * x ** 16
def make_phase_func(num_coeffs):
  func_str = 'def phase_func(x,'
  for i in range(num_coeffs):
    func_str = func_str + 'a' + str(2*(i+1)) 
    if i < num_coeffs - 1:
      func_str = func_str + ','
  func_str = func_str + '): return '
  for i in range(num_coeffs):
    func_str = func_str + 'a' + str(2*(i+1)) + '*x**' + str(2*(i+1))
    if i < num_coeffs - 1:
      func_str = func_str + ' + '
  ldict = {}
  print(func_str)
  exec(func_str, globals(), ldict)
  return ldict['phase_func']

# Initializes parameters used in the simulation and optimization.
def initialize_params(args):

  theta_base = args.theta_base
  phi_base = 0.0 # Phi angle for full field simulation. Currently unused.

  # Define the `params` dictionary.
  params = dict({})

  # Number of optimizable phase coefficients
  params['num_coeffs'] = args.num_coeffs
  params['phase_func'] = make_phase_func(params['num_coeffs'])

  # Units and tensor dimensions.
  params['nanometers'] = 1E-9
  params['degrees']    = np.pi / 180

  # Upsampling for Fourier optics propagation
  params['upsample']      = 1
  params['normalize_psf'] = args.normalize_psf

  # Sensor parameters
  params['magnification'] = args.mag       # Image magnification
  params['sensor_pixel']  = 5.86E-6        # Meters
  params['sensor_height'] = 1216           # Sensor pixels
  params['sensor_width']  = 1936           # Sensor pixels
  params['a_poisson']     = args.a_poisson # Poisson noise component
  params['b_sqrt']        = args.b_sqrt    # Gaussian noise standard deviation

  # Focal length
  params['f'] = 1E-3

  # Tensor shape parameters and upsampling.
  lambda_base = [606.0, 511.0, 462.0]
  params['lambda_base'] = lambda_base # Screen wavelength
  params['lambda_base_weights'] = np.array([1.0, 1.0, 1.0])
  params['theta_base'] = theta_base
  params['phi_base'] = [0.0]

  # PSF grid shape.
  # dim is set to work with the offset PSF training scheme
  if args.offset:
      dim = np.int(2 * (np.size(params['theta_base']) - 1) - 1)
  else:
      dim = 5  # <-- TODO: Hack to get image size to be 720 x 720
  psfs_grid_shape = [dim, dim]
  params['psfs_grid_shape'] = psfs_grid_shape

  # Square input image width based on max field angle (20 degrees)
  image_width = params['f'] * np.tan(20.0 * np.pi / 180.0) * np.sqrt(2)
  image_width = image_width * params['magnification'] / params['sensor_pixel']
  params['image_width'] = np.int(2*dim * np.ceil(image_width / (2*dim) ))

  if args.conv == 'patch_size':
      # Patch sized image for training efficiency
      params['psf_width'] = (params['image_width'] // dim)
      assert(params['psf_width'] % 2 == 0)
      params['hw'] = (params['psf_width']) // 2
      params['load_width'] = (params['image_width'] // params['psfs_grid_shape'][0]) + 2*params['psf_width']
      params['network_width'] = (params['image_width'] // params['psfs_grid_shape'][0]) + params['psf_width']
      params['out_width'] = (params['image_width'] // params['psfs_grid_shape'][0])
  elif args.conv == 'full_size':
      # Full size image for inference
      params['psf_width'] = (params['image_width'] // 2)
      print(params['psf_width'])
      assert(params['psf_width'] % 2 == 0)
      params['hw'] = (params['psf_width']) // 2
      params['load_width'] = params['image_width'] + 2*params['psf_width']
      params['network_width'] = params['image_width'] + params['psf_width']
      params['out_width'] = params['image_width']
  else:
    assert 0
      
  print('Image width: {}'.format(params['image_width']))
  print('PSF width: {}'.format(params['psf_width']))
  print('Load width: {}'.format(params['load_width']))
  print('Network width: {}'.format(params['network_width']))
  print('Out width: {}'.format(params['out_width']))

  params['batchSize'] = np.size(lambda_base) * np.size(theta_base) * np.size(phi_base)
  batchSize = params['batchSize']
  fwhm = np.array([35.0, 34.0, 21.0])  # Screen fwhm
  params['sigma'] = fwhm / 2.355
  calib_fwhm = np.array([15.0, 30.0, 14.0])  # Calibration fwhm
  params['calib_sigma'] = calib_fwhm / 2.355
  num_pixels = 1429  # Needed for 0.5 mm diameter aperture
  params['pixels_aperture'] = num_pixels
  pixelsX = num_pixels
  pixelsY = num_pixels
  params['pixelsX'] = pixelsX
  params['pixelsY'] = pixelsY
  params['upsample'] = 1

  # Simulation grid.
  params['wavelength_nominal'] = 452E-9
  params['pitch'] = 350E-9
  params['Lx'] = 1 * params['pitch']
  params['Ly'] = params['Lx']
  dx = params['Lx'] # grid resolution along x
  dy = params['Ly'] # grid resolution along x
  xa = np.linspace(0, pixelsX - 1, pixelsX) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, pixelsY - 1, pixelsY) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya, xa)
  params['x_mesh'] = x_mesh
  params['y_mesh'] = y_mesh

  # Wavelengths and field angles.
  lam0 = params['nanometers'] * tf.convert_to_tensor(np.repeat(lambda_base, np.size(theta_base) * np.size(phi_base)), dtype = tf.float32)
  lam0 = lam0[:, tf.newaxis, tf.newaxis]
  lam0 = tf.tile(lam0, multiples = (1, pixelsX, pixelsY))
  params['lam0'] = lam0

  theta = params['degrees'] * tf.convert_to_tensor(np.tile(theta_base, np.size(lambda_base) * np.size(phi_base)), dtype = tf.float32)
  theta = theta[:, tf.newaxis, tf.newaxis]
  theta = tf.tile(theta, multiples = (1, pixelsX, pixelsY))
  params['theta'] = theta

  phi = np.repeat(phi_base, np.size(theta_base))
  phi = np.tile(phi, np.size(lambda_base))
  phi = params['degrees'] * tf.convert_to_tensor(phi, dtype = tf.float32)
  phi = phi[:, tf.newaxis, tf.newaxis]
  phi = tf.tile(phi, multiples = (1, pixelsX, pixelsY))
  params['phi'] = phi

  # Propagation parameters.
  params['propagator'] = make_propagator(params)
  params['input'] = define_input_fields(params)

  # Metasurface proxy phase model.
  params['phase_to_structure_coeffs'] = [-0.1484, 0.6809, 0.2923]
  params['structure_to_phase_coeffs'] = [6.051, -0.02033, 2.26, 1.371E-5, -0.002947, 0.797]
  params['use_proxy_phase'] = True

  # Compute the PSFs on the full field grid without exploiting azimuthal symmetry.
  params['full_field'] = False  # Not currently used

  # Use a predefined phase pattern (cubic, log-asphere, shifted axicon)
  params['use_general_phase'] = args.use_general_phase

  # Manufacturing considerations.
  params['fab_tolerancing'] = False #True
  params['fab_error_global'] = 0.03 # +/- 6% duty cycle variation globally (2*sigma)
  params['fab_error_local'] = 0.015 # +/- 3% duty cycle variation locally (2*sigma)

  return params

# Makes the recicrpocal space propagator to use for the specified input conditions.
def make_propagator(params):
  
  batchSize = params['batchSize']
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  upsample = params['upsample']

  # Propagator definition.
  k = 2 * np.pi / params['lam0'][:, 0, 0]
  k = k[:, np.newaxis, np.newaxis]
  samp = params['upsample'] * pixelsX
  k = tf.tile(k, multiples = (1, 2 * samp - 1, 2 * samp - 1))
  k = tf.cast(k, dtype = tf.complex64)  
  k_xlist_pos = 2 * np.pi * np.linspace(0, 1 / (2 *  params['Lx'] / params['upsample']), samp)  
  front = k_xlist_pos[-(samp - 1):]
  front = -front[::-1]
  k_xlist = np.hstack((front, k_xlist_pos))
  k_x = np.kron(k_xlist, np.ones((2 * samp - 1, 1)))
  k_x = k_x[np.newaxis, :, :]
  k_y = np.transpose(k_x, axes = [0, 2, 1])
  k_x = tf.convert_to_tensor(k_x, dtype = tf.complex64)
  k_x = tf.tile(k_x, multiples = (batchSize, 1, 1))
  k_y = tf.convert_to_tensor(k_y, dtype = tf.complex64)
  k_y = tf.tile(k_y, multiples = (batchSize, 1, 1))
  k_z_arg = tf.square(k) - (tf.square(k_x) + tf.square(k_y))
  k_z = tf.sqrt(k_z_arg)

  # Find shift amount
  theta = params['theta'][:, 0, 0]
  theta = theta[:, np.newaxis, np.newaxis]
  y0 = np.tan(theta) * params['f']
  y0 = tf.tile(y0, multiples = (1, 2 * samp - 1, 2 * samp - 1))
  y0 = tf.cast(y0, dtype = tf.complex64)

  phi = params['phi'][:, 0, 0]
  phi = phi[:, np.newaxis, np.newaxis]
  x0 = np.tan(phi) * params['f']
  x0 = tf.tile(x0, multiples = (1, 2 * samp - 1, 2 * samp - 1))
  x0 = tf.cast(x0, dtype = tf.complex64)

  propagator_arg = 1j * (k_z * params['f'] + k_x * x0 + k_y * y0)
  propagator = tf.exp(propagator_arg)

  return propagator

# Propagate the specified fields to the sensor plane.
def propagate(field, params):
  # Field has dimensions of (batchSize, pixelsX, pixelsY)
  # Each element corresponds to the zero order planewave component on the output
  propagator = params['propagator']

  # Zero pad `field` to be a stack of 2n-1 x 2n-1 matrices
  # Put batch parameter last for padding then transpose back
  _, _, m = field.shape
  n = params['upsample'] * m
  field = tf.transpose(field, perm = [1, 2, 0])
  field_real = tf.math.real(field)
  field_imag = tf.math.imag(field)
  field_real = tf.image.resize(field_real, [n, n], method = 'nearest')
  field_imag = tf.image.resize(field_imag, [n, n], method = 'nearest')
  field = tf.cast(field_real, dtype = tf.complex64) + 1j * tf.cast(field_imag, dtype = tf.complex64)
  field = tf.image.resize_with_crop_or_pad(field, 2 * n - 1, 2 * n - 1)
  field = tf.transpose(field, perm = [2, 0, 1])

  field_freq = tf.signal.fftshift(tf.signal.fft2d(field), axes = (1, 2))
  field_filtered = tf.signal.ifftshift(field_freq * propagator, axes = (1, 2))
  out = tf.signal.ifft2d(field_filtered)

  # Crop back down to n x n matrices
  out = tf.transpose(out, perm = [1, 2, 0])
  out = tf.image.resize_with_crop_or_pad(out, n, n)
  out = tf.transpose(out, perm = [2, 0, 1])
  return out

# Defines the input electric fields for the given wavelengths and field angles.
def define_input_fields(params):

  # Define the cartesian cross section
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  dx = params['Lx'] # grid resolution along x
  dy = params['Ly'] # grid resolution along y
  xa = np.linspace(0, pixelsX - 1, pixelsX) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, pixelsY - 1, pixelsY) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya, xa)
  x_mesh = x_mesh[np.newaxis, :, :]
  y_mesh = y_mesh[np.newaxis, :, :]
  lam_phase_test = params['lam0'][:, 0, 0]
  lam_phase_test = lam_phase_test[:, tf.newaxis, tf.newaxis]
  theta_phase_test = params['theta'][:, 0, 0]
  theta_phase_test = theta_phase_test[:, tf.newaxis, tf.newaxis]
  phi_phase_test = params['phi'][:, 0, 0]
  phi_phase_test = phi_phase_test[:, tf.newaxis, tf.newaxis]
  phase_def = 2 * np.pi  / lam_phase_test * (np.sin(theta_phase_test) * x_mesh + np.sin(phi_phase_test) * y_mesh)
  phase_def = tf.cast(phase_def, dtype = tf.complex64)

  return tf.exp(1j * phase_def)

# Generates a phase distribution modelling a metasurface given some phase coefficients.
def metasurface_phase_generator(phase_coeffs, params):
  x_mesh = params['x_mesh']
  y_mesh = params['y_mesh']
  x_mesh = x_mesh[np.newaxis, :, :]
  y_mesh = y_mesh[np.newaxis, :, :]
  phase_def = tf.zeros(shape = np.shape(x_mesh), dtype = tf.float32)
  r_phase = np.sqrt(x_mesh ** 2 + y_mesh ** 2) / (params['pixels_aperture'] * params['Lx'] / 2.0)
  if params['use_general_phase'] == True:
    phase_def = params['general_phase']
  else:
    for j in range(np.size(phase_coeffs.numpy())):
      power = tf.constant(2 * (j + 1), dtype =  tf.float32)
      r_power = tf.math.pow(r_phase, power)
      phase_def = phase_def + phase_coeffs[j] * r_power
  
  phase_def = tf.math.floormod(phase_def, 2 * np.pi)
  if params['use_proxy_phase'] == True:
    # Determine the duty cycle distribution first.
    duty = duty_cycle_from_phase(phase_def, params)

    # Accounts for global and local process variations in grating duty cycle.
    if params['fab_tolerancing'] == True:
      global_error = tf.random.normal(shape = [1], mean = 0.0, stddev = params['fab_error_global'], dtype = tf.float32)
      local_error = tf.random.normal(shape = tf.shape(duty), mean = 0.0, stddev = params['fab_error_local'], dtype = tf.float32)
      duty = duty + global_error + local_error

      # Duty cycle is fit to this range and querying outside is not physically meaningful so we need to clip it.
      duty = tf.clip_by_value(duty, clip_value_min = 0.3, clip_value_max = 0.82)

    phase_def = phase_from_duty_and_lambda(duty, params)
  else:
    phase_def = phase_def * params['wavelength_nominal'] / params['lam0']

  mask = ((x_mesh ** 2 + y_mesh ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)
  phase_def = phase_def * mask
  return phase_def

# Calculates the required duty cycle distribution at the nominal wavelength given
# a specified phase function using a pre-fit polynomial proxy for the mapping.
def duty_cycle_from_phase(phase, params):
  phase = phase / (2 * np.pi)
  p = params['phase_to_structure_coeffs']
  return p[0] * phase ** 2 + p[1] * phase + p[2]

# Calculates the phase shift for a distribution of diameters at all the desired
# simulation wavelengths using a pre-fit polynomial proxy for the mapping.
def phase_from_duty_and_lambda(duty, params):
  p = params['structure_to_phase_coeffs']
  lam = params['lam0'] / params['nanometers']
  phase = p[0] + p[1]*lam + p[2]*duty + p[3]*lam**2 + p[4]*lam*duty + p[5]*duty**2
  return phase * 2 * np.pi

# Finds the intensity at the sensor given the input fields.
def compute_intensity_at_sensor(field, params):
  coherent_psf = propagate(params['input'] * field, params)
  return tf.math.abs(coherent_psf) ** 2

# Determines the PSF from the intensity at the sensor, accounting for image magnification.
def calculate_psf(intensity, params):
  # Transpose for subsequent reshaping
  intensity = tf.transpose(intensity, perm = [1, 2, 0])
  aperture = params['pixels_aperture']
  sensor_pixel = params['sensor_pixel']
  magnification = params['magnification']
  period = params['Lx']

  # Determine PSF shape after optical magnification
  mag_width = int(np.round(aperture * period * magnification / sensor_pixel))
  mag_intensity = tf.image.resize(intensity, [mag_width, mag_width], method='bilinear') # Sample onto sensor pixels

  # Maintain same energy as before optical magnification
  denom = tf.math.reduce_sum(mag_intensity, axis = [0, 1], keepdims = False)
  denom = denom[tf.newaxis, tf.newaxis, :]
  mag_intensity = mag_intensity * tf.math.reduce_sum(intensity, axis = [0, 1], keepdims = True) / denom

  # Crop to sensor dimensions
  sensor_psf = mag_intensity
  #sensor_psf = tf.image.resize_with_crop_or_pad(sensor_psf, params['sensor_height'], params['sensor_width'])
  sensor_psf = tf.transpose(sensor_psf, perm = [2, 0, 1])
  sensor_psf = tf.clip_by_value(sensor_psf, 0.0, 1.0)
  return sensor_psf

# Defines a metasurface, including phase and amplitude variation.
def define_metasurface(phase_var, params):
  phase_def = metasurface_phase_generator(phase_var, params)  
  phase_def = tf.cast(phase_def, dtype = tf.complex64)
  amp = ((params['x_mesh'] ** 2 + params['y_mesh'] ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)
  I = 1.0 / np.sum(amp)
  E_amp = np.sqrt(I)
  return amp * E_amp * tf.exp(1j * phase_def)

# Shifts the raw PSF to be centered, cropped to the patch size, and stacked
# along the channels dimension
def shift_and_segment_psf(psf, params):
  # Calculate the shift amounts for each PSF.
  b, h, w = psf.shape
  shifted_psf = psf

  # Reshape the PSFs based on the color channel.
  psf_channels_shape = (params['batchSize'] // (np.size(params['theta_base']) * np.size(params['phi_base'])), 
                        np.size(params['theta_base']) * np.size(params['phi_base']) ,
                        h, w)
  shifted_psf_c_channels = tf.reshape(shifted_psf, shape = psf_channels_shape)
  shifted_psf_c_channels = tf.transpose(shifted_psf_c_channels, perm = (1, 2, 3, 0))

  samples = np.size(params['lambda_base']) // 3
  for j in range(np.size(params['theta_base']) * np.size(params['phi_base'])):
    psfs_j = shifted_psf_c_channels[j, :, :, :]
    for k in range(3):
      psfs_jk = psfs_j[:, :, k * samples : (k + 1) * samples]
      psfs_jk_avg = tf.math.reduce_sum(psfs_jk, axis = 2, keepdims = False)
      psfs_jk_avg = psfs_jk_avg[:, :, tf.newaxis]
      if k == 0:
        psfs_channels = psfs_jk_avg
      else:
        psfs_channels = tf.concat([psfs_channels, psfs_jk_avg], axis = 2)
    
    psfs_channels_expanded = psfs_channels[tf.newaxis, :, :, :]
    if j == 0:
      psfs_thetas_channels = psfs_channels_expanded
    else:
      psfs_thetas_channels = tf.concat([psfs_thetas_channels, psfs_channels_expanded], axis = 0)

  psfs_thetas_channels = psfs_thetas_channels[:, h // 2 - params['hw'] : h // 2 + params['hw'],
                                                 w // 2 - params['hw'] : w // 2 + params['hw'], :]

  # Normalize to unit power per channel since multiple wavelengths are now combined into each channel
  if params['normalize_psf']:
      psfs_thetas_channels_sum = tf.math.reduce_sum(psfs_thetas_channels, axis = (1, 2), keepdims = True)
      psfs_thetas_channels = psfs_thetas_channels / psfs_thetas_channels_sum
  return psfs_thetas_channels


# Rotate PSF (non-SVOLA)
def rotate_psfs(psf, params, rotate=True):
  #psfs_grid_shape = params['psfs_grid_shape']
  #rotations = np.zeros(np.prod(psfs_grid_shape))
  psfs = shift_and_segment_psf(psf, params)
  rot_angle = 0.0
  if rotate:
    angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], dtype=np.float32)
    rot_angle = (np.random.choice(angles) * np.pi / 180.0).astype(np.float32)
  rot_angles = tf.fill([np.size(params['theta_base']) * np.size(params['phi_base'])], rot_angle)
  psfs_rot = tfa.image.rotate(psfs, angles = rot_angles, interpolation = 'NEAREST')
  return psfs_rot

# PSF patches are determined by rotating them into the different patch regions
# for subsequent SVOLA convolution.
def rotate_psf_patches(psf, params):
  psfs_grid_shape = params['psfs_grid_shape']
  rotations = np.zeros(np.prod(psfs_grid_shape))
  psfs = shift_and_segment_psf(psf, params)

  # Iterate through all positions in the PSF grid.
  mid_y = (psfs_grid_shape[0] - 1) // 2
  mid_x = (psfs_grid_shape[1] - 1) // 2
  for i in range(psfs_grid_shape[0]):
    for j in range(psfs_grid_shape[1]):
      r_idx = i - mid_y
      c_idx = j - mid_x

      if params['full_field'] == True:
        index = psfs_grid_shape[0] * j + i
        psf_ij = psfs[index, :, :, :]
      else:
        # Calculate the required rotation angle.
        rotations[i * psfs_grid_shape[0] + j] = np.arctan2(-r_idx, c_idx) + np.pi / 2

        # Set the PSF based on the normalized radial distance. 
        psf_ij = psfs[max(abs(r_idx), abs(c_idx)),:,:,:]
        
      psf_ij = psf_ij[tf.newaxis, :, :, :]

      if (i == 0 and j == 0):
        psf_patches = psf_ij
      else:
        psf_patches = tf.concat([psf_patches, psf_ij], axis = 0)
  
  # Apply the rotations as a batch operation.
  psf_patches = tfa.image.rotate(psf_patches, angles = rotations, interpolation = 'NEAREST')
  return psf_patches


def get_psfs(phase_var, params, conv_mode, aug_rotate):
  metasurface_mask = define_metasurface(phase_var, params)
  intensity = compute_intensity_at_sensor(metasurface_mask, params)
  psf = calculate_psf(intensity, params)
  psfs_single = rotate_psfs(psf, params, rotate=False)
  psfs_conv = rotate_psfs(psf, params, rotate=aug_rotate)
  return psfs_single, psfs_conv


# Applies Poisson noise and adds Gaussian noise.
def sensor_noise(input_layer, params, clip = (1E-20,1.)):

  # Apply Poisson noise.
  if (params['a_poisson'] > 0):
      a_poisson_tf = tf.constant(params['a_poisson'], dtype = tf.float32)

      input_layer = tf.clip_by_value(input_layer, clip[0], 100.0)
      p = tfp.distributions.Poisson(rate = input_layer / a_poisson_tf, validate_args = True)
      sampled = tfp.monte_carlo.expectation(f = lambda x: x, samples = p.sample(1), log_prob = p.log_prob, use_reparameterization = False)
      output = sampled * a_poisson_tf
  else:
      output = input_layer

  # Add Gaussian readout noise.
  gauss_noise = tf.random.normal(shape=tf.shape(output), mean = 0.0, stddev = params['b_sqrt'], dtype = tf.float32)
  output = output + gauss_noise

  # Clipping.
  output = tf.clip_by_value(output, clip[0], clip[1])
  return output


# Samples wavelengths from a random normal distribution centered about the peak
# wavelengths in the spectra based on the FWHM of each peak.
def randomize_wavelengths(params, lambda_base, sigma):
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  thetas = params['theta_base']
  phis = params['phi_base']
  lambdas = np.random.normal(lambda_base, sigma)
  lam0 = params['nanometers'] * tf.convert_to_tensor(np.repeat(lambdas, np.size(thetas) * np.size(phis)), dtype = tf.float32)
  lam0 = lam0[:, tf.newaxis, tf.newaxis]
  lam0 = tf.tile(lam0, multiples = (1, pixelsX, pixelsY))
  params['lam0'] = lam0

# Reset wavelengths back to nominal wavelength
def set_wavelengths(params, lambda_base):
  lam0 = params['nanometers'] * tf.convert_to_tensor(np.repeat(lambda_base, np.size(params['theta_base']) * np.size(params['phi_base'])), dtype = tf.float32)
  lam0 = lam0[:, tf.newaxis, tf.newaxis]
  lam0 = tf.tile(lam0, multiples = (1, params['pixelsX'], params['pixelsY']))
  params['lam0'] = lam0


## General Phase Functions ##

# Calculates the phase for a log-asphere based on the s1 and s2 parameters.
def log_asphere_phase(s1, s2, params):
  x_mesh = params['x_mesh']
  y_mesh = params['y_mesh']
  x_mesh = x_mesh[np.newaxis, :, :]
  y_mesh = y_mesh[np.newaxis, :, :]
  r_phase = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
  R = params['pixels_aperture'] * params['Lx'] / 2.0  # Aperture radius
  quo = (s2 - s1) / R ** 2
  quo_large = s1 + quo * r_phase**2
  term1 = np.pi / params['wavelength_nominal'] / quo
  term2 = np.log(2 * quo * (np.sqrt(r_phase**2 + quo_large**2) + quo_large) + 1) - np.log(4*quo*s1 + 1)
  phase_def = -term1 * term2
  phase_def = tf.convert_to_tensor(phase_def, dtype = tf.float32)
  mask = ((x_mesh ** 2 + y_mesh ** 2) < R ** 2)
  phase_def = phase_def * mask
  return phase_def


# Calculates the phase for a shifted axicon based on the s1 and s2 parameters.
def shifted_axicon_phase(s1, s2, params):
  x_mesh = params['x_mesh']
  y_mesh = params['y_mesh']
  x_mesh = x_mesh[np.newaxis, :, :]
  y_mesh = y_mesh[np.newaxis, :, :]
  r_phase = np.sqrt(x_mesh ** 2 + y_mesh ** 2)
  R = params['pixels_aperture'] * params['Lx'] / 2.0  # Aperture radius
  samples = 1 * params['pixels_aperture']
  dr = R / samples
  phase_def = np.zeros((1, params['pixels_aperture'], params['pixels_aperture']))
  for j in range(params['pixels_aperture']):
    for k in range(params['pixels_aperture']):
      r_max = r_phase[0, j, k]
      if r_max < R:
        if j <= params['pixels_aperture'] // 2 and k <= params['pixels_aperture'] // 2:
          r_vector = np.linspace(0, r_max, np.int(samples * r_max / R))
          numerator = r_vector * dr
          denominator = np.sqrt(r_vector ** 2 + (s1 + (s2 - s1) * r_vector / R) ** 2)
          integrand = numerator / denominator
          phase_def[0, j, k] = np.sum(integrand)
        else: # Copy the previously computed result
          phase_def[0, j, k] = phase_def[0, min(j, params['pixels_aperture'] - j - 1), \
                                            min(k, params['pixels_aperture'] - k - 1)]
  phase_def = -2 * np.pi / params['wavelength_nominal'] * phase_def
  phase_def = tf.convert_to_tensor(phase_def, dtype = tf.float32)
  mask = ((x_mesh ** 2 + y_mesh ** 2) < R ** 2)
  phase_def = phase_def * mask
  return phase_def


# Calculates the phase for a cubic phase mask with a hyperboloidal lens term assuming focusing
# for the specified wavelength.
def cubic_phase(alpha, wavelength, params):
  x_mesh = params['x_mesh']
  y_mesh = params['y_mesh']
  x_mesh = x_mesh[np.newaxis, :, :]
  y_mesh = y_mesh[np.newaxis, :, :]

  # As we intend for the focusing term to be for the provided 'wavelength' parameter, we need to scale
  # the focal length because we are effectively designing an intentionally defocused lens at the 
  # nominal wavelength. Output phase from this function needs to be the phase at the nominal wavelength.
  f = params['f'] * wavelength / params['wavelength_nominal']

  R = params['pixels_aperture'] * params['Lx'] / 2.0  # Aperture radius
  focusing_term = 2 * np.pi / params['wavelength_nominal'] * (f - np.sqrt(x_mesh ** 2 + y_mesh ** 2 + f ** 2))
  edof_term = alpha / R ** 3 * (x_mesh ** 3 + y_mesh ** 3)
  phase_def = focusing_term + edof_term
  phase_def = tf.convert_to_tensor(phase_def, dtype = tf.float32)
  mask = ((x_mesh ** 2 + y_mesh ** 2) < R ** 2)
  phase_def = phase_def * mask
  return phase_def


# Calculates the phase for a hyperboloidal lens term assuming focusing for the specified wavelength.
def hyperboidal_phase(wavelength, params):
  x_mesh = params['x_mesh']
  y_mesh = params['y_mesh']
  x_mesh = x_mesh[np.newaxis, :, :]
  y_mesh = y_mesh[np.newaxis, :, :]

  # As we intend for the focusing term to be for the provided 'wavelength' parameter, we need to scale
  # the focal length because we are effectively designing an intentionally defocused lens at the 
  # nominal wavelength. Output phase from this function needs to be the phase at the nominal wavelength.
  f = params['f'] * wavelength / params['wavelength_nominal']

  R = params['pixels_aperture'] * params['Lx'] / 2.0  # Aperture radius
  phase_def = 2 * np.pi / params['wavelength_nominal'] * (f - np.sqrt(x_mesh ** 2 + y_mesh ** 2 + f ** 2))
  phase_def = tf.convert_to_tensor(phase_def, dtype = tf.float32)
  mask = ((x_mesh ** 2 + y_mesh ** 2) < R ** 2)
  phase_def = phase_def * mask
  return phase_def
