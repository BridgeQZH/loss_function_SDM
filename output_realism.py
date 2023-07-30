import math
import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

angles = [0]

def convert_to_grayscale(image):
   """Converts an image to grayscale.

   image_path -- path to the image file
   """
   # Open the image
   # image = Image.open(image_path)

   # Convert the image to grayscale
   grayscale_image = image.convert('L')

   # Calculate GLCM for each angle
   glcms = []
   for angle in angles:
       glcm = greycomatrix(grayscale_image, [1], [angle], levels=256, symmetric=True, normed=True)
       glcms.append(glcm)

   # Calculate desired GLCM properties (e.g., contrast, energy, etc.)
   properties = ['contrast', 'energy', 'homogeneity']
   glcm_features = []
   for glcm in glcms:
       features = [greycoprops(glcm, prop).ravel()[0] for prop in properties]
       glcm_features.append(features)

   return glcm_features


def calculate_gabor_energy(image, ksize, sigma, theta, lambda_val, gamma):
   # Convert the image to grayscale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
   # Normalize the grayscale image to the range [0, 1]
   gray = gray.astype(np.float32) / 255.0
  
   # Create the Gabor kernel
   kernel = cv2.getGaborKernel(ksize, sigma, theta, lambda_val, gamma)
  
   # Apply the Gabor kernel to the image
   filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
  
   # Calculate the Gabor energy
   energy = np.mean(filtered**2)
  
   return energy

def calculate_canny_edge_density(image):
   # Convert the image to grayscale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
   # Apply Canny edge detection
   edges = cv2.Canny(gray, 100, 200)  # Adjust the threshold values as per your requirement
  
   # Calculate the edge density
   height, width = edges.shape[:2]
   total_pixels = height * width
   edge_pixels = cv2.countNonZero(edges)
   edge_density = edge_pixels / total_pixels
  
   return edge_density

def decode_img_latents(latents):
  latents = 1 / 0.18215 * latents

  with torch.no_grad():
    imgs = vae.decode(latents)

  imgs = (imgs / 2 + 0.5).clamp(0, 1)
  imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
  imgs = (imgs * 255).round().astype('uint8')
  pil_images = [Image.fromarray(image) for image in imgs]
  return pil_images

def output_realism(img):

   image_np = np.array(img)

   edge_density_base_value = 0.11623
   GLCM_Contrast_base_value = 418.309
   GLCM_energy_base_value = 0.04643
   Variance_Blur_Measure_base_value = 3317.952377
   spectrum_base_value = 96.773

   # Get those values
   # image_width, image_height = img.size
   # image_format = img.format
   # image_mode = img.mode

   f2 = np.fft.fft2(img)
   fshift = np.fft.fftshift(f2)
   # Calculate magnitude spectrum (amplitude)
   magnitude_spectrum = np.abs(fshift)

   # Add a small constant to avoid zeros
   epsilon = 1e-8
   magnitude_spectrum += epsilon

   # Calculate logarithm with added constant
   log_magnitude_spectrum = 20 * np.log(magnitude_spectrum)
   # Calculate magnitude spectrum (amplitude)
   # magnitude_spectrum = 20 * np.log(np.abs(fshift))

   # Calculate statistics
   mean_spectrum = np.mean(log_magnitude_spectrum)
   edge_density = calculate_canny_edge_density(image_np)
   glcm_features = convert_to_grayscale(img)
   gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
   Variance_Blur_Measure = np.var(gray)
  
   after_edge_density = edge_density / edge_density_base_value
   after_GLCM_Contrast = glcm_features[0][0] / GLCM_Contrast_base_value

   after_GLCM_energy = glcm_features[0][1] / GLCM_energy_base_value
   after_Variance_Blur_Measure = Variance_Blur_Measure / Variance_Blur_Measure_base_value
   after_spectrum = spectrum_base_value/abs(mean_spectrum)

   after_GLCM_energy = 1.0 / after_GLCM_energy
   after_Variance_Blur_Measure = 1.0 / after_Variance_Blur_Measure
   after_spectrum = 1.0 / after_spectrum
   # After considering
   weight_edge_density = 1.0/0.637340 # Adjustable
   weight_GLCM_Contrast = 1.0/0.43478 # Adjustable
   weight_GLCM_energy = 1.0/1.03037
   weight_variance_blur_measure = 1.0/1.1126
   weight_spectrum = 1.0/1.01640
   a = after_edge_density * weight_edge_density
   b = after_GLCM_Contrast * weight_GLCM_Contrast
   c = after_GLCM_energy * weight_GLCM_energy
   d = after_Variance_Blur_Measure * weight_variance_blur_measure
   e = after_spectrum * weight_spectrum

   print(0.5*math.sin(math.pi*72/180)*(a*b+a*e+b*d+c*d+e*c))

temp_img = decode_img_latents(temp_latent)
temp_realism = output_realism(temp_img)
# output
temp_realism = 1.0 / temp_realism
