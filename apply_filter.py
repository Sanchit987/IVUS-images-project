import pydicom
import matplotlib.pyplot as plt
import pylibjpeg
import numpy as np
import cv2
from PIL import Image
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.signal.signaltools import wiener
from skimage.filters import median
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.restoration import denoise_bilateral
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
import os



#os.chdir('IVUS Project')
files = []
for file in os.listdir("Sample Real Images"):
    if file.endswith(".dcm"):
        files.append(file)


NUMBER_OF_IMAGES = len(files)
for i in range(NUMBER_OF_IMAGES):
    IMAGE_PATH = os.path.join("Sample Real Images/", files[i])
    img = pydicom.read_file(IMAGE_PATH)
    img = np.array(img.pixel_array)

    """ Converting RGB Image to Grayscale image and saving it """
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join("IVUS Project/Grayscale Images/", files[i][:-4] + ".png"), grayscale)

    """ Applying Anisotropic Diffusion Filter and saving it """
    img_anisotropic = anisotropic_diffusion(grayscale)
    cv2.imwrite(os.path.join("IVUS Project/Anisotropic Filter Images/", files[i][:-4] + ".png"), img_anisotropic)

    """ Applying wiener filter and saving it """
    img_wiener = wiener(grayscale)
    cv2.imwrite(os.path.join("IVUS Project/Wiener Filter Images/", files[i][:-4] + ".png"), img_wiener)

    """ Applying NonLocal Means filter and saving it """
    sigma_est = np.mean(estimate_sigma(grayscale, multichannel=False))
    img_ncm = denoise_nl_means(grayscale,
                              h=1.15 * sigma_est,
                              fast_mode=False,
                              patch_size=5,
                              patch_distance=3,
                              preserve_range=True,
                              multichannel=False)
    cv2.imwrite(os.path.join("IVUS Project/Non-Local Means Filter Images/", files[i][:-4] + ".png"), img_ncm)

    """ Applying median filter and saving it """
    # Disk creates a circular structuring element, similar to a mask with specific radius
    img_median = median(grayscale, disk(3), mode='constant', cval=0.0)
    cv2.imwrite(os.path.join("IVUS Project/Median Filter Images/", files[i][:-4] + ".png"), img_median)

    """ Applying bilateral filter and saving it """
    img_bilateral = denoise_bilateral(grayscale,
                                      sigma_color=0.05,
                                      sigma_spatial=15,
                                      multichannel=False)
    # sigma_color = float - Sigma for grey or color value.
    # For large sigma_color values the filter becomes closer to gaussian blur.
    # sigma_spatial: float. Standard ev. for range distance. Increasing this smooths larger features.
    cv2.imwrite(os.path.join("IVUS Project/Bilateral Filter Images/", files[i][:-4] + ".png"), img_bilateral)

    """ Applying gaussian filter and saving it """
    img_gaussian = gaussian(grayscale, sigma=1, mode='constant', cval=0.0)
    # sigma defines the std dev of the gaussian kernel.
    cv2.imwrite(os.path.join("IVUS Project/Gaussian Filter Images/", files[i][:-4] + ".png"), img_gaussian)

    """ Applying total variance filter and saving it """
    img_denoise = denoise_tv_chambolle(grayscale, weight=0.1, eps=0.0002, n_iter_max=200, multichannel=False)
    cv2.imwrite(os.path.join("IVUS Project/Total Variance Filter Images/", files[i][:-4] + ".png"), img_denoise)


    """ Applying hybrid filters """
    # 1. Wiener + Anisotropic filter
    img_wiener_anisotropic = anisotropic_diffusion(img_wiener)
    cv2.imwrite(os.path.join("IVUS Project/Wiener + Anisotropic Filter Images/", files[i][:-4] + ".png"), img_wiener_anisotropic)

    # 2. Non-local Means filter + Median filter
    img_ncm_median = median(img_ncm, disk(3), mode='constant', cval=0.0)
    cv2.imwrite(os.path.join("IVUS Project/Non-local Means filter + Median Filter Images/", files[i][:-4] + ".png"), img_ncm_median)






