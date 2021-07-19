import pydicom
import matplotlib.pyplot as plt
import pylibjpeg
import numpy as np
import cv2
from PIL import Image
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.signal.signaltools import wiener
import os


os.chdir('IVUS Project')
files = []
for file in os.listdir("Sample Real Images"):
    if file.endswith(".dcm"):
        files.append(file)

NUMBER_OF_IMAGES = len(files)
for i in range(NUMBER_OF_IMAGES):
    IMAGE_PATH = os.path.join("Sample Real Images/", files[i])
    img = pydicom.read_file(IMAGE_PATH)
    img = np.array(img.pixel_array)

    # Converting RGB Image to Grayscale image and saving it
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join("Grayscale Images/", files[i][:-4] + ".png"), grayscale)

    # Applying Anisotropic Diffusion Filter and saving it
    img_c = anisotropic_diffusion(grayscale)
    cv2.imwrite(os.path.join("Anisotropic Filter Images/", files[i][:-4] + ".png"), img_c)

    # Applying wiener filter and saving it
    img_d = wiener(grayscale)
    cv2.imwrite(os.path.join("Wiener Filter Images/", files[i][:-4] + ".png"), img_d)




