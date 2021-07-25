# IVUS-images-project


1. Open terminal and type `pip install -r requirements.txt`
2. `Sample Real Images` directory contains some sample dicom images on which further preprocessing is done.
3. `apply_filter.py` will create the necessary filters by transforming images from `Sample Real Images` directory and save filters inside `IVUS Project` directory.
4. `calculate_scores.ipynb` is a jupyter notebook which will calculate evaluation metrics such as PSNR, MSE, SSIM and save it into an excel sheet `data.xlsx` and csv file `data.csv`.  


> Python 3.7