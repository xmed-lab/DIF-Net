import os
import json
import scipy
import numpy as np
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


def load_names():
    with open('info.json', 'r') as f:
        info = json.load(f)
        names = []
        for s in ['train', 'test', 'eval']:
            names += info[s]
        return names


def resample():
    os.makedirs('./resampled', exist_ok=True)
    ref_spacing = np.array([0.8, 0.8, 0.8])

    for name in tqdm(load_names(), ncols=50):
        a, b = name.split('-')
        path = f'/home/ylindw/workspace/dataset_zhaowei/{a}/{b}.mhd'

        itk_img = sitk.ReadImage(path)
        spacing = np.array(itk_img.GetSpacing()) * 1000
        image = sitk.GetArrayFromImage(itk_img)
        image = image.transpose(2, 1, 0)

        image = np.clip(image, a_min=-400, a_max=500)
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.float32)

        scaling = spacing / ref_spacing
        image = scipy.ndimage.zoom(
            image, 
            scaling, 
            order=3, 
            prefilter=False
        )

        image = (image * 255).astype(np.uint8)
        save_path = f'./resampled/{name}.nii.gz'
        save_nifti(image, save_path)


def crop_pad():
    os.makedirs('./processed', exist_ok=True)
    files = glob('./resampled/*.nii.gz')
    for file in tqdm(files, ncols=50):
        name = file.split('/')[-1].split('.')[0]
        image = read_nifti(file)
            
        # w, h
        if image.shape[0] > 256: # crop
            p = image.shape[0] // 2 - 128
            image = image[p:p+256, p:p+256, :]
        elif image.shape[0] < 256: # padding
            image_tmp = np.full([256, 256, image.shape[-1]], fill_value=0, dtype=np.uint8)
            p = 128 - image.shape[0] // 2
            l = image.shape[0]
            image_tmp[p:p+l, p:p+l, :] = image
            image = image_tmp

        # d
        if image.shape[-1] > 256: # crop
            p = image.shape[-1] // 2 - 128
            image = image[..., p:p+256]
        elif image.shape[-1] < 256: # padding
            image_tmp = np.full(list(image.shape[:2]) + [256], fill_value=0, dtype=np.uint8)
            p = 128 - image.shape[-1] // 2
            l = image.shape[-1]
            image_tmp[..., p:p+l] = image
            image = image_tmp

        save_path = f'./processed/{name}.nii.gz'
        save_nifti(image, save_path)

            
if __name__ == '__main__':
    resample()
    crop_pad()
        