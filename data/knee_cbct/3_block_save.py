import os
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


def generate_blocks():
    block_list = []
    base = np.mgrid[:64, :64, :64] * 4 # 3, 64 ^ 3
    base = base.reshape(3, -1)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                offset = np.array([x, y, z])
                block = base + offset[:, None]
                block_list.append(block)
    return block_list


if __name__ == '__main__':
    os.makedirs('./blocks/', exist_ok=True)

    block_list = generate_blocks()
    blocks = np.stack(block_list, axis=0) # K, 3, N^3
    blocks = blocks.transpose(0, 2, 1).astype(float) / 255 # K, N^3, 3
    np.save('./blocks/blocks.npy', blocks)
    
    files = glob(f'processed/*.nii.gz')
    for file in tqdm(files, ncols=50):
        name = file.split('/')[-1].split('.')[0]
        data_path = f'./processed/{name}.nii.gz'
        image = read_nifti(data_path)

        save_dir = f'./blocks/{name}/'
        os.makedirs(save_dir, exist_ok=True)
        for k, block in enumerate(block_list):
            block = block.reshape(3, -1).transpose(1, 0)
            image_block = image[block[:, 0], block[:, 1], block[:, 2]]
            np.save(os.path.join(save_dir, f'block_{k}.npy'), image_block)
