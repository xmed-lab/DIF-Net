import os
import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import SimpleITK as sitk
import matplotlib.pyplot as plt

import tigre
from tigre.utilities.geometry import Geometry



def save_projections(save_dir, projections, angles):
    n_row = len(projections) // 10
    projections = projections.copy()
    projections = (projections - projections.min()) / (projections.max() - projections.min())

    for i in range(len(projections)):
        angle = int((angles[i] / np.pi) * 180)
        plt.subplot(n_row, 10, i + 1)
        plt.imshow(projections[i] * 255, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{angle}')
        plt.axis('off')

    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(save_dir, 'visualization.png'), dpi=500)



class ConeGeometry_special(Geometry):
    '''
    Cone beam CT geometry.
    '''
    def __init__(self, data):
        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data['DSD'] / 1000  # Distance Source Detector      (m)
        self.DSO = data['DSO'] / 1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data['nDetector'])         # number of pixels              (px)
        self.dDetector = np.array(data['dDetector']) / 1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector     # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data['nVoxel'][::-1])         # number of voxels              (vx)
        self.dVoxel = np.array(data['dVoxel'][::-1]) / 1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel              # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data['offOrigin'][::-1]) / 1000        # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data['offDetector'][1], data['offDetector'][0], 0]) / 1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data['accuracy']  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data['mode']          # parallel, cone                ...
        self.filter = data['filter']


def read_nifti(path):
    itk_img = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(itk_img)
    return image


def save_nifti(image, path):
    out = sitk.GetImageFromArray(image)
    sitk.WriteImage(out, path)


def generate_data(ct_path, cfg_path, save_dir, visualize=False):
    # if os.path.exists(os.path.join(save_dir, 'all.pickle')):
    #     print(' -- skip', ct_path)
    #     return

    image = read_nifti(ct_path).astype(np.float32) / 255.

    with open(cfg_path, 'r') as f:
        data = yaml.safe_load(f)
        data['image'] = image.copy()

    '''--- generate training data ---'''
    # generate training angles
    data['train'] = {
        'angles': np.linspace(
            0, 
            data['totalAngle'] / 180 * np.pi, 
            data['numTrain'] + 1
        )[:-1] + data['startAngle'] / 180 * np.pi
    }
    
    # generate projections
    geo = ConeGeometry_special(data)
    data['train']['projections'] = tigre.Ax(
        image.transpose(2, 1, 0).copy(),
        geo, 
        data['train']['angles']
    )[:, ::-1, :]

    '''--- save data ---'''
    if visualize:
        save_projections(save_dir, data['train']['projections'], data['train']['angles'])

    save_path = os.path.join(save_dir, 'all.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(data['train'], f, pickle.HIGHEST_PROTOCOL)

    print(f' -- save processed file to {save_path}')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser(description='project')
    parser.add_argument('-n', '--name', type=str, default='FL-140400.nii.gz')
    args = parser.parse_args()
    
    cfg_path = os.path.join('./config.yaml')
    name = args.name.replace('.nii.gz', '')
    ct_path = f'./processed/{name}.nii.gz'
    save_dir = os.path.join(f'./projections/{name}')
    os.makedirs(save_dir, exist_ok=True)
    generate_data(ct_path, cfg_path, save_dir, visualize=True)
