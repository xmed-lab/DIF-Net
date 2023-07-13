import os
import pickle
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    check_max = False
    max_value = 0.10
    max_list = []
    for name in os.listdir('./projections/'):
        with open(f'./projections/{name}/all.pickle', 'rb') as f:
            data = pickle.load(f)
            projs = data['projections'] # K, res^2

            if check_max:
                max_list.append(projs.max())
                print(np.mean(max_list))
            else:
                projs /= max_value
                projs *= 255
                projs = projs.astype(int).astype(np.uint8)
                angles = data['angles'] # K,

        if not check_max:
            save_head = './projections_normalized/'
            os.makedirs(save_head, exist_ok=True)
            save_path = os.path.join(save_head, f'{name}.pickle')
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'projections': projs,
                    'angles': angles
                }, f, pickle.HIGHEST_PROTOCOL)
