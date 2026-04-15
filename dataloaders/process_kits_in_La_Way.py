import pathlib

import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd
import SimpleITK as sitk

output_size =[112, 112, 80]

loadPath = r"E:\4_Dataset\KiTS19\kits19\data\\"
savePath = r"E:\4_Dataset\KiTS19\KiTS19_withLAway"

def covert_h5():
    listt = glob(r"E:\4_Dataset\KiTS19\kits19\data\\*\\imaging.nii.gz")
    for item in tqdm(listt):
        name = pathlib.Path(item).parent.name
        image = sitk.ReadImage(item)
        label = sitk.ReadImage(item.replace('imaging.nii.gz', 'segmentation.nii.gz'))

        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)
        print(np.count_nonzero(label==1))
        # print(np.count_nonzero(label == 255))
        # label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        f = h5py.File(savePath+f'/{name}.h5', 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()