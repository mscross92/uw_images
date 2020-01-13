import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys
import json


types = ['2','3','4','5','6','7','8','9','10','11','12'] # included images


class TurbidPatches(data.Dataset):

    def __init__(self, train=True, transform=None, download=False):
        self.train = train
        self.transform = transform

    def read_image_file(self, data_dir,val_set,tr=True):
        """Return a Tensor containing the patches
        """
        # read validation patch labels
        val_p = np.loadtxt(val_set,delimiter=',')

        patches = []
        labels = []
        counter = 0
        for nn in types:
            # load patch sprite for included image type
            sequence_path = os.path.join(data_dir, '/',str(nn))+'.png'
            print(sequence_path)
            image = cv2.imread(sequence_path, 0)
            h, w = image.shape

            # iterate through all patches in sprite
            n_patches = int(h / w)
            for i in range(n_patches):
                # add relevant patches to dataset
                if tr:
                    if i not in val_p:
                        patch = image[i * (w): (i + 1) * (w), 0:w]
                        patch = cv2.resize(patch, (32, 32))
                        patch = np.array(patch, dtype=np.uint8)
                        patches.append(patch)
                        labels.append(i+counter)
                else:
                    if i in val_p:
                        patch = image[i * (w): (i + 1) * (w), 0:w]
                        patch = cv2.resize(patch, (32, 32))
                        patch = np.array(patch, dtype=np.uint8)
                        patches.append(patch)
                        labels.append(i+counter)

            counter += n_patches
                
        print(counter)
        return torch.ByteTensor(np.array(patches, dtype=np.uint8)), torch.LongTensor(labels)


if __name__ == '__main__':
    # need to be specified
    try:
        path_to_patches_dir = sys.argv[1]
        output_dir  = sys.argv[2]
        val_set_path  = sys.argv[3]
    except:
        print("Wrong input format.")
        sys.exit(1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tPatches = TurbidPatches()

    images, labels = tPatches.read_image_file(path_to_patches_dir,val_set_path,True)
    trr = 'train'
    with open(os.path.join(output_dir, trr + '.pt'), 'wb') as f:
        torch.save((images, labels), f)
    print('Data saved (set =',trr,')')

    images, labels = tPatches.read_image_file(path_to_patches_dir,val_set_path,False)
    trr = 'val'
    with open(os.path.join(output_dir, trr + '.pt'), 'wb') as f:
        torch.save((images, labels), f)
    print('Data saved (set =',trr,')')

