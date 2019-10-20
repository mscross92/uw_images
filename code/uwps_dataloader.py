import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys
import json


class UWPS(data.Dataset):

    def __init__(self, train=True, download=False):
        self.train = train

    def read_image_file(self, data_dir):
        """Return a Tensor containing the patches
        """
        # patches = []
        patches = None
        labels = []
        counter = 0
        # hpatches_sequences = [x[1] for x in os.walk(data_dir)][0]
        # for directory in hpatches_sequences:
        directory = os.fsencode(data_dir)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".png"):
                sequence_path = os.path.join(data_dir, filename)
                image = cv2.imread(sequence_path, 0)
                h, w = image.shape
                n_patches = int(h / w)
                for i in range(n_patches):
                    patch = image[i * (w): (i + 1) * (w), 0:w]
                    patch = cv2.resize(patch, (32, 32))

                    patch = np.array(patch, dtype=np.uint8)
                    patch = torch.ByteTensor(patch).cuda()
                    if type(patches) == type(None):
                        patches = patch
                    else:
                        patches = torch.cat([patches,patch], dim=0) # concat list of tensors
                    # patches.append(patch)
                    labels.append(counter)
                # counter += n_patches
                counter += 1
        print(counter)
        return torch.ByteTensor(np.array(patches, dtype=np.uint8)), torch.LongTensor(labels)
        return patches, torch.LongTensor(labels)

if __name__ == '__main__':
    # need to be specified
    try:
        path_to_data_dir = sys.argv[1]
        output_dir  = sys.argv[2]
    except:
        print("Wrong input format.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    uw_patches = UWPS()
    images, labels = uw_patches.read_image_file(path_to_data_dir)
    with open(os.path.join(output_dir, 'train.pt'), 'wb') as f:
        torch.save((images, labels), f)
    print(len(np.array(labels)), 'patches from',len(np.unique(np.array(labels))),'sequences saved')
