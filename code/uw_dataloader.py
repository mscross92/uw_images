import os
import numpy as np
import torch
import torch.utils.data as data
import cv2
import sys
import json
from scipy.stats import truncnorm
import torchvision.transforms as transforms
import random

def get_truncated_normal(mean=0, sd=25, low=-65, upp=65):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class TURBID(data.Dataset):

    def __init__(self, train=True, transform=None, download=False):
        self.train = train
        self.transform = transform

    def read_image_file(self, data_dir, n_images):
       
        """Reads images and extracts training patches, performing offline augmentation
        """
        ptchs = None
        labels = []
        counter = 0

        for nn in range(1,n_images+1):
            imgs = []

            # load image
            pth = data_dir + '/' + str(nn) +'_gray.png'
            print(pth)
            gray = cv2.imread(pth,0)
            imgs.append(gray)
            
            (h,w) = gray.shape
            
            # get stack of blurred images
            # sds = np.linspace(1, 4, 8)
            sds = get_truncated_normal(mean=0, sd=5, low=1, upp=10).rvs(4)
            for sd in sds:
                blurred = cv2.GaussianBlur(gray,(21,21),sd)
                imgs.append(blurred)

            # get feature points
            fps_str = data_dir + '/fps_'+str(nn)+'.txt'
            lines = [line.strip() for line in open(fps_str)]
            for line in lines:
                lst = line.split(',')
                p = cv2.KeyPoint(x=float(lst[0]), y=float(lst[1]), _size=float(lst[2]), _angle=float(lst[3]),
                                _response=float(lst[4]), _octave=int(lst[5]), _class_id=int(lst[6]))

                ps = []

                # extract feature details
                (y,x) = p.pt
                s = p.size
                if s<12:
                    s = 12

                for gray in imgs: # extract patches from original and each blurred image
                    # original patch
                    ptch = gray[int(x-0.5*s):int(x-0.5*s)+int(s),int(y-0.5*s):int(y-0.5*s)+int(s)]
                    ptch_1 = cv2.resize(ptch, (32, 32))
                    ptch_1 = np.array(ptch_1, dtype=np.uint8)
                    ps.append(ptch_1)
                    labels.append(counter)

                    # perspective transform patch
                    pts1 = np.float32([[0,0],[s,0],[0,s],[s,s]])
                    pts2 = np.float32([[-random.randint(0,15),-random.randint(0,15)],[s+random.randint(0,15),-random.randint(0,15)],[-random.randint(0,15),s+random.randint(0,15)],[s+random.randint(0,15),s+random.randint(0,15)]])
                    xmin=np.max((pts2[0,0],pts2[2,0]))
                    xmax=np.min((pts2[1,0],pts2[3,0]))
                    ymin=np.max((pts2[0,1],pts2[1,1]))
                    ymax=np.min((pts2[2,1],pts2[3,1]))
                    x_dif = (xmax-xmin)
                    y_dif = (ymax-ymin)
                    sz=np.min((x_dif,y_dif))
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.CenterCrop(sz),
                        transforms.Resize(32)])
                    M = cv2.getPerspectiveTransform(pts1,pts2)
                    M[0, 2] -= xmin
                    M[1, 2] -= ymin
                    ptch = cv2.warpPerspective(ptch,M,(x_dif,y_dif))
                    ptch = np.array(transform(ptch), dtype=np.uint8)
                    ps.append(ptch)
                    labels.append(counter)

                    # rotate patch
                    r = get_truncated_normal().rvs() # sample angles from normal distribution
                    M = cv2.getRotationMatrix2D((y,x), r, 1.0) # rotate about patch center
                    rotated = cv2.warpAffine(gray, M, (w, h))
                    ptch = rotated[int(x-0.5*s):int(x-0.5*s)+int(s),int(y-0.5*s):int(y-0.5*s)+int(s)]
                    ptch = cv2.resize(ptch, (32, 32))
                    ptch = np.array(ptch, dtype=np.uint8)
                    ps.append(ptch)
                    labels.append(counter)

                
                ps = np.array(ps)
                ps = ps.astype('uint8')

                
                ps = torch.ByteTensor(ps).cuda()

                if type(ptchs) == type(None):
                    ptchs = ps
                else:
                    ptchs = torch.cat([ptchs,ps], dim=0) # concat list of tensors

                counter += 1

                    
        print(len(ptchs),'patches created from',counter,'features and',nn,'images')
        # ptchs = torch.cat(ptchs, dim=2) # concat list of tensors
        
        return ptchs, torch.LongTensor(labels)


if __name__ == '__main__':
    # need to be specified
    try:
        path_to_imgs_dir = sys.argv[1]
        output_dir  = sys.argv[2]
        no_imgs  = int(sys.argv[3])
    except:
        print("Wrong input format. Try python HPatchesDatasetCreator.py path_to_hpatches path_to_splits_json output_dir")
        sys.exit(1)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    t = "train"
    trbd = TURBID()
    images, labels = trbd.read_image_file(path_to_imgs_dir, no_imgs)
    with open(os.path.join(output_dir, 'turbid_imgs.pt'), 'wb') as f:
        torch.save((images, labels), f)
    print(t, 'images saved')
