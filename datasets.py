from __future__ import absolute_import, division

import numpy as np
import cv2
from torch.utils.data import Dataset


__all__ = ['Pair']


class Pair(Dataset):

    def __init__(self, seqs_rgb, seqs_i, transforms=None,
                 pairs_per_seq=1):
        super(Pair, self).__init__()
        self.seqs_rgb = seqs_rgb
        self.seqs_i = seqs_i
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        self.indices = np.random.permutation(len(seqs_rgb))
        self.return_meta = getattr(seqs_rgb, 'return_meta', False)

    def __getitem__(self, index):
        index = self.indices[index % len(self.indices)]

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs_rgb[index]
            infrared_files = self.seqs_i[index]
            vis_ratios = meta.get('cover', None)
        else:
            img_files, anno = self.seqs_rgb[index][:2]
            infrared_files, anno = self.seqs_i[index][:2]
            vis_ratios = None
        
        # filter out noisy frames
        val_rgb_indices = self._filter(
            cv2.imread(img_files[0], cv2.IMREAD_COLOR),
            anno, vis_ratios)
        val_i_indices = self._filter(
            cv2.imread(infrared_files[0], cv2.IMREAD_COLOR),
            anno, vis_ratios)
        if len(val_rgb_indices) < 2:
            index = np.random.choice(len(self))
            return self.__getitem__(index)

        # sample a frame pair
        rand_z_rgb, rand_x_rgb = self._sample_pair(val_rgb_indices)

        z_rgb = cv2.imread(img_files[rand_z_rgb], cv2.IMREAD_COLOR)
        x_rgb = cv2.imread(img_files[rand_x_rgb], cv2.IMREAD_COLOR)
        z_rgb = cv2.cvtColor(z_rgb, cv2.COLOR_BGR2RGB)
        x_rgb = cv2.cvtColor(x_rgb, cv2.COLOR_BGR2RGB)
        
        box_z_rgb = anno[rand_z_rgb]
        box_x_rgb = anno[rand_x_rgb]

        rand_z_i, rand_x_i = self._sample_pair(val_i_indices)

        z_i = cv2.imread(img_files[rand_z_i], cv2.IMREAD_COLOR)
        x_i = cv2.imread(img_files[rand_x_i], cv2.IMREAD_COLOR)
        z_i = cv2.cvtColor(z_i, cv2.COLOR_BGR2RGB)
        x_i = cv2.cvtColor(x_i, cv2.COLOR_BGR2RGB)
        
        box_z_i = anno[rand_z_i]
        box_x_i = anno[rand_x_i]


        item = (z_rgb, x_rgb, box_z_rgb, box_x_rgb, z_i, x_i, box_z_i, box_x_i)
        if self.transforms is not None:
            item = self.transforms(*item)
        
        return item
    
    def __len__(self):
        return len(self.indices) * self.pairs_per_seq
    
    def _sample_pair(self, indices):
        n = len(indices)
        assert n > 0

        if n == 1:
            return indices[0], indices[0]
        elif n == 2:
            return indices[0], indices[1]
        else:
            for i in range(100):
                rand_z, rand_x = np.sort(
                    np.random.choice(indices, 2, replace=False))
                if rand_x - rand_z < 100:
                    break
            else:
                rand_z = np.random.choice(indices)
                rand_x = rand_z

            return rand_z, rand_x
    
    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = (vis_ratios > max(1, vis_ratios.max() * 0.3))
        else:
            c8 = np.ones_like(c1)
        
        mask = np.logical_and.reduce(
            (c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices
