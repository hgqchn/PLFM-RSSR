import sys
import os
from torch.utils import data
from torchvision.transforms.functional import normalize
import random

from simplesr.data.data_utils import paired_paths_from_folder, paired_paths_from_lmdb
from simplesr.utils.img_utils import img_from_bytes,img_array_to_tensor,img_augment,paired_random_crop_numpy
from simplesr.utils.color_utils import rgb2ycbcr
from simplesr.data.file_client import FileClient

class PairedImageDataset(data.Dataset):

    def __init__(
        self,
        dataroot_gt,
        dataroot_lq,
        phase,
        scale,
        name='',
        io_backend_opt=None,
        lq_filename_tmpl='{}',
        gt_size=None,
        use_augment=False,
        color=None,
        mean=None,
        std=None,
    ):
        super(PairedImageDataset, self).__init__()
        self.name = name
        self.dataroot_gt = dataroot_gt
        self.dataroot_lq = dataroot_lq
        self.io_backend_opt = io_backend_opt.copy()
        self.phase = phase
        self.scale = scale
        self.filename_tmpl = lq_filename_tmpl

        self.gt_size = gt_size
        self.use_augment = use_augment
        self.color = color
        self.mean = mean
        self.std = std


        self.file_client = None

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.dataroot_lq, self.dataroot_gt]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.dataroot_lq, self.dataroot_gt], ['lq', 'gt'])
        else:
            self.paths = paired_paths_from_folder(
                [self.dataroot_lq, self.dataroot_gt], ['lq', 'gt'], self.filename_tmpl
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)



        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = img_from_bytes(img_bytes)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = img_from_bytes(img_bytes)
        original_shapes = (img_gt.shape, img_lq.shape)
        mode = 0


        if self.phase == 'train':
            if self.gt_size:

                img_gt, img_lq = paired_random_crop_numpy(img_gt, img_lq, self.gt_size, self.scale, gt_path)

            if self.use_augment:
                mode=random.randint(0, 7)
                img_gt=img_augment(img_gt,mode=mode)
                img_lq=img_augment(img_lq,mode=mode)


        if self.color == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]


        img_gt = img_array_to_tensor(img_gt)
        img_lq = img_array_to_tensor(img_lq)


        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        sample = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        return self._finish_sample(sample, index, mode, original_shapes)

    def _finish_sample(self, sample, index, augmentation_mode, original_shapes):
        return sample

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    pass
