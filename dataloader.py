from PIL import Image
from pathlib import Path
import torch.utils.data as data
import numpy as np
import opencv_transforms.functional as FF
import cv2
import random
import torch

class FlatFolderDataset(data.Dataset):
    # Get all images from the root and transform
    def __init__(self, root, transform, args, palette_info=False):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.ncluster = args.ncluster
        self.paths = list(Path(self.root).glob('*'))
        self.palette_info = palette_info
        if palette_info:
            self.paths = list(Path(self.root + '/train').glob('*'))
            self.data_path = Path(self.root+'/palette/{:d}_1'.format(args.ncluster)).expanduser()
            self.palette_dict = torch.load(self.data_path/'palette_db_{:d}_1'.format(args.ncluster))

    def __getitem__(self, index):
        random_seed = np.random.randint(2147483647)
        random.seed(random_seed)
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = np.asarray(img)
        try:
            img_rgb = img[:, 0:512, :]
        except IndexError as e:
            print(str(path), e)
        # img_edge = img[:, 512:, :]

        # make color palette #
        img_rgb = self.transform(img_rgb) # to make color_cluster shape as img_rgb's
        img_rgb = self.make_tensor(img_rgb)

        img_edge = Image.open(str(path)).convert('L')
        img_edge = np.asarray(img_edge)
        img_edge = img_edge[:, 512:, np.newaxis]

        random.seed(random_seed)
        img_edge = self.transform(img_edge)
        img_edge = self.make_tensor(img_edge)

        if self.palette_info:
            palette = self.palette_dict[path.stem]
            palette = palette.reshape((-1, 1, 1))

            return img_edge, img_rgb, palette

        else:
            return img_edge, img_rgb

    def make_tensor(self, img):
        img = FF.to_tensor(img)
        img = FF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return img

    def __len__(self):
        return len(self.paths)