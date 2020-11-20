import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_noise = os.path.join(opt.dataroot, opt.phase + '_noise')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.noise_paths = sorted(make_dataset(self.dir_noise, opt.max_dataset_size))   

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.noise_size = len(self.noise_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.input_nc = opt.input_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
            if self.noise_size >= 1:
                index_noise = index % self.noise_size
            index_patchnoiseA = index % self.A_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
            if self.noise_size >= 1:
                index_noise = random.randint(0, self.noise_size - 1)
            index_patchnoiseA = random.randint(0, self.A_size - 1)
        B_path = self.B_paths[index_B]
        if self.noise_size >= 1:
            noise_path = self.noise_paths[index_noise]
        else:
            noise_path = A_path

        patchnoiseA_path = self.A_paths[index_patchnoiseA]

        if self.input_nc == 1:
            A_img = Image.open(A_path).convert('L')
            B_img = Image.open(B_path).convert('L')
            noise_img = Image.open(noise_path).convert('L')
            patchnoiseA_img = Image.open(patchnoiseA_path).convert('L')
        else:
            assert self.input_nc == 3
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('RGB')
            noise_img = Image.open(noise_path).convert('RGB')
            patchnoiseA_img = Image.open(patchnoiseA_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        noise = self.transform_A(noise_img)
        patchnoiseA = self.transform_A(patchnoiseA_img)

        return {'A': A, 'B': B, 'noise':noise, 'A_paths': A_path, 'B_paths': B_path, 'noise_paths': noise_path, 'patchnoiseA':patchnoiseA}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
