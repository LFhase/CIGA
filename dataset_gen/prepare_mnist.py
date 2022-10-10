# Compute superpixels for MNIST/CIFAR-10 using SLIC algorithm
# https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic

import numpy as np
import random
import os
import scipy
import pickle
from skimage.segmentation import slic
from torchvision import datasets
import multiprocessing as mp
import scipy.ndimage
import scipy.spatial
import argparse
import datetime

import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
# import torchvision.datasets.utils as dataset_utils


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


class ColoredMNIST(datasets.VisionDataset):
    """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """

    def __init__(self, root='../data', env='train1', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform, target_transform=target_transform)

        self.prepare_colored_mnist()
        if env in ['train1', 'train2', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')
        data = []
        labels = []
        transform = transforms.ToTensor()
        for (img, label) in self.data_label_tuples:
            data.append(transform(img))
            labels.append(label)

        data = torch.stack(data, dim=0)
        # print(data[0])
        # print(sum(data[0]))
        labels = torch.LongTensor(labels).to(data.device)
        data = data.permute(0, 2, 3, 1)
        if 'train' in env:
            self.train_data = data
            self.train_labels = labels
        else:
            self.test_data = data
            self.test_labels = labels

    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train1.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'train2.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train1_set = []
        train2_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            if idx < 20000:
                # 20% in the first training environment
                if np.random.uniform() < 0.2:
                    color_red = not color_red
            elif idx < 40000:
                # 10% in the first training environment
                if np.random.uniform() < 0.1:
                    color_red = not color_red
            else:
                # 90% in the test environment
                if np.random.uniform() < 0.9:
                    color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if idx < 20000:
                train1_set.append((Image.fromarray(colored_arr), binary_label))
            elif idx < 40000:
                train2_set.append((Image.fromarray(colored_arr), binary_label))
            else:
                test_set.append((Image.fromarray(colored_arr), binary_label))

            # Debug
            # print('original label', type(label), label)
            # print('binary label', binary_label)
            # print('assigned color', 'red' if color_red else 'green')
            # plt.imshow(colored_arr)
            # plt.show()
            # break

        os.mkdir(colored_mnist_dir)
        torch.save(train1_set, os.path.join(colored_mnist_dir, 'train1.pt'))
        torch.save(train2_set, os.path.join(colored_mnist_dir, 'train2.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract SLIC superpixels from images')
    parser.add_argument('-D', '--dataset', type=str, default='mnist', choices=['mnist', 'cmnist', 'cifar10'])
    parser.add_argument('-d', '--data_dir', type=str, default='../data', help='path to the dataset')
    parser.add_argument('-o', '--out_dir', type=str, default='../data', help='path where to save superpixels')
    parser.add_argument('-s', '--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of parallel threads')
    parser.add_argument('-n', '--n_sp', type=int, default=75, help='max number of superpixels per image')
    parser.add_argument('-c',
                        '--compactness',
                        type=int,
                        default=0.25,
                        help='compactness of the SLIC algorithm '
                        '(Balances color proximity and space proximity): '
                        '0.25 is a good value for MNIST '
                        'and 10 for color images like CIFAR-10')
    parser.add_argument('--seed', type=int, default=111, help='seed for shuffling nodes')
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    return args


def process_image(params):

    img, index, n_images, args, to_print, shuffle = params
    if img.dtype == np.uint8:
        # assert img.dtype == np.uint8, img.dtype
        img = (img / 255.).astype(np.float32)
    else:
        assert img.dtype == np.float32

    n_sp_extracted = args.n_sp + 1  # number of actually extracted superpixels (can be different from requested in SLIC)
    n_sp_query = args.n_sp + (
        20 if args.dataset in ['mnist'] else 50
    )  # number of superpixels we ask to extract (larger to extract more superpixels - closer to the desired n_sp)
    # print(len(img.shape) > 2)
    while n_sp_extracted > args.n_sp:
        # superpixels = slic(img, n_segments=n_sp_query, compactness=args.compactness, multichannel=len(img.shape) > 2)
        superpixels = slic(img, n_segments=n_sp_query, compactness=args.compactness, channel_axis=-1)
        sp_indices = np.unique(superpixels)
        n_sp_extracted = len(sp_indices)
        n_sp_query -= 1  # reducing the number of superpixels until we get <= n superpixels
    # print(superpixels)
    assert n_sp_extracted <= args.n_sp and n_sp_extracted > 0, (args.split, index, n_sp_extracted, args.n_sp)
    # assert n_sp_extracted == np.max(superpixels) + 1, ('superpixel indices', np.unique(superpixels)
    #                                                   )  # make sure superpixel indices are numbers from 0 to n-1

    if shuffle:
        ind = np.random.permutation(n_sp_extracted)
    else:
        ind = np.arange(n_sp_extracted)

    sp_order = sp_indices[ind].astype(np.int32)
    if len(img.shape) == 2:
        img = img[:, :, None]

    n_ch = 1 if img.shape[2] == 1 else 3

    sp_intensity, sp_coord = [], []
    for seg in sp_order:
        mask = (superpixels == seg).squeeze()
        avg_value = np.zeros(n_ch)
        for c in range(n_ch):
            avg_value[c] = np.mean(img[:, :, c][mask])
        cntr = np.array(scipy.ndimage.measurements.center_of_mass(mask))  # row, col
        sp_intensity.append(avg_value)
        sp_coord.append(cntr)
    sp_intensity = np.array(sp_intensity, np.float32)
    sp_coord = np.array(sp_coord, np.float32)
    if to_print:
        print('image={}/{}, shape={}, min={:.2f}, max={:.2f}, n_sp={}'.format(index + 1, n_images, img.shape, img.min(),
                                                                              img.max(), sp_intensity.shape[0]))

    return sp_intensity, sp_coord, sp_order, superpixels


if __name__ == '__main__':

    dt = datetime.datetime.now()
    print('start time:', dt)

    args = parse_args()

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)  # to make node random permutation reproducible (not tested)

    # Read image data using torchvision
    is_train = args.split.lower() == 'train'
    if args.dataset == 'mnist':
        data = datasets.MNIST(args.data_dir, train=is_train, download=True)
        assert args.compactness < 10, ('high compactness can result in bad superpixels on MNIST')
        assert args.n_sp > 1 and args.n_sp < 28 * 28, (
            'the number of superpixels cannot exceed the total number of pixels or be too small')
    elif args.dataset == 'cifar10':
        data = datasets.CIFAR10(args.data_dir, train=is_train, download=True)
        assert args.compactness > 1, ('low compactness can result in bad superpixels on CIFAR-10')
        assert args.n_sp > 1 and args.n_sp < 32 * 32, (
            'the number of superpixels cannot exceed the total number of pixels or be too small')
    elif args.dataset == 'cmnist':
        data = ColoredMNIST(root='../data', env='all_train' if is_train else 'test')
    else:
        raise NotImplementedError('unsupported dataset: ' + args.dataset)
    # print(data.train.size())
    images = data.train_data if is_train else data.test_data
    labels = data.train_labels if is_train else data.test_labels

    if not isinstance(images, np.ndarray):
        images = images.numpy()
    if isinstance(labels, list):
        labels = np.array(labels)
    if not isinstance(labels, np.ndarray):
        labels = labels.numpy()

    n_images = len(labels)

    if args.threads <= 0:
        sp_data = []
        for i in range(n_images):
            sp_data.append(process_image((images[i], i, n_images, args, True, True)))
    else:
        with mp.Pool(processes=args.threads) as pool:
            sp_data = pool.map(process_image, [(images[i], i, n_images, args, True, True) for i in range(n_images)])

    superpixels = [sp_data[i][3] for i in range(n_images)]
    sp_data = [sp_data[i][:3] for i in range(n_images)]
    args.out_dir = os.path.join(args.out_dir,"CMNISTSP")
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    with open('%s/%s_%dsp_%s.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump((labels.astype(np.int32), sp_data), f, protocol=2)
    with open('%s/%s_%dsp_%s_superpixels.pkl' % (args.out_dir, args.dataset, args.n_sp, args.split), 'wb') as f:
        pickle.dump(superpixels, f, protocol=2)

    print('done in {}'.format(datetime.datetime.now() - dt))
