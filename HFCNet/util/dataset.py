import os
import os.path
import cv2
import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)

# output a image_label list
def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if len(line_split) != 2:
            raise (RuntimeError("Image list file read line error : " + line + "\n"))
        image_name = os.path.join(line_split[0])
        label_name = os.path.join(line_split[1])
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class ORSSD(Dataset):
    def __init__(self, root, mode, transform=None):
        self.mode = mode
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        if mode == 'train':
            self.image_paths = [os.path.join(root, mode+'-images', prefix + '.png') for prefix in self.prefixes]
            # self.image_paths = [os.path.join(root, mode+'-images', prefix + '.jpg') for prefix in self.prefixes]
        else:
            self.image_paths = [os.path.join(root, mode+'-images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, mode+'-labels', prefix + '.png') for prefix in self.prefixes]
        self.transform = transform
        # if mode == 'train':
        #     self.edge_paths = [os.path.join(root, mode+'-edges', prefix + '.png') for prefix in self.prefixes]
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        # edge = self.label_transformation(flip_rot(Image.open(self.edge_paths[index])))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        img_H, img_W = label.shape  # 为了保存原始图像尺寸大小
        img_size = torch.Tensor([img_H, img_W])

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, img_size
        # if self.mode == 'train':
        #     return image, label, edge, name
        # else:
        # return image, label, name
    def __len__(self):
        return len(self.prefixes)

class EORSSD(Dataset):
    def __init__(self, root, mode, transform=None):
        self.mode = mode
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        if mode == 'train':
            # self.image_paths = [os.path.join(root, mode+'-images', prefix + '.png') for prefix in self.prefixes]
            self.image_paths = [os.path.join(root, mode+'-images', prefix + '.jpg') for prefix in self.prefixes]
        else:
            self.image_paths = [os.path.join(root, mode+'-images', prefix + '.jpg') for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, mode+'-labels', prefix + '.png') for prefix in self.prefixes]
        self.transform = transform
        # if mode == 'train':
        #     self.edge_paths = [os.path.join(root, mode+'-edges', prefix + '.png') for prefix in self.prefixes]
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        # edge = self.label_transformation(flip_rot(Image.open(self.edge_paths[index])))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        img_H, img_W = label.shape  # 为了保存原始图像尺寸大小
        img_size = torch.Tensor([img_H, img_W])

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, img_size
        # if self.mode == 'train':
        #     return image, label, edge, name
        # else:
        # return image, label, name
    def __len__(self):
        return len(self.prefixes)
    

class ORSI4199(Dataset):
    def __init__(self, root, mode, transform=None):
        self.mode = mode
        self.prefixes = [line.strip() for line in open(os.path.join(root, mode+'.txt'))]
        if mode == 'train':
            # self.image_paths = [os.path.join(root, mode+'-images', prefix + '.png') for prefix in self.prefixes]
            self.image_paths = [os.path.join(root, 'images', prefix) for prefix in self.prefixes]
        else:
            self.image_paths = [os.path.join(root, 'images', prefix) for prefix in self.prefixes]
        self.label_paths = [os.path.join(root, 'gt', prefix[:-4] + '.png') for prefix in self.prefixes]
        self.transform = transform
        # if mode == 'train':
        #     self.edge_paths = [os.path.join(root, mode+'-edges', prefix + '.png') for prefix in self.prefixes]
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        # edge = self.label_transformation(flip_rot(Image.open(self.edge_paths[index])))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        if image is None:
            print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        img_H, img_W = label.shape  # 为了保存原始图像尺寸大小
        img_size = torch.Tensor([img_H, img_W])

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, img_size
        # if self.mode == 'train':
        #     return image, label, edge, name
        # else:
        # return image, label, name
    def __len__(self):
        return len(self.prefixes)