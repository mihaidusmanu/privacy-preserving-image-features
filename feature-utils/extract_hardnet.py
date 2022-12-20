# Adapted from https://github.com/vcg-uvic/image-matching-benchmark-baselines/blob/master/extract_descriptors_hardnet.py.

import argparse

import numpy as np

import os

import cv2

import sys

import shutil

import sqlite3

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms

import tqdm

from extract_patches.core import extract_patches


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim = 1) + self.eps)
        x= x / norm.unsqueeze(-1).expand_as(x)
        return x


class HardNet(nn.Module):
    """HardNet model definition
    """
    def __init__(self):
        super(HardNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias = False),
            nn.BatchNorm2d(128, affine=False),
        )
    
    def input_norm(self,x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)
    
    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


def get_transforms():
    transform = transforms.Compose([
        transforms.Lambda(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)),
        transforms.Lambda(lambda x: np.reshape(x, (32, 32, 1))),
        transforms.ToTensor()
    ])

    return transform


def recover_database_images_and_ids(database_path):
    # Connect to the database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cursor.execute('SELECT name, image_id FROM images;')
    for row in cursor:
        images[row[1]] = row[0]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='path to the images'
    )
    parser.add_argument(
        '--mr_size', type=float, default=12.0,
        help='patch size in image is mr_size * pt.size'
    )
    parser.add_argument(
        '--weights_path', type=str, default='feature-utils/checkpoint_liberty_with_aug.pth',
        help='path to the model weights'
    )
    parser.add_argument(
        '--batch_size', type=int, default=512,
        help='path to the model weights'
    )
    args = parser.parse_args()
    if args.image_path is None:
        args.image_path = args.dataset_path

    # Dataset paths.
    paths = types.SimpleNamespace()
    paths.sift_database_path = os.path.join(args.dataset_path, 'sift-features.db')
    paths.database_path = os.path.join(args.dataset_path, 'hardnet-features.db')

    # Copy SIFT database.
    if os.path.exists(paths.database_path):
        raise FileExistsError('Database already exists at %s.' % paths.database_path)
    shutil.copy(paths.sift_database_path, paths.database_path)

    # PyTorch settings.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_grad_enabled(False)

    # Transforms and network.
    transforms = get_transforms()

    model = HardNet()
    model.load_state_dict(torch.load(args.weights_path, map_location=device)['state_dict'])
    model = model.to(device)
    model.eval()

    # Recover list of images.
    images = recover_database_images_and_ids(paths.database_path)

    # Connect to database.
    connection = sqlite3.connect(paths.database_path)
    cursor = connection.cursor()
    
    cursor.execute('DELETE FROM descriptors;')
    connection.commit()

    cursor.execute('SELECT image_id, rows, cols, data FROM keypoints;')
    raw_keypoints = cursor.fetchall()
    for row in tqdm.tqdm(raw_keypoints):
        assert(row[2] == 6)
        image_id = row[0]
        image_relative_path = images[image_id]
        if row[1] == 0:
            keypoints = np.zeros([0, 6])
        else:
            keypoints = np.frombuffer(row[-1], dtype=np.float32).reshape(row[1], row[2])
        
        keypoints = np.copy(keypoints)
        # In COLMAP, the upper left pixel has the coordinate (0.5, 0.5).
        keypoints[:, 0] = keypoints[:, 0] - .5
        keypoints[:, 1] = keypoints[:, 1] - .5

        # Extract patches.
        image = cv2.cvtColor(
            cv2.imread(os.path.join(args.image_path, image_relative_path)),
            cv2.COLOR_BGR2RGB
        )

        patches = extract_patches(
            keypoints, image, 32, args.mr_size, 'xyA'
        )

        # Extract descriptors.
        descriptors = np.zeros((len(patches), 128), dtype=np.float32)
        for i in range(0, len(patches), args.batch_size):
            data_a = patches[i : i + args.batch_size]
            data_a = torch.stack(
                [transforms(patch) for patch in data_a]
            ).to(device)
            # Predict
            out_a = model(data_a)
            descriptors[i : i + args.batch_size] = out_a.cpu().numpy()

        # Insert into database.
        cursor.execute(
            'INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);',
            (image_id, descriptors.shape[0], descriptors.shape[1], descriptors.tobytes())
        )
    connection.commit()

    # Close connection to database.
    cursor.close()
    connection.close()
