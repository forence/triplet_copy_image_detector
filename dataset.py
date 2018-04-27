from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
"""
References: Pytorch/torchvision/datasets & 
            adambielski/siamese-triplet/datasets 
            
"""

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(root):
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    labels = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
                    labels.append(class_to_idx[target])

    return images, labels


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class TripletDatasets(Dataset):
    def __init__(self, root, transform=None):
        classes, class_to_idx = find_classes(root)
        images, labels = make_dataset(root, class_to_idx)
        self.root = root
        self.transform = transform
        self.images = images
        self.labels = labels
        self.labels_set = set(labels)
        self.classes = classes              # ['file1', 'file2', 'file3']
        self.class_to_idx = class_to_idx    # {'file1': 0, 'file2': 1, 'file3': 2}
        self.label_to_idx = {label: np.where(np.array(labels) == label)[0]
                             for label in self.labels_set}  # {0: pos1, pos2... 1: pos3, pos4...}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        anchor_image, anchor_label = pil_loader(self.images[idx]), self.labels[idx]

        positive_idx = idx
        while positive_idx == idx:
            positive_idx = np.random.choice(self.label_to_idx[anchor_label])
        positive_image = pil_loader(self.images[positive_idx])

        negative_label = np.random.choice(list(self.labels_set - set([anchor_label])))
        negative_idx = np.random.choice(self.label_to_idx[negative_label])
        negative_image = pil_loader(self.images[negative_idx])

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image





