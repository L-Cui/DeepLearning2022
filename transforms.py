import random
import torch
import torchvision.transforms as Trans
from torchvision.transforms import functional as F
import numpy as np
from PIL import ImageFilter

class SesemiTransform:
    """
    Torchvision-style transform to apply SESEMI augmentation to image.
    """

    classes = ('0', '90', '180', '270', 'hflip', 'vflip')

    def __call__(self, x):
        tf_type = random.randint(0, len(self.classes) - 1)
        if tf_type == 0:
            x = x
        elif tf_type == 1:
            x = Trans.functional.rotate(x, 90)
        elif tf_type == 2:
            x = Trans.functional.rotate(x, 180)
        elif tf_type == 3:
            x = Trans.functional.rotate(x, 270)
        elif tf_type == 4:
            x = Trans.functional.hflip(x)
        elif tf_type == 5:
            x = Trans.functional.rotate(x, 180)
            x = Trans.functional.hflip(x)
        return x, tf_type

def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = Trans.Normalize(mean=self.mean, std=self.std)
    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
# class Cutout(object):
#     def __init__(self, n_holes, length):
#         self.n_holes = n_holes
#         self.length = length

#     def __call__(self, img, target):
#         h = img.size(1)
#         w = img.size(2)

#         mask = np.ones((h,w), np.float32)

#         for n in range(self.n_holes):
#             y = np.random.randint(h)
#             x = np.random.randint(w)

#             y1 = np.clip(y - self.length // 2, 0, h)
#             y2 = np.clip(y + self.length // 2, 0, h)
#             x1 = np.clip(x - self.length // 2, 0, w)
#             x2 = np.clip(x + self.length // 2, 0, w)

#             mask[y1:y2, x1:x2] = 0.

#         mask = torch.from_numpy(mask)
#         mask = mask.expand_as(img)
#         img = img * mask

#         return img, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
