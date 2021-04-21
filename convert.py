import os
from PIL import Image
import cv2
import numpy as np
from linize import renderFaceMask
from faceSeg import getSegmentedHair
import random

imgPath = 'D:\\github_repo\\PencilFace\\test.jpg'
dataset_dir = os.getcwd() + os.sep + 'dataset' + os.sep
pencil = dataset_dir + 'pencilface' + os.sep
line = dataset_dir + 'linedface' + os.sep


def convert2Lined(imgPath: str):
    f1 = True
    f2 = True
    _, filename = os.path.split(imgPath)
    faceMask = renderFaceMask(imgPath)
    if random.randint(0, 1) == 1:
        f1 = False
    if random.randint(0, 1) == 1:
        f2 = False
    hairMask = getSegmentedHair(imgPath, f1, f2)
    kernel = np.ones((5, 5), np.uint8)
    faceMask = cv2.dilate(faceMask, kernel)

    mask = np.clip(faceMask + hairMask, 0, 255)
    mask = 255 - mask

    cv2.imwrite(line + filename, mask)
    # cv2.imwrite('mask.jpg',mask)
    # return mask


def convert2PencilFace(imgPath: str):
    # img = io.imread(imgPath)
    img = np.asarray(Image.open(imgPath).convert('L')).astype('float')
    _, filename = os.path.split(imgPath)
    depth = 10.
    grad = np.gradient(img)
    grad_x, grad_y = grad
    grad_x = grad_x * depth / 100.
    grad_y = grad_y * depth / 100.

    A = np.sqrt(grad_x**2 + grad_y**2 + 1.)
    uni_y = grad_y / A
    uni_x = grad_x / A
    uni_z = 1. / A

    vec_el = np.pi / 2.2
    vec_az = np.pi / 4.
    dx = np.cos(vec_el) * np.cos(vec_az)
    dy = np.cos(vec_el) * np.cos(vec_az)
    dz = np.sin(vec_el)

    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)
    b = b.clip(0, 255)

    b = b.astype('uint8')
    b = cv2.resize(b, (256, 256))
    if len(b.shape) == 1:
        b = cv2.merge([b, b, b])
    cv2.imwrite(pencil + filename, b)
    # return b
