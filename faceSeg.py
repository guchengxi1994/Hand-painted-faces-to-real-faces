import os
from PIL import Image

import cv2
import numpy as np
import torch
from torchvision import transforms
import copy

from utils.nets.MobileNetV2_unet import MobileNetV2_unet

BASEDIR = os.getcwd()

# load model
modelPath = BASEDIR + os.sep + 'models' + os.sep + 'model.pt'
mobileModelPath = BASEDIR + os.sep + 'models' + os.sep + 'mobilenet_v2.pth.tar'
useGPU = False
if useGPU:
    device = torch.device("gpu")
else:
    device = torch.device("cpu")


def load_model():
    # print(device)
    model = MobileNetV2_unet(mobileModelPath, device=str(device)).to(device)

    state_dict = torch.load(modelPath, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


def getSegmentedFace(imgPath: str):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(device)

    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)

    mask = mask[0]
    # cv2.imwrite('mask.jpg',mask)
    return mask


def getSegmentedHair(imgPath: str, hole: bool = True, noised: bool = True):
    mask = getSegmentedFace(imgPath)
    if len(mask.shape) == 3:
        mask = mask[0]
    mask[mask == 2] = 255
    mask[mask != 255] = 0
    mask = np.array(mask, dtype=np.uint8)
    mask = cv2.resize(mask, (256, 256))  # reshape

    if hole:
        kernel = np.ones((15, 15), np.uint8)
        erosion = cv2.erode(mask, kernel)
        mask = mask - erosion

    r = copy.deepcopy(mask)
    g = copy.deepcopy(mask)
    b = copy.deepcopy(mask)

    if noised:  # random salt & pepper noise use numpy
        nr = np.random.randint(0, 2, (256, 256)) * 255
        ng = np.random.randint(0, 2, (256, 256)) * 255
        nb = np.random.randint(0, 2, (256, 256)) * 255

        r = np.clip(nr + r, 0, 255)
        g = np.clip(ng + g, 0, 255)
        b = np.clip(nb + b, 0, 255)

    res = np.array(cv2.merge([r, g, b]), dtype=np.uint8)
    cv2.imwrite('mask.jpg', res)
    return res


# getSegmentedHair('D:\\github_repo\\PencilFace\\test.jpg')
