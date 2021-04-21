import os
import glob
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
from convert import *
from skimage import io

__cpus__ = multiprocessing.cpu_count()

dataset_dir = os.getcwd() + os.sep + 'dataset' + os.sep
real = dataset_dir + 'realface' + os.sep
pencil = dataset_dir + 'pencilface' + os.sep
line = dataset_dir + 'linedface' + os.sep


def convert1D23D(p: str):
    img = cv2.imread(p)
    img = cv2.resize(img,(256,256))
    if len(img.shape) == 3:
        pass
    else:
        img = cv2.merge([img, img, img])
    cv2.imwrite(p, img)


def generate(path: str):
    realImgs = glob.glob(path + "*.png")
    pool = Pool(__cpus__ - 1)
    pool_list = []
    for i in realImgs:
        rs1 = pool.apply_async(convert2Lined, (i, ))
        rs2 = pool.apply_async(convert2PencilFace, (i, ))
        pool_list.append(rs1)
        pool_list.append(rs2)

    for pr in tqdm(pool_list):
        pr.get()


def modify(path: str):
    imgs = glob.glob(path + "*.png")
    pool = Pool(__cpus__ - 1)
    pool_list = []
    for i in imgs:
        rs = pool.apply_async(convert1D23D, (i, ))
        pool_list.append(rs)

    for pr in tqdm(pool_list):
        pr.get()


if __name__ == "__main__":
    # generate(real)
    modify(dataset_dir + 'realface2' + os.sep)
