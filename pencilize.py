'''
Descripttion: 
version: 
Author: xiaoshuyui
email: guchengxi1994@qq.com
Date: 2021-04-18 08:59:11
LastEditors: xiaoshuyui
LastEditTime: 2021-04-18 09:07:38
'''
from PIL import Image
import numpy as np
from skimage import io
a = np.asarray(
    Image.open('D:\\pencilface\\PencilFace\\test.jpg').convert('L')).astype(
        'float')
depth = 10.
grad = np.gradient(a)
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

print(type(b))

io.imsave('D:\\pencilface\\PencilFace\\pencil.jpg', b)
