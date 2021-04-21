'''
Descripttion: 
version: 
Author: xiaoshuyui
email: guchengxi1994@qq.com
Date: 2021-04-18 09:18:53
LastEditors: xiaoshuyui
LastEditTime: 2021-04-18 09:20:00
'''
import dlib
from skimage import io
import numpy as np
import cv2

predictor_path = './shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def renderFaceMask(imgPath: str):
    img = io.imread(imgPath)
    img = cv2.resize(img,(256,256))
    faces = detector(img, 1)
    imgShape = img.shape
    im = np.ones(imgShape)
    _color = [255, 255, 255]
    for i, _ in enumerate(faces):
        shape = predictor(img, faces[i])
        for i in range(1, 18):  # 脸颊
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 1:
                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

        for i in range(18, 23):  # 左边眉毛
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 18:
                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

        for i in range(23, 28):  # 右边眉毛
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 23:
                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

        for i in range(28, 37):  # 鼻子
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 28:
                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)
                if i == 36:
                    _d = shape.parts()[31 - 1]
                    cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

        for i in range(37, 43):  # 左边眼睛
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 37:
                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)
                if i == 42:
                    _d = shape.parts()[37 - 1]
                    cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

        for i in range(43, 49):  # 左边眼睛
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 43:
                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)
                if i == 48:
                    _d = shape.parts()[43 - 1]
                    cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

        for i in range(49, 61):  # 外嘴唇
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 49:

                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)
                if i == 60:
                    _d = shape.parts()[49 - 1]
                    cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

        for i in range(61, 69):  # 内嘴唇
            d = shape.parts()[i - 1]
            im[d.y, d.x] = _color

            if i > 61:
                _d = shape.parts()[i - 2]
                cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)
                if i == 68:
                    _d = shape.parts()[61 - 1]
                    cv2.line(im, (d.x, d.y), (_d.x, _d.y), _color)

    # io.imsave('renderedFace.jpg', im)
    return im


# renderFaceMask('./test.jpg')
