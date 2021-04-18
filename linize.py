'''
Descripttion: 
version: 
Author: xiaoshuyui
email: guchengxi1994@qq.com
Date: 2021-04-18 09:18:53
LastEditors: xiaoshuyui
LastEditTime: 2021-04-18 09:20:00
'''
import cv2#导入opencv库

#读取原灰度图片
image=cv2.imread("test.jpg")
# cv2.imshow("image", image)#将原图片命名为“image”显示出来

#图像的阈值分割处理，即将图像处理成非黑即白的二值图像
ret,image1 = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)  # binary（黑白二值），ret代表阈值，80是低阈值，255是高阈值
# cv2.imshow('image1', image1)#将阈值分割处理后的图片命名为“image1”显示出来

#二值图像的反色处理，将图片像素取反
(height,width,_) = image1.shape #返回图片大小
image2 = image1.copy()
for i in range(height):
    for j in range(width):
        image2[i,j] = (255-image1[i,j]) 
# cv2.imshow('image2', image2)#将反色处理后的图片命名为“image2”显示出来

#边缘提取，使用Canny函数
image2_3 = cv2.Canny(image2,80,255) #设置80为低阈值，255为高阈值
# cv2.imshow('image2_3', image2_3)#将边缘提取后的图片命名为“image2_3”显示出来

#再次对图像进行反色处理使提取的边缘为黑色，其余部分为白色，方法同image2
height1,width1 = image2_3.shape
image3 = image2_3.copy()
for i in range(height1):
    for j in range(width1):
        image3[i,j] = (255-image2_3[i,j]) 
cv2.imshow('image3', image3)#将边缘提取后反色处理的图片命名为“image3”显示出来
cv2.waitKey(0)# 等待键盘输入，不输入则无限等待
cv2.destroyAllWindows()  #销毁所有窗口