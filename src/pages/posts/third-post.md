---
layout: '@/templates/BasePost.astro'
title: OpenCv Python入门到入土
description: 安装Opencv、基本图像操作方法
pubDate: 2024-11-21T00:00:00Z
imgSrc: '/assets/images/opencv.jpeg'
imgAlt: 'Image post 4'
---

友情链接[this page](../sixth-post/).

##  写在前面
因为经常需要用到opencv，又经常忘记很多用过的方法，所以写个文章给记下来好了。

## 1 安装OpenCv
通常安装opencv只需要这么一行命令就可以做到了。

    pip3 install opencv-python
在不那么通常的情况下，我们需要使用Anaconda来安装，这个以后再说。

在非常不通常的情况下，我们需要直接下载源码编译安装，这是及其痛苦的。

## 2 基本图像操作方法
你应该知道：

什么是数组
在C语言中数组的存储方式
图片就是数组

## 2.1 从摄像头获取一张图片
    import cv2
    import numpy
    cap = cv2.VideoCapture(0) #创建VideoCapture对象
    
    while(True): #创建无限循环，用于播放每一帧图像

    ret, frame = cap.read() #读取图像的每一帧
    print(frame)
 
    cv2.imshow('frame',frame) #显示帧
 
    #等待1毫秒，判断此期间有无按键按下，以及按键的值是否是Esc键
 
    if cv2.waitKey(1) & 0xFF == 27:
 
        break #中断循环

    cap.release() #释放ideoCapture对象
    
    cv2.destroyAllWindows()
## 2.1.2 从文件中读取一张图片或是视频中的一帧

    imgFile = "../images/sakara.jpg"  # 读取文件的路径
    img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)
    img2 = cv2.imread(imgFile, flags=0)  # flags=0 读取为灰度图像
## 2.2 改变图片的大小
    # 读取图片
    
    image = cv2.imread('image.jpg')
    
    cv2.imshow('Original Image', image)
    
    # 让我们使用新的宽度和高度缩小图像
    
    down_width = 300
    
    down_height = 200
    
    down_points = (down_width, down_height)

    resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
传递给resize函数的是 图像对象，宽/高的元组 ，缩放a方法（这里用线性

## 2.3 把图片叠在另一张图片上
很简单，用numpy自带的数组索引就可以

black_img[0:10, 0:10] = img[0:10, 0:10]

把 后面 img 叠在 black_img 上面
## 3 实例
## 3.1

    import time
    import cv2
    import numpy as np
    path = "sakara.png"
    wei = 640
    hei = 480
    cut_times = 640
    # 对图像进行预处理的函数，传递参数：宽度，高度，图像数组，切片次数
    img = cv2.imread(path, 1)
    
    def pre_make(weight, height, img, img_cut_times):

    if img_cut_times > weight:
        raise ValueError("cut times is more than img weight")
    else:
        pass
 
    size_of_pic = (weight, height)
    # 需要识别的图片大小
 
    # 图片切片数量
    cut = 0
    point_list = []
    df = weight/(img_cut_times)
    while cut < img_cut_times:
        cut += 1
        point_list.append(int(df*cut))
    # 计算需要裁剪的点位置
 
    img = cv2.resize(img, size_of_pic, interpolation=cv2.INTER_LINEAR)
    black_img = np.zeros((height, weight, 3), np.uint8)
    black_img.fill(0)
    # 生成一张纯黑图片
 
    # 要 进 来 力 ！
    for position in point_list:
 
        black_img[0:height, (weight-position):weight] = img[0:height, 0:position]
        cv2.imshow("cancan",black_img)
        cv2.waitKey(1)
 
    # 黑 化 力 ！
    black_img.fill(0)
 
    # 要 出 去 力 ！
    for position in point_list:
 
        black_img[0:height, 0:(weight-position)] = img[0:height, position:weight]
        cv2.imshow("cancan", black_img)
        cv2.waitKey(1)
        black_img.fill(0)

    pre_make(wei, hei, img, cut_times)
## 3.2
“””旨在自动计算亮度，调节图片亮度“””
    
    import cv2
    import numpy as np
    
    def calculate_brightness(image):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算灰度图的平均像素值，即亮度值
    brightness = np.mean(gray_image)
    return brightness
    
    def adjust_brightness(image, target_brightness):
    # 计算当前图像亮度
    current_brightness = calculate_brightness(image)
    
    # 计算亮度调整系数
    brightness_ratio = target_brightness / current_brightness
    
    # 调整图像亮度
    adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_ratio, beta=0)
    
    return adjusted_image
    
    if __name__ == “__main__”:
    # 读取图像
    #image_path = “IMG/photo/10.jpg”
    #image = cv2.imread(image_path)
    cap = cv2.VideoCapture(0)
    while True:
    ret,image = cap.read()
    image = cv2.resize(image, (720,480), interpolation=cv2.INTER_LINEAR)
    # 自动计算图像亮度
    brightness = calculate_brightness(image)
    print(“当前图像亮度:”, brightness)
    
    # 设置目标亮度（可以根据需要进行调整）
    target_brightness = 190
    
    # 调整图像亮度
    adjusted_image = adjust_brightness(image, target_brightness)
    
    # 显示原始图像和调整后的图像
    cv2.imshow(“Original Image”, image)
    cv2.imshow(“Adjusted Image”, adjusted_image)
    
    # 保存调整后的图像
    cv2.imwrite(“adjusted_image.jpg”, adjusted_image)
    
    cv2.waitKey(1)
    cv2.destroyAllWindows()