# -*- coding: utf-8 -*-
import cv2
import glob

file = "F:/Feature/MISAW/Video/*.mp4"
files = sorted(glob.glob(file)) #匹配给定路径下的所有pattern，并以列表形式返回
# print(files)
for f in files:
    vc = cv2.VideoCapture(f)  # 进行视频的载入
    i = 0
    while vc.isOpened(): # 判断载入的视频是否可以打开
        (ret, frame) = vc.read() # 进行单张图片的读取，ret的值为True或者Flase， frame表示读入的图片
        file_name = "F:/Feature/MISAW/Image/" + f.split('\\')[1].replace('.mp4', '_') + str("{:0>6d}".format(i)) + ".jpg" # 数字左边补零,小数报错
        if ret == True:
            cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY])# cv2.IMWRITE_JPEG_QUALITY类型为 long ,必须转换成 intcv2.IMWRITE_PNG_COMPRESSION, 从0到9 压缩级别越高图像越小
        else:
            break
        i += 1
print("complete")

'''import cv2

mp4 = cv2.VideoCapture("F:/Feature/MISAW/Video/6_4.mp4")  # 读取视频
is_opened = mp4.isOpened()  # 判断是否打开
print(is_opened)
fps = mp4.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
print(fps)
widght = mp4.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取视频的宽度
height = mp4.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取视频的高度
print(str(widght) + "x" + str(height))
i = 0
while is_opened:
    i += 1
    (flag, frame) = mp4.read()  # 读取图片
    file_name = "F:/Feature//MISAW/Image/" + "6_4_" + str(i) + ".jpg"
    print(file_name)
    if flag == True:
        cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY])  # 保存图片
    else:
        break
print("complete")

mp4 = cv2.VideoCapture("C:/Research/MISAW/Video/6_4.mp4")  # 读取视频
i = 0
while mp4.isOpened():
    i += 1
    (flag, frame) = mp4.read()  # 读取图片
    print(frame)
    break
    if flag == True:
        cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY])  # 保存图片
    else:
        break
print("complete")'''
