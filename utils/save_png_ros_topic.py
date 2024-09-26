#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from datetime import datetime  # 导入时间模块

# 初始化 ROS 节点
rospy.init_node('image_saver')

# 创建 CvBridge 对象
bridge = CvBridge()

# 定义全局变量来存储图像
rgb_image = None
depth_image = None
is_recording = False
video_writer = None

# 设置图像保存的路径
rgb_image_path = './rgb.png'
depth_image_path = './depth.png'

# 订阅 RGB 和深度图像的回调函数
def rgb_callback(msg):
    global rgb_image
    rgb_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    if is_recording and video_writer is not None:
        video_writer.write(rgb_image)  # 将帧写入视频

def depth_callback(msg):
    global depth_image
    depth_image = bridge.imgmsg_to_cv2(msg, "16UC1")

# 订阅 RGB 和深度图像
rospy.Subscriber('/camera/color/image_raw', Image, rgb_callback)
rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, depth_callback)

def save_images():
    print("Saving images...")
    global rgb_image, depth_image
    if rgb_image is not None and depth_image is not None:
        # 保存 RGB 图像
        cv2.imwrite(rgb_image_path, rgb_image)
        # 保存深度图像
        cv2.imwrite(depth_image_path, depth_image)
        # 显示保存路径
        rospy.loginfo("Images saved to: %s and %s", rgb_image_path, depth_image_path)

def start_recording():
    global is_recording, video_writer
    if rgb_image is not None:
        height, width, _ = rgb_image.shape
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f'./video_output_{timestamp}.mp4'
        # 创建 VideoWriter 对象
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        is_recording = True
        print("Recording started... Saving to:", video_path)

def stop_recording():
    global is_recording, video_writer
    if is_recording:
        is_recording = False
        if video_writer is not None:
            video_writer.release()  # 停止录制
            video_writer = None
        print("Recording stopped.")

# 主循环，监听键盘输入
def key_listener():
    print("Press 'p' to save images, 'r' to start recording, 's' to stop recording, 'q' to quit.")
    while not rospy.is_shutdown():
        key = input()  # 在终端监听输入
        if key == 'p':
            save_images()
        elif key == 'r':
            start_recording()
        elif key == 's':
            stop_recording()
        elif key == 'q':
            print("Quitting...")
            if is_recording:
                stop_recording()  # 退出前停止录制
            break

if __name__ == '__main__':
    try:
        key_listener()
    except rospy.ROSInterruptException:
        pass
