import argparse
import subprocess
from pathlib import Path

import numpy as np
from skimage.io import imsave
from tqdm import tqdm

from dataset.database import parse_database_name, get_ref_point_cloud
from estimator import name2estimator
from eval import visualize_intermediate_results
from utils.base_utils import load_cfg, project_points
from utils.draw_utils import pts_range_to_bbox_pts, draw_bbox_3d
from utils.pose_utils import pnp

import rospy
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped

import time
import cv2 
from scipy.spatial.transform import Rotation as R # 提取旋转矩阵

USE_VIRTUAL_K = True  # 是否使用伪K进行姿态估计

class PoseEstimatorNode:
    def __init__(self, args):
        rospy.loginfo("初始化 PoseEstimatorNode...")

        self.args = args
        self.cfg = load_cfg(args.cfg)
        self.ref_database = parse_database_name(args.database)
        self.estimator = name2estimator[self.cfg['type']](self.cfg)
        self.estimator.build(self.ref_database, split_type='all')

        rospy.loginfo("加载参考点云...")
        self.object_pts = get_ref_point_cloud(self.ref_database)
        self.object_bbox_3d = pts_range_to_bbox_pts(np.max(self.object_pts, 0), np.min(self.object_pts, 0))

        rospy.loginfo(f"创建输出目录: {args.output}")
        self.output_dir = Path(args.output)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        (self.output_dir / 'images_out_smooth').mkdir(exist_ok=True, parents=True)

        self.bridge = CvBridge()
        self.K = None  # 相机内参
        self.pose_init = None
        self.hist_pts = []

        # 订阅话题
        rospy.loginfo("订阅相机话题...")
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # 创建发布者
        rospy.loginfo("创建图像发布者...")
        self.bbox_pub = rospy.Publisher("/pose_estimation/bbox_image", Image, queue_size=10)

        rospy.loginfo("创建物体姿态发布者...")
        self.pose_pub = rospy.Publisher("/gen6d/pose", PoseStamped, queue_size=10)

    def camera_info_callback(self, msg):
        # 获取相机内参
        self.K = np.array(msg.K).reshape(3, 3)
        # rospy.loginfo(f"接收到相机内参: \n{self.K}")

    def image_callback(self, msg):
        global USE_VIRTUAL_K

        if self.K is None:
            rospy.logwarn("相机内参尚未获取，等待中...")
            return

        # 转换ROS图像消息为OpenCV格式
        # rospy.loginfo("接收到图像数据，进行姿态估计...")
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
        if USE_VIRTUAL_K:
            # 生成伪K
            h, w, _ = img.shape
            f = np.sqrt(h**2 + w**2)
            K = np.asarray([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
            # rospy.loginfo(f"使用伪内参进行姿态估计，内参矩阵为: \n{K}")

            # 记录开始时间
            start_time = time.time()
            
            # 进行姿态估计
            if self.pose_init is not None:
                # rospy.loginfo("初始化已存在，执行精细化...")
                self.estimator.cfg['refine_iter'] = 1  # 仅在初始化后进行一次精细化
            else:
                # rospy.loginfo("执行初步姿态估计...")
                pass
            pose_pr, inter_results = self.estimator.predict(img, K, pose_init=self.pose_init)
            # 计算结束时间
            end_time = time.time()
            inference_time = end_time - start_time
            fps = 1.0 / inference_time

            # 赋值
            self.pose_init = pose_pr

            pts, _ = project_points(self.object_bbox_3d, pose_pr, K)
            bbox_img = draw_bbox_3d(img, pts, (0, 0, 255))

            self.hist_pts.append(pts)
            pts_ = weighted_pts(self.hist_pts, weight_num=self.args.num, std_inv=self.args.std)
            pose_ = pnp(self.object_bbox_3d, pts_, K)
            pts__, _ = project_points(self.object_bbox_3d, pose_, K)
            bbox_img_ = draw_bbox_3d(img, pts__, (0, 0, 255))

            """
                提取旋转矩阵 | 计算欧拉角
            """
            # 提取旋转矩阵
            rotation_matrix = pose_pr[:, :3]  # 取前3列
            # 将旋转矩阵转换为欧拉角
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=True)  # 使用XYZ顺序并转换为度数
            quaternion = rotation.as_quat()  # 以(x, y, z, w)格式返回四元数
            #rospy.loginfo(f"物体的欧拉角 (XYZ顺序): \n{euler_angles}")
            #rospy.loginfo(f"提取的旋转矩阵: \n{rotation_matrix}")
            rospy.loginfo(f"物体的四元数: {quaternion}")
            """
                绘制 fps | 绘制欧拉角信息
            """
            # 在图像上绘制fps
            cv2.putText(bbox_img_, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 添加欧拉角信息到图像
            euler_text = f"Pitch:{euler_angles[0]:.2f}, Yaw:{euler_angles[1]:.2f}, Roll:{euler_angles[2]:.2f}"
            cv2.putText(bbox_img_, euler_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            """
                发布图像信息 | 发布姿态信息
            """
            bbox_img_msg = self.bridge.cv2_to_imgmsg(bbox_img_, encoding='bgr8')
            self.bbox_pub.publish(bbox_img_msg)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = "camera_color_optical_frame"  # 设置为相机的坐标系或其他合适的坐标系
            pose_msg.pose.position.x = pose_pr[0, 3]  # 提取平移部分
            pose_msg.pose.position.y = pose_pr[1, 3]
            pose_msg.pose.position.z = pose_pr[2, 3]
            pose_msg.pose.orientation.x = quaternion[0]
            pose_msg.pose.orientation.y = quaternion[1]
            pose_msg.pose.orientation.z = quaternion[2]
            pose_msg.pose.orientation.w = quaternion[3]
            self.pose_pub.publish(pose_msg)
        else:
            # 进行姿态估计
            if self.pose_init is not None:
                self.estimator.cfg['refine_iter'] = 1  # 仅在初始化后进行一次精细化
            else:
                # rospy.loginfo("执行初步姿态估计...")
                pass

            # 使用相机的实际K进行姿态估计
            pose_pr, inter_results = self.estimator.predict(img, self.K, pose_init=self.pose_init)
            self.pose_init = pose_pr

            # 绘制3D包围框
            pts, _ = project_points(self.object_bbox_3d, pose_pr, self.K)
            bbox_img = draw_bbox_3d(img, pts, (0, 0, 255))

            self.hist_pts.append(pts)
            pts_ = weighted_pts(self.hist_pts, weight_num=self.args.num, std_inv=self.args.std)
            pose_ = pnp(self.object_bbox_3d, pts_, self.K)
            pts__, _ = project_points(self.object_bbox_3d, pose_, self.K)
            bbox_img_ = draw_bbox_3d(img, pts__, (0, 0, 255))

            # rospy.loginfo("绘制并发布包围框图像...")
            # 将 bbox_img 转换为 ROS 图像消息并发布
            bbox_img_msg = self.bridge.cv2_to_imgmsg(bbox_img_, encoding='bgr8')
            self.bbox_pub.publish(bbox_img_msg)            


def weighted_pts(pts_list, weight_num=10, std_inv=10):
    # rospy.loginfo(f"计算加权历史点，权重数量: {weight_num}, 标准差倒数: {std_inv}")
    weights = np.exp(-(np.arange(weight_num) / std_inv) ** 2)[::-1]
    pose_num = len(pts_list)
    if pose_num < weight_num:
        weights = weights[-pose_num:]
    else:
        pts_list = pts_list[-weight_num:]
    pts = np.sum(np.asarray(pts_list) * weights[:, None, None], 0) / np.sum(weights)
    return pts

if __name__ == "__main__":
    rospy.init_node('gen6d_pose_estimator', anonymous=True)

    # 处理参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/gen6d_pretrain.yaml')
    parser.add_argument('--database', type=str, default="custom/mouse")
    parser.add_argument('--output', type=str, default="data/custom/mouse/test")
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--std', type=float, default=2.5)
    args = parser.parse_args()

    # ROS node
    rospy.loginfo("启动节点 gen6d_pose_estimator...")
    poseEstimatorNode = PoseEstimatorNode(args)
    rospy.spin()
