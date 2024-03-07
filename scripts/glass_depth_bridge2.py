import rospy
import numpy as np
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, Image, PointField
import tf
import time
from scipy.spatial.transform import Rotation as R
import tf2_ros
import threading
import os
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge
import struct
import cv2

trans = []
rot = []

pcd_data = []
rgb_data = []
img_rgb = []

start_tf_listen = False



def uv_vec2cloud(uv_vec, depth_img, depth_scale = 1e-3):
    '''
        uv_vec: n×2,
        depth_image: rows * cols
    '''
    default_cam_intri = np.asarray(
    [[521.85359567,   0.        , 321.18647073],
    [0.        , 521.7098714 , 233.81475134],
    [0.        ,   0.        ,   1.        ]])
    fx = default_cam_intri[0, 0]
    fy = default_cam_intri[1, 1]
    cx = default_cam_intri[0, 2]
    cy = default_cam_intri[1, 2]
    cloud = np.zeros((len(uv_vec), 3))
    cloud[:, 2] = depth_img[uv_vec[:, 1].astype(int), uv_vec[:, 0].astype(int)] * depth_scale
    cloud[:, 0] = (uv_vec[:, 0] - cx) * (cloud[:, 2] / fx)
    cloud[:, 1] = (uv_vec[:, 1] - cy) * (cloud[:, 2] / fy)
    return cloud

def depth2cloud(img_depth, cam_intri_inv = None):
    # default unit of depth and point cloud is mm.
    # if cam_intri_inv is None:
    #     cam_intri_inv = np.zeros((1, 1, 3, 3))
    #     cam_intri_inv[0, 0] = np.linalg.inv(default_cam_intri)
    uv_vec = np.transpose(np.mgrid[0:480, 0:640], (1, 2, 0))[:, :, [1, 0]].reshape((-1, 2))
    point_cloud = uv_vec2cloud(uv_vec, img_depth, depth_scale=1)
    return point_cloud

def rgb_callback(image):
    global rgb_data, img_rgb
    try:
        bridge = CvBridge()
        img_rgb = bridge.imgmsg_to_cv2(image, "bgr8")
        rgb_data = bridge.imgmsg_to_cv2(image, "bgr8").reshape((-1,3))
        # print(np.shape(cv_image))
    except Exception as e:
        print(e)
    

def pcd_callback(pcd_msg):
    global pcd_data
    pcd_data = np.array(list(pcl2.read_points(pcd_msg, skip_nans=True))).reshape((-1,3))

def depth_callback(depth_image):
    global pcd_data
    try:
        bridge = CvBridge()
        depth = bridge.imgmsg_to_cv2(depth_image,desired_encoding="16UC1")
        pcd_data = depth2cloud(depth)/1000# mm to m
    except Exception as e:
        print(e)
    


def publish_pcd(pub_pcd):
    global pcd_data,trans,rot
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "world"
    pcd = np.copy(pcd_data)
    R_world_camera = R.from_quat(rot).as_matrix()
    T_world_camera = np.array(trans).reshape((-1,3))
    pcd_send = np.matmul(R_world_camera, pcd.T).T
    pcd_send = pcd_send + T_world_camera
    pub_pcd.publish(pcl2.create_cloud_xyz32(header=header, points=pcd_send))

def publish_pcd_with_rgb(pub_pcd):
    global pcd_data,trans,rot,rgb_data
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "world"
    pcd = np.copy(pcd_data)
    R_world_camera = R.from_quat(rot).as_matrix()
    T_world_camera = np.array(trans).reshape((-1,3))
    pcd_send = np.matmul(R_world_camera, pcd.T).T
    pcd_send = pcd_send + T_world_camera
   
    rgb_send = np.copy(rgb_data)
    
    
    pcd_with_rgb_send = []
    num_points = np.shape(pcd_send)[0]



    for i in range(num_points):
        x = pcd_send[i, 0]
        y = pcd_send[i, 1]
        z = pcd_send[i, 2]
        r = rgb_send[i, 0]
        g = rgb_send[i, 1]
        b = rgb_send[i, 2]
        a = int(255)
        rgb = struct.unpack('I', struct.pack('BBBB', r, g, b, a))[0]
        pt = [x, y, z, rgb]
        pcd_with_rgb_send.append(pt)


    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        # PointField('rgb', 12, PointField.UINT32, 1),
        PointField('rgba', 12, PointField.UINT32, 1),
    ]
    pub_pcd.publish(pcl2.create_cloud(header=header, fields=fields, points=pcd_with_rgb_send))
    return pcd_with_rgb_send

    





def thread_job():
    rospy.spin()


if __name__ == "__main__":
    rospy.init_node("glass_depth_bridge")
    # sub_pcd = rospy.Subscriber("/camera/depth/points",PointCloud2, callback=pcd_callback)
    sub_tf = tf.TransformListener()
    sub_rgb  = rospy.Subscriber("/camera/color/image_raw", Image, callback=rgb_callback)
    sub_depth  = rospy.Subscriber("/camera/depth/image_raw", Image, callback=depth_callback)
    pub_pcd = rospy.Publisher("/points_in_world",PointCloud2,queue_size=10)
    rate = rospy.Rate(10)

    sub_pcd_thread = threading.Thread(target=thread_job)
    sub_pcd_thread.start()    
    time.sleep(2)


    while not rospy.is_shutdown():
        try:
            if not start_tf_listen:
                start_tf_listen = True
            (trans, rot) = sub_tf.lookupTransform("/world",
                                                  "/camera_color_optical_frame",
                                                  rospy.Time(0))
            pcd_with_rgb = []
            if start_tf_listen:
                # publish_pcd(pub_pcd=pub_pcd)
                pcd_with_rgb = publish_pcd_with_rgb(pub_pcd=pub_pcd)
            rate.sleep()
            cv2.imshow("Press Q to Save",img_rgb)
            key = cv2.waitKey(1)
            if key == ord("q"):
                save_path = "/media/yuxuan/My Passport/test_Samcon/pcd_data/"
                np.save(save_path+"pcd3.npy", pcd_with_rgb)
                np.save(save_path+"trans3.npy", trans)
                np.save(save_path+"R3.npy", rot)
                np.save(save_path+"img3.npy", img_rgb)


        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
    