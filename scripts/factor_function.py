import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import pyransac3d as pyrsc

default_cam_intri = np.asarray(
    [[521.85359567,   0.        , 321.18647073],
    [0.        , 521.7098714 , 233.81475134],
    [0.        ,   0.        ,   1.        ]])
rgbd_tracker_paras = np.load('/home/yuxuan/Project/SamconPros/src/samcon_perception/scripts/rgbd_tracker_paras.npy', allow_pickle=True).item()

def uv_vec2cloud(uv_vec, depth_img, depth_scale = 1e-3):
    '''
        uv_vec: n×2,
        depth_image: rows * cols
    '''
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
    if cam_intri_inv is None:
        cam_intri_inv = np.zeros((1, 1, 3, 3))
        cam_intri_inv[0, 0] = np.linalg.inv(default_cam_intri)
    uv_vec = np.transpose(np.mgrid[0:480, 0:640], (1, 2, 0))[:, :, [1, 0]].reshape((-1, 2))
    point_cloud = uv_vec2cloud(uv_vec, img_depth, depth_scale=1)
    return point_cloud

def cam1_to_cam2_xyz(uv_vec_1, z_vec_1, cam1_matrix, T_cam1_in_cam2):
    uv_vec_1 = uv_vec_1.reshape((-1, 2))
    num_points = uv_vec_1.shape[0]
    uv_vec_1 = np.asarray(uv_vec_1)
    uvz_vec_1 = np.ones((num_points, 3))
    uvz_vec_1[:, :2] = uv_vec_1
    uvz_vec_1 *= np.reshape(z_vec_1, (-1, 1))
    p_vec_1 = np.ones((4, num_points))
    p_vec_1[:3, :] = np.matmul(np.linalg.inv(cam1_matrix), uvz_vec_1.T)
    p_vec_2 = np.matmul(T_cam1_in_cam2, p_vec_1)
    return p_vec_2

def tracker_to_rgbd_without_depth(uv_tracker, img_depth, rgbd_tracker_paras):
    uv_tracker = uv_tracker.reshape([1, -1])
    cam_matrix_tracker = rgbd_tracker_paras.get('cam_matrix_tracker')
    T_tracker_in_depth = np.linalg.inv(rgbd_tracker_paras.get('T_mat_depth_in_tracker'))
    points_in_depth = depth2cloud(img_depth)

    start = time.time()
    p_eye_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([0]),
                                      cam1_matrix=cam_matrix_tracker,
                                      T_cam1_in_cam2=T_tracker_in_depth)[:3].T
    p_virtual_point_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([1000]),
                                                cam1_matrix=cam_matrix_tracker,
                                                T_cam1_in_cam2=T_tracker_in_depth)[:3].T

    indices = np.arange(points_in_depth.shape[0])
    np.random.shuffle(indices)

    points_in_depth = points_in_depth[indices[:10000]]
    indices = indices[:10000]

    valid_indices = np.abs(points_in_depth[:, 2]) > 100
    points_in_depth = points_in_depth[valid_indices]
    indices = indices[valid_indices]

    distance = np.linalg.norm(np.cross(points_in_depth - p_eye_in_depth,
                                       points_in_depth-p_virtual_point_in_depth), axis=-1)
    distance /= np.linalg.norm(p_eye_in_depth - p_virtual_point_in_depth, axis=-1)
    index_min = indices[np.argmin(distance)]
    v = int(np.floor(index_min / img_depth.shape[1]))
    u = index_min - v * img_depth.shape[1]
    uv_est_in_depth = np.asarray([u, v])
    print('Computing time of 2D gaze to 3D gaze: {:0f} ms'.format(1000 * (time.time() - start)))
    return uv_est_in_depth
def uvz2xyz(u, v, d, cam_intri_inv= None):
    if cam_intri_inv is None:
        cam_intri_inv = np.linalg.inv(default_cam_intri)
    uvd_vec = np.asarray([u, v, 1]) * d
    xyz = np.matmul(cam_intri_inv, uvd_vec)
    return xyz
def tracker_to_rgbd_xyz(gaze_2d,img_depth,depth_scale=1000):
    '''
    Input:
    gaze 2d: u in [0, 1], v in [0, 1]
    image depth: unit mm
    Output:
    xyz: unit m
    '''
    uv_in_tracker = gaze_2d * np.asarray([1920, 1080])
    u, v = tracker_to_rgbd_without_depth(uv_in_tracker, img_depth, rgbd_tracker_paras)
    d = img_depth[v, u] / depth_scale
    return uvz2xyz(u, v, d), (u,v)

def fifo_vec(data_vec, data):
    data_vec[0:-1,:] = data_vec[1:,:]
    data_vec[-1,:] = data
    return data_vec