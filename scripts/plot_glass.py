import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import struct
from scipy.spatial.transform import Rotation as R
from factor_function import *
save_path = "/media/yuxuan/My Passport/test_Samcon/pcd_data/"

idx = "2"

pcd_with_rgb = np.load(save_path+"pcd{}.npy".format(idx))
trans = np.load(save_path+"trans{}.npy".format(idx))
R_mat = np.load(save_path+"R{}.npy".format(idx))
img_rgb = np.load(save_path+"img.npy".format(idx))
img_glass = np.load(save_path+"imgglass{}.npy".format(idx))

color = []
for i in range(np.shape(pcd_with_rgb)[0]):
    rgba = struct.unpack('BBBB', struct.pack('I', int(pcd_with_rgb[i,3])))
    color.append(rgba)
color = np.array(color)[:,0:3]
color = color/255

pcd = pcd_with_rgb[:,0:3]
pcd_in_world = pcd.reshape(np.shape(img_rgb)).reshape((-1,3))



print("v:{},u:{}".format(np.shape(img_glass)[1],np.shape(img_glass)[0]))
print("v:{},u:{}".format(np.shape(img_rgb)[1], np.shape(img_rgb)[0]))



uv_vec_glass = np.transpose(np.mgrid[0:1080, 0:1920], (1, 2, 0)).reshape((-1,2))
uv_vec_cam = np.transpose(np.mgrid[0:480, 0:640], (1, 2, 0)).reshape((-1,2))



fig = plt.figure()
grid = plt.GridSpec(1,2,wspace=0.5,hspace=0.5)
ax1 = plt.subplot(grid[0,0])
ax2 = plt.subplot(grid[0,1])

ax1.imshow(img_glass)
ax1.set_xlabel('v')
ax1.set_ylabel('u')
ax2.imshow(img_rgb)
ax2.set_xlabel('v')
ax2.set_ylabel('u')

fig2 = plt.figure()
ax3 = fig2.add_subplot(111,projection='3d')
ax3.scatter(
    pcd_in_world[0::20,0],
    pcd_in_world[0::20,1],
    pcd_in_world[0::20,2],
    c=color[0::20],
    s=5,
    alpha=0.1
)





gaze_in_glass = []
gaze_in_depth = []
gaze_3d = []
def on_press(event):
    global gaze_in_glass, gaze_in_depth, gaze_3d
    global ax1, ax2
    global R_mat
    '''
    0-----v
    |
    |
    u
    '''
    u = int(event.ydata)
    v = int(event.xdata)
    gaze_in_glass.append([v, u])
    ax1.scatter(v,u,color='b',alpha=0.5)
    print("Gaze at:[v,u]",v,u)

    "----From Kuangen"
    uv_tracker = np.array(gaze_in_glass[-1]).reshape([1,-1])
    cam_matrix_tracker = rgbd_tracker_paras.get('cam_matrix_tracker')
    T_tracker_in_depth = np.linalg.inv(rgbd_tracker_paras.get('T_mat_depth_in_tracker'))
    p_eye_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([0]),
                                      cam1_matrix=cam_matrix_tracker,
                                      T_cam1_in_cam2=T_tracker_in_depth)[:3].T
    eular = R.from_quat(R_mat).as_euler('xyz',degrees=True)
    R_cam = R.from_euler('xyz',[eular[0], eular[1], eular[2]],degrees=True).as_matrix()
    p_eye_in_depth = np.matmul(R_cam, p_eye_in_depth.T/1000).T+trans
    p_virtual_point_in_depth = cam1_to_cam2_xyz(uv_tracker, np.asarray([1500]),
                                                cam1_matrix=cam_matrix_tracker,
                                                T_cam1_in_cam2=T_tracker_in_depth)[:3].T
    p_virtual_point_in_depth = np.matmul(R_cam, p_virtual_point_in_depth.T/1000).T+trans

    distance = np.linalg.norm(
        np.cross(
            pcd_in_world-p_eye_in_depth,
            pcd_in_world-p_virtual_point_in_depth
        ),axis=-1
    )
    distance/=np.linalg.norm(p_eye_in_depth-p_virtual_point_in_depth,axis=-1)
    u, v = uv_vec_cam[np.argmin(distance)]
    ax2.scatter(v,u,color='g',alpha=0.5)
    print("In Depth [v,u]:",[v,u])
    gaze_in_depth.append([v,u])

    p_gaze3d = pcd_in_world[np.argmin(distance)]
    ax3.scatter(
        p_gaze3d[0],
        p_gaze3d[1],
        p_gaze3d[2],
        c='r',
        marker='o',
        s=100
    )
    gaze_3d.append(p_gaze3d)

fig.canvas.mpl_connect('button_press_event', on_press)
plt.ion()




try:
    while(len(gaze_in_glass)<10):
        plt.pause(0.5)
except KeyboardInterrupt:
    plt.close()


plt.ioff()
gaze_in_glass = np.array(gaze_in_glass)
median_gaze_in_glass = np.median(gaze_in_glass,axis=0).astype(np.int16)
gaze_in_depth = np.array(gaze_in_depth)
median_gaze_in_depth = np.median(gaze_in_depth,axis=0).astype(np.int16)
circle1 = plt.Circle(median_gaze_in_glass, radius=40, color='r', fill=False)
circle2 = plt.Circle(median_gaze_in_depth, radius=20, color='r', fill=False)
ax1.add_patch(circle1)
ax2.add_patch(circle2)
ax1.scatter(median_gaze_in_glass[0], median_gaze_in_glass[1],marker="*", color="r",
            linewidths=5)
ax2.scatter(median_gaze_in_depth[0], median_gaze_in_depth[1],marker="*", color="r",
            linewidths=5)
plt.show(block=True)
gaze_3d = np.array(gaze_3d)
# np.save(save_path+"gazeglass{}.npy".format(idx), gaze_in_glass)
# np.save(save_path+"gazedepth{}.npy".format(idx), gaze_in_depth)
# np.save(save_path+"gaze3d{}.npy".format(idx), gaze_3d)
