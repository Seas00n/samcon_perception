import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import struct
from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from scipy.spatial import ConvexHull
import cv2

save_path = "/media/yuxuan/My Passport/test_Samcon/pcd_data/"

idx = ""

pcd_with_rgb = np.load(save_path+"pcd{}.npy".format(idx))
pcd = pcd_with_rgb[:,0:3]
color = []
for i in range(np.shape(pcd_with_rgb)[0]):
    rgba = struct.unpack('BBBB', struct.pack('I', int(pcd_with_rgb[i,3])))
    color.append(rgba)
color = np.array(color)[:,0:3]
color = color/255

print(np.shape(color))
print(np.shape(pcd))

trans = np.load(save_path+"trans{}.npy".format(idx))
R_mat = np.load(save_path+"R{}.npy".format(idx))
img_rgb = np.load(save_path+"img.npy".format(idx))

eular = R.from_quat(R_mat).as_euler('xyz',degrees=True)
R_z = R.from_euler('xyz',[0, 0, -eular[2]],degrees=True).as_matrix()
pcd = np.matmul(R_z, (pcd-trans).T).T

pcd_chosen_id = np.where(
    np.logical_and(
        np.abs(pcd[:,0])<0.2,
        pcd[:,1]<1,
        pcd[:,2]<0.5
    )
)[0]

pcd = pcd[pcd_chosen_id,:]
color = color[pcd_chosen_id,:]

pcd_delete_id = np.where(
    np.linalg.norm(pcd,axis=1)<0.1
)[0]

pcd = np.delete(pcd, pcd_delete_id, axis=0)
color = np.delete(color, pcd_delete_id, axis=0)

gaze_glass = np.load(save_path+"gazeglass{}.npy".format(idx))
gaze_depth = np.load(save_path+"gazedepth{}.npy".format(idx))
gaze_3d = np.load(save_path+"gaze3d{}.npy".format(idx))

gaze_3d_in_world = gaze_3d
gaze_3d_in_track = np.matmul(R_z, (gaze_3d-trans).T).T

#####classification
median_gaze_3d = np.median(gaze_3d_in_track, axis=0)
idx_next_stance = np.where(np.abs(pcd[:,2]-median_gaze_3d[2])<0.02)[0]
pcd_stance = pcd[idx_next_stance,:]
#####belief visualization
mu_x = np.mean(gaze_3d_in_track[:,0], axis=0)
mu_y = np.mean(gaze_3d_in_track[:,1], axis=0)
sigma_x = np.std(gaze_3d_in_track[:,0], axis=0)
sigma_y = np.std(gaze_3d_in_track[:,1], axis=0)
rv = multivariate_normal([mu_x,mu_y], [[sigma_x,0],[0,sigma_y]])
x = np.linspace(np.min(pcd_stance[:,0]),np.max(pcd_stance[:,0]),30)
y = np.linspace(np.min(pcd_stance[:,1]),np.max(pcd_stance[:,1]),30)
X, Y = np.meshgrid(x,y)
Z = np.ones_like(X)*median_gaze_3d[2]
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
pd = rv.pdf(pos)
#####cam pose visualization
hull = ConvexHull(gaze_3d_in_track[:,0:2])




cmap = 'viridis'
def color_map(data, cmap):
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256/cmo.N
    
    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i*k), int((i+1)*k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255*(data-dmin)/(dmax-dmin))
    return cs[data]

pd_color = color_map(pd, cmap)[:,:,0:3]


#######################

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pcd[0::5,0], 
           pcd[0::5,1],
           pcd[0::5,2],
           c=color[0::5,:],
           s=10,alpha=0.05,marker='o')

for i in range(np.shape(gaze_3d_in_track)[0]):
    u, v = np.mgrid[0:2*np.pi:10j,0:2*np.pi:10j]
    r = 0.005
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    ax.plot_surface(
        x+gaze_3d_in_track[i,0],
        y+gaze_3d_in_track[i,1],
        z+gaze_3d_in_track[i,2]+0.02,
        edgecolors='b',
        alpha=0.5
    )
u, v = np.mgrid[0:2*np.pi:10j,0:2*np.pi:10j]
r = 0.01
x = r*np.cos(u)*np.sin(v)
y = r*np.sin(u)*np.sin(v)
z = r*np.cos(v)
ax.plot_surface(
    x+median_gaze_3d[0],
    y+median_gaze_3d[1],
    z+median_gaze_3d[2]+0.04,
    edgecolors='r'
)

ax.scatter(X.reshape((-1,1)),
           Y.reshape((-1,1)),
           Z.reshape((-1,1))-0.01,
           c=pd_color.reshape((-1,3)),s=5,alpha=0.5)


for i in hull.simplices:
    ax.plot3D(
        gaze_3d_in_track[i,0],
        gaze_3d_in_track[i,1],
        gaze_3d_in_track[i,2]+0.02,
        color='black',
        lw=2
    )
    faces = Poly3DCollection([
        np.array([[0,0,0],
                  [gaze_3d_in_track[i,0][0],gaze_3d_in_track[i,1][0],gaze_3d_in_track[i,2][0]+0.02],
                  [gaze_3d_in_track[i,0][1],gaze_3d_in_track[i,1][1],gaze_3d_in_track[i,2][1]+0.02]])
    ])
    faces.set_edgecolor('')
    faces.set_alpha(0.1)
    ax.add_collection3d(faces)

plane_zaxis = median_gaze_3d/np.linalg.norm(median_gaze_3d)
axis_theta = np.arccos(np.dot(np.array([0,0,1]),plane_zaxis))
axis_v1v2 = np.cross(np.array([0,0,1]),plane_zaxis)
axis_v1v2 = axis_v1v2/np.linalg.norm(axis_v1v2)
R_mat,_ = cv2.Rodrigues(axis_theta*axis_v1v2)
points_cam = np.array([
    [0.02,-0.02,-0.02,0.02],
    [0.02,0.02,-0.02,-0.02],
    [0.1,0.1,0.1,0.1]
])
points_cam = np.matmul(R_mat,points_cam).T
for i in range(np.shape(points_cam)[0]):
    if i < np.shape(points_cam)[0]-1:
        faces = Poly3DCollection([
            np.array([[0,0,0],
                      [points_cam[i,0],points_cam[i,1],points_cam[i,2]],
                      [points_cam[i+1,0],points_cam[i+1,1],points_cam[i+1,2]]])
        ])
    else:
        faces = Poly3DCollection([
            np.array([[0,0,0],
                      [points_cam[i,0],points_cam[i,1],points_cam[i,2]],
                      [points_cam[0,0],points_cam[0,1],points_cam[0,2]]])
        ])
    faces.set_edgecolor('black')
    faces.set_alpha(0.2)
    ax.add_collection3d(faces)

# ax._axis3don = False
# ax.set_facecolor("gray")
ax.set_xlim3d([-0.3,0.3])
ax.set_ylim3d([0,0.6])
ax.set_zlim3d([-1,0])
plt.show()