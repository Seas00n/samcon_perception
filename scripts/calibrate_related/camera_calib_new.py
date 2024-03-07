import cv2
import numpy as np
import os
import glob

def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    rimg_copy = cv2.resize(rimg,(WIDTH,HEIGHT))
    imgcat = np.zeros((HEIGHT*2+10, WIDTH,3))
    imgcat[:HEIGHT,:,:] = limg
    imgcat[-HEIGHT:,:,:] = rimg_copy
    for i in range(int(WIDTH / 32)):
        imgcat[:,i*32,:] = 255 
    return imgcat.astype('uint8')

save_path = '/media/yuxuan/My Passport/testTobbi/'
leftpath = glob.glob(save_path+"frame_*_left.jpg")
rightpath = glob.glob(save_path+"frame_*_right.jpg")
CHECKERBOARD = (8,6)  #棋盘格内角点数
square_size = (24,24)   #棋盘格大小，单位mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
imgpoints_l = []    #存放左图像坐标系下角点位置
imgpoints_r = []    #存放左图像坐标系下角点位置
objpoints = []   #存放世界坐标系下角点位置
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
# objp[0,:,0] *= square_size[0]
# objp[0,:,1] *= square_size[1]


for ii in range(len(leftpath)):
    img_l = cv2.imread(leftpath[ii])
    gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    img_r = cv2.imread(rightpath[ii])
    gray_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD)   #检测棋盘格内角点
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD)
    if ret_l and ret_r:
        objpoints.append(objp)
        corners2_l = cv2.cornerSubPix(gray_l,corners_l,(11,11),(-1,-1),criteria) 
        imgpoints_l.append(corners2_l)
        corners2_r = cv2.cornerSubPix(gray_r,corners_r,(11,11),(-1,-1),criteria)
        imgpoints_r.append(corners2_r)
        img = cv2.drawChessboardCorners(img_l, CHECKERBOARD, corners2_l,ret_l)
        cv2.imshow('img',img)
        cv2.waitKey(10)
cv2.destroyAllWindows()
print("MonoCalibrate...........")
ret, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1],None,None)  #先分别做单目标定
ret, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1],None,None)
print("StereoCalibrate............")
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
flags |= cv2.CALIB_USE_INTRINSIC_GUESS
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
    cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, gray_r.shape[::-1],
                        criteria=criteria,flags=flags)   #再做双目标定

print("stereoCalibrate : \n")
print("Camera matrix left : \n")
print(cameraMatrix1)
print("distCoeffs left  : \n")
print(distCoeffs1)
print("cameraMatrix right : \n")
print(cameraMatrix2)
print("distCoeffs right : \n")
print(distCoeffs2)
print("R : \n")
print(R)
print("T : \n")
print(T)
print("E : \n")
print(E)
print("F : \n")
print(F)

left_image = cv2.imread(leftpath[10])
right_image = cv2.imread(rightpath[10])
imgcat_source = cat2images(left_image,right_image)
HEIGHT = right_image.shape[0]
WIDTH = right_image.shape[1]
cv2.imshow('source_l&r', imgcat_source)
cv2.waitKey(0)
cv2.destroyAllWindows()

(R_l, R_r, P_l, P_r, Q, validPixROI1, validPixROI2) = \
    cv2.stereoRectify(cameraMatrix1, dist_l, cameraMatrix2, dist_r, np.array([WIDTH,HEIGHT]), R, T) #计算旋转矩阵和投影矩阵

(map1, map2) = \
    cv2.initUndistortRectifyMap(cameraMatrix1, dist_l, R_l, P_l, np.array([WIDTH,HEIGHT]), cv2.CV_32FC1) #计算校正查找映射表
rect_left_image = cv2.remap(left_image, map1, map2, cv2.INTER_CUBIC) #重映射
rect_right_image = cv2.remap(right_image, map1, map2, cv2.INTER_CUBIC)
imgcat_out = cat2images(rect_left_image,rect_right_image)
cv2.imshow('remap_l&r', imgcat_out)
cv2.waitKey(0)
cv2.destroyAllWindows()