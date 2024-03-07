# live_scene_and_gaze.py : A demo for video streaming and synchronized gaze
#
# Copyright (C) 2021  Davide De Tommaso
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import av
import cv2
import numpy as np
from tobiiglassesctrl import TobiiGlassesController
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from samcon_perception.msg import Glass_Gaze

rospy.init_node("glass_publisher",anonymous=True)
image_pub = rospy.Publisher("/glass/color/image_raw", Image, queue_size=10)
info_pub = rospy.Publisher("/glass/color/camera_info",CameraInfo,queue_size=10)
gaze_pub = rospy.Publisher("/glass/gaze",Glass_Gaze,queue_size=10)


save_path = "/media/yuxuan/My Passport/test_Samcon/pcd_data/"


ipv4_address = "192.168.71.50"

tobiiglasses = TobiiGlassesController(ipv4_address)
cap = cv2.VideoCapture("rtsp://%s:8554/live/scene" % ipv4_address)

tobiiglasses.start_streaming()

def camera_info():
    msg = CameraInfo()
    msg.header.stamp = rospy.Time.now()
    msg.height = 1920
    msg.width = 1080
    msg.distortion_model = "plumb_bob"
    msg.D = [0,0,0,0,0,0]
    msg.K = [0,0,0,0,0,0,0,0,0]
    msg.R =  [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.P = [446.8370666503906, 0.0, 333.6257734769097, 0.0, 0.0, 451.9144592285156, 237.7315547766393, 0.0, 0.0, 0.0, 1.0, 0.0]
    msg.binning_x = 0
    msg.binning_y = 0
    msg.roi.x_offset = 0
    msg.roi.y_offset = 0
    msg.roi.height = 0
    msg.roi.width = 0
    msg.roi.do_rectify = False
    return msg



bridge = CvBridge()
while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        height, width = img.shape[:2]

        header = Header(stamp=rospy.Time.now())
        header.frame_id = "camera_color_optical_frame"
        img_msg = bridge.cv2_to_imgmsg(img, encoding='bgr8')
        img_msg.header = header
        image_pub.publish(img_msg)
        info_msg = camera_info()
        info_pub.publish(info_msg)
        
        
        data_gp  = tobiiglasses.get_data()['gp']
        if data_gp['ts'] > 0:
            cv2.circle(img,(int(data_gp['gp'][0]*width),int(data_gp['gp'][1]*height)), 60, (0,0,255), 6)
            gaze_msg = Glass_Gaze()
            gaze_msg.header = header
            gaze_msg.width = width
            gaze_msg.height = height
            gaze_msg.u_2d = data_gp['gp'][0]
            gaze_msg.v_2d = data_gp['gp'][1]
            gaze_pub.publish(gaze_msg)
        cv2.namedWindow('Tobii Pro Glasses 2 - Live Scene', cv2.WINDOW_NORMAL)
        cv2.imshow('Tobii Pro Glasses 2 - Live Scene',img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    elif key == ord("s"):
        np.save(save_path+"imgglass.npy",img)
cv2.destroyAllWindows()

tobiiglasses.stop_streaming()
tobiiglasses.close()
