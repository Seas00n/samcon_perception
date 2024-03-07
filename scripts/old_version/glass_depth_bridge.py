import rospy
import numpy as np
import cv2
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from samcon_perception.msg import Glass_Gaze, Glass_Gait
from factor_function import *


class Glass_Depth_Bridge:
    def __init__(self):
        self.bridge = CvBridge()
        self.tracker_img_sub = message_filters.Subscriber('/glass/color/image_raw', Image)
        self.depth_img_sub = message_filters.Subscriber('/camera/depth/image_raw',Image)
        self.depth_rgb_sub = message_filters.Subscriber('/camera/color/image_raw',Image)
        self.gaze_sub = rospy.Subscriber('/glass/gaze',Glass_Gaze,self.gaze_callback)
        self.gait_pub = rospy.Publisher('/glass/gait',Glass_Gait,queue_size=10)
        self.gait_msg = Glass_Gait()
        self.gaze_buffer = np.zeros((10,3))
        self.median_gaze_2d = np.zeros((2,))
        self.gait_list = np.zeros((2,3))
        self.gaze_2d = np.zeros((2,))
        self.gaze_3d = np.zeros((3,))
        sync = message_filters.ApproximateTimeSynchronizer([self.tracker_img_sub,
                                                            self.depth_img_sub,
                                                            self.depth_rgb_sub],10,1)
        sync.registerCallback(self.sync_callback)

        self.tracker_mutex = False
        self.depth_mutex = False

        self.frames_tracker = [np.zeros(0,)]
        self.frames_depth = [np.zeros(0,)]
        self.frames_depth_color = [np.zeros(0,)]

        self.phase = 0 #touch

    def gaze_callback(self,data:Glass_Gaze):
        print("Gaze2d in tracker:{},{}".format(data.u_2d,data.v_2d))
        new_gaze = np.array([self.phase,data.u_2d,data.v_2d])
        self.gaze_buffer = fifo_vec(self.gaze_buffer, new_gaze)
        touch_list = np.diff(self.gaze_buffer[:,0])
        if self.gaze_buffer[-1,0]==0: #last touch
            if np.sum(self.gaze_buffer[:,0])>0:# stance and swing
               last_heel_strike = np.where(touch_list==-1)[0][-1]
               if np.shape(self.gaze_buffer)[0]-last_heel_strike>=5:
                    self.median_gaze_2d = np.median(self.gaze_buffer[last_heel_strike:,:],axis=0)
            else:
                self.median_gaze_2d = np.median(self.gaze_buffer[-6:,1:],axis=0)
        return
    
    def tracker_img_callback(self,data:Image):
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.tracker_mutex = True
            self.frames_tracker[0] = cv_image
            self.tracker_mutex = False
        except Exception as e:
            print(e)
        return
    
    def depth_img_callback(self, data:Image):
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth_mutex = True
            self.frames_depth[0] = np.array(cv_image, dtype=np.float32)
            self.depth_mutex = False
        except Exception as e:
            print(e)
        return

    def depth_rgb_callback(self,data:Image):
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.circle(cv_image,(int(self.gaze_2d[0]),data.height-int(self.gaze_2d[1])),
                       40, (0,0,255), 6)
            self.frames_depth_color[0] = cv_image
        except Exception as e:
            print(e)
        return

    def sync_callback(self,data1,data2,data3):
        self.tracker_img_callback(data1)
        self.depth_img_callback(data2)    
        self.depth_rgb_callback(data3)
        self.gaze_3d, self.gaze_2d = tracker_to_rgbd_xyz(self.median_gaze_2d, self.frames_depth[0])
        print("Gaze2d in depth:{},{}".format(self.gaze_2d[0],self.gaze_2d[1]))

    def check_touch(self):
        if self.phase == 0:
            self.phase = 0
        else:
            self.phase = 1
        return


    
    def gait_update(self):
        if not self.depth_mutex and not self.tracker_mutex:
            # self.gaze_3d, self.gaze_2d = tracker_to_rgbd_xyz(self.median_gaze_2d, self.frames_depth[0])
            # print("Gaze2d in depth:{},{}".format(self.gaze_2d[0],self.gaze_2d[1]))
            if self.phase == 0:
                self.gait_list[1,:] = self.gaze_3d
            else:
                self.gait_list[0,:] = self.gait_list[1,:]
        self.gait_msg.header = Header()
        self.gait_msg.header.stamp = rospy.Time.now()
        self.gait_msg.last_step_x = self.gait_list[0,0]
        self.gait_msg.last_step_y = self.gait_list[0,1]
        self.gait_msg.last_step_z = self.gait_list[0,2]
        self.gait_msg.next_step_x = self.gait_list[1,0]
        self.gait_msg.next_step_y = self.gait_list[1,1]
        self.gait_msg.next_step_z = self.gait_list[1,2]
        self.gait_pub.publish(self.gait_msg)
        return
    

if __name__ == '__main__':
    rospy.init_node('glass_depth_bridge', anonymous=True)
    brg = Glass_Depth_Bridge()
    rate = rospy.Rate(10)

    
    try:
        while not rospy.is_shutdown():
            if np.shape(brg.frames_depth[0])[0]>0 and np.shape(brg.frames_tracker[0])[0]>0:
                # both sub receive image
                brg.gait_update()
                if not brg.depth_mutex:
                    depth_rgb_img = brg.frames_depth_color[0]
                    cv2.imshow("depth_rgb", depth_rgb_img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            rate.sleep()
    except rospy.ROSInternalException:
        pass
    cv2.destroyAllWindows()
