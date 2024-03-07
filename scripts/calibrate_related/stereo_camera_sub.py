import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import os

input("Clear Current Data?")
save_path = "/media/yuxuan/My Passport/testTobbi/"
img_list = os.listdir(save_path)
for f in img_list:
    os.remove(save_path+f)

class VideoRecorder:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback_left)
        self.image_sub2 = rospy.Subscriber('/glass/color/image_raw', Image, self.image_callback_right)
        self.frames_right = []
        self.frames_left = []
        self.recording = False
        self.save_count = 0

 
    def image_callback_left(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.recording:
                self.frames_left.append(cv_image)
        except Exception as e:
            print(e)
 
    def image_callback_right(self, msg):
        try: 
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.recording:
                self.frames_right.append(cv_image)
        except Exception as e:
            print(e)

    def start_recording(self):
        self.frames_right = []
        self.frames_left = []
        self.recording = True
 
    def stop_recording(self):
        self.recording = False
        if self.frames_left and self.frames_right:
            self.save_frames()
 
    def save_frames(self):
        min_len = min(len(self.frames_left),len(self.frames_right))
        if min_len > 20:
            min_len = 20
        for i in range(min_len):
            filename = save_path+'frame_{:04d}_right.jpg'.format(self.save_count)
            cv2.imwrite(filename, self.frames_right[i])
            filename = save_path+'frame_{:04d}_left.jpg'.format(self.save_count)
            cv2.imwrite(filename, self.frames_left[i])
            print('Saved {} frames.'.format(self.save_count))
            print(np.shape(self.frames_right[i]))
            print(np.shape(self.frames_left[i]))
            self.save_count += 1

if __name__ == '__main__':
    rospy.init_node('video_recorder_node', anonymous=True)
    recorder = VideoRecorder()
 
    try:
        while not rospy.is_shutdown():
            cmd = input("Enter 's' to begin recording or 'q' to stop recording: ")
            if cmd == 's':
                recorder.start_recording()
            elif cmd == 'q':
                recorder.stop_recording()
    except rospy.ROSInterruptException:
        pass