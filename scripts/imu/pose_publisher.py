import rospy
from std_msgs.msg import String
import tf2_ros
import gatt
import sys
import numpy as np
import cv2
from scipy import io
from sensor_msgs.msg import Imu
import time
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import TransformStamped



def publish_imu(imu_pub,imu_data):
    imu = Imu()
    imu.header.frame_id = "world"
    imu.header.stamp = rospy.Time.now()
    imu.linear_acceleration.x = imu_data[0]
    imu.linear_acceleration.y = imu_data[1]
    imu.linear_acceleration.z = imu_data[2]
    imu.angular_velocity.x = imu_data[3]*np.pi/180
    imu.angular_velocity.y = imu_data[4]*np.pi/180
    imu.angular_velocity.z = imu_data[5]*np.pi/180
    imu.orientation.x = imu_data[9]
    imu.orientation.y = imu_data[10]
    imu.orientation.z = imu_data[11]
    imu.orientation.w = imu_data[12]
    imu_pub.publish(imu)
    return imu

def publish_tf(tf_pub, imu_data):
    qx = imu_data[9]
    qy = imu_data[10]
    qz = imu_data[11]
    qw = imu_data[12]
    R_world_imu = R.from_quat([qx,qy,qz,qw]).as_matrix()
    R_imu_cam = R.from_euler('xyz',[0,0,90],degrees=True).as_matrix()
    R_world_cam = np.matmul(R_world_imu,R_imu_cam)
    tf_world_cam = TransformStamped()
    tf_world_cam.header.frame_id = "base"
    tf_world_cam.header.stamp = rospy.Time.now()
    tf_world_cam.child_frame_id = "camera_link"
    q_cam = R.from_matrix(R_world_cam).as_quat()
    tf_world_cam.transform.rotation.x = q_cam[0]
    tf_world_cam.transform.rotation.y = q_cam[1]
    tf_world_cam.transform.rotation.z = q_cam[2]
    tf_world_cam.transform.rotation.w = q_cam[3]
    tf_pub.sendTransform(tf_world_cam)

def publish_static_tf(tf_pub):
    tf_world_base = TransformStamped()
    tf_world_base.header.frame_id = "world"
    tf_world_base.header.stamp = rospy.Time.now()
    tf_world_base.child_frame_id = "base"
    tf_world_base.transform.translation.z = 1
    q_base = R.from_euler("xyz",[0,0,0]).as_quat()
    tf_world_base.transform.rotation.x = q_base[0]
    tf_world_base.transform.rotation.y = q_base[1]
    tf_world_base.transform.rotation.z = q_base[2]
    tf_world_base.transform.rotation.w = q_base[3]
    tf_pub.sendTransform(tf_world_base)


if __name__ == "__main__":
    rospy.init_node("pose_publisher", anonymous=True)
    tf_imu_camera_pub = tf2_ros.TransformBroadcaster()
    tf_world_base_pub = tf2_ros.StaticTransformBroadcaster()

    imu_head_pub = rospy.Publisher("imu_head_pub",Imu, queue_size = 10)

    rate = rospy.Rate(100)

    imu_head_bf = np.memmap("/home/yuxuan/Project/SamconPros/src/samcon_perception/scripts/imu/imu_head.npy", dtype='float32', mode='r',
                               shape=(13,))
    
    
    img = np.zeros((300, 300), np.uint8)
    # 浅灰色背景
    img.fill(200)

    t0 = time.time()

    while not rospy.is_shutdown():
        try:
            imu_head_data = np.copy(imu_head_bf[0:])
            imu_head = publish_imu(imu_head_pub,imu_data = imu_head_data)
            publish_static_tf(tf_world_base_pub)
            publish_tf(tf_imu_camera_pub,imu_head_data)
            rospy.loginfo("Head:x=%.2f, y=%.2f, z=%.2f",
                        imu_head_bf[6], imu_head_bf[7], imu_head_bf[8])

            cv2.imshow("Press q to stop imu", img)
            if cv2.waitKey(1) == ord('q'):
                break
        except Exception as e:
            rospy.logerr("Exception:%s", e)
            break
        time.sleep(0.005)
