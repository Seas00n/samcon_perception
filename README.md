# Samcon_perception

## 眼动仪标定：

使用前需要对每个受试者在windows系统下进行眼动仪标定

标定软件：https://connect.tobii.com/s/g2-downloads?language=en_US

标定参考软件内部指导，技术问题微信问客服（阳光森林）

标定时的主要问题是眼睛距离眼睛的距离一般需要调整，后面可以去眼镜店买鼻托

## 眼动仪配置：
终端下载第三方包
```
$ pip install tobiiglassesctrl
```

进入samcon_perception，看是否能成功导入包
```
$ python3
>>> from tobiiglassesctrl import TobiiGlassesController
```

## 眼动仪设备连接：
电脑搜索所有wifi，可以看到一个TG02B-xxxx的设备热点，输入密码连接
```
密码: TobiiGlasses
```

也可以选择有线连接，传输延时更低，但是目前没有配置成功

## 眼动仪程序运行：

在`SamconPros`下运行catkin build samcon_perception 即可，`samcon_controllers`为倒立摆等相关行走算法，如果后续继续开发可以参考，目前没想好lipm等怎么与眼动仪结合

保证连接后运行glass_publisher.py
```
$ source /SamconPros/devel/setup.bash
$ rosrun glass_publisher.py
```

如果连接正常就可以看到眼动仪的图像了，不正常就检查wifi连接
```
$ ping 192.168.71.50
```

## 深度相机驱动安装

奥比中光驱动，有问题淘宝问卖Orbbec Astra Mini的客服，注意相关包和ros版本对应
```
https://github.com/orbbec/ros_astra_camera
```
记得source一下setup.bash

```
$ roslaunch astra_camera astra.launch
```
用rviz查看相关消息

注意修改astra.launch文件中的以下参数才能把rgb和深度点云对齐
```
<arg name="enable_point_cloud_xyzrgb" default="true"/>
```
（或者直接把samcon_perception/launch文件夹下的astra.launch文件拷贝到ros_astra_camera/launch下替换就行）

## IMU购买和安装
淘宝IM948,9轴带壳
```
https://www.yuque.com/cxqwork/lkw3sg/yqa3e0?
```
使用蓝牙或串口连接均可（安装位置或驱动运行有问题直接联系我）

```
$ roslaunch samcon_perception multi_imu.launch
```
此时用rviz可以看到

## Samcon_perception算法测试

眼动仪和深度相机的支架设计图在文件夹有，一定保护好支架，因为两个相机的外参矩阵目前仍然需要使用先前的参数，重新标定难度较大，如果有兴趣可以尝试

眼动仪的算法主要在`plot_glass.py`和`plot_classification.py`内，目前项目只需要这两个即可

其他算法在`FPP`文件夹内，以及同名github仓库


运行眼动仪会发布glass消息和image消息
```
rosrun glass_publisher.py
```

运行深度相机会发布image消息和与rgb对准的点云消息
```
roslaunch astra_camera  astra.launch
```

运行imu会发布imu消息，并且发布tf消息，根据imu的角度得到相机坐标系到世界坐标系的变换
```
roslaunch samcon_perception multi_imu.launch
```

运行glass_depth_bridge2.py用来存储世界坐标系下的点云，该点云也被用在plot_glass.py和plot_classification.py中进行意图分类


## 任务：
将glass_depth_bridge2.py和plot_glass.py和plot_classification.py的逻辑整理为新的
glass_depth_bridge3.py可以实时进行眼动仪视线和深度信息的融合以及上几节楼梯的分类，后面结合假肢阻抗控制调节阻抗参数