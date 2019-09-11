import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import ros_numpy

'''
def callback(data):
    nu=ros_numpy.numpify(data)
    print(nu)
'''
def callback(data):
    pc=ros_numpy.numpify(data)




def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/camera/depth/points', PointCloud2, callback)
    rospy.spin()





if __name__ == '__main__':
    listener()


    
