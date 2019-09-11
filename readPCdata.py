import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import pptk
def callback(data):
    pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
    pc_list = []
    for p in pc:
        pc_list.append( [p[0],p[1],p[2]] )

    p = pcl.PointCloud()
    p.from_list(pc_list)
    p.to_file("Kinect.pcd")
    a = np.asarray(p)
    #pptk.viewer(a)

def listener():
    rospy.init_node('listener', anonymous=True)
    #rate = rospy.Rate(10)
    rospy.Subscriber('/camera/depth/points', PointCloud2, callback)
    #rate.sleep()
    rospy.spin()


if __name__ == '__main__':
    listener()