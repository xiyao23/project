import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
def callback(msg):
    print("Received an image!")
    bridge = CvBridge()
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg)
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        cv2.imwrite('040.jpg', cv2_img)


def listener():
    #creat a topic node
    rospy.init_node('listener', anonymous=True)
    #Define subscribing data as image
    rospy.Subscriber("/camera/depth/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()