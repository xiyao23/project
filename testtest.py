import rospy
import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import ros_numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pypcd
import pptk
import random
#import pcl.pcl_visualization
import argparse
import tf
import roslib
import math
import threading
import time
from sklearn.cluster import KMeans
import cv2
import os

pcloud = pcl.load('cokecan.pcd')
pcloud_array=np.asarray(pcloud)
pptk.viewer(pcloud_array)