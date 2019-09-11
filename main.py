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


template='beer.pcd'
def callback(data):
    objectNum = 2
    maxIteration = 30
    tolerance = 0.0005
    controlPoints = 500
    turtlename = 'beer'
    Imgtemp='beer.jpg'
    pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
    pc_list = []
    for p in pc:
        pc_list.append( [p[0],p[1],p[2]] )

    p = pcl.PointCloud()
    p.from_list(pc_list)
    p.to_file("inliers.pcd")
    a = np.asarray(p)
    x = a[:, 0]
    y = a[:, 1]
    z = a[:, 2]
    p_array=np.asarray(p)
    #print(p_array)
    #pptk.viewer(p_array)


    PC1= reader()
    PC2,Muti_PC2= TargetPreprocessing(p)


    if objectNum==1:
        Source_array, Target_array = Downsample(PC1, PC2)
        R, T, A, E = ICPRegistration(Source_array, Target_array, maxIteration, tolerance, controlPoints)
        MaxP = np.max(Target_array, axis=0)
        MinP = np.min(Target_array, axis=0)
        objSize = np.linalg.norm(MaxP - MinP)
        print('the object size is: ')
        print(objSize)
        #pptk.viewer(A,Target_array)

        data_x = np.delete(Target_array, 2, axis=1)
        # print(data_x)
        '''
        sample_x = random.sample(range(data_x.shape[0]), 300)
        Xdata = np.array([data_x[i] for i in sample_x])
        
        A1=A.T
        T1=Target_array.T
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(A1[0], A1[1], A1[2], c='red')
        ax.scatter(T1[0], T1[1], T1[2], c='blue')
        ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
        ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
        ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
        plt.show()
        '''

        print('------begin loading, Be careful------')
        print('------begin loading, Be careful------')
        print('------begin loading, Be careful------')
        print(len(data_x))
        '''
        Xdata1=data_x.T
        plt.figure(1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')  
        plt.scatter(Xdata1[0], Xdata1[1])
        #plt.show()
        #plt.savefig("40.jpg")
        '''
        '''
        xf=open('X_train.txt','a')
        np.savetxt(xf,Xdata, fmt='%f', delimiter=',')
        xf.write('???\n')
        xf.close()
        yf=open('Y_train.txt','a')
        yf.write('1,')
        yf.close()
        '''

        # R, T, A = ICPRegistration(Source_array, Target_array, maxIteration=50, tolerance=0.0005, controlPoints=500)
        # print('the corresponding transform is:')
        # print(T)
        # print(R)
        # print(A)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        E = np.array([x, y, z])
        print('yaw,pitch,roll is:')
        print(E)
        ori_Target = np.mean(Target_array, axis=0)
        print('orignal poisition:')
        print(ori_Target)

        PosTarget = np.mean(A, axis=0)
        print('Estimation Position:')
        print(PosTarget)
        DisE = np.linalg.norm(ori_Target - PosTarget)
        print('Euclidean Distance')
        print(DisE)
        offset = DisE / objSize
        print('offset value is:')
        print(offset)

        ExpData1=open('KinectDis.txt','a')
        ExpData1.write(str(offset))
        ExpData1.write('\n')
        ExpData1.close()
        ExpData2=open('KinectEuler.txt','a')
        ExpData2.write(str(E))
        ExpData2.write('\n')
        ExpData2.close()

        # a,d,p=tf.transformations.rotation_from_matrix(R)
        a = tf.transformations.quaternion_from_euler(x, y, z)
        print(a)

        br.sendTransform((PosTarget[0], PosTarget[1], PosTarget[2]), tf.transformations.quaternion_from_euler(x, y, z),
                         rospy.Time.now(), turtlename, "camera_link")
        #os.remove('1.jpg')
        #os.remove('2.jpg')
        print('program finish')
        #time.sleep(35)

    if objectNum==2:
        fa1,fa2=MutiObSeg(Muti_PC2)
        Source = cv2.imread(Imgtemp)
        img1 = cv2.imread('1.jpg',0)
        img2 = cv2.imread('2.jpg',0)
        img1C = Canny(img1)
        img2C = Canny(img2)
        cv2.imwrite('3.jpg', img1C)
        cv2.imwrite('4.jpg', img2C)
        #im1 = cv2.imread('1.jpg')
        #im2 = cv2.imread('2.jpg')
        MPNr1 = Flann(Source, img1C)
        MPNr2 = Flann(Source, img2C)
        if(MPNr1==MPNr2):
            print('do it again')
            MPNr1 = Flann(Source, img1C)
            MPNr2 = Flann(Source, img2C)
        print(MPNr1)
        print(MPNr2)

        #s1=compare_Sim(Source,im1)
        #s2=compare_Sim(Source,im2)
        #cv2.imwrite('bowl.jpg',img2C)
        Source_array, Target_array = Downsample(PC1, PC2)
        if MPNr1>MPNr2:
            Source_array, fa1_array = Downsample(PC1, fa1)
            #R, T, A, E = ICPRegistration(Source_array, fa1_array, maxIteration, tolerance, controlPoints)
            Target_array=fa1_array
        if MPNr1<MPNr2:
            Source_array, fa2_array = Downsample(PC1, fa2)
            #R, T, A, E = ICPRegistration(Source_array, fa2_array, maxIteration, tolerance, controlPoints)
            Target_array=fa2_array
        R, T, A, E = ICPRegistration(Source_array,Target_array, maxIteration, tolerance, controlPoints)
        MaxP = np.max(Target_array, axis=0)
        MinP = np.min(Target_array, axis=0)
        objSize = np.linalg.norm(MaxP - MinP)
        print('the object size is: ')
        print(objSize)

        data_x = np.delete(Target_array, 2, axis=1)
        # print(data_x)
        '''
        sample_x = random.sample(range(data_x.shape[0]), 300)
        Xdata = np.array([data_x[i] for i in sample_x])
        '''

        print('------begin loading, Be careful------')
        print('------begin loading, Be careful------')
        print('------begin loading, Be careful------')
        print(len(data_x))
        '''
        Xdata1=data_x.T
        plt.figure(1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')  
        plt.scatter(Xdata1[0], Xdata1[1])
        #plt.show()
        #plt.savefig("40.jpg")
        '''
        '''
        xf=open('X_train.txt','a')
        np.savetxt(xf,Xdata, fmt='%f', delimiter=',')
        xf.write('???\n')
        xf.close()
        yf=open('Y_train.txt','a')
        yf.write('1,')
        yf.close()
        '''

        # R, T, A = ICPRegistration(Source_array, Target_array, maxIteration=50, tolerance=0.0005, controlPoints=500)
        # print('the corresponding transform is:')
        # print(T)
        # print(R)
        # print(A)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        E = np.array([x, y, z])
        print('yaw,pitch,roll is:')
        print(E)
        ori_Target = np.mean(Target_array, axis=0)
        print('orignal poisition:')
        print(ori_Target)

        PosTarget = np.mean(A, axis=0)
        print('Estimation Position:')
        print(PosTarget)
        DisE = np.linalg.norm(ori_Target - PosTarget)
        print('Euclidean Distance')
        print(DisE)
        offset = DisE /objSize
        print('offset value is:')
        print(offset)
        '''
        ExpData1=open('CokecanDis.txt','a')
        ExpData1.write(str(DisE))
        ExpData1.write('\n')
        ExpData1.close()
        ExpData2=open('CokecanEuler.txt','a')
        ExpData2.write(str(E))
        ExpData2.write('\n')
        ExpData2.close()
        '''
        # a,d,p=tf.transformations.rotation_from_matrix(R)
        a = tf.transformations.quaternion_from_euler(x, y, z)
        print(a)

        br.sendTransform((PosTarget[0], PosTarget[1], PosTarget[2]), tf.transformations.quaternion_from_euler(x, y, z),
                         rospy.Time.now(), turtlename, "camera_link")
        #os.remove('1.jpg')
        #os.remove('2.jpg')
        print('program finish')
        #time.sleep(100)


'''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y,z, c='red')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
'''




def listener():
    #creat a topic for object tracking
    rospy.init_node('listener', anonymous=True)
    #Define the subscribing data as Pointcloud
    rospy.Subscriber('/camera/depth/points', PointCloud2, callback)
    #rate.sleep()
    rospy.spin()


def reader():
    pcloud = pcl.load(template)
    #pcloud = PCdata
    pcloud_array=np.asarray(pcloud)
    #pptk.viewer(pcloud_array)
    # create a passthrough filter
    passthrough = pcloud.make_passthrough_filter()
    # set a specificed filtering direction Z axis
    passthrough.set_filter_field_name('z')
    # set a range from 0 to 2.5
    passthrough.set_filter_limits(0,2.5)
    filtered = passthrough.filter()
    pf = np.asarray(filtered)
    #pptk.viewer(pf)
    #segmentation Plane model
    # estimate normals
    seg = filtered.make_segmenter_normals(ksearch=50)
    #Set the coefficients of the estimated model to be optimized
    seg.set_optimize_coefficients(True)
    # set model type
    seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    # set weight of normals
    seg.set_normal_distance_weight(0.1)
    # Set estimation method
    seg.set_method_type(pcl.SAC_RANSAC)
    # Set iteration times
    seg.set_max_iterations(100)
    # define the range of inliers in the model
    seg.set_distance_threshold(0.1)
    [inliers_plane, coefficients_plane] = seg.segment()
    cloud_plane = filtered.extract(inliers_plane, True)
    cplane=np.asarray(cloud_plane)
    #pptk.viewer(cplane)

    #statistical outlier removal
    data3 = cloud_plane.make_statistical_outlier_filter()
    data3.set_mean_k(60)
    data3.set_std_dev_mul_thresh(0.05)
    #data3.set_negative(True)
    data3_final=data3.filter()
    #pcl.save(data3_final, "object1.pcd")
    data4 = data3_final.make_statistical_outlier_filter()
    data4.set_mean_k(60)
    data4.set_std_dev_mul_thresh(0.5)
    data4_final=data4.filter()
    data4_array=np.asarray(data4_final)

    #pptk.viewer(data4_array)
    return data4_final


def TargetPreprocessing(PCdata):
    #SourceCloud=PCsource
    #TargetCloud=pcl.load('inliers.pcd')
    TargetCloud=PCdata
    tc_array = np.asarray(TargetCloud)
    #pptk.viewer(tc_array)
    #sc_array = np.asarray(SourceCloud)
    #pptk.viewer(sc_array)
    passthrough = TargetCloud.make_passthrough_filter()
    passthrough.set_filter_field_name('z')
    passthrough.set_filter_limits(0, 2.5)
    Targetfiltered = passthrough.filter()
    Tfl=np.asarray(Targetfiltered)
    #pptk.viewer(Tfl)
    seg = Targetfiltered.make_segmenter_normals(ksearch=50)
    seg.set_optimize_coefficients(True)
    seg.set_model_type(pcl.SACMODEL_CYLINDER)
    seg.set_normal_distance_weight(0.1)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_max_iterations(100)
    seg.set_distance_threshold(0.1)
    [inliers_plane, coefficients_plane] = seg.segment()
    cloud_plane = Targetfiltered.extract(inliers_plane, True)
    cplane = np.asarray(cloud_plane)
    #pptk.viewer(cplane)
    Muti_Target=cloud_plane
    data3 = cloud_plane.make_statistical_outlier_filter()
    data3.set_mean_k(60)
    data3.set_std_dev_mul_thresh(0.05)
    # data3.set_negative(True)
    #data3.set_mean_k(800)
    #data3.set_std_dev_mul_thresh(0.001)

    Target_half = data3.filter()
    data4=Target_half.make_statistical_outlier_filter()
    data4.set_mean_k(60)
    data4.set_std_dev_mul_thresh(0.5)

    Target_final=data4.filter()

    #pcl.save(Target_final, "mobject.pcd")

    Target_array = np.asarray(Target_final)
    #pptk.viewer(Target_array)
    '''
    #pptk.viewer(Target_array)
    data4 = Target_final1.make_statistical_outlier_filter()
    data4.set_mean_k(60)
    data4.set_std_dev_mul_thresh(0.1)
    data4.set_negative(True)
    Target_final2 = data4.filter()

    Target_array2 = np.asarray(Target_final2)
    #pptk.viewer(Target_array2)
    '''


    return Target_final,Muti_Target




def Downsample(PCsource,PCdata):
    SourceCloud = PCsource
    TargetCloud = PCdata
    '''
    data4 = SourceCloud.make_statistical_outlier_filter()
    data4.set_mean_k(60)
    data4.set_std_dev_mul_thresh(0.1)
    data4.set_negative(True)
    #SourceCloud = data4.filter()
    '''

    SourcePC = np.asarray(SourceCloud)
    TargetPC = np.asarray(TargetCloud)
    #pptk.viewer(SourcePC)
    #pptk.viewer(TargetPC)
    #print(TargetPC)

    #print(TargetPC.shape[0])
    #Downsample
    obj_source = SourceCloud.make_voxel_grid_filter()
    leaf1 = 0.003
    obj_source.set_leaf_size(leaf1, leaf1, leaf1)
    Source_object = obj_source.filter()
    obj_target = TargetCloud.make_voxel_grid_filter()
    leaf2 = 0.003
    obj_target.set_leaf_size(leaf2, leaf2, leaf2)
    Target_object = obj_target.filter()
    SourcePC=np.asarray(Source_object)
    TargetPC=np.asarray(Target_object)
    #pptk.viewer(SourcePC)
    #pptk.viewer(TargetPC)
    #Estimate normals for Target
    return SourcePC,TargetPC




def nearest_point(P, Q):
    P = np.array(P)
    Q = np.array(Q)
    dis = np.zeros(P.shape[0])
    index = np.zeros(Q.shape[0], dtype = np.int)

    for i in range(P.shape[0]):
        minDis = np.inf
        for j in range(Q.shape[0]):
            tmp = np.linalg.norm(P[i] - Q[j], ord = 2)
            if minDis > tmp:
                minDis = tmp
                index[i] = j
        dis[i] = minDis
    return dis, index

def find_optimal_transform(P, Q):
    meanP = np.mean(P, axis = 0)
    meanQ = np.mean(Q, axis = 0)
    Premain = P - meanP
    Qremain = Q - meanQ

    W = np.dot(Qremain.T, Premain)
    U, S, VT = np.linalg.svd(W)
    R = np.dot(U, VT)
    if np.linalg.det(R) < 0:
       R[2, :] *= -1

    T = meanQ.T - np.dot(R, meanP.T)
    return R, T

def ICPRegistration(src, dst, maxIteration, tolerance, controlPoints):
    A = np.array(src)
    B = np.array(dst)
    lastErr = 0
    if (A.shape[0] != B.shape[0]):
        minAB= min(A.shape[0], B.shape[0])
        length = min(minAB, controlPoints)
        sampleA = random.sample(range(A.shape[0]), length)
        sampleB = random.sample(range(B.shape[0]), length)
        P = np.array([A[i] for i in sampleA])
        Q = np.array([B[i] for i in sampleB])
    else:
        length = A.shape[0]
        if (length > controlPoints):
            sampleA = random.sample(range(A.shape[0]), length)
            sampleB = random.sample(range(B.shape[0]), length)
            P = np.array([A[i] for i in sampleA])
            Q = np.array([B[i] for i in sampleB])
        else :
            P = A
            Q = B
    count=0

    for i in range(maxIteration):
        print("Iteration : " + str(i) + " and Error Rate : " + str(lastErr))
        count=count+1
        dis, index = nearest_point(P, Q)
        R, T = find_optimal_transform(P, Q[index,:])
        A = np.dot(R, A.T).T + np.array([T for j in range(A.shape[0])])
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])

        meanErr = np.sum(dis) / dis.shape[0]
        if abs(lastErr - meanErr) < tolerance: break
        lastErr = meanErr

    R, T = find_optimal_transform(A,np.array(src))
    return R, T, A,lastErr


def main():
    SourcePC,TargetPC=Downsample()
    print(SourcePC)
    print(TargetPC)
    pptk.viewer(SourcePC)
    pptk.viewer(TargetPC)
    R,T,A= ICPRegistration(SourcePC, TargetPC, maxIteration=50, tolerance=0.001, controlPoints=1000)
    print('the corresponding transform is:')
    print(T)
    print(R)
    print(A)



    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    E=np.array([x,y,z])
    print(E)


    turtlename = 'object1'
    PosTarget = np.mean(A, axis=0)
    print(PosTarget[1])
    #a,d,p=tf.transformations.rotation_from_matrix(R)
    a=tf.transformations.quaternion_from_euler(x, y, z)
    print(a)


    br.sendTransform((PosTarget[0], PosTarget[1], PosTarget[2]), tf.transformations.quaternion_from_euler(x,y,z),
                     300, turtlename, "camera_link")
    #br.sendTransform((PosTarget[0], PosTarget[1], PosTarget[2]), tf.transformations.quaternion_from_euler(x,y,z),
                     #rospy.Time.now(30), turtlename, "camera_link")




    #print(A)

#def TrackingPosition():


'''
threads = []

threads.append(threading.Thread(target=TargetPreprocessing))
threads.append(threading.Thread(target=main))
'''


def MutiObSeg(PC):
    #TargetPC = pcl.load('mobject.pcd')
    TargetPC=PC
    Target = np.asarray(TargetPC)
    Targett = Target.T
    print(Targett)
    print(Targett[0])
    x = Targett[0]
    y = Targett[1]
    z = Targett[2]
    # kmeans_model = KMeans(n_clusters=3).fit(Target)
    colors = ['b', 'g', 'r']
    shapes = ['o', 's', 'D']
    labels = ['A', 'B', 'C']
    kmeans_model, x_result, y_result, z_result = kmeans_building(Target, x, y, z, 2, labels, colors, shapes)
    #print(kmeans_model)

    x = np.array(x_result[0])
    # np.array(x_result[1])

    # print(x)
    #print(x_result[1])
    #print(y_result[0])
    #print(y_result[1])
    X1 = np.array(list(zip(x_result[0], y_result[0], z_result[0]))).reshape(len(x_result[0]), 3)
    X2 = np.array(list(zip(x_result[1], y_result[1], z_result[1]))).reshape(len(x_result[1]), 3)
    y1 = np.array(list(zip(x_result[0], y_result[0]))).reshape(len(x_result[0]), 2)
    y2 = np.array(list(zip(x_result[1], y_result[1]))).reshape(len(x_result[1]), 2)
    #print(X1)
    #print(X2)
    #print(y1)
    #print(y2)
    #pptk.viewer(X1)
    #pptk.viewer(X2)

    p1 = pcl.PointCloud(X1)
    p2 = pcl.PointCloud(X2)

    pt1 = p1.make_statistical_outlier_filter()
    pt1.set_mean_k(60)
    pt1.set_std_dev_mul_thresh(0.05)
    Ft1_half = pt1.filter()
    pt1 = Ft1_half.make_statistical_outlier_filter()
    pt1.set_mean_k(60)
    pt1.set_std_dev_mul_thresh(0.5)
    Ft1 = pt1.filter()

    pt2 = p2.make_statistical_outlier_filter()
    pt2.set_mean_k(60)
    pt2.set_std_dev_mul_thresh(0.05)
    Ft2_half = pt2.filter()
    pt2 = Ft2_half.make_statistical_outlier_filter()
    pt2.set_mean_k(60)
    pt2.set_std_dev_mul_thresh(0.5)
    Ft2 = pt2.filter()


    X1=np.asarray(Ft1)
    X2=np.asarray(Ft2)

    #pptk.viewer(X1)
    #pptk.viewer(X2)
    # pptk.viewer(fa2)

    fa1 = X1.T
    fa2 = X2.T
    #os.remove('1.jpg')
    #os.remove('2.jpg')


    plt.figure(1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.scatter(fa1[0], fa1[1])
    plt.savefig("1.jpg")
    plt.show()
    plt.figure(2)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.scatter(fa2[0], fa2[1])
    plt.savefig("2.jpg")
    plt.show()

    return Ft1,Ft2




def kmeans_building(X, x, y, z, types_num, types, colors, shapes):
    X = np.array(list(zip(x, y, z))).reshape(len(x), 3)

    kmeans_model = KMeans(n_clusters=types_num).fit(X)

    x_result = [];
    y_result = [];
    z_result = []
    for i in range(types_num):
        temp = [];
        temp1 = [];
        temp2 = []
        x_result.append(temp)
        y_result.append(temp1)
        z_result.append(temp2)
    for i, l in enumerate(kmeans_model.labels_):
        x_result[l].append(x[i])
        y_result[l].append(y[i])
        z_result[l].append(z[i])
    '''
        plt.scatter(x, y, z, c=colors[l],marker=shapes[l])
    for i in range(len(list(kmeans_model.cluster_centers_))):
        plt.scatter(list(list(kmeans_model.cluster_centers_)[i])[0],list(list(kmeans_model.cluster_centers_)[i])[1],c=colors[i],marker=shapes[i],label=types[i])
    plt.legend()
    '''

    return kmeans_model, x_result, y_result, z_result


def classify_hist_with_split(image1, image2, size=(256, 256)):

    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data



def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])

    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree



def compare_Sim(img1,img2):


    n = classify_hist_with_split(img1, img2)
    print('Three histogram algorithm similarity')
    print(n)
    return n

def Canny(img):
    img1 = cv2.GaussianBlur(img, (3, 3), 0)
    can = cv2.Canny(img1, 50, 150)
    _, Thr_img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)
    return gradient

def Flann(source,target):
    template = source  # queryImage
    #target = cv2.imread('bowl.jpg')  # trainImage

    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(target, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []

    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return len(good)
if __name__ == '__main__':
    br = tf.TransformBroadcaster()


    listener()
    #reader()

    #TargetPreprocessing()
    #Downsample()

    #main()
    #locate()


    #rospy.init_node('camera_link')

    #HandlePose()





