import numpy as np
import pcl
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import pptk
def MutiObSeg():
    TargetPC= pcl.load('mobject.pcd')
    Target=np.asarray(TargetPC)
    Targett=Target.T
    print(Targett)
    print(Targett[0])
    x=Targett[0]
    y=Targett[1]
    z=Targett[2]
    #kmeans_model = KMeans(n_clusters=3).fit(Target)
    colors = ['b', 'g', 'r']
    shapes = ['o', 's', 'D']
    labels = ['A', 'B', 'C']
    kmeans_model,x_result,y_result,z_result= kmeans_building(Target,x, y, z, 2, labels, colors, shapes)
    print(kmeans_model)

    x=np.array(x_result[0])
    #np.array(x_result[1])


    #print(x)
    print(x_result[1])
    print(y_result[0])
    print(y_result[1])
    X1 = np.array(list(zip(x_result[0], y_result[0], z_result[0]))).reshape(len(x_result[0]), 3)
    X2 = np.array(list(zip(x_result[1], y_result[1], z_result[1]))).reshape(len(x_result[1]), 3)
    y1 = np.array(list(zip(x_result[0], y_result[0]))).reshape(len(x_result[0]), 2)
    y2 = np.array(list(zip(x_result[1], y_result[1]))).reshape(len(x_result[1]), 2)
    print(X1)
    print(X2)
    print(y1)
    print(y2)
    #pptk.viewer(X1)
    #pptk.viewer(X2)

    p1 = pcl.PointCloud(X1)
    p2 = pcl.PointCloud(X2)
    pt1 = p1.make_statistical_outlier_filter()
    pt1.set_mean_k(60)
    pt1.set_std_dev_mul_thresh(0.5)
    Ft1 = pt1.filter()
    pt2 = p2.make_statistical_outlier_filter()
    pt2.set_mean_k(60)
    pt2.set_std_dev_mul_thresh(0.5)
    Ft2 = pt2.filter()
    fa1=np.asarray(Ft1)
    fa2=np.asarray(Ft2)
    pptk.viewer(fa1)
    pptk.viewer(fa2)
    #pptk.viewer(fa2)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    fa1=fa1.T
    fa2=fa2.T


    plt.figure(1)
    plt.title('object1')
    plt.scatter(fa1[0], fa1[1])
    plt.show()
    plt.figure(2)
    plt.title('object2')
    plt.scatter(fa2[0], fa2[1])
    plt.show()


    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_result, y_result, z_result, c='red')
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()
    
    X=Target
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c='red')
    plt.show()

    #print(metrics.calinski_harabaz_score(X, y_pred))
    '''







def kmeans_building(X,x,y,z,types_num,types,colors,shapes):
    X = np.array(list(zip(x, y,z))).reshape(len(x), 3)


    kmeans_model = KMeans(n_clusters=types_num).fit(X)


   
    x_result=[]; y_result=[];z_result=[]
    for i in range(types_num):
        temp=[]; temp1=[]; temp2=[]
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


    return kmeans_model,x_result,y_result,z_result





if __name__ == '__main__':
    MutiObSeg()