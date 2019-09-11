import numpy as np
import matplotlib.pyplot as plt
with open(r'KinectDis.txt','r') as file:
    content_list = file.readlines()

for i in range(0,len(content_list)):
    content_list[i] = float(content_list[i].strip('\n'))
plt.figure()
plt.title('Offset value for Kinect sample')
x=[]
array=np.array(content_list)
mean=np.mean(content_list)
print(mean)
for j in range(0,100):
    x.append(1)
plt.plot(content_list,'blue')
plt.plot(x,'red')
plt.xlim(0, 100)
plt.ylim(-0.5, 1.5)
#print(content_list[i])
plt.show()
#print(content_list)