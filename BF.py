import cv2
from matplotlib import pyplot as plt

template=cv2.imread("template.jpg",0)
target=cv2.imread("coke.jpg",0)
orb=cv2.ORB_create()
kp1,des1=orb.detectAndCompute(template,None)
kp2,des2=orb.detectAndCompute(target,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
mathces=bf.match(des1,des2)
mathces=sorted(mathces,key=lambda x:x.distance)
result= cv2.drawMatches(template,kp1,target,kp2,mathces[:40],None,flags=2)
plt.imshow(result),plt.show()