import cv2
import numpy as np

original_img = cv2.imread("11.jpg", 0)


img1 = cv2.GaussianBlur(original_img, (3, 3), 0)
canny = cv2.Canny(img1, 50, 150)


_, Thr_img = cv2.threshold(original_img, 210, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
gradient = cv2.morphologyEx(Thr_img, cv2.MORPH_GRADIENT, kernel)

#cv2.imshow("original_img", original_img)
cv2.imshow("gradient", gradient)
#cv2.imshow('Canny', canny)
cv2.imwrite("bowl1.jpg", gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()
