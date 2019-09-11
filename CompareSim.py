import cv2
import numpy as np



def aHash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0
    hash_str = ''
    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]

    avg = s / 64

    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str



def dHash(img):

    img = cv2.resize(img, (9, 8))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''

    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str



def pHash(img):

    img = cv2.resize(img, (32, 32))


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dct = cv2.dct(np.float32(gray))

    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

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



def cmpHash(hash1, hash2):
    n = 0

    if len(hash1)!=len(hash2):
        return -1

    for i in range(len(hash1)):

        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread('1.jpg')
img2 = cv2.imread('coke1.jpg')


hash1 = aHash(img1)
hash2 = aHash(img2)
n = cmpHash(hash1, hash2)
print('Mean hash Similarity:')
print(n)

hash1 = dHash(img1)
hash2 = dHash(img2)
n = cmpHash(hash1, hash2)
print('Difference hash similarity:')
print(n)
hash1 = pHash(img1)
hash2 = pHash(img2)
n = cmpHash(hash1, hash2)
print('Perceived hash similarity:')
print(n)
n = classify_hist_with_split(img1, img2)
print('Three histogram algorithm similarity')
print(n)