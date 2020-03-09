import os
import cv2
import time
import matplotlib.pyplot as plt

def show(img, title):
    print('image {} has shape: {}'.format(title, img.shape))
    plt.figure(title)
    plt.imshow(img)
    plt.show() 

def saveImage(image, src_file, src_dir, target_dir):
    target_file = src_file.replace(src_dir, target_dir)
    target_file_dir = os.path.dirname(target_file)
    if not os.path.exists(target_file_dir):
        print('creating directory {}'.format(target_file_dir))
        os.makedirs(target_file_dir)
    print('saving {}'.format(target_file))
    cv2.imwrite(target_file,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

class Timer(object):
    def __init__(self):
        self.start = time.time()
    
    def getTime(self):
        self.finish = time.time()
        return (self.finish - self.start)