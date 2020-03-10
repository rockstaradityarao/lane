# @Author  : Aditya Rao
# @File    : runOne.py
import sys
import os
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
from utils import *
from hough.hough import *
from cnn.cnn import *
import cv2
import glob

# process arguments
debug=True
my_program = sys.argv[0]
src_dir = sys.argv[1] if (len(sys.argv) > 1) else './data/tmp/0.jpg'
model_dir = sys.argv[2] if (len(sys.argv) > 2) else './model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt'
hough_dir = sys.argv[3] if (len(sys.argv) > 3) else src_dir + '_hough'
cnn_dir = sys.argv[4] if (len(sys.argv) > 4) else src_dir + '_cnn'
print('Running program: {} {} {} {} {}'.format(my_program, src_dir, model_dir, hough_dir, cnn_dir))
assert os.path.exists(src_dir), '{:s} not exist'.format(src_dir)
os.makedirs(cnn_dir, exist_ok=True)
os.makedirs(hough_dir, exist_ok=True)

#sys.path.insert(0, './cnn/config')
#sys.path.insert(0, './cnn/lanenet_model')
#sys.path.insert(0, './cnn')
#sys.path.insert(0, './hough')
sys.path.insert(0, '.')
src_files = [src_dir]
assert (len(src_files) > 0), 'images not found in directory {}'.format(src_dir)

image_source = src_files[0]
image = Image.open(image_source)
if image.mode != 'RGB':
    image = image.convert('RGB')
image_array = asarray(image)

if debug: show(image_array, 'source image')
hough = Hough(debug)
hough_image = hough.process(image_array)
hough_target = os.path.join(hough_dir,os.path.basename(image_source))
cv2.imwrite(hough_target,cv2.cvtColor(hough_image, cv2.COLOR_RGB2BGR))

cnn = CNN(debug, model_dir)
cnn_image = cnn.process(image_array)
cnn_target = os.path.join(cnn_dir,os.path.basename(image_source))
cv2.imwrite(cnn_target,cv2.cvtColor(cnn_image, cv2.COLOR_RGB2BGR))

