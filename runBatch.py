# @Author  : Aditya Rao
# @File    : runBatch.py
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
debug=False
my_program = sys.argv[0]
src_dir = sys.argv[1] if (len(sys.argv) > 1) else './data'
model_dir = sys.argv[2] if (len(sys.argv) > 2) else './model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt'
hough_dir = sys.argv[3] if (len(sys.argv) > 3) else src_dir + '_hough'
cnn_dir = sys.argv[4] if (len(sys.argv) > 4) else src_dir + '_cnn'
log_file = open('./batch.log','w') 
print('Running program: {} {} {} {} {}'.format(my_program, src_dir, model_dir, hough_dir, cnn_dir))
assert os.path.exists(src_dir), '{:s} not exist'.format(src_dir)
os.makedirs(cnn_dir, exist_ok=True)
os.makedirs(hough_dir, exist_ok=True)

#sys.path.insert(0, './cnn/config')
#sys.path.insert(0, './cnn/lanenet_model')
#sys.path.insert(0, './cnn')
#sys.path.insert(0, './hough')
sys.path.insert(0, '.')
src_files = glob.glob('{:s}/*/*.jpg'.format(src_dir), recursive=True)
assert (len(src_files) > 0), 'images not found in directory {}'.format(src_dir)
hough = Hough(debug)
cnn = CNN(debug, model_dir)

for image_source in src_files:
    print('processing {}'.format(image_source))
    image = Image.open(image_source)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_array = asarray(image)

    if debug: show(image_array, 'source image')
    hough_timer = Timer()
    hough_image = hough.process(image_array)
    hough_duration = hough_timer.getTime()
    saveImage(hough_image, image_source, src_dir, hough_dir)

    cnn_timer = Timer()
    cnn_image = cnn.process(image_array)
    cnn_duration = cnn_timer.getTime()
    saveImage(cnn_image, image_source, src_dir, cnn_dir)
    text = '{}\t{:.5f}\t{:.5f}'.format(image_source, hough_duration, cnn_duration)
    log_file.write('{}\n'.format(text))
    log_file.flush()
    print(text)
log_file.close()
