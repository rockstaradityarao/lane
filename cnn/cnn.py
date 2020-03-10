# @Author  : Aditya Rao
# @File    : cnn.py
import cv2
import numpy as np
from .config import global_config
from .lanenet_model import lanenet
from .lanenet_model import lanenet_postprocess
import tensorflow as tf
tf.compat.v1.disable_v2_behavior() 
CFG = global_config.cfg
from utils import *
import matplotlib.pyplot as plt


class CNN(object):
    def __init__(self, debug, weights_path):
        self.debug = debug
        self.weights_path = weights_path
        self.input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

        net = lanenet.LaneNet(phase='test', net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()

        self.saver = tf.compat.v1.train.Saver()
        # Set sess configuration
        sess_config = tf.ConfigProto()
        #sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self.sess = tf.Session(config=sess_config)
        
        self.saver.restore(sess=self.sess, save_path=self.weights_path)

    @staticmethod    
    def minmax_scale(input_arr):
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)
        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)
        return output_arr

    def process(self, image):
        resized_image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        if self.debug: show(resized_image, 'cnn - resized image')
        resized_image = resized_image / 127.5 - 1.0
        if self.debug: show(resized_image, 'cnn - remapped image')
        with self.sess.as_default():
            binary_seg_image, instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [resized_image]}
            )
            postprocess_result = self.postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=image
            )
            if self.debug:
                mask_image = postprocess_result['mask_image']

                for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
                    instance_seg_image[0][:, :, i] = CNN.minmax_scale(instance_seg_image[0][:, :, i])
                embedding_image = np.array(instance_seg_image[0], np.uint8)

                show(binary_seg_image[0] * 255, 'cnn - binary image')
                show(mask_image[:, :, (2, 1, 0)], 'cnn - masked image')
                show(embedding_image[:, :, (2, 1, 0)], 'cnn - instance image')
                show(image, 'cnn - final image')
            return image