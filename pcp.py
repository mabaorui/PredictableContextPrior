# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:44:22 2020

@author: Administrator
"""
#############
#多个模型4^3个局部 + 全局网络 + pointnet + 一个模型一个模型finetune + 场景训练测试 + 53服务器 3D Scene Dataset *****
import numpy as np
import tensorflow as tf 
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import os 
import shutil
import random
import math
import scipy.io as sio
import time
import argparse
#from im2mesh.utils import libmcubes
import trimesh
from scipy.spatial import cKDTree
from plyfile import PlyData
from plyfile import PlyElement
from skimage.measure import marching_cubes_lewiner



parser = argparse.ArgumentParser()
parser.add_argument('--train',action='store_true', default=False)
parser.add_argument('--finetune',action='store_true', default=False)
parser.add_argument('--test',action='store_true', default=False)
parser.add_argument("--save_idx", type=int, default=-1)
parser.add_argument('--input_ply_file', type=str, default="test.ply")
parser.add_argument('--data_dir', type=str, default="test.ply")
parser.add_argument('--CUDA', type=int, default=0)
parser.add_argument('--OUTPUT_DIR_LOCAL', type=str, default="test.ply")
parser.add_argument('--OUTPUT_DIR_GLOBAL', type=str, default="test.ply")
a = parser.parse_args()

cuda_idx = str(a.CUDA)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= cuda_idx

class_idx = '03211117'
name = 'totempole'
BS = 1
primitives = 1
POINT_NUM = 400
POINT_NUM_GT = 10000

part_vox_size = 6
OUTPUT_DIR = a.OUTPUT_DIR_LOCAL
OUTPUT_DIR_FINETUNE = a.OUTPUT_DIR_GLOBAL
LR = 0.0001
START = 0
SHAPE_NUM = 8000
BD_EMPTY = 0.05
TRAIN = a.train
bd = 0.55

if(TRAIN):
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR)
    if os.path.exists(OUTPUT_DIR_FINETUNE):
        shutil.rmtree(OUTPUT_DIR_FINETUNE)
        print ('test_res_dir: deleted and then created!')
    os.makedirs(OUTPUT_DIR_FINETUNE)



def normal_points(ps_gt, ps, translation = False):
    tt =  0
    if((np.max(ps_gt[:,0])-np.min(ps_gt[:,0])))>(np.max(ps_gt[:,1])-np.min(ps_gt[:,1])):
        tt = (np.max(ps_gt[:,0])-np.min(ps_gt[:,0]))
    else:
        tt = (np.max(ps_gt[:,1])-np.min(ps_gt[:,1]))
    if(tt < (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))):
        tt = (np.max(ps_gt[:,2])-np.min(ps_gt[:,2]))
    #print('tt:',tt)
    tt = 10/(10*tt)
    ps_gt = ps_gt*tt
    ps = ps*tt
    if(translation):
        t = np.mean(ps_gt,axis = 0)
        ps_gt = ps_gt - t
        ps = ps - t
    #print('normal_gt:',np.max(ps_gt),np.min(ps_gt))
    #print('normal:',np.max(ps),np.min(ps))
    return ps_gt, ps

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
    num_outputs: int
  
  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs
def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs
def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)


      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def eval_pointcloud(pointcloud, pointcloud_tgt,
                        normals=None, normals_tgt=None):
        ''' Evaluates a point cloud.

        Args:
            pointcloud (numpy array): predicted point cloud
            pointcloud_tgt (numpy array): target point cloud
            normals (numpy array): predicted normals
            normals_tgt (numpy array): target normals
        '''
        # Return maximum losses if pointcloud is empty


        pointcloud = np.asarray(pointcloud)
        pointcloud_tgt = np.asarray(pointcloud_tgt)

        # Completeness: how far are the points of the target point cloud
        # from thre predicted point cloud
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        completeness2 = completeness**2

        completeness = completeness.mean()
        completeness2 = completeness2.mean()
        completeness_normals = np.absolute(completeness_normals).mean()

        # Accuracy: how far are th points of the predicted pointcloud
        # from the target pointcloud
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        
        accuracy2 = accuracy**2

        accuracy = accuracy.mean()
        accuracy2 = accuracy2.mean()
        accuracy_normals = np.absolute(accuracy_normals).mean()

        # Chamfer distance
        chamferL2 = 0.5 * (completeness2 + accuracy2)
        #print(completeness,accuracy,completeness2,accuracy2)
        #print('chamferL2:',chamferL2)
        normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
        )
        chamferL1 = 0.5 * (completeness + accuracy)
        print('chamferL2:',chamferL2,'accuracy:',accuracy,'normals_correctness:',normals_correctness,'chamferL1:',chamferL1)
        return normals_correctness, chamferL1, chamferL2
    

def safe_norm_np(x, epsilon=1e-12, axis=1):
    return np.sqrt(np.sum(x*x, axis=axis) + epsilon)

def safe_norm(x, epsilon=1e-12, axis=None):
  return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis) + epsilon)
  #return tf.reduce_sum(x ** 2, axis=axis)

def boundingbox(x,y,z):
    return min(x),max(x),min(y),max(y),min(z),max(z)


def get_data_from_filename(filename):
    load_data = np.load(filename)
    point = np.asarray(load_data['sample_near']).reshape(-1,POINT_NUM,3)
    sample = np.asarray(load_data['sample']).reshape(-1,POINT_NUM,3)
    rt = random.randint(0,sample.shape[0]-1)
    #rt = random.randint(0,int((sample.shape[0]-1)/5))
    sample = sample[rt,:,:].reshape(BS, POINT_NUM, 3)
    point = point[rt,:,:].reshape(BS, POINT_NUM, 3)

    
    
    #print('input_points_bs:',filename)
    #print(input_points_bs)
    return point.astype(np.float32), sample.astype(np.float32)

def sample_query_points(input_ply_file):
    data = PlyData.read(a.data_dir + input_ply_file)
    v = data['vertex'].data
    v = np.asarray(v)
    print(v.shape)

    #rt = np.random.choice(v.shape, 50000, replace = False)

    points = []
    for i in range(v.shape[0]):
        points.append(np.array([v[i][0],v[i][1],v[i][2]]))
    points = np.asarray(points)
    pointcloud_s =points.astype(np.float32)
    print('pointcloud sparse:',pointcloud_s.shape[0])
    
    pointcloud_s_t = pointcloud_s - np.array([np.min(pointcloud_s[:,0]),np.min(pointcloud_s[:,1]),np.min(pointcloud_s[:,2])])
    pointcloud_s_t = pointcloud_s_t / (np.array([np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0])]))
    trans = np.array([np.min(pointcloud_s[:,0]),np.min(pointcloud_s[:,1]),np.min(pointcloud_s[:,2])])
    scal = np.array([np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]) - np.min(pointcloud_s[:,0])])
    pointcloud_s = pointcloud_s_t
    
    print(np.min(pointcloud_s[:,0]), np.max(pointcloud_s[:,0]))
    print(np.min(pointcloud_s[:,1]), np.max(pointcloud_s[:,1]))
    print(np.min(pointcloud_s[:,2]), np.max(pointcloud_s[:,2]))
    
    sample = []
    sample_near = []
    sample_near_o = []
    sample_dis = []
    sample_vec = []
    gt_kd_tree = cKDTree(pointcloud_s)
    for i in range(int(1000000/pointcloud_s.shape[0])):
        
        pnts = pointcloud_s
        ptree = cKDTree(pnts)
        i = 0
        sigmas = []
        for p in np.array_split(pnts,100,axis=0):
            d = ptree.query(p,51)
            sigmas.append(d[0][:,-1])
        
            i = i+1
        
        sigmas = np.concatenate(sigmas)
        sigmas_big = 0.2 * np.ones_like(sigmas)
        sigmas = sigmas
        
        #tt = pnts + 0.5*0.25*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        tt = pnts + 0.5*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        #tt = pnts + 1*np.expand_dims(sigmas_big,-1) * np.random.normal(0.0, 1.0, size=pnts.shape)
        sample.append(tt)
        distances, vertex_ids = gt_kd_tree.query(tt, p=2, k = 1)
    

        vertex_ids = np.asarray(vertex_ids)
        print('distances:',distances.shape)
        #print(vertex_ids)

        sample_near.append(pointcloud_s[vertex_ids].reshape(-1,3))
        
        



    
    sample = np.asarray(sample).reshape(-1,3)
    sample_near = np.asarray(sample_near).reshape(-1,3)
    np.savez_compressed(a.data_dir + input_ply_file , sample = sample, sample_near=sample_near,pointcloud_s = pointcloud_s, trans = trans, scal = scal)
    sample_all = sample.reshape(-1,3)
    sample_near_all = sample_near.reshape(-1,3)
    sample_part = [[] for i in range(part_vox_size*part_vox_size*part_vox_size)]
    sample_near_part = [[] for i in range(part_vox_size*part_vox_size*part_vox_size)]
    bd_max_x = np.max(pointcloud_s[:,0])
    bd_max_y = np.max(pointcloud_s[:,1])
    bd_max_z = np.max(pointcloud_s[:,2])
    bd_min_x = np.min(pointcloud_s[:,0])
    bd_min_y = np.min(pointcloud_s[:,1])
    bd_min_z = np.min(pointcloud_s[:,2])
    for l in range(sample_near_all.shape[0]):
        ex = sample_near_all[l,0] - bd_min_x
        ix = int(math.floor(ex/((bd_max_x- bd_min_x)/(part_vox_size))))
        #print(ex,ix)
        ey = sample_near_all[l,1] - bd_min_y
        iy = int(math.floor(ey/((bd_max_y- bd_min_y)/(part_vox_size))))
        ez = sample_near_all[l,2] - bd_min_z
        iz = int(math.floor(ez/((bd_max_z- bd_min_z)/(part_vox_size))))
        ix = np.clip(ix,0,part_vox_size-1)
        iy = np.clip(iy,0,part_vox_size-1)
        iz = np.clip(iz,0,part_vox_size-1)
        #print(ix,iy,iz)
        sample_part[ix*(part_vox_size)*(part_vox_size)+iy*(part_vox_size)+iz].append(sample_all[l])
        sample_near_part[ix*(part_vox_size)*(part_vox_size)+iy*(part_vox_size)+iz].append(sample_near_all[l])
      
        
    for iv in range(len(sample_part)):
            #print(np.asarray(sample[iv]).shape)
            np.savez(a.data_dir + input_ply_file + '_' + str(iv), sample = sample_part[iv],sample_near = sample_near_part[iv])
if(a.train):
    sample_query_points(a.input_ply_file)
mm = 0
files = []
files_path = []

files.append(a.input_ply_file)
files_path.append(a.data_dir + a.input_ply_file)



points_all = []
samples_all = []

if(a.train):
    for fi in range(len(files_path)):
        print(files_path[fi])
        for i in range(part_vox_size*part_vox_size*part_vox_size):
        #for i in range(100):
            if(os.path.exists(files_path[fi] + '_{}.npz'.format(i))):
                print(i)
                load_data = np.load(files_path[fi] + '_{}.npz'.format(i))
                sample_near = np.asarray(load_data['sample_near'])
                sampler = np.asarray(load_data['sample'])
                print(sample_near.shape[0])
                if(sample_near.shape[0]>=POINT_NUM):
                    print(sample_near.shape[0])
                    sample_near,sampler = normal_points(sample_near,sampler,True)
                    tt = int(math.floor(sample_near.shape[0]*1.0/POINT_NUM))
                    tt = (tt*POINT_NUM)
                    #print(sample_near[0:tt,:].shape)
                    points_all.append(sample_near[0:tt,:])
                    samples_all.append(sampler[0:tt,:])
            #print(points_all[i].shape)
   
SHAPE_NUM = len(files)
#SHAPE_NUM = 26
print('SHAPE_NUM:',SHAPE_NUM)


points_target = tf.placeholder(tf.float32, shape=[BS,POINT_NUM,3])
input_points_3d = tf.placeholder(tf.float32, shape=[BS,POINT_NUM,3])
normal_gt = tf.placeholder(tf.float32, shape=[BS,None,3])
points_target_num = tf.placeholder(tf.int32, shape=[1,1])
points_input_num = tf.placeholder(tf.int32, shape=[1,1])
points_cd = tf.placeholder(tf.float32, shape=[BS,None,3])

def local_decoder(feature,input_points_3d):
    with tf.variable_scope('local', reuse=tf.AUTO_REUSE):
        feature_f = tf.nn.relu(tf.layers.dense(feature,512))
        net = tf.nn.relu(tf.layers.dense(input_points_3d, 512))
        net = tf.concat([net,feature_f],2)
        print('net:',net)
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            for i in range(8):
                with tf.variable_scope("resnetBlockFC_%d" % i ):
                    b_initializer=tf.constant_initializer(0.0)
                    w_initializer = tf.random_normal_initializer(mean=0.0,stddev=np.sqrt(2) / np.sqrt(512))
                    net = tf.layers.dense(tf.nn.relu(net),512,kernel_initializer=w_initializer,bias_initializer=b_initializer)
                    
        b_initializer=tf.constant_initializer(-0.5)
        w_initializer = tf.random_normal_initializer(mean=2*np.sqrt(np.pi) / np.sqrt(512), stddev = 0.000001)
        print('net:',net)
        sdf = tf.layers.dense(tf.nn.relu(net),1,kernel_initializer=w_initializer,bias_initializer=b_initializer)
        print('sdf',sdf)
        
        grad = tf.gradients(ys=sdf, xs=input_points_3d) 
        print('grad',grad)
        print(grad[0])
        normal_p_lenght = tf.expand_dims(safe_norm(grad[0],axis = -1),-1)
        print('normal_p_lenght',normal_p_lenght)
        grad_norm = grad[0]/normal_p_lenght
        print('grad_norm',grad_norm)
        return sdf,grad_norm

input_points_3d_global = tf.placeholder(tf.float32, shape=[BS,None,3])
points_target_global = tf.placeholder(tf.float32, shape=[BS,None,3])
feature_global = tf.placeholder(tf.float32, shape=[BS,None,SHAPE_NUM])
#with tf.variable_scope('pointnet', reuse=tf.AUTO_REUSE):
#    input_image = tf.expand_dims(points_target_global,-1)
#    net = conv2d(input_image, 64, [1,3], padding='VALID', stride = [1,1], is_training = True, scope = 'conv1')
#    net = conv2d(input_image, 128, [1,3], padding='VALID', stride = [1,1], is_training = True, scope = 'conv2')
#    net = conv2d(input_image, 1024, [1,3], padding='VALID', stride = [1,1], is_training = True, scope = 'conv3')
#    net = max_pool2d(net,[POINT_NUM,1], padding = 'VALID', scope = 'maxpool')
#    net = tf.reshape(net,[1,-1])
#    net = fully_connected(net, 512, is_training = True, scope = 'fc1')
#    net = fully_connected(net, 256, is_training = True, scope = 'fc2')
#    feature_global = net
#feature_global = tf.tile(tf.expand_dims(feature_global,1),[1,POINT_NUM,1])   
def global_decoder(feature_global_f,input_points_3d_global_f):
    with tf.variable_scope('global', reuse=tf.AUTO_REUSE):
        feature_g = tf.nn.relu(tf.layers.dense(feature_global_f,512))
        net_g = tf.nn.relu(tf.layers.dense(input_points_3d_global_f, 512))
        #print(net_g,feature_g)
        net_g = tf.concat([net_g,feature_g],2)
        for i in range(8):
            net_g = tf.layers.dense(tf.nn.relu(net_g),512)
                    
        feature_output = tf.layers.dense(tf.nn.relu(net_g),SHAPE_NUM)
        d_output = tf.layers.dense(tf.nn.relu(net_g),3)
        sdf_g,grad_norm_g = local_decoder(feature_output,input_points_3d_global_f+d_output)
        g_points_g = input_points_3d_global_f - sdf_g * grad_norm_g
        return g_points_g, sdf_g

g_points_g, sdf_g = global_decoder(feature_global,input_points_3d_global)
loss_g = tf.reduce_mean(tf.norm((points_target_global-g_points_g), axis=-1))
        



with tf.variable_scope('pointnet', reuse=tf.AUTO_REUSE):
    input_image = tf.expand_dims(points_target,-1)
    net = conv2d(input_image, 64, [1,3], padding='VALID', stride = [1,1], is_training = True, scope = 'conv1')
    net = conv2d(input_image, 128, [1,3], padding='VALID', stride = [1,1], is_training = True, scope = 'conv2')
    net = conv2d(input_image, 1024, [1,3], padding='VALID', stride = [1,1], is_training = True, scope = 'conv3')
    net = max_pool2d(net,[POINT_NUM,1], padding = 'VALID', scope = 'maxpool')
    net = tf.reshape(net,[1,-1])
    net = fully_connected(net, 512, is_training = True, scope = 'fc1')
    net = fully_connected(net, 256, is_training = True, scope = 'fc2')
    feature = net
feature = tf.tile(tf.expand_dims(feature,1),[1,POINT_NUM,1])       
sdf,grad_norm = local_decoder(feature,input_points_3d)
g_points = input_points_3d - sdf * grad_norm


loss = tf.reduce_mean(tf.norm((points_target, g_points), axis=-1))

t_vars = tf.trainable_variables()
optim = tf.train.AdamOptimizer(learning_rate=LR, beta1=0.9)
loss_grads_and_vars = optim.compute_gradients(loss, var_list=t_vars)
loss_optim = optim.apply_gradients(loss_grads_and_vars)

global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global')
loss_grads_and_vars_g = optim.compute_gradients(loss_g, var_list=global_vars)
#loss_grads_and_vars_g = optim.compute_gradients(loss_g, var_list=t_vars)
loss_optim_g = optim.apply_gradients(loss_grads_and_vars_g)


config = tf.ConfigProto(allow_soft_placement=False) 
saver_restore = tf.train.Saver(var_list=t_vars)
saver = tf.train.Saver(max_to_keep=2000000)




with tf.Session(config=config) as sess:
    feature_bs_all = []
    for i in range(SHAPE_NUM):
        tt = []
        for j in range(int(POINT_NUM)):
            t = np.zeros(SHAPE_NUM)
            
            t[i] = 1
            tt.append(t)
        feature_bs_all.append(tt)
    feature_bs_all = np.asarray(feature_bs_all)
    #print(feature_bs_all,feature_bs_all.shape)
    if(TRAIN):
        print('train start')
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        print('data shape:',len(points_all))
        for bi in range(500):
            epoch_index = np.random.choice(len(points_all)-1, len(points_all)-1, replace = False)
            for epoch in epoch_index:


                points = points_all[epoch].reshape(-1,POINT_NUM,3)
                samples = samples_all[epoch].reshape(-1,POINT_NUM,3)

                rt = random.randint(0,samples.shape[0]-1)
                sample = samples[rt,:].reshape(BS, POINT_NUM, 3)
                point = points[rt,:].reshape(BS, POINT_NUM, 3)
              
                _, loss_c,g_points_g_c = sess.run([loss_optim,loss,g_points],feed_dict={points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs,
                                     points_target:point,input_points_3d:sample})
                
            if(bi%100 == 0):
                print('model:',bi,'epoch:',epoch,'loss:',loss_c)
                saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=bi)
    
        saver.save(sess, os.path.join(OUTPUT_DIR, "model"), global_step=bi)
    if(a.finetune):
        print('finuetune')
        POINT_NUM_GT_bs = np.array(POINT_NUM_GT).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        points_all = []
        samples_all = []
        for epoch in range(1):
            print('epoch:',epoch)
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            checkpoint = tf.train.get_checkpoint_state(OUTPUT_DIR).all_model_checkpoint_paths
            print(checkpoint[a.save_idx])
            
            
            saver.restore(sess, checkpoint[a.save_idx])
            print(files_path[0] + '.npz')  
            load_data = np.load(files_path[0] + '.npz')
            
            points = load_data['sample_near'].reshape(-1,3)
            samples = load_data['sample'].reshape(-1,3)
            SP_NUM = points.shape[0]
            for bi in range(100010):
                feature_bs = feature_bs_all[0]
                rt = np.random.choice(SP_NUM, POINT_NUM, replace = False)  
                #rt = random.randint(0,samples.shape[0]-1)
                sample = samples[rt,:].reshape(BS, POINT_NUM, 3)
                point = points[rt,:].reshape(BS, POINT_NUM, 3)
           
                feature_bs_t = feature_bs.reshape(BS,POINT_NUM,SHAPE_NUM)
                _, loss_c = sess.run([loss_optim_g,loss_g],feed_dict={feature_global:feature_bs_t,points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs,
                                     points_target_global:point,input_points_3d_global:sample})
                    
                if(bi%100000 == 0):
                    print('model:',bi,'epoch:',epoch,'loss:',loss_c)
                    saver.save(sess, os.path.join(OUTPUT_DIR_FINETUNE, "model"), global_step=bi)
            #saver.save(sess, os.path.join(OUTPUT_DIR_FINETUNE, "model"), global_step=epoch)
    
    if(a.test):
        """ feature_bs = []
        for j in range(vox_size*vox_size):
            t = np.zeros(SHAPE_NUM)
            t[0] = 1
            feature_bs.append(t)
        feature_bs = np.asarray(feature_bs)
        sdf_c = sess.run([sdf_g],feed_dict={input_points_3d_global:input_points_2d_bs_t,feature_global:feature_bs_t,
                                 points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs}) """
        print('test start')
       
        
        s = np.arange(-bd,bd, (2*bd)/128)
            
        print(s.shape[0])
        vox_size = s.shape[0]
        POINT_NUM_GT_bs = np.array(vox_size).reshape(1,1)
        points_input_num_bs = np.array(POINT_NUM).reshape(1,1)
        
        
        
        
        POINT_NUM_GT_bs = np.array(vox_size*vox_size).reshape(1,1)

        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(OUTPUT_DIR_FINETUNE).all_model_checkpoint_paths
        print(checkpoint[a.save_idx])
        saver.restore(sess, checkpoint[a.save_idx])

        #saver.restore(sess, a.out_dir + 'model-0')
        
        point_sparse = np.load(a.data_dir + a.input_ply_file + '.npz')['pointcloud_s']
        

        
        input_points_2d_bs = []

        bd_max = [np.max(point_sparse[:,0]), np.max(point_sparse[:,1]), np.max(point_sparse[:,2])] 
        bd_min = [np.min(point_sparse[:,0]), np.min(point_sparse[:,1]),np.min(point_sparse[:,2])] 
        bd_max  = np.asarray(bd_max) + 0.05
        bd_min = np.asarray(bd_min) - 0.05
        sx = np.arange(bd_min[0], bd_max[0], (bd_max[0] - bd_min[0])/vox_size)
        sy = np.arange(bd_min[1], bd_max[1], (bd_max[1] - bd_min[1])/vox_size)
        sz = np.arange(bd_min[2], bd_max[2], (bd_max[2] - bd_min[2])/vox_size)
        print(bd_max)
        print(bd_min)
        for i in sx:
            for j in sy:
                for k in sz:
                    input_points_2d_bs.append(np.asarray([i,j,k]))
        input_points_2d_bs = np.asarray(input_points_2d_bs)
        input_points_2d_bs = input_points_2d_bs.reshape((vox_size,vox_size,vox_size,3))
                    
        vox = []
        feature_bs = []
        for j in range(vox_size*vox_size):
            t = np.zeros(SHAPE_NUM)
            t[0] = 1
            feature_bs.append(t)
        feature_bs = np.asarray(feature_bs)
        for i in range(input_points_2d_bs.shape[0]):
            
            input_points_2d_bs_t = input_points_2d_bs[i,:,:,:]
            input_points_2d_bs_t = input_points_2d_bs_t.reshape(BS, vox_size*vox_size, 3)
            feature_bs_t = feature_bs.reshape(BS,vox_size*vox_size,SHAPE_NUM)
            sdf_c = sess.run([sdf_g],feed_dict={input_points_3d_global:input_points_2d_bs_t,feature_global:feature_bs_t,
                                 points_target_num:POINT_NUM_GT_bs,points_input_num:points_input_num_bs})
            vox.append(sdf_c)

            
        vox = np.asarray(vox)
        #vis_single_points(moved_points, 'moved_points.ply')
        #print('vox',np.min(vox),np.max(vox),np.mean(vox))
        vox = vox.reshape((vox_size,vox_size,vox_size))
        vox_max = np.max(vox.reshape((-1)))
        vox_min = np.min(vox.reshape((-1)))
        print('max_min:',vox_max,vox_min,np.mean(vox))
        
        #threshs = [0.001,0.0015,0.002,0.0025,0.005]
        threshs = [0.005]
        for thresh in threshs:
            print(np.sum(vox>thresh),np.sum(vox<thresh))
            
            if(np.sum(vox>0.0)<np.sum(vox<0.0)):
                thresh = -thresh
            #vertices, triangles = libmcubes.marching_cubes(vox, thresh)
            vertices, triangles, _, _ = marching_cubes_lewiner(vox, thresh)
            if(vertices.shape[0]<10 or triangles.shape[0]<10):
                print('no sur---------------------------------------------')
                continue
            if(np.sum(vox>0.0)>np.sum(vox<0.0)):
                triangles_t = []
                for it in range(triangles.shape[0]):
                    tt = np.array([triangles[it,2],triangles[it,1],triangles[it,0]])
                    triangles_t.append(tt)
                triangles_t = np.asarray(triangles_t)
            else:
                triangles_t = triangles
                triangles_t = np.asarray(triangles_t)

            vertices -= 0.5
            # Undo padding
            vertices -= 1
            # Normalize to bounding box
            vertices /= np.array([vox_size-1, vox_size-1, vox_size-1])
            vertices = (bd_max-bd_min) * vertices + bd_min
            mesh = trimesh.Trimesh(vertices, triangles_t,
                            vertex_normals=None,
                            process=False)
            
            
            loc_data = np.load(a.data_dir + a.input_ply_file + '.npz')
            vertices = vertices * loc_data['scal'] + loc_data['trans']
            mesh = trimesh.Trimesh(vertices, triangles_t,
                                vertex_normals=None,
                                process=False)
            mesh.export(OUTPUT_DIR_FINETUNE +  '/PCL_' + a.input_ply_file + '_'+ str(thresh) + '.off')
            
        
            

    
    