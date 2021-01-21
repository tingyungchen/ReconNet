import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 
import numpy as np
# import cPickle
# import ipdb

# Change the name of the phi file depending on the MR
# 8/2 4->1
MR_CASE = 4#1:0_01,2:0_04,3:0.10,4:0.25
if MR_CASE == 1:
    m = 10
elif MR_CASE == 2:
    m = 43
elif MR_CASE == 3:
    m = 109
else:
    m = 272
batch_size = 128

def new_fc_layer(bottom, output_size, name ):
    shape = bottom.get_shape().as_list()
    print ('new_fc_layer',shape)
    dim = np.prod( shape[1:] )
    x = tf.reshape( bottom, [-1, dim])
    input_size = dim

    with tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable(
                "W",
                shape=[input_size, output_size],
                # initializer=tf.contrib.layers.xavier_initializer()
                initializer=tf.contrib.layers.xavier_initializer())
        b = tf.compat.v1.get_variable(
                "b",
                shape=[output_size],
                initializer=tf.constant_initializer(0.))
        fc = tf.nn.bias_add( tf.matmul(x, w), b)
    return w, fc
    
def new_conv_layer(bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
    with tf.compat.v1.variable_scope( name ):
        w = tf.compat.v1.get_variable(
                "W",
                shape=filter_shape,
                initializer=tf.random_normal_initializer(0., 0.01)) 
                #tf.random_normal_initializer(mean=0.0, stddev=0.05, seed=None)
        b = tf.compat.v1.get_variable(
                "b",
                shape=filter_shape[-1],#-1就是給python依照輸入自己算，這邊其實就是整個filter用同個weight
                initializer=tf.constant_initializer(0.))
        shape = bottom.get_shape().as_list()
        # print ('new_conv_layer',shape)
        conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
        ofmap = tf.nn.bias_add(conv, b)
        
# This is (mostly) a special case of tf.add where bias is restricted to 1-D. Broadcasting is supported, 
# so value may have any number of dimensions. Unlike tf.add, the type of bias is allowed to differ from value in the case where both types are quantized.
    return ofmap #relu

def leaky_relu( bottom, leak=0.0,name=None,is_training=None):
    return tf.maximum(leak*bottom, bottom)

def build_reconstruction(images, is_train, reuse=None,iters=None ): #這是generator
    batch_size = images.get_shape().as_list()[0]
    image_size = images.get_shape().as_list()[1] * images.get_shape().as_list()[2]
    with tf.compat.v1.variable_scope('generator',reuse=reuse):
        
        # jointly learned Measurment MTX Method-> use two fc layer (just like autoencoder)
        # images(inputs) is 1089
        flattenimg = tf.reshape(images,[batch_size,image_size])
        # w1, fc1 = new_bernoulli_fc_layer(flattenimg, m, name="fc1")
        w1, fc1 = new_fc_layer(flattenimg, m, name="fc1")
        w, fc2 = new_fc_layer(fc1, 33*33, name="fc2")
        fc2 = tf.reshape(fc2, [batch_size, 33, 33, 1])
        
        conv1 = new_conv_layer(fc2, [11,11,1,64], stride=1, name="conv1" )
        bn1 = leaky_relu(conv1, is_training=is_train)
        print(bn1)
        conv2 = new_conv_layer(bn1, [1,1,64,32], stride=1, name="conv2" )
        bn2 = leaky_relu(conv2, is_training=is_train)
        print(bn2)
        conv3 = new_conv_layer(bn2, [7,7,32,1], stride=1, name="conv3")
        bn3 = leaky_relu(conv3, is_training=is_train)
        print(bn3)
        conv4 = new_conv_layer(bn3, [11,11,1,64], stride=1, name="conv4" )
        bn4 = leaky_relu(conv4, is_training=is_train)
        print(bn4)
        conv5 = new_conv_layer(bn4, [1,1,64,32], stride=1, name="conv5" )
        bn5 = leaky_relu(conv5, is_training=is_train)
        print(bn5)
        conv6 = new_conv_layer(bn5, [7,7,32,1], stride=1, name="conv6")
        bn6 = leaky_relu(conv6, is_training=is_train)
        print(bn6)
    # return fc2, conv1, bn1, conv2, bn2, conv3, bn3 #, conv4, bn4, conv5, bn5, conv6, bn6 
    return fc2, bn1, bn1, bn6

def build_adversarial(images, is_train, reuse=None, keep_prob=1.0): #這是discriminator
    with tf.compat.v1.variable_scope('discriminator', reuse=reuse):

        conv1 = new_conv_layer(images, [4,4,1,4], stride=2, name="conv1" ) #注意這裡stride=2???
        bn1 = leaky_relu(batch_norm_wrapper(conv1, is_training=is_train))
        conv2 = new_conv_layer(bn1, [4,4,4,4], stride=2, name="conv2")
        bn2 = leaky_relu(batch_norm_wrapper(conv2, is_training=is_train))
        conv3 = new_conv_layer(bn2, [4,4,4,4], stride=2, name="conv3")
        bn3 = leaky_relu(batch_norm_wrapper(conv3, is_training=is_train))

        w, output = new_fc_layer( bn3, output_size=1, name='output') #a fully connected layer maps the feature maps to a single probability value
        # output = tf.nn.dropout(output, keep_prob)
        output_sig = tf.sigmoid(output)
    return output[:,0], output_sig[:,0]

#這個是另外我自己放的def block，注意tf.bool和python的bool不可相容
def batch_norm_wrapper(inputs, is_training=False, decay = 1):
    """
    每个BN层，引入了4个变量 scale beta pop_mean pop_var，其中:
    scale beta 是可训练的，训练结束后被保存为模型参数
    pop_mean pop_var 是不可训练，只在训练中进行统计，
    pop_mean pop_var 最终保存为模型的变量。在测试时重构的计算图会接入该变量，只要载入训练参数即可。
    """
    epsilon=1e-05
    scale = tf.Variable(tf.ones([inputs.get_shape()[1],inputs.get_shape()[2],inputs.get_shape()[3]])) #記得輸入正確的shape
    beta = tf.Variable(tf.zeros([inputs.get_shape()[1],inputs.get_shape()[2],inputs.get_shape()[3]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[1],inputs.get_shape()[2],inputs.get_shape()[3]]), trainable=False) 
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[1],inputs.get_shape()[2],inputs.get_shape()[3]]), trainable=False)

    if is_training is not False:  #tf的dtype不能通用python
        # 以下为训练时的BN计算图构造
        # batch_mean、batch_var在一个batch里的每一层，在前向传播时会计算一次，
        # 在反传时通过它来计算本层输入加权和的梯度，仅仅作为整个网络传递梯度的功能。在训练结束后被废弃。
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        
        # 通过移动指数平均的方式，把每一个batch的统计量汇总进来，更新总体统计量的估计值pop_mean、pop_var
        # assign构建计算图一个operation，即把pop_mean * decay + batch_mean * (1 - decay) 赋值给pop_mean
        train_mean = tf.compat.v1.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.compat.v1.assign(pop_var,pop_var * decay + batch_var * (1 - decay))

        #control_dependencies是一個上下文管理器，以下代表 上文的 train_mean, train_var 會先被執行
        # 确保本层的train_mean、train_var这两个operation都执行了，才进行BN。
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,batch_mean, batch_var, beta, scale, epsilon)
    else:
        # 以下为测试时的BN计算图构造，即直接载入已训练模型的beta, scale,已训练模型中保存的pop_mean, pop_var
        return tf.nn.batch_normalization(inputs,pop_mean, pop_var, beta, scale, epsilon )