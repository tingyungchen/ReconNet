import os
import numpy as np
import csgan_learnedMM
import h5py

import tensorflow as tf
import numpy as np
# import tensorflow as tf
# tf.disable_v2_behavior()

momentum = 0.9
batch_size = 128
lambda_recon = 1
lambda_adv = 0.0001
max_iters = 50000
lr1 = 0.001
lr2 = 0.0001
lr3 = 0.00001
# Change these values before running code. .
# MR = 0.01, 0.04, 0.10, 0.25 and corresponding m = 10, 43, 109, 272

MR_CASE = csgan_learnedMM.MR_CASE  #1:0_01,2:0_04,3:0.10,4:0.25

# jointly learned Measurment MTX Method,先用SRCNN的資料夾產生 generate_test & generate_train
training_data = './training_im_h5_file_learnedMM/train.h5' 
validation_data = './training_im_h5_file_learnedMM/test.h5'
if MR_CASE == 1:
    checkpoint_file = 'checkpoints_learnedMM/mr_0_01/mr_0_01_50000.ckpt'
    m = 11
elif MR_CASE == 2:
    checkpoint_file = 'checkpoints_learnedMM/mr_0_04/mr_0_04_50000.ckpt'
    m = 41
elif MR_CASE == 3:
    checkpoint_file = 'checkpoints_learnedMM/mr_0_10/mr_0_10_50000.ckpt'
    m = 103
else:
    checkpoint_file = 'checkpoints_learnedMM/mr_0_25/mr_0_25_50000.ckpt'
    m = 272

with h5py.File(training_data) as hf:
    # print (hf.keys())
    data = np.array(hf.get('data'))
    print ('data.shape',data.shape)
    label = np.array(hf.get('label')) 
    print ('label.shape',label.shape)

with h5py.File(validation_data) as hf:
    # print (hf.keys())
    valid_data = np.array(hf.get('data'))
    print ('valid_data.shape',valid_data.shape)
    valid_label = np.array(hf.get('label'))
    print ('valid_label.shape',valid_label.shape)

with tf.Graph().as_default(): #如果你新開一張 graph 使用，你可以使用with tf.Graph().as_default()如此，你以下所宣告的節點就會在此 graph 上

    is_train = tf.placeholder(tf.bool) #宣告placeholder
    learning_rate = tf.placeholder( tf.float32, [])
    images_tf = tf.placeholder( tf.float32, [batch_size, 33, 33, 1], name="images")    
    input_img = tf.placeholder( tf.float32, [batch_size, 33, 33, 1], name='input_img')#to do learned MM method 
    keep_prob = tf.placeholder(tf.float32, [])
    # iters_tf = tf.placeholder( tf.int33 )
    
    graph = tf.get_default_graph()
    
    labels_D = tf.concat( [tf.ones([batch_size]), tf.zeros([batch_size])], 0) #(0,[...])就是把tf.ones 和 tf.zeros直接相接
    labels_G = tf.ones([batch_size])
    
    bn1, bn2, bn3, reconstruction_ori  = csgan_learnedMM.build_reconstruction(input_img, is_train)
    loss_recon = tf.divide(tf.reduce_sum(tf.square(tf.subtract(images_tf, reconstruction_ori))), batch_size)
    #tf.reduce_sum 就是降維總和，這邊沒指定dim，就是矩陣內所有元素相加
    #the average reconstruction error over all the training image blocks

    adversarial_pos, adversarial_pos_sig = csgan_learnedMM.build_adversarial(images_tf, is_train)#真圖
    adversarial_neg, adversarial_neg_sig = csgan_learnedMM.build_adversarial(reconstruction_ori, is_train, reuse=True)#假圖 # I changed this from reconstruction to reconstruction_ori. No idea which is right
    adversarial_all = tf.concat( [adversarial_pos, adversarial_neg], 0)
    
    loss_adv_D = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_all, labels= labels_D))#注意adversarial_all 在build_adversarial中是尚未經過sigmoid的東西，但這邊的loss函數有包含sigmoid功能
    loss_adv_G = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=adversarial_neg, labels= labels_G))
    
    loss_G = loss_recon * lambda_recon + loss_adv_G * lambda_adv
    loss_D = loss_adv_D
    
    # with tf.GradientTape() as tape_G:
    #     tape_G.watch(images_tf)
    #     dy_dx = tape_G.gradient(reconstruction_ori, images_tf)
    #     print(dy_dx)
    
#下面就是宣告optimizer，和抓取一些參數值，與執行code
    var_G = list(filter( lambda x: x.name.startswith('generator'), tf.all_variables()))
    var_D = list(filter( lambda x: x.name.startswith('discriminator'), tf.trainable_variables()))

    for v in var_G:
        print (v.name)

    for v in var_D:
        print (v.name)

    W_G = list(filter(lambda x: x.name.endswith('W:0'), var_G))
    W_D = list(filter(lambda x: x.name.endswith('W:0'), var_D))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #其實paper裡面是G更新兩次才換更新D，但是其實效果差不多
    # optimizer_G = tf.train.MomentumOptimizer( learning_rate, momentum )  # = SGD + momentum
    optimizer_G = tf.train.AdamOptimizer( learning_rate, momentum )
    train_op = optimizer_G.minimize(loss_G, var_list=var_G)
    # optimizer_G = tf.keras.optimizers.Adam( learning_rate, momentum )
    # train_op = optimizer_G.minimize(loss_G, var_list=var_G)
    
    # optimizer_D = tf.train.MomentumOptimizer( learning_rate, momentum )  # = SGD + momentum
    optimizer_D = tf.train.AdamOptimizer( learning_rate, momentum )
    train_op_D = optimizer_D.minimize(loss_D, var_list=var_D)
    
    saver = tf.train.Saver(max_to_keep=10)
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    iters = 0

    loss_D_val = 0.
    loss_G_val = 0.

    for iters in range(max_iters+1):
        batch_idx = iters % 170 #21760/128 = 170
        valid_batch_idx = iters % 8 #這個的值是小於 (valid_data的數量/batch_size),而且要注意這樣有些圖片train不到

        if iters <  90000:
            learning_rate_val =lr1 #0.0001
        elif iters >= 90000 and iters < 95000:
            learning_rate_val =lr2 #0.00001
        else:
            learning_rate_val =lr3 #0.000001

        gradients, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, recon_ori_vals, bn1_val,bn2_val,bn3_val = sess.run(
                [train_op, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction_ori, bn1,bn2,bn3],
                feed_dict={
                    images_tf: label[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    input_img: data[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    learning_rate: learning_rate_val,
                    # iters: iters_tf,
                    is_train: True,
                    keep_prob: 0.5
                    })
        
        _, loss_D_val, adv_pos_val, adv_pos_val_sig, adv_neg_val, adv_neg_val_sig = sess.run(
                [train_op_D, loss_D, adversarial_pos, adversarial_pos_sig, adversarial_neg, adversarial_neg_sig],
                feed_dict={
                    images_tf: label[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    input_img: data[batch_idx*batch_size: (batch_idx+1)*batch_size, :, :, :],
                    learning_rate: learning_rate_val/100.0,
                    is_train: True,
                    keep_prob: 0.5
                    })

        # Validation:
        if iters % 10 == 0: #幾個iters就印一次loss
            loss_recon_test, adv_pos_val, adv_pos_val_sig, adv_neg_val, adv_neg_val_sig =  sess.run([loss_recon, adversarial_pos, adversarial_pos_sig, adversarial_neg, adversarial_neg_sig],
                                                                                             feed_dict={images_tf : valid_label[valid_batch_idx*batch_size: (valid_batch_idx+1)*batch_size, :, :, :], 
                                                                                                        input_img : valid_data[valid_batch_idx*batch_size: (valid_batch_idx+1)*batch_size, :, :, :], 
                                                                                                        is_train: False,
                                                                                                        keep_prob: 1.0
                                                                                                   })
            # 如果你想檢查參數更新有沒有動:                                                                                        
            # fc1_w = graph.get_tensor_by_name('generator/fc1/W:0')
            # print('fc1_w:',sess.run(fc1_w))                                                                                           
            print ("Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_test, "Gen ADV Loss:", loss_adv_G_val,  "Dis Loss:", loss_D_val)
            print (adv_pos_val_sig.mean(), adv_neg_val_sig.mean())
            # print('gradients:',gradients)
            # print('measuremwnt mtx:',bn1_val)
        # if (iters % 100 == 0) or (iters + 1) == max_iters:
        # 	saver.save(sess, checkpoint_file+str(iters))
        if (iters % 100 == 0) or (iters) == max_iters:
            saver.save(sess, checkpoint_file)
            np.save(checkpoint_file + 'bn1',bn1_val) #tf的tensor可直接存成npy檔，不用轉
    


	
        
        

    