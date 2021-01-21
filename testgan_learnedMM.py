import numpy as np 
import tensorflow as tf 
import cv2
import os
import sys
import csgan_learnedMM
import time
# Change these values before running code. 
# MR = 0.01, 0.04, 0.10, 0.25 and corresponding m = 10, 43, 109, 272

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MR_CASE = csgan_learnedMM.MR_CASE  #1:0_01,2:0_04,3:0.10,4:0.25

if MR_CASE == 1:
	checkpointPath = './checkpoints_learnedMM/mr_0_01/'
	matdir = './recon_images/mr_0_01/'

	m = 10
elif MR_CASE == 2:
	checkpointPath = './checkpoints_learnedMM/mr_0_04/'
	matdir = './recon_images/mr_0_04/'
	
	m = 43
elif MR_CASE == 3:
	checkpointPath = './checkpoints_learnedMM/mr_0_10/'
	matdir = './recon_images/mr_0_10/'

	m = 109
else:
	checkpointPath = './checkpoints_final_learnedMM/mr_0_25/'
	matdir = './recon_images/mr_0_25/'

	m = 272

inputDir = 'test_images/'
blockSize = 33
batch_size = 1

imList = os.listdir(inputDir)
print (imList)

with tf.Graph().as_default():
	images_tf = tf.placeholder( tf.float32, [batch_size, 33, 33, 1], name="images")
	cs_meas = tf.placeholder( tf.float32, [batch_size, 33, 33, 1], name='cs_meas')
	is_train = tf.placeholder( tf.bool )

	fc2, bn2, conv3, reconstruction_ori  = csgan_learnedMM.build_reconstruction(cs_meas, is_train)
	# saver = tf.train.import_meta_graph('./checkpoints_final/mr_0_25/mr_0_25_50000.ckpt.meta') #載入網路結構
	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, tf.train.latest_checkpoint(checkpointPath)) #載入最近次儲存ckpt
	# print model parameters to txt file
	np.set_printoptions(threshold=sys.maxsize) # without printing truncate
	# f = open('model_param_txt/weights/ReconNet_bias1.txt', 'w')
	# param = sess.run("generator/conv1/b:0")
	# # --------------------------------------Quantization--------------------------------------
	# max_param = tf.argmax(param, 0)
	# max_param = tf.dtypes.cast(max_param, tf.float32) # all arguments parse into tf.quantization.quantize must be float32
	# min_param = tf.argmin(param, 0)
	# min_param = tf.dtypes.cast(min_param, tf.float32)
	# quan_param = tf.quantization.quantize(param, min_param, max_param, tf.qint8, mode='MIN_COMBINED',round_mode='HALF_AWAY_FROM_ZERO', name=None)
	# # ----------------------------------------------------------------------------------------
	# print('--------------------')
	# print('quan_param info.:', quan_param.output)
	# print('--------------------')
	# an_array = quan_param.output.eval(session=tf.Session())
	# f.write(np.array2string(an_array))
	# f.close()
	
	# print('generator/fc1/b:0:',sess.run("generator/fc1/b:0"))
	print_model_param_flag = 0
	pre_layer_output = 0
	if print_model_param_flag:
		f = open('model_param_txt/weights/ReconNet_bias1.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv1/b:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_bias2.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv2/b:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_bias3.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv3/b:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_bias4.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv4/b:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_bias5.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv5/b:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_bias6.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv6/b:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_conv1.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv1/W:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_conv2.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv2/W:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_conv3.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv3/W:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_conv4.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv4/W:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_conv5.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv5/W:0")))
		f.close()
		f = open('model_param_txt/weights/ReconNet_conv6.txt', 'w')
		f.write(np.array2string(sess.run("generator/conv6/W:0")))
		f.close()

	psnr = np.array([])

	idx = 0
	take_time=0
	for imName in imList:
		# Read image
		# im = cv2.imread(inputDir + imName,cv2.IMREAD_UNCHANGED)
		im = cv2.imread(inputDir + imName,0)
		# cv2.imshow('array_img',im)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		# im = cv2.resize(im,(256,256))
		im = cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
		[height, width] = im.shape

		# print ('im',im.shape)
		# Determine the size of zero pad
		rowPad = blockSize - (height % blockSize)
		colPad = blockSize - (width % blockSize)
		# print ('Pad',rowPad, colPad)

		imPad = np.concatenate((im, np.zeros([rowPad, width])), axis=0)
		imPad = np.concatenate((imPad, np.zeros([height+rowPad, colPad])),axis=1)
		# print ('imPad',imPad.shape)

		numBlocksRow = (height + rowPad)/blockSize
		numBlocksCol = (width + colPad)/blockSize

		outputImPad = np.zeros([height+rowPad, width+colPad])
		start_time=time.time()
		for i in range(int(numBlocksRow)):
			for j in range(int(numBlocksCol)):

				# Break into blocks
				block = imPad[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize]
				block = np.hstack(block)
				block = np.reshape(block, [blockSize*blockSize, 1])
				# blockCS = phi.dot(block)
				# blockCS = np.reshape(block, [1, 33, 33, 1]) # Reshape to 4D tensor
				blockIm = np.reshape(block, [1,blockSize,blockSize,1])
				
				
				# Feed blocks to the trained network
				reconstruction_ori_val = sess.run([reconstruction_ori], feed_dict={cs_meas: blockIm, is_train: False}) #feed 回去前面的csgan.build_reconstruction
				# reconstruction_ori_val = sess.run(fc2, feed_dict={cs_meas: blockIm, is_train: False}) #feed 回去前面的csgan.build_reconstruction
				reconstruction_op = np.reshape(reconstruction_ori_val, [33, 33])
				
				# ------------------------------print output feature map-----------------------------------
				# 如果要print出output feature map, 因為graph初始化時沒有初始化ofmap, 因此在這邊要再用sess.run feed一次dict進去才可以拿出來看!
				if i ==0 & j ==0:	
					if pre_layer_output:
						f = open('model_param_txt/activations/ofmap_conv1.txt', 'w')
						f.write(np.array2string(sess.run(conv1, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_conv2.txt', 'w')
						f.write(np.array2string(sess.run(conv2, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_conv3.txt', 'w')
						f.write(np.array2string(sess.run(conv3, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_conv4.txt', 'w')
						f.write(np.array2string(sess.run(conv4, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_conv5.txt', 'w')
						f.write(np.array2string(sess.run(conv5, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_conv6.txt', 'w')
						f.write(np.array2string(sess.run(conv6, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_relu1.txt', 'w')
						f.write(np.array2string(sess.run(bn1, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_relu2.txt', 'w')
						f.write(np.array2string(sess.run(bn2, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_relu3.txt', 'w')
						f.write(np.array2string(sess.run(bn3, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_relu4.txt', 'w')
						f.write(np.array2string(sess.run(bn4, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_relu5.txt', 'w')
						f.write(np.array2string(sess.run(bn5, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
						f = open('model_param_txt/activations/ofmap_relu6.txt', 'w')
						f.write(np.array2string(sess.run(reconstruction_ori, feed_dict={cs_meas: blockIm, is_train: False})))
						f.close()
				# --------------------------------------------------------------------------------------------

				# Re-arrange output into image
				outputImPad[i*blockSize:(i+1)*blockSize,j*blockSize:(j+1)*blockSize] = reconstruction_op
				outputIm = outputImPad[0:height,0:width]
		take_time=take_time+time.time()-start_time       
		print(" Average processing Time of current frame:",time.time()-start_time , "s")
		print ('outputIm',outputIm.shape[0])
		rmse = np.sqrt(np.mean(np.square(outputIm - im)))
		# temp_psnr = 20*np.log10(1./rmse)
		# if temp_psnr > 26:
		psnr = np.append(psnr, 20*np.log10(1./rmse))
		cv2.imwrite(matdir+imName+".png",(outputIm*255.)) # to store no format file(so that keeping float32 format)
		idx = idx +1
	print (psnr)
	print ('--------------')
	print ('psnr.mean():',psnr.mean())
	print(" Processing Time of all frames:", take_time/len(imList), "s")


	
	

	