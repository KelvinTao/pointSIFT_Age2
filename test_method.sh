import tensorflow as tf
tf.add_to_collection('losses', tf.constant(2.2))
with tf.Session()as sess:
    print(sess.run(tf.get_collection('losses')))
    print(sess.run(tf.add_n(tf.get_collection('losses'))))

scp -r *log taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log
scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log/* .
tensorboard --logdir=train_log
http://taoxianming.local:6006 

scp -r *log taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_one_GPU
scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_one_GPU/* .

scp -r  script taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/script_3D
scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/script_3D/script .



scp -r 2 taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_one_GPU
scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_one_GPU/2 .
cd 2 
tensorboard --logdir=train_log

rm 2/*
##
cd /share_bio/nas5/liufan_group/taoxm/script_3D
cd /data/taoxm/pointSIFT_age/RS_age
scp -r script taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/script_3D

cd /Users/taoxianming/Documents/reference
scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/script_3D/script .

##scp log file
##118.1
cd /share_bio/nas5/liufan_group/taoxm/log_seg3D
###rm -r *

#79.35
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bb_NN_dense_rgb
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_regress_dense_rgb_aug_dp_svm
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_regress_dense_rgb_aug_dp_svm_group_smpws
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp_avgvs_grpws2
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bb_NN_dense_rgb_aug_dp_smpws_finetune
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bb_NN_dense_rgb_aug_dp_avgvs_grpws2_11_smpws_finetune
cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp_avgvs_grpws7
scp -r *log taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_seg3D

##local
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bb_NN_dense_rgb
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bbOK_NN_dense_rgb_aug_dp
#cd /Users/taoxianming/Documents/reference/result/log/logbz_10_range_bbOK_regress_dense_rgb_aug_dp_svm
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bbOK_regress_dense_rgb_aug_dp_svm_group_smpws
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bbOK_NN_dense_rgb_aug_dp
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bbOK_NN_dense_rgb_aug_dp_avgvs_grpws2
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bb_NN_dense_rgb_aug_dp_smpws_finetune
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bb_NN_dense_rgb_aug_dp_avgvs_grpws2_11_smpws_finetune
cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bbOK_NN_dense_rgb_aug_dp_avgvs_grpws7
###rm -r *
scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_seg3D/* .
tensorboard --logdir=train_log
http://taoxianming.local:6006
tensorboard --logdir=test_log


import numpy as np
import tensorflow as tf
r=np.array([[9,9,9],[8,8,8]])
c=[]
c.append(r)
c.append(r)
c.append(r)

with tf.Session()as sess:
	print(ages)
    print(sess.run(tf.concat(c, axis=1)))
    #print(sess.run(tf.reduce_mean(c)))


## train time and steps
train: 3112
test: 1000
batch: 10, 
steps: 1 epoch: 3112/10=311.2 steps
learning_rate 1e-4 1e-2, 1e-3, 1e-4, 1e-5
1h: 5 epochs
1000epoches: 200 h, 8.3 day
loss:

