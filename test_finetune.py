    train_log_file = 'age.ckpt'
    
    #官方下载的检查点文件路径
    checkpoint_file = 'tdage/pretrained/vgg_16.ckpt'
    
    #设置batch_size
    batch_size  = 64
    
    learning_rate = lr

    n_train = len(train_labels)
    n_test = len(val_labels)
    #迭代轮数
    training_epochs =epoch
    
    #save_batch = 30
    #display_batch = 30
    if not tf.gfile.Exists(train_log_dir):
        tf.gfile.MakeDirs(train_log_dir)
                        
from nets import vgg_l2 as vgg
#获取模型参数的命名空间
arg_scope = vgg.vgg_arg_scope()



import tensorflow as tf 
from tensorflow.python import pywrap_tensorflow
slim = tf.contrib.slim
import model.pointSIFT_pointnet_age_dense_rgb_finetune as AGE_MODEL




model_dir = "/data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp_avgvs_grpws2/model_pred"
ckpt = tf.train.get_checkpoint_state(model_dir)
ckpt_path = ckpt.model_checkpoint_path
reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
param_dict = reader.get_variable_to_shape_map()
''' 
for key, val in param_dict.items():
    try:
        print(key)
        print(val)
    except:
        pass
'''

        with tf.Graph().as_default():
            self.build_graph()




save = tf.train.Saver(max_to_keep=1) 

with  slim.arg_scope(param_dict):  
    self.build_graph()
    params = slim.get_variables_to_restore(exclude=['layer8_dense'])
    restorer = tf.train.Saver(params)        
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint(train_log_dir)
        save.restore(sess,ckpt)