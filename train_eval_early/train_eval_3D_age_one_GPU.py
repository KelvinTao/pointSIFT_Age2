from __future__ import division
from __future__ import print_function

import os
import numpy as np
import time
import tensorflow as tf
import argparse
import tf_utils.provider as provider
import models.pointSIFT_pointnet_age as AGE_MODEL

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run[default: 1000]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size during training[default: 32')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate[default: 1e-3]')
parser.add_argument('--save_path', default='model_param', help='model param path')
parser.add_argument('--data_path', default='data', help='scannet dataset path')
parser.add_argument('--train_log_path', default='log/pointSIFT_train')
parser.add_argument('--test_log_path', default='log/pointSIFT_test')
parser.add_argument('--gpu_num', type=int, default=1, help='number of GPU to train')

# basic params..
FLAGS = parser.parse_args()
BATCH_SZ = FLAGS.batch_size
LEARNING_RATE = FLAGS.learning_rate
MAX_EPOCH = FLAGS.max_epoch
SAVE_PATH = FLAGS.save_path
DATA_PATH = FLAGS.data_path
TRAIN_LOG_PATH = FLAGS.train_log_path
TEST_LOG_PATH = FLAGS.test_log_path
GPU_NUM = FLAGS.gpu_num
BATCH_PER_GPU = BATCH_SZ // GPU_NUM
##classfication predict: logits
NUM_CLASS = 100*2

# lr params..
DECAY_STEP = 200000
DECAY_RATE = 0.7

# bn params..
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

class ClassTrainer(object):
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.train_sz = 0
        self.test_sz = 0
        self.point_sz = 8192

        # batch loader init....
        self.batch_loader = None
        self.batch_sz = BATCH_SZ

        # net param...
        self.point_pl = None
        self.label_pl = None
        self.smpws_pl = None
        self.is_train_pl = None
        self.ave_MAE_pl = None
        self.net = None
        self.end_point = None
        self.bn_decay = None
        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.predict = None
        self.MAE = None
        self.MAE_wt = None
        self.batch = None  # int record the training step..
        self.gvs = None
        # summary
        self.MAE_summary = None
        self.MAE_weighted_summary=None

        # list for multi gpu tower..
        self.tower_grads = []
        self.net_gpu = []
        self.total_loss_gpu_list = []

    def calweights(self,c_age):
        ##NUM_CLASS=Nage_thresh x 2
        ##input: c_age, Nsample x Nage_thresh x 2 ,catrgorical age
        ##output: 
        ##sample_weights: calculate sample weights in c_age set
        ##Nsample,age for each sample
        labeln=np.sum(c_age[:,:,0],axis=1).astype(int)
        ##age distribution, age_i: index1, index2 
        ##[[index1,index2,...],[index7,index10,...],[]];Nage
        labes=list(range(min(labeln),max(labeln)+1))
        mydict=list(map(lambda x:[i for i,j in enumerate(labeln) if j==x],labes))
        ##age hist: age_i: Num_age_i
        lenth=list(map(lambda x:len(x),mydict))
        lenths=list(np.asarray(lenth)[np.asarray(lenth)>0])
        ##average num at one age
        assert c_age.shape[0]==sum(lenth)
        #assert len(c_age)==sum(lenth)
        meanle=sum(lenth)//len(lenths)
        ##Nsample
        sample_weights=np.ones(len(labeln))
        for i in range(len(mydict)):
            if lenth[i]>0:
                sample_weights[mydict[i]]=min(round(meanle/lenth[i],2),25)
        return sample_weights

    def load_data_npz(self,npz_path):
        d = np.load(npz_path)
        return [d["points_set"],d["age_label_set"]]
        ##return [d["points_set"][0:3,...],d["age_label_set"][0:3,...]]

    def load_data(self):
        ##load train and test; 
        ##list:["image"], d["age_label"]
        ##image: Nsample x Npoints x Nchannel; channel: X Y Z (+ R G B)
        ##label: Nsample x N_age_thresh x 2: age_thresh:0,1,2,...,99; [>,<]age_thresh
        train_data=self.load_data_npz(DATA_PATH+'/RS.train.4108.8192.npz')#[["image"], ["age_label"]]
        test_data=self.load_data_npz(DATA_PATH+'/RS.test.1000.npz')##list[[np],[np]]
        ##train sample weights
        costweight=self.calweights(train_data[1])
        train_data.append(costweight)
        self.train_data=train_data #['image','age_label','sample_weights'] 
        ##test sample weights
        maeweight=self.calweights(test_data[1])
        test_data.append(maeweight) ##['image','age_label','mae_weights'] 
        self.test_data=test_data
        self.train_sz = self.train_data[0].shape[0]
        self.test_sz = self.test_data[0].shape[0]
        print('train size %d and test size %d' % (self.train_sz, self.test_sz))
        #training logits length
        #NUM_CLASS =int(train_data[1].shape[1]*2)


    def get_learning_rate(self):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE,
                                                   self.batch * BATCH_SZ,
                                                   DECAY_STEP,
                                                   DECAY_RATE,
                                                   staircase=True)
        learning_rate = tf.maximum(learning_rate, 1e-5)
        tf.summary.scalar('learning rate', learning_rate)
        return learning_rate

    def get_bn_decay(self):
        bn_momentum = tf.train.exponential_decay(BN_INIT_DECAY,
                                                 self.batch * BATCH_SZ,
                                                 BN_DECAY_DECAY_STEP,
                                                 BN_DECAY_DECAY_RATE,
                                                 staircase=True)
        bn_momentum = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
        tf.summary.scalar('bn_decay', bn_momentum)
        return bn_momentum

    def get_batch_wdp(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, 3))
        batch_label = np.zeros((bsize, NUM_CLASS//2,2), dtype=np.int32)
        batch_smpw = np.zeros((bsize), dtype=np.float32)
        ps0, seg0, smpw0 = dataset  #[list]
        for i in range(bsize):
            idxsUse=idxs[i + start_idx]
            batch_data[i, ...] = ps = ps0[idxsUse,...]
            batch_label[i, ...] = seg0[idxsUse,...] #seg =
            batch_smpw[i] = smpw0[idxsUse] #smpw =
            ##dropout part points
            #dropout_ratio = np.random.random() * 0.5#0.875  # 0-0.875
            #drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]
            #batch_data[i, drop_idx, :] = batch_data[i, 0, :]
            #batch_label[i, drop_idx] = batch_label[i, 0]
            #batch_smpw[i, drop_idx] *= 0
        return batch_data, batch_label, batch_smpw

    def get_batch(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_sz, 3))
        batch_label = np.zeros((bsize, NUM_CLASS//2,2), dtype=np.int32)
        batch_smpw = np.zeros((bsize), dtype=np.float32)
        ps0, seg0, smpw0 = dataset #[list]
        for i in range(bsize):
            idxsUse=idxs[i + start_idx]
            batch_data[i, ...] = ps0[idxsUse,...]
            batch_label[i, ...] = seg0[idxsUse,...]
            batch_smpw[i] = smpw0[idxsUse]
        return batch_data, batch_label, batch_smpw

    @staticmethod
    def ave_gradient(tower_grad):
        ave_gradient = []
        for gpu_data in zip(*tower_grad):
            grads = []
            for g, k in gpu_data:
                t_g = tf.expand_dims(g, axis=0)
                grads.append(t_g)
            grad = tf.concat(grads, axis=0)
            grad = tf.reduce_mean(grad, axis=0)
            key = gpu_data[0][1]
            ave_gradient.append((grad, key))
        return ave_gradient

    # cpu part of graph
    def build_g_cpu(self):
        self.batch = tf.Variable(0, name='batch', trainable=False)
        self.point_pl, self.label_pl, self.smpws_pl = AGE_MODEL.placeholder_inputs(self.batch_sz, self.point_sz,NUM_CLASS)
        self.is_train_pl = tf.placeholder(dtype=tf.bool, shape=())
        self.ave_MAE_pl = tf.placeholder(dtype=tf.float32, shape=())
        self.optimizer = tf.train.AdamOptimizer(self.get_learning_rate())
        self.bn_decay = self.get_bn_decay()
        AGE_MODEL.get_model(self.point_pl, self.is_train_pl, num_class=NUM_CLASS, bn_decay=self.bn_decay)

    # graph for each gpu, reuse params...
    def build_g_gpu(self, gpu_idx):
        print("build graph in gpu %d" % gpu_idx)
        with tf.device('/gpu:%d' % gpu_idx), tf.name_scope('gpu_%d' % gpu_idx) as scope:
            point_cloud_slice = tf.slice(self.point_pl, [gpu_idx * BATCH_PER_GPU, 0, 0], [BATCH_PER_GPU, -1, -1])
            label_slice = tf.slice(self.label_pl, [gpu_idx * BATCH_PER_GPU, 0,0], [BATCH_PER_GPU, -1,-1])
            smpws_slice = tf.slice(self.smpws_pl, [gpu_idx * BATCH_PER_GPU], [BATCH_PER_GPU])
            ##model and loss, net is predicted logits
            self.net, end_point = AGE_MODEL.get_model(point_cloud_slice, self.is_train_pl, num_class=NUM_CLASS, bn_decay=self.bn_decay)
            self.loss = AGE_MODEL.get_loss(self.net,label_slice,num_class=NUM_CLASS,smpws=smpws_slice)
            ##gradients
            self.gvs = self.optimizer.compute_gradients(self.loss)

    def build_graph(self):
        with tf.device('/cpu:0'):
            self.build_g_cpu()
            self.tower_grads = []
            ##GPU graph
            #for i in range(GPU_NUM):
            for i in range(1):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self.build_g_gpu(i)
            # get training op
            self.train_op = self.optimizer.apply_gradients(self.gvs, global_step=self.batch)
            ## predict
            self.predict, self.MAE, self.MAE_wt =AGE_MODEL.eval_pred(self.net,input_labels=self.label_pl,num_class=NUM_CLASS)
            tf.summary.scalar('MAE', self.MAE)
            tf.summary.scalar('loss', self.loss)

    def training(self):
        with tf.Graph().as_default():
            self.build_graph()
            # merge operator (for tensorboard)
            merged = tf.summary.merge_all()
            iter_in_epoch = self.train_sz // self.batch_sz
            saver = tf.train.Saver()
            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            ##accuracy judgement/////////////
            best_MAE = 1000
            with tf.Session(config=config) as sess:
                train_writer = tf.summary.FileWriter(TRAIN_LOG_PATH, sess.graph)
                evaluate_writer = tf.summary.FileWriter(TEST_LOG_PATH, sess.graph)
                sess.run(tf.global_variables_initializer())
                epoch_sz = MAX_EPOCH
                tic = time.time()
                for epoch in range(epoch_sz):
                    ave_loss = 0
                    train_idxs = np.arange(0, self.train_data[0].shape[0])
                    np.random.shuffle(train_idxs)
                    for _iter in range(iter_in_epoch):
                        start_idx = _iter * self.batch_sz
                        end_idx = (_iter + 1) * self.batch_sz
                        batch_data, batch_label, batch_smpw = self.get_batch_wdp(self.train_data, train_idxs,
                                                                                 start_idx, end_idx)
                        #aug_data = provider.rotate_point_cloud_z(batch_data)
                        aug_data = provider.rotate_perturbation_point_cloud(batch_data)
                        aug_data = provider.shift_point_cloud(aug_data)
                        aug_data = provider.random_scale_point_cloud(aug_data)
                        aug_data = provider.jitter_point_cloud(aug_data)
                        aug_data = provider.random_point_dropout(aug_data,max_dropout_ratio=0.2)
                        loss, _, summary, step = sess.run([self.loss, self.train_op, merged, self.batch],
                                                          feed_dict={self.point_pl: aug_data,
                                                                     self.label_pl: batch_label,
                                                                     self.smpws_pl: batch_smpw,
                                                                     self.is_train_pl: True})
                        ave_loss += loss
                        train_writer.add_summary(summary, step)
                    ave_loss /= iter_in_epoch
                    #train_writer.add_summary(ave_loss)
                    print("epoch %d , Train average loss is %f take %.3f s" % (epoch + 1, ave_loss, time.time() - tic))
                    tic = time.time()
                    if (epoch + 1) % 5 == 0:
                        ## predict and evaluate
                        maewt,predict,mae = self.evaluate_one_epoch(sess, evaluate_writer, step, epoch)
                        if maewt < best_MAE:
                            _path = saver.save(sess, os.path.join(SAVE_PATH, "best_age_model_%d.ckpt" % (epoch + 1)))
                            print("epoch %d, best saved in file: " % (epoch + 1), _path)
                            best_MAE = maewt
                            np.savetxt(os.path.join(SAVE_PATH, 'best_age_pred_%d.txt'% (epoch + 1)),predict,delimiter=',',fmt='%.2f')
                ###predict finally
                _ ,predict,_ = self.evaluate_one_epoch(sess, evaluate_writer, step, epoch)
                _path = saver.save(sess, os.path.join(SAVE_PATH, 'train_base_age_model.ckpt'))
                np.savetxt(os.path.join(SAVE_PATH, 'train_base_age_pred_%d.txt'% (epoch + 1)),predict,delimiter=',',fmt='%.2f')
                print("Model saved in file: ", _path)

    def evaluate_one_epoch(self, sess, test_writer, step, epoch):
        print("---EVALUATE %d EPOCH---" % (epoch + 1))
        is_training = False
        loss_test=0
        MAE=0
        MAE_wt=0       
        ##all test data
        points_test,label_test,smpw_test=self.test_data
        ###
        sample_num=points_test.shape[0]
        #logits_all=np.zeros((sample_num,NUM_CLASS))
        predict=np.zeros((sample_num))
        ##
        iter_num=sample_num//self.batch_sz
        last_num=sample_num % self.batch_sz ##not work, 0 is ok
        if last_num>0:iter_num+=1
        ##prediction 
        test_idxs = np.arange(0, sample_num)
        for _iter in range(iter_num):
            start_idx = _iter * self.batch_sz
            end_idx = (_iter + 1) * self.batch_sz
            if last_num>0 and _iter==(iter_num-1):
                end_idx = _iter * self.batch_sz + last_num
            ##get batch data
            batch_data, batch_label, batch_smpw=self.get_batch(self.test_data, test_idxs, start_idx, end_idx)
            net, loss, mae, pred, mae_wt  = sess.run([self.net,self.loss,self.MAE,self.predict,self.MAE_wt], feed_dict={self.point_pl: batch_data,
                                                  self.label_pl: batch_label,
                                                  self.smpws_pl: batch_smpw,
                                                  self.is_train_pl: is_training})
            loss_test+=loss
            MAE+=mae
            MAE_wt+=mae_wt
            predict[start_idx:end_idx]=pred
        loss_test /=iter_num
        MAE/=iter_num
        MAE_wt/=iter_num
        #logits_all=tf.concat(logits_all0, axis=0)
        #loss_test=AGE_MODEL.get_loss(logits_all,label_test,num_class=NUM_CLASS,smpws=smpw_test)
        #logits_all=np.asarray(logits_all0)
        #self.predict, self.MAE, MAE_wt=AGE_MODEL.eval_pred(logits_all,label_test,num_class=NUM_CLASS,wt=smpw_test)
        print("Testset: loss is %.3f " % loss_test)
        print("Testset: MAE is %.3f " % MAE)
        print("Testset: MAE_wt is %.3f " % MAE_wt)
        ##
        loss_test_summary = tf.summary.Summary(value=[tf.summary.Summary.Value(tag="Test loss", simple_value=loss_test)])
        MAE_summary = tf.summary.Summary(value=[tf.summary.Summary.Value(tag="Test MAE", simple_value=MAE)])
        MAE_weighted_summary = tf.summary.Summary(value=[tf.summary.Summary.Value(tag="Test MAE_weighted", simple_value=MAE_wt)])
        test_writer.add_summary(loss_test_summary,step)
        test_writer.add_summary(MAE_summary, step)
        test_writer.add_summary(MAE_weighted_summary, step)
        return MAE_wt,predict,MAE

if __name__ == '__main__':
    trainer = ClassTrainer()
    trainer.load_data()
    trainer.training()

