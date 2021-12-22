##scp log file
##118.1
cd /share_bio/nas5/liufan_group/taoxm/log_seg3D
###rm -r *

#79.35
#cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bb_NN_dense_rgb
cd /data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp
scp -r *log taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_seg3D
scp -r plot taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_seg3D

##local
#cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bb_NN_dense_rgb
cd /Users/taoxianming/Documents/reference/result/log/bz_10_range_bbOK_NN_dense_rgb_aug_dp
###rm -r *
#scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_seg3D/* .
scp -r taoxm@192.168.118.1:/share_bio/nas5/liufan_group/taoxm/log_seg3D/plot .
tensorboard --logdir=train_log
http://taoxianming.local:6006
tensorboard --logdir=test_log


##
import numpy as np
path0='/data/taoxm/pointSIFT_age/RS_age/data'
a=np.load(path0+'/sampleTest1000page_NN_idage_xyzrgb_21000.npz')['ages']
a=np.array([int(i) for i in a])+52
np.savetxt(path0+'/sampleTest1000page.age.txt', a, fmt='%.2f')
###
path='/data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp'
epoch1=75;epoch2=25;
pred1=np.loadtxt(path+'/model_pred/age_pred_'+str(epoch1)+'.txt')+52
#pred1=np.loadtxt(path+'/model_pred/best_age_pred_'+str(epoch)+'.txt')+52
pred2=np.loadtxt(path+'/model_pred/best_age_pred_'+str(epoch2)+'.txt')+52
##
np.mean(np.abs(pred1-a))
np.corrcoef(pred1,a)
#all=np.vstack((pred,a))
#np.corrcoef(all)
np.mean(np.abs(pred2-a))
np.corrcoef(pred2,a)

##
library(Metrics)
library(ggplot)
path0='/data/taoxm/pointSIFT_age/RS_age/data'
tar=read.csv(paste0(path0,'/sampleTest1000page.age.txt'),head=F)[,1]
path='/data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bbOK_NN_dense_rgb_aug_dp'
epoch1=75;epoch2=25;
pred=read.csv(paste0(path,'/model_pred/age_pred_',epoch1,'.txt'),head=F)[,1]+52
predB=read.csv(paste0(path,'/model_pred/best_age_pred_',epoch2,'.txt'),head=F)[,1]+52
mae(predB,tar)
mae(pred,tar)
##
p=ggplot(data.frame(x=tar,y=pred), aes(x=x, y=y))+geom_point()+
geom_abline(intercept=0,slope=1,color='green',size=.5,linetype=2)
ggsave(filname=paste0(path,'/plot/age_pred_',epoch1,'.jpg'),p)
p=ggplot(data.frame(x=tar,y=predB), aes(x=x, y=y))+geom_point()+
geom_abline(intercept=0,slope=1,color='green',size=.5,linetype=2)
ggsave(filname=paste0(path,'/plot/best_age_pred_',epoch2,'.jpg'),p)







