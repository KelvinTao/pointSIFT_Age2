

import numpy as np
a=np.load('/data/taoxm/pointSIFT_age/RS_age/data/sampleTest1000page_NN_idage_xyzrgb_21000.npz')['ages']
a=np.array([int(i) for i in a])+52
epoch=130
pred1=np.loadtxt('/data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bb_NN_dense_rgb/model_pred/age_pred_'+str(epoch)+'.txt')+52
epoch=76
pred2=np.loadtxt('/data/taoxm/pointSIFT_age/RS_age/result/bz_10_range_bb_NN_dense_rgb/model_pred/best_age_pred_'+str(epoch)+'.txt')+52
##
np.mean(np.abs(pred1-a))
np.corrcoef(pred1,a)
#all=np.vstack((pred,a))
#np.corrcoef(all)

np.mean(np.abs(pred2-a))
np.corrcoef(pred2,a)

