ageRange_batchBalance_NN_dense
ageRange:50-90, real 52-90
batchBalance: resampling, batch_size=10, age range to 10 group, each group get 1, not use
batchBalanceOK: resampling, batch_size=10, age range to 10 group, each group get 1, ok
NN: classification, 52-90, NUM_CLASS=38*2, 
dense: the same to origin of pointSIFT, because of graph size
dp: dropout a part of points
addrgb: add color feature
aug: sample augmentation: rotate, shift, scale

### perform well script
train_eval_3D_age_one_GPU_ageRange_batchBalance_NN_dense_addrgb.py
##
train_eval_3D_age_one_GPU_ageRange_batchBalance_NN_dense_addrgb_aug_dp.py
train_eval_3D_age_one_GPU_ageRange_batchBalanceOK_NN_dense_addrgb_aug_dp.py


###
add more points?
learning_rate 1e-2, 1e-3, 1e-4, 1e-5
