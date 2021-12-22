#source activate /data/taoxm/AI/envs/dpGPU
#bash train_eval_3D_age_one_GPU_4.sh > log/RS_bz10_one_GPU.log
#bash train_eval_3D_age_one_GPU_4.sh > log/RS_bz10_one_GPU_2.log
#bash train_eval_3D_age_one_GPU_4.sh > log/RS_bz10_one_GPU_range.log
#bash train_eval_3D_age_one_GPU_4.sh > log/RS_bz5_one_GPU_range_bb.log
#bash train_eval_3D_age_one_GPU_4.sh > log/RS_bz10_one_GPU_range_bb_2.log
#bash train_eval_3D_age_one_GPU_4_NN.sh > log/RS_bz10_one_GPU_range_bb_NN.log
#bash train_eval_3D_age_one_GPU_4_NN_dense.sh > log/RS_bz10_one_GPU_range_bb_NN_dense.log
#bash train_eval_3D_age_one_GPU_4_NN_dense.sh > log/RS_bz20_one_GPU_range_bb_NN_dense.log
#bash train_eval_3D_age_one_GPU_4_NN_dense.sh > log/RS_bz10_one_GPU_range_bb_NN_dense.log
#bash train_eval_3D_age_one_GPU_4_NN_dense.sh > log/RS_bz10_one_GPU_range_bb_NN_dense_rgb.log
bash run_3D_age_one_GPU_4_NN_dense.sh > log/RS_bz10_one_GPU_range_bbOK_NN_dense_rgb_aug_dp.log
#source deactivate
