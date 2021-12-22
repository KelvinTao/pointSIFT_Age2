###conda activate pointSIFT##run at aommand line
#bash this file
#source activate /data/taoxm/AI/envs/dpGPU ##first here run, bz10 folder
rootPath=/data/taoxm/pointSIFT_age/RS_age
cd $rootPath/script
data_path=$rootPath/data
batch_size=10  ## must: TestData Size (1000 here)//batch_size is integer
#result=result/bz_${batch_size}_2
#result=result/bz_${batch_size}_range
#result=result/bz_${batch_size}_range_bb_2
result=result/bz_${batch_size}_range_bb_NN_rgb
save_path=$rootPath/$result/model_pred
train_log_path=$rootPath/$result/train_log
test_log_path=$rootPath/$result/test_log
mkdir -p $save_path
mkdir -p $train_log_path
mkdir -p $test_log_path
#CUDA_VISIBLE_DEVICES=0,1\
#python train_eval_3D_age_one_GPU.py --save_path $save_path --data_path $data_path \
#python train_eval_3D_age_one_GPU_age_range.py --save_path $save_path --data_path $data_path \
#python train_eval_3D_age_one_GPU_ageRange_batchBalance.py --save_path $save_path --data_path $data_path \
python train_eval_3D_age_one_GPU_ageRange_batchBalance_NN_dense_addrgb.py --save_path $save_path --data_path $data_path \
--train_log_path $train_log_path --test_log_path $test_log_path --batch_size $batch_size \
--gpu_num 1 \
--learning_rate 1e-4

###conda deactivate  ##run at aommand line
#source deactivate
#tensorboard --logdir=$train_log_path
