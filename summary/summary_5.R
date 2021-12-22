##
path='/data/taoxm/pointSIFT_age/RS_age'
##age 
phe=read.csv(paste0(path,'/phe/sampleTest1000page.csv'))
pred=read.table(paste0(path,'/result/bz_10/model_pred/best_age_pred_10.txt'),head=F)
##
library(Metrics)
MAD=mae(phe$Age,pred[,1])

##
train=read.csv(paste0(path,'/phe/sampleTrain4108.csv'))
a=hist(train$Age)
b=hist(phe$Age)
