##
path='/data/taoxm/pointSIFT_age/RS_age/'
fileNames=list.files(paste0(path,'/points/21000_xyzrgb'),'.xyz.txt')
names=as.numeric(sub('.xyz.txt','',fileNames))#4332
##use unique samples for each person
ageId=read.csv(paste0(path,'/phe/pageAndOtherUnique_age.csv'))
sampleUse=merge(names,ageId,by.x=1,by.y=1)
table(round(sampleUse$Age))
##
sampleUse=sampleUse[round(sampleUse$Age)<=90,]
sampleUse$AgeNN=round(sampleUse$Age)-52  #4112
write.csv(sampleUse,file=paste0(path,'/phe/sampleUse_NN_rgb_21000.csv'),row.names=F,quote=F)


##page samples
load(paste0(path,'/phe/RS_page2693.RData'))
index=match(pheMat[,1],sampleUse[,1])##page index in sampleUse
cross=index[!is.na(index)]##2607 overlaped
length(cross)#2148; 
##choose 1000 of page as test set
set.seed(100)
testIndex=cross[sample(length(cross),1000)]  
##
testSet=merge(sampleUse[testIndex,],pheMat,by.x=1,by.y=1)
trainSet=sampleUse[-testIndex,]#3112
##
write.csv(testSet,file=paste0(path,'/phe/sampleTest1000page_NN_rgb_21000.csv'),row.names=F,quote=F)
write.csv(trainSet,file=paste0(path,'/phe/sampleTrain_NN_rgb_21000.csv'),row.names=F,quote=F)





