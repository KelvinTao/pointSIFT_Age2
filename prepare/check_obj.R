##
path='/data/taoxm/pointSIFT_age/RS_age'
fileNames=list.files(paste0(path,'/points'),'.txt')
names=as.numeric(sub('.8192.txt','',fileNames))
##use unique samples for each person
ageId=read.csv(paste0(path,'/phe/pageAndOtherUnique_age.csv'))
#ageId0=read.table(paste0(path,'/phe/RS_IDAgeSex_ok2_repeatedImages.txt'),sep='\t',head=T)
##samples less in origin 3D obj, not page samples
ageId[,1][is.na(match(ageId[,1],names))]
sampleUse=merge(names,ageId,by.x=1,by.y=1)
##
lessObj=ageId[,1][is.na(match(ageId[,1],names))]
write.csv(lessObj,file=paste0(path,'/phe/lessObjID.csv'),row.names=F,quote=F)
##

##
write.csv(sampleUse,file=paste0(path,'/phe/sampleUse4756.csv'),row.names=F,quote=F)


##page samples
load(paste0(path,'/phe/RS_page2693.RData'))
index=match(pheMat[,1],sampleUse[,1])##page index in sampleUse
cross=index[!is.na(index)]##2607
##choose 1000 of page as test set
set.seed(100)
testIndex=cross[sample(length(cross),1000)]  
##
testSet=merge(sampleUse[testIndex,],pheMat,by.x=1,by.y=1)
trainSet=sampleUse[-testIndex,]
##
write.csv(testSet,file=paste0(path,'/phe/sampleTest1000.csv'),row.names=F,quote=F)
write.csv(trainSet,file=paste0(path,'/phe/sampleTrain3756.csv'),row.names=F,quote=F)









###2607
cross1=merge(ageId,pheMat,by.x=1,by.y=1)
cross2=merge(sampleUse,pheMat,by.x=1,by.y=1)
