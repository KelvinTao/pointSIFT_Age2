##
readObj<-function(objFile){
	obj=read.table(objFile,stringsAsFactors=F,fill=T)
	v=apply(obj[which(obj[,1]=='v'),-1],2,as.numeric)
	#vn=apply(obj[which(obj[,1]=='vn'),-1],2,as.numeric)
	#vt=apply(obj[which(obj[,1]=='vt'),c(-1,-4)],2,as.numeric)
	#f=obj[which(obj[,1]=='f'),-1]
	#return(list(v=v,vn=vn,vt=vt,f=f))
	return(v)
}

###
#resPath='/data/taoxm/pointSIFT_age/RS_age/points'
##
#npoints=8192
path='/Users/taoxianming/Documents/reference/obj_test'
names=gsub('.obj','',list.files(path,'*.obj'))
files=paste0(path,'/',names,'.obj')
statAll=NULL
for(fi in seq_along(files)){
  print(paste0(fi,':'))
  ##
  v=readObj(files[fi])
  stat=c(apply(v,2,min),apply(v,2,max),
    apply(v,2,function(x) (min(x)+max(x))/2),
    apply(v,2,mean))
  quant=as.vector(apply(v,2,function(x) quantile(x,seq(0.25, 0.75, 0.25))))
  statAll=rbind(statAll,c(stat,quant))
  ##sort
  #xOrder=order(v[,1])
  #vX=v[xOrder,]
  ##sort by Y, up to down
  if(F){
    yOrder=order(v[,2],decreasing = T)
    vY=v[yOrder,]
    print(nrow(v))
    gap=floor(nrow(v)/npoints)
    print(gap)
    print('min--max')
    print(min(vY[,2]))
    print(max(vY[,2]))
  }
}
statAll=as.data.frame(statAll)
names(statAll)=c(
  paste0(c('x','y','z'),'Min'),
  paste0(c('x','y','z'),'Max'),
  paste0(c('x','y','z'),'middle_axis'),
  paste0(c('x','y','z'),'Mean'),
  paste0(c('0.25','0.5','0.75'),'x'),
  paste0(c('0.25','0.5','0.75'),'y'),
  paste0(c('0.25','0.5','0.75'),'z'))
##
write.csv(statAll,paste0(path,'/xyz.stat.csv'))
