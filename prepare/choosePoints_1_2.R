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
resPath='/data/taoxm/pointSIFT_age/RS_age/points'
##
npoints=8192
path='/data/taoxm/data/3Ddata/RS/obj'
names=gsub('.obj','',list.files(path,'*.obj'))
files=paste0(path,'/',names,'.obj')
for(fi in seq_along(files)){
  print(paste0(fi,':'))
  ##
  v=readObj(files[fi])
  yCenter=0
  xCenter=0
  zCenter=median(v[,3])
  print(paste('xMin: ',as.character(min(v[,1]))))
  print(paste('xMedian: ',as.character(median(v[,1]))))
  print(paste('xMax: ',as.character(max(v[,1]))))
  print(paste('yMin:',as.character(min(v[,2]))))
  print(paste('yMedian:',as.character(median(v[,2]))))
  print(paste('yMax:',as.character(max(v[,2]))))
  print(paste('zMin: ',as.character(min(v[,3]))))
  print(paste('zMedian: ',as.character(median(v[,3]))))
  print(paste('zMax: ',as.character(max(v[,3]))))
  #print(zCenter)
  if(fi>10)break
  ##sort
  #xOrder=order(v[,1])
  #vX=v[xOrder,]
  ##sort by Y, up to down
  yOrder=order(v[,2],decreasing = T)
  vY=v[yOrder,]
  print(nrow(v))
  #gap=floor(nrow(v)/npoints)
  print(gap)
  #print('min--max')
  #print(min(vY[,2]))
  #print(max(vY[,2]))
  #print((nrow(v)-gap*npoints)/nrow(v))
  ###
  #rowUse=(1:npoints)*gap
  #vUse=vY[rowUse,]
  #write.table(vUse,file=paste0(resPath,'/',names[fi],'.8192.txt'),
 # 	row.names=F,col.names=F,quote=F,sep=' ')
  #break
}



