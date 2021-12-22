###
library(data.table)
getColor<-function(col){
  f1=as.numeric(unlist(strsplit(col,'/')))
  indexV=((1:length(col))-1)*3+1
  f1VC=cbind(f1[indexV],f1[indexV+1])
  return(f1VC)
}
readObj<-function(objFile){
    #obj=read.table(objFile,stringsAsFactors=F,fill=T)
    #v=apply(obj[which(obj[,1]=='v'),-1],2,as.numeric)
    obj=fread(objFile,stringsAsFactors=F,fill=T,data.table=F)
    f=obj[which(obj[,1]=='f'),2:4]
    fVC=rbind(getColor(f[,1]),getColor(f[,2]),getColor(f[,3]))
    fVC=fVC[!duplicated(fVC[,1]),]
    vc=fVC[order(fVC[,1]),]
    v=apply(obj[which(obj[,1]=='v'),2:4],2,as.numeric)
    return(list(v,vc))
}

fillNA<-function(rgb){
  for (c in 1:3){
    rgb[which(is.na(rgb[,c])),c]=median(rgb[!is.na(rgb[,c]),c])
  }
  return(rgb)
}

##
path='/data/taoxm/data/3Ddata/RS/'
resPath='/data/taoxm/pointSIFT_age/RS_age/'
#path='/Users/taoxianming/Documents/reference/obj_test/'
#resPath='/Users/taoxianming/Documents/reference/'
##

phe=read.csv(paste0(resPath,'/phe/pageAndOtherUnique_age.csv'))[,c(1,4)]
npoints=21000
##library(rgl)
###get result
for (i in 1:nrow(phe)){
  name=phe[i,1]
  #name='131112084431'
  if (name==131024000000){name='131024000000'}
  #objFile=paste0(path,'/obj/',name,'.obj')
  vvc=readObj(paste0(path,'/obj/',name,'.obj'))
  v=vvc[[1]]
  vc=vvc[[2]]
  ##
  rgb=fread(paste0(path,'/rgb/',name,'.txt'),data.table=F)
  ###
  v=v[vc[,1],]
  rgb=rgb[vc[,2],]
  ##x, y, z, region by quantile
  sex=phe[i,2]
  ##cut region
  xth=quantile(v[,1],c(0.15, 0.85))
  zth=quantile(v[,3],0.5)
  #xth=quantile(v[,1],c(0.25, 0.75))
  if (sex==2){
    yth=quantile(v[,2],c(0.15, 0.9))
  }else{
    yth=quantile(v[,2],c(0.2, 0.95))
  }
  index=(v[,1]>=xth[1]) & (v[,1]<=xth[2]) &
  (v[,2]>=yth[1]) & (v[,2]<=yth[2]) & (v[,3]>=zth)
  v=v[index,]
  rgb=rgb[index,]
  print(paste0(i,' : ',name,' : ',nrow(v)))
  ##choose points
  ##sort by Y, down to up
  if(nrow(v)>=npoints){
    index2=order(v[,2],decreasing = F)
    v=v[index2,][1:npoints,]
    rgb=rgb[index2,][1:npoints,]
    rgb=fillNA(rgb)
    write.table(v,file=paste0(resPath,'/points/',npoints,'_xyzrgb/',name,'.xyz.txt'),
    row.names=F,col.names=F,quote=F,sep=' ')
    write.table(rgb,file=paste0(resPath,'/points/',npoints,'_xyzrgb/',name,'.rgb.txt'),
    row.names=F,col.names=F,quote=F,sep=' ')
  }
}



