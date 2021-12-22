##make RGB matrix
library(imager)
library(data.table)
##get RGB
getRGB<-function(objFile,bmpFile,type='rgb'){
	#bj=read.table(objFile,stringsAsFactors=F,fill=T)
	obj=fread(objFile,fill=T,data.table=F)
	vt=apply(obj[which(obj[,1]=='vt'),c(-1,-4)],2,as.numeric)
	rgb=round(load.image(bmpFile)*255)#0-1 for r g b #rgb=read.bmp(bmpFile)#/255##255,shape not ok
	#if (type='hsv')rgb=RGBtoHSV(rgb)
	h=nrow(rgb);v=ncol(rgb);
	pos=round(cbind(h*vt[,1],v*(1-vt[,2])))
	rgb1=unlist(apply(pos,1,function(x)return(c(rgb[x[1],x[2],,]))))
	xi=(1:(length(rgb1)/3))*3-2
	yi=xi+1
	zi=xi+2
	rgb=as.data.frame(cbind(rgb1[xi],rgb1[yi],rgb1[zi]))
	return(rgb)
}

###
#path='/Users/taoxianming/Documents/face_3D/twinsUK/3D/MAP/'
#path='/Users/taoxianming/Documents/face_3D/RS/3D/LOC/'
#id='131012090026'
#objbmp=paste0(path,id,c('.obj','.bmp'))
#rgb=getRGB(objbmp[1],objbmp[2])

###
path='/data/taoxm/data/3Ddata/RS'
###
objPath=paste0(path,'/obj')
bmpPath='/mnt/img/RS/RS3D-tangkun-Process/origin/bmp'
##
rgbPath=paste0(path,'/rgb')
###
names=list.files(objPath,'*.obj')
##
objFiles=paste0(objPath,'/',names)
bmpFiles=paste0(bmpPath,'/',gsub('obj','bmp',names))
rgbFiles=paste0(rgbPath,'/',gsub('obj','txt',names))
##
for(i in seq_along(names)){
  print(i)
  rgb=getRGB(objFiles[i],bmpFiles[i])
  write.table(rgb,file=rgbFiles[i],sep=' ',row.names=F,col.names=F,quote=F)
}




