###
readObj<-function(objFile){
        obj=read.table(objFile,stringsAsFactors=F,fill=T)
        v=apply(obj[which(obj[,1]=='v'),-1],2,as.numeric)
        #vn=apply(obj[which(obj[,1]=='vn'),-1],2,as.numeric)
        #vt=apply(obj[which(obj[,1]=='vt'),c(-1,-4)],2,as.numeric)
        #f=obj[which(obj[,1]=='f'),-1]
        #return(list(v=v,vn=vn,vt=vt,f=f))
        return(v)
}

##
path0='/Users/taoxianming/Documents/reference/obj_test'
##
phe=read.csv('/Users/taoxianming/Documents/reference/phe/pageAndOtherUnique_age.csv')[,c(1,4)]
npoints=21000
pNum=NULL
library(rgl)
###get result
for (i in 24:nrow(phe)){
  print(i)
  file_i=paste0(path0,'/obj/',phe[i,1],'.obj')
  v=readObj(file_i)
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
  pNum=c(pNum,nrow(v))
  ##choose points
  ##sort by Y, down to up
  if(T){
    #r=nrow(v)
    #start=round((r-npoints)/2);end=start+npoints
    vY=v[order(v[,2],decreasing = F),]
    #v=vY[start:end,]
    v=vY[1:npoints,]
  }
  if(T){
  par3d(windowRect = c(0,0,1920, 1080))
  next3d();rgl.viewpoint(zoom=0.5);
  plot3d(v,size=2,col='pink',box=F,aspect=F,type='p',axes=T,
      xlab='',ylab='',zlab='')#,lit=F);#ps,axes
  jpgFile=paste0(path0,'/jpg_choose_more/',phe[i,1],'.',nrow(v),'.',sex,'.jpg')
  rgl.snapshot(jpgFile)
  }
  #break
}

