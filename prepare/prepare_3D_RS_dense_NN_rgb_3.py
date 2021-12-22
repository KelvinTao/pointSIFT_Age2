import numpy as np
import glob

# train data
## collect points
def mkdata(NUM_CLASS,ids,ageNN):
    #ids=np.loadtxt(filePath,dtype=str,skiprows=1,delimiter=',',usecols=0)
    points_set=[]
    rgb_set=[]
    for i,idi in enumerate(ids):
        points_set.append(np.loadtxt(path+'/points/21000_xyzrgb/'+idi+'.xyz.txt',delimiter=' '))
        rgb_set.append(np.loadtxt(path+'/points/21000_xyzrgb/'+idi+'.rgb.txt',delimiter=' '))#
        print(i)
        #if i>10:break
    ## produce label
    ##age_threths:0-99; age range: 0-100
    #ageNN=np.round(np.loadtxt(filePath,skiprows=1,delimiter=',',usecols=4))
    ageNN=np.array([int(i) for i in ageNN])
    assert not len(ageNN[ageNN<0])
    age_label_set=[]
    rows=NUM_CLASS//2
    for a in ageNN:
        age_label=np.zeros((rows,2))
        age_label[0:int(a),0]=1
        age_label[int(a):,1]=1
        age_label_set.append(age_label)
    return ids,np.array(points_set),np.array(rgb_set),np.array(age_label_set)

def getUse(path,filePath):
  files=glob.glob(path+'/points/21000_xyzrgb/*.xyz.txt')
  imgIds=[i.replace('.xyz.txt','').replace(path+'/points/21000_xyzrgb/','') for i in files]
  ##train
  #filePath=path+'/phe/sampleTrain_NN_21000.csv'
  idAge=np.loadtxt(filePath,dtype=str,skiprows=1,delimiter=',',usecols=[0,4])
  tarIds=[i.replace('1.31024e+11','131024000000') for i in idAge[:,0]]
  idAge[:,0]=tarIds
  ##
  idUse=list(set(imgIds).intersection(set(tarIds)))
  ageUse=[]
  for idi in idUse:
    for j in range(idAge.shape[0]):
      if idAge[j,0]==idi:
        ageUse.append(idAge[j,1])
  return idUse,ageUse
##

##
path='/data/taoxm/pointSIFT_age/RS_age'
NUM_CLASS=38*2
##train data
filePath=path+'/phe/sampleTrain_NN_rgb_21000.csv'
idUse,ageUse=getUse(path,filePath)
ids,points,rgb,label=mkdata(NUM_CLASS,idUse,ageUse)
np.savez(path+'/data/sampleTrain_NN_idage_xyzrgb_21000.npz',ids=idUse,ages=ageUse,rgb_set=rgb,points_set=points,age_label_set=label)
##test
filePath=path+'/phe/sampleTest1000page_NN_rgb_21000.csv'
idUse,ageUse=getUse(path,filePath)
ids,points,rgb,label=mkdata(NUM_CLASS,idUse,ageUse)
np.savez(path+'/data/sampleTest1000page_NN_idage_xyzrgb_21000.npz',ids=idUse,ages=ageUse,rgb_set=rgb,points_set=points,age_label_set=label)





