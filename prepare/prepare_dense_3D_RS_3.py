import numpy as np

##
NUM_CLASS=100*2

# train data
## collect points
def mkdata(filePath):
    ids=np.loadtxt(filePath,dtype=str,skiprows=1,delimiter=',',usecols=0)
    points_set=[]
    for i,idi in enumerate(ids):
        if idi=='1.31024e+11':idi='131024000000'
        points_set.append(np.loadtxt(path+'/points/'+idi+'.txt',delimiter=' '))
        print(i)
        #if i>10:break
    ## produce label
    ##age_threths:0-99; age range: 0-100
    age=np.round(np.loadtxt(filePath,skiprows=1,delimiter=',',usecols=2))
    age[age>100]=100;
    assert not len(age[age<=0])
    age_label_set=[]
    rows=NUM_CLASS//2
    for a in age:
        age_label=np.zeros((rows,2))
        age_label[0:int(a),0]=1
        age_label[int(a):,1]=0
        age_label_set.append(age_label)
    
    return np.array(points_set),np.array(age_label_set)


path='/data/taoxm/pointSIFT_age/RS_age'
##train
filePath=path+'/phe/sampleTrain4108.csv'
points,label=mkdata(filePath)
np.savez(path+'/data/RS.train.4108.8192.npz',points_set=points,age_label_set=label)
##test
filePath=path+'/phe/sampleTest1000page.csv'
points,label=mkdata(filePath)
np.savez(path+'/data/RS.test.1000.npz',points_set=points,age_label_set=label)


