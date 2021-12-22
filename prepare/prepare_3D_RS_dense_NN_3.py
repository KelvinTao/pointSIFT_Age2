import numpy as np

# train data
## collect points
def mkdata(filePath,NUM_CLASS):
    ids=np.loadtxt(filePath,dtype=str,skiprows=1,delimiter=',',usecols=0)
    points_set=[]
    for i,idi in enumerate(ids):
        if idi=='1.31024e+11':idi='131024000000'
        #points_set.append(np.loadtxt(path+'/points/21000/'+idi+'.8192.txt',delimiter=' '))
        points_set.append(np.loadtxt(path+'/points/21000/'+idi+'.txt',delimiter=' '))
        print(i)
        #if i>10:break
    ## produce label
    ##age_threths:0-99; age range: 0-100
    ageNN=np.round(np.loadtxt(filePath,skiprows=1,delimiter=',',usecols=4))
    assert not len(ageNN[ageNN<0])
    age_label_set=[]
    rows=NUM_CLASS//2
    for a in ageNN:
        age_label=np.zeros((rows,2))
        age_label[0:int(a),0]=1
        age_label[int(a):,1]=1
        age_label_set.append(age_label)
    return np.array(points_set),np.array(age_label_set)

##
path='/data/taoxm/pointSIFT_age/RS_age'
NUM_CLASS=38*2
##train
filePath=path+'/phe/sampleTrain_NN_21000.csv'
points,label=mkdata(filePath,NUM_CLASS)
np.savez(path+'/data/sampleTrain_NN_21000.npz',points_set=points,age_label_set=label)
##test
filePath=path+'/phe/sampleTest1000page_NN_21000.csv'
points,label=mkdata(filePath,NUM_CLASS)
np.savez(path+'/data/sampleTest1000page_NN_21000.npz',points_set=points,age_label_set=label)

