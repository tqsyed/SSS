import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
import utils
import random
def get_semi_dataset(path,dataset,percent,logger):
    logger.debug("Load Semi supervised dataset==>%s" % (dataset))
    if dataset=='credit':
        drop_column=['Class','Time']
        target='Class'
    elif dataset=='redwine':
        drop_column=['quality']
        target='quality'
    elif dataset=='paysim':
        drop_column=['isFraud']
        target='isFraud'
    elif dataset=='optical':
        drop_column=['Class','0','39']
        target='Class'
    else:
        drop_column=['Class']
        target='Class'


    data = pd.read_csv(path)
    #################Smote##########################
    y = data[target].values
    data = data.drop(drop_column, axis = 1)
    x = data.values
    sm = SMOTE()
    x, y = sm.fit_resample(x, y)
    data = pd.DataFrame(x, columns=data.columns)
    df_y = pd.DataFrame(y)
    data[target]=df_y
    drop_column = target
    # data.to_csv('osPendigits.csv',index=False)
    ##################################################
    anomaly = data[data.Class == 1]
    normal = data[data.Class == 0]
    percent=int(len(data)*(percent/100))
    if percent>len(anomaly) or percent>len(normal):
        raise Exception('percent should not exceed any class.')

    normal = normal.head(percent)
    anomaly = anomaly.head(percent)

    y=np.append(normal[target].values,anomaly[target].values)
    normal = normal.drop(drop_column, axis = 1)
    anomaly = anomaly.drop(drop_column, axis = 1)
    ##for noise only
    # normal = (normal-normal.min())/(normal.max()-normal.min())
    # anomaly = (anomaly-anomaly.min())/(anomaly.max()-anomaly.min())
    x = np.vstack([normal.values,anomaly.values])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify = y, random_state=3) #4,11,12,14,18
    x_train, x_test = utils.normalize(x_train, x_test)
    print("Load Semi dataset")
    return x_train, x_test, y_train, y_test



def get_full_dataset(path,dataset,logger):
    logger.debug("Load Full dataset==>%s" % (dataset))
    if dataset=='credit':
        drop_column=['Class','Time']
        target='Class'
    elif dataset=='redwine':
        drop_column=['quality']
        target='quality'
    elif dataset=='paysim':
        drop_column=['isFraud']
        target='isFraud'
    elif dataset=='optical':
        drop_column=['Class','0','39']
        target='Class'
    else:
        drop_column=['Class']
        target='Class'

    data = pd.read_csv(path)
    y = data[target].values
    data = data.drop(drop_column, axis = 1)
    x = data.values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify = y, random_state=3) #4,11,12,14,18
    x_train, x_test = utils.normalize(x_train, x_test)
    return x_train, x_test, y_train, y_test

def get_noisy_data(path,dataset):
    if dataset=='credit':
        drop_column=['Class','Time']
    elif dataset=='redwine':
        drop_column=['quality']
    elif dataset=='paysim':
        drop_column=['isFraud']
    elif dataset=='optical':
        drop_column=['Class','0','39']
    else:
        drop_column=['Class']

    data = pd.read_csv(path)
    data = data.drop(drop_column, axis = 1)
    normData = (data-data.min())/(data.max()-data.min())
    x = normData.values
    noiseData = np.array(np.zeros(x.shape[1]))
    noiseLabel = np.array([])
    h=x
    # for i in tqdm(range(len(x))):
    for j in tqdm(range(4)):
        if j==0:
            noiseValue = h
        else:
            if j==1:
                # sigma=0.04*(x.max()-x.min())
                sigma=0.16*(x.max()-x.min())
                # sigma=0.33*(x.max()-x.min())
            elif j==2:
                # sigma=0.16*(x.max()-x.min())
                sigma=0.33*(x.max()-x.min())
                # sigma=0.66*(x.max()-x.min())
            elif j==3:
                # sigma = 0.32*(x.max()-x.min())
                sigma = 0.66*(x.max()-x.min())
                # sigma=0.99*(x.max()-x.min())
            noiseValue = h[:,:]+ np.random.normal(0,sigma,x.shape[1])
                # sigma=0.42*(x.max()-x.min())
            # elif j==4:
            #      sigma=0.36*(x.max()-x.min())
            #      # sigma=0.56*(x.max()-x.min())
            # elif j==5:
            #     sigma=0.45*(x.max()-x.min())
            #     # sigma=0.70*(x.max()-x.min())
            # elif j==6:
            #     sigma = 0.54*(x.max()-x.min())
            #     # sigma = 0.84*(x.max()-x.min())
            # elif j==7:
            #     sigma = 0.66*(x.max()-x.min())
            #     # sigma = 0.99*(x.max()-x.min())
            # if j==1:
            #     # sigma=0.04*(x.max()-x.min())
            #     sigma=0.08*(x.max()-x.min())
            #     # sigma=0.01*(x.max()-x.min())
            # elif j==2:
            #     # sigma=0.06*(x.max()-x.min())
            #     sigma=0.12*(x.max()-x.min())
            #     # sigma=0.08*(x.max()-x.min())
            # elif j==3:
            #     # sigma = 0.08*(x.max()-x.min())
            #     sigma=0.16*(x.max()-x.min())
            #     # sigma=0.15*(x.max()-x.min())
            # elif j==4:
            #     # sigma=0.10*(x.max()-x.min())
            #     sigma=0.20*(x.max()-x.min())
            #     # sigma=0.22*(x.max()-x.min())
            # elif j==5:
            #     # sigma=0.12*(x.max()-x.min())
            #     sigma=0.24*(x.max()-x.min())
            #     # sigma=0.29*(x.max()-x.min())
            # elif j==6:
            #     # sigma = 0.14*(x.max()-x.min())
            #     sigma = 0.28*(x.max()-x.min())
            #     # sigma=0.36*(x.max()-x.min())
            # elif j==7:
            #     # sigma = 0.16*(x.max()-x.min())
            #     sigma = 0.32*(x.max()-x.min())
            #     # sigma=0.43*(x.max()-x.min())
            # elif j==8:
            #     # sigma=0.18*(x.max()-x.min())
            #     sigma=0.36*(x.max()-x.min())
            #     # sigma=0.50*(x.max()-x.min())
            # elif j==9:
            #     # sigma=0.20*(x.max()-x.min())
            #     sigma=0.40*(x.max()-x.min())
            #     # sigma=0.57*(x.max()-x.min())
            # elif j==10:
            #     # sigma = 0.22*(x.max()-x.min())
            #     sigma=0.44*(x.max()-x.min())
            #     # sigma=0.64*(x.max()-x.min())
            # elif j==11:
            #     # sigma=0.24*(x.max()-x.min())
            #     sigma=0.48*(x.max()-x.min())
            #     # sigma=0.71*(x.max()-x.min())
            # elif j==12:
            #     # sigma=0.26*(x.max()-x.min())
            #     sigma=0.52*(x.max()-x.min())
            #     # sigma=0.78*(x.max()-x.min())
            # elif j==13:
            #     # sigma = 0.28*(x.max()-x.min())
            #     sigma = 0.56*(x.max()-x.min())
            #     # sigma=0.85*(x.max()-x.min())
            # elif j==14:
            #     # sigma = 0.30*(x.max()-x.min())
            #     sigma = 0.60*(x.max()-x.min())
            #     # sigma=0.92*(x.max()-x.min())
            # elif j==15:
            #     # sigma = 0.32*(x.max()-x.min())
            #     sigma = 0.66*(x.max()-x.min())
            #     # sigma = 0.99*(x.max()-x.min())


        label=np.full((h.shape[0]), j)

        noiseLabel=np.append(noiseLabel,label)
        #             noiseLabel=np.vstack([noiseLabel,j+1])
        noiseData= np.vstack([noiseData,noiseValue])

    noiseData = np.delete(noiseData, 0, 0)
    x_train, x_test, y_train, y_test = train_test_split(noiseData, noiseLabel, test_size=0.2, stratify = noiseLabel, random_state=3)
    return x_train, x_test, y_train, y_test

def get_rotation_data(path,dataset):
    if dataset=='credit':
        drop_column=['Class','Time']
    elif dataset=='redwine':
        drop_column=['quality']
    elif dataset=='paysim':
        drop_column=['isFraud']
    elif dataset=='optical':
        drop_column=['Class','0','39']
    else:
        drop_column=['Class']

    data = pd.read_csv(path)
    x = data.drop(drop_column, axis = 1).values
    noiseData = np.array(np.zeros(x.shape[1]))
    noiseLabel = np.array([])
    h=x
    # for i in tqdm(range(len(x))):
    for j in tqdm(range(4)):
        if j==0:
            noiseValue = h
        else:
            noiseValue = h[:,get_rotation_array(j,x.shape[1])]

        label=np.full((h.shape[0]), j)

        noiseLabel=np.append(noiseLabel,label)
        #             noiseLabel=np.vstack([noiseLabel,j+1])
        noiseData= np.vstack([noiseData,noiseValue])

    noiseData = np.delete(noiseData, 0, 0)
    x_train, x_test, y_train, y_test = train_test_split(noiseData, noiseLabel, test_size=0.2, stratify = noiseLabel, random_state=3)
    x_train, x_test = utils.normalize(x_train, x_test)
    return x_train, x_test, y_train, y_test

# def rotation_followed_by_noise(path,dataset):
#     if dataset=='credit':
#         drop_column=['Class','Time']
#     elif dataset=='redwine':
#         drop_column=['quality']
#     elif dataset=='paysim':
#         drop_column=['isFraud']
#     elif dataset=='optical':
#         drop_column=['Class','0','39']
#     else:
#         drop_column=['Class']
#
#     data = pd.read_csv(path)
#     data = data.drop(drop_column, axis = 1)
#     normData = (data-data.min())/(data.max()-data.min())
#     x = normData.values
#     noiseData = np.array(np.zeros(x.shape[1]))
#     noiseLabel = np.array([])
#     h=x
#     classLabel=0
#     for j in tqdm(range(4)):
#         if j==0:
#             noiseValue = h
#         else:
#             if j==1:
#                 sigma=0.04*(x.max()-x.min())
#                 # sigma=0.14*(x.max()-x.min())
#             elif j==2:
#                 sigma=0.16*(x.max()-x.min())
#                 # sigma=0.28*(x.max()-x.min())
#             elif j==3:
#                 sigma = 0.32*(x.max()-x.min())
#             noiseValue = h[:,:]+ np.random.normal(0,sigma,x.shape[1])
#
#         for i in tqdm(range(4)):
#             if i==0:
#                 rotationArray = noiseValue
#             else:
#                 rotationArray = noiseValue[:,get_rotation_array(i,x.shape[1])]
#
#             label=np.full((h.shape[0]), classLabel)
#             classLabel+=1
#             noiseLabel=np.append(noiseLabel,label)
#             noiseData= np.vstack([noiseData,rotationArray])
#
#     noiseData = np.delete(noiseData, 0, 0)
#     x_train, x_test, y_train, y_test = train_test_split(noiseData, noiseLabel, test_size=0.2, stratify = noiseLabel, random_state=3)
#     return x_train, x_test, y_train, y_test

##
# noise followed by rotation:
# R0N0
# R0N1
# R0N2
# .
# R2N4
# .
# R4N4
##
# def noise_followed_by_rotaion(path,dataset):
#     if dataset=='credit':
#         drop_column=['Class','Time']
#     elif dataset=='redwine':
#         drop_column=['quality']
#     elif dataset=='paysim':
#         drop_column=['isFraud']
#     elif dataset=='optical':
#         drop_column=['Class','0','39']
#     else:
#         drop_column=['Class']
#
#     data = pd.read_csv(path)
#     data = data.drop(drop_column, axis = 1)
#     normData = (data-data.min())/(data.max()-data.min())
#     x = normData.values
#     noiseData = np.array(np.zeros(x.shape[1]))
#     noiseLabel = np.array([])
#     h=x
#     classLabel=0
#     for i in tqdm(range(4)):
#         if i==0:
#             rotationArray = h
#         else:
#             rotationArray = h[:,get_rotation_array(i,x.shape[1])]
#         for j in tqdm(range(4)):
#             if j==0:
#                 noiseValue = rotationArray
#             else:
#                 if j==1:
#                     sigma=0.04*(x.max()-x.min())
#                     # sigma=0.14*(x.max()-x.min())
#                 elif j==2:
#                     sigma=0.16*(x.max()-x.min())
#                     # sigma=0.28*(x.max()-x.min())
#                 elif j==3:
#                     sigma = 0.32*(x.max()-x.min())
#                 noiseValue = rotationArray[:,:]+ np.random.normal(0,sigma,x.shape[1])
#
#             label=np.full((h.shape[0]), classLabel)
#             classLabel+=1
#             noiseLabel=np.append(noiseLabel,label)
#             noiseData= np.vstack([noiseData,noiseValue])
#
#     noiseData = np.delete(noiseData, 0, 0)
#     x_train, x_test, y_train, y_test = train_test_split(noiseData, noiseLabel, test_size=0.2, stratify = noiseLabel, random_state=3)
#     return x_train, x_test, y_train, y_test
#
#
# def get_mortgage_full(path):
#     # print("Complete data load")
#     # trian_path=path+"train.csv"
#     # train_data = pd.read_csv(trian_path)
#     # train_data.fillna(0,inplace=True)
#     # x = train_data.drop(['TARGET'], axis = 1).values
#     # y = train_data['TARGET'].values
#     # test_path=path+"test.csv"
#     # test_data = pd.read_csv(test_path)
#     # test_data.fillna(0,inplace=True)
#     # x = np.vstack([x,(test_data.drop(['TARGET'], axis = 1).values)])
#     # y = np.append(y,test_data['TARGET'].values)
#     # sm = SMOTE()
#     # x, y = sm.fit_resample(x, y)
#     # train_data = train_data.drop(['TARGET'], axis = 1)
#     # data = pd.DataFrame(x, columns=train_data.columns)
#     # df_y = pd.DataFrame(y)
#     # data['TARGET']=df_y
#     # data.to_csv('oversampleMortgage2.csv',index=False)
#     train_data = pd.read_csv('oversampleMortgage.csv')
#     x = train_data.drop(['TARGET'], axis = 1).values
#     y = train_data['TARGET'].values
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify = y, random_state=3)
#     return x_train, x_test, y_train, y_test
#
# def get_anomaly(path = '../creditcard.csv'):
#     data = pd.read_csv(path)
#     data = data[data.Class == 1]
#     x = data.drop(['Class','Time'], axis = 1).values
#     y = data['Class'].values
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify = y, random_state=3) #4,11,12,14,18
#     return x_train, x_test, y_train, y_test
#
# def get_mortgage_noisy_data(path = '../creditcard.csv'):
#     print("Noise Mortgage Data Load start")
#     # trian_path=path+"train.csv"
#     # train_data = pd.read_csv(trian_path)
#     # train_data.fillna(0,inplace=True)
#     # x = train_data.drop(['TARGET'], axis = 1).values
#     # test_path=path+"test.csv"
#     # test_data = pd.read_csv(test_path)
#     # test_data.fillna(0,inplace=True)
#     # x = np.vstack([x,(test_data.drop(['TARGET'], axis = 1).values)])
#     train_data = pd.read_csv('oversampleMortgage.csv')
#     train_data = train_data.drop(['TARGET'], axis = 1)
#     normData = (train_data-train_data.min())/(train_data.max()-train_data.min())
#     x = normData.values
#     noiseData = np.array(np.zeros(x.shape[1]))
#     noiseLabel = np.array([])
#     h=x
#     # for i in tqdm(range(len(x))):
#     for j in tqdm(range(4)):
#         if j==0:
#             noiseValue = h
#         else:
#             noiseValue = h[:,:]+ np.random.normal(0,j+2,x.shape[1])
#
#         label=np.full((h.shape[0]), j)
#
#         noiseLabel=np.append(noiseLabel,label)
#         #             noiseLabel=np.vstack([noiseLabel,j+1])
#         noiseData= np.vstack([noiseData,noiseValue])
#
#     noiseData = np.delete(noiseData, 0, 0)
#     x_train, x_test, y_train, y_test = train_test_split(noiseData, noiseLabel, test_size=0.2, stratify = noiseLabel, random_state=3)
#     return x_train, x_test, y_train, y_test
#
# def get_mortgage_rotation_data(path):
#     # print("Mortgage Data Load start")
#     # trian_path=path+"train.csv"
#     # train_data = pd.read_csv(trian_path)
#     # train_data.fillna(0,inplace=True)
#     # x = train_data.drop(['TARGET'], axis = 1).values
#     # test_path=path+"test.csv"
#     # test_data = pd.read_csv(test_path)
#     # test_data.fillna(0,inplace=True)
#     # x = np.vstack([x,(test_data.drop(['TARGET'], axis = 1).values)])
#     train_data = pd.read_csv('oversampleMortgage.csv')
#     x = train_data.drop(['TARGET'], axis = 1).values
#     print("Mortgage Data Load start(",x.shape)
#     # noiseData = np.array(np.zeros(x.shape[1]))
#     noiseData = np.empty((4*len(x), x.shape[1]))
#     noiseLabel = np.array([])
#     h=x
#     # for i in tqdm(range(len(x))):
#     for j in tqdm(range(4)):
#         if j==0:
#             noiseValue=noiseData[:len(x)]
#         elif j==1:
#             noiseValue=noiseData[len(x):len(x)*2]
#         elif j==2:
#             noiseValue=noiseData[len(x)*2:len(x)*3]
#         else:
#             noiseValue=noiseData[len(x)*3:]
#
#         if j==0:
#             noiseValue[:] = h
#         else:
#             noiseValue[:] = h[:,get_rotation_array(j,x.shape[1])]
#
#         label=np.full((h.shape[0]), j)
#
#         noiseLabel=np.append(noiseLabel,label)
#         #             noiseLabel=np.vstack([noiseLabel,j+1])
#         # noiseData= np.vstack([noiseData,noiseValue])
#
#     # noiseData = np.delete(noiseData, 0, 0)
#     print("Mortgage Data Load End(",noiseData.shape)
#     np.random.shuffle(noiseData)
#     test, training = noiseData[:452297,:], noiseData[452297:,:]
#     data = pd.DataFrame(noiseData)
#     df_y = pd.DataFrame(noiseLabel)
#     data['TARGET']=df_y
#     msk = np.random.rand(len(data)) < 0.5
#     train = data[msk]
#     test = data[~msk]
#     # data = data.iloc[np.random.permutation(len(data))]
#     train = train.reindex(np.random.permutation(train.index))
#     test = test.reindex(np.random.permutation(test.index))
#     train=train.append(test)
#     data=train
#     X = data.drop(['TARGET'], axis = 1).values
#     Y = data['TARGET'].values
#     x_train, x_test, y_train, y_test = shuffle(X, Y, 5)
#     print("xTrain:[",x_train.shape,"],y_train[",y_train.shape,"],xTest:[",x_test.shape,"],y_test[",y_test.shape)
#     # x_train, x_test, y_train, y_test = train_test_split(noiseData, noiseLabel, test_size=0.2, stratify = noiseLabel, random_state=3)
#     return x_train, x_test, y_train, y_test

def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion) #should be int
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

def get_rotation_array(index,size):
    ##########
    # arr = [i for i in range(size)]
    # random.shuffle(arr)
    ########
    ###
    # arr=[]
    # rotate=size//3
    # if index == 1:
    #     arr = [*range(size-rotate, size, 1)]+[*range(0, size-rotate, 1)]
    # elif index == 2:
    #     arr = [*range(size-(rotate*2), size, 1)]+[*range(0, size-(rotate*2), 1)]
    # else:
    #     arr = [*range(size-(rotate*3), size, 1)]+[*range(0, size-(rotate*3), 1)]
    #     if(rotate*3==size):
    #         arr.reverse()
    #########
    arr = [*range(0,size)]
    jumps=size//3
    arr = (arr[-jumps:] + arr[:-jumps]) ## 4 steps
    arr = (arr[-jumps:] + arr[:-jumps]) ## 8 steps
    # arr = (arr[-jumps:] + arr[:-jumps]) ## 16 steps
    if index > 1:
        arr = (arr[-jumps:] + arr[:-jumps]) ## 4 steps
        arr = (arr[-jumps:] + arr[:-jumps]) ## 8 steps
        # arr = (arr[-jumps:] + arr[:-jumps]) ## 16 steps
    if index >2:
        arr = (arr[-jumps:] + arr[:-jumps]) ## 4 steps
        arr = (arr[-jumps:] + arr[:-jumps]) ## 8 steps
        # arr = (arr[-jumps:] + arr[:-jumps]) ## 16 steps

    return arr
# print("x----",x[0])

# xyz=np.array([])
# yz=np.array([])
# print("clolumns---",noiseValue)
# for i in range(2):
#     noise = x[0,[3,10,11,13,17]] + np.random.normal(0,i+2,5)
#     print("noise----",noise)
#     print("test----",x[0,[3,10,11,13,17]] + noise)
#     noiseValue = noiseValue + noise
#     print("afteraddingNois----",noiseValue)
#     print("x----",x[0])
#     xy = np.append(x[0],noise)
#     yz=np.append(yz,i+1)
#     if len(xyz) == 0:
#         xyz=np.append(xyz,xy)
#     else:
#         xyz= np.vstack([xyz,xy])

# print(xy)
