
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch
import utils,dataloader
from model import Net,FinalClassifier,Blocks
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging


def run_epoch(epoch,model, opt, criterion,data,is_self,is_fully_supervised,block,do_train=True):
    if is_self:
        model['SelfSupervised'].train() if do_train else model['SelfSupervised'].eval()
    else:
        model['Classifier'].train() if do_train else model['Classifier'].eval()
        if is_fully_supervised:
           model['FeatureExtractor'].train() if do_train else model['FeatureExtractor'].eval()
        else:
            model['FeatureExtractor'].eval()

    losses = []
    loop = tqdm(data)
    loop.set_description('Epoch {}'.format(e))
    for i, (inputs, labels) in enumerate(loop):
        if do_train:
            if is_self:
                opt['SelfSupervised'].zero_grad()
            else:
                opt['Classifier'].zero_grad()
                if is_fully_supervised:
                    opt['FeatureExtractor'].zero_grad()

        if is_self:
            y_hat = model['SelfSupervised'](inputs)
        else:
            y_hat = model['FeatureExtractor'](inputs,is_self)
            # y_hat = model['FeatureExtractor'](inputs,is_self,block)
            y_hat = model['Classifier'](y_hat)


        loss = criterion(y_hat, labels)

        if do_train:
            loss.backward()
            if is_self:
                opt['SelfSupervised'].step()
            else:
                opt['Classifier'].step()
                if is_fully_supervised:
                    opt['FeatureExtractor'].step()

        losses.append(loss.cpu().data.numpy())

    if epoch%10 == 0:
        logger.debug("Epoch(%d) %s epoch average loss : %f" % (epoch,do_train, np.mean(losses)))
    return np.mean(losses)

if __name__ == '__main__':
    # Create and configure logger
    logging.basicConfig(filename="Logging.log",
                        format='%(asctime)s %(message)s',
                        filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--self', default=False)
    parser.add_argument('-type', '--algo-type', type=str)
    parser.add_argument('-save','--save-network',type=str)
    parser.add_argument('-load', '--load-supervised', type=str)
    parser.add_argument('-path', '--data-path', type=str)
    parser.add_argument('-data', '--data-type', type=str)
    parser.add_argument('-e','--epoch', type=int, default=5)
    parser.add_argument('-fully','--full-supervised', type=int)
    parser.add_argument('-semi','--s-supervised', type=int)
    parser.add_argument('-p','--percent', type=int)
    parser.add_argument('-b','--block', type=int)
    # parser.add_argument('-s', '--samples', type=int, default=1, help="How many samples of connectivity/masks to average logits over during inference")
    # parser.add_argument('-r', '--resample-every', type=int, default=20, help="For efficiency we can choose to resample orders/masks only once every this many steps")
    args = parser.parse_args()
    self= True if args.self == 'True' else False
    f_upervised = True if args.full_supervised == 1 else False
    semi_supervised = True if args.s_supervised == 1 else False
    algo=args.algo_type
    load_path=args.load_supervised
    save_path=args.save_network+str(args.epoch)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug("----------------------------------------------------")
    logger.debug("----------------------------------------------------")
    logger.debug("-------------Experiment Started with Following parameters----------------" )
    logger.debug("SelfSupervised==>%s" % (self))
    logger.debug("Algo==>%s" % (algo))
    logger.debug("Data==>%s" % (args.data_type))
    logger.debug("Load Path==>%s" % (load_path))
    logger.debug("Saved path==>%s" % (args.save_network))
    logger.debug("epoch==>%d" % (args.epoch))
    logger.debug("FullySupervised==>%s" % (f_upervised))
    logger.debug("Semi-Supervised==>%s" % (semi_supervised))
    logger.debug("Percent==>%s" % (args.percent))
    logger.debug("Blocks==>%s" % (args.block))
    logger.debug("device==>%s" % (device))
    logger.debug("----------------------------------------------------")
    if self:
        logger.debug("Yes self")
    else:
        logger.debug("No self")

    if semi_supervised:
        logger.debug("Yes semi")
    else:
        logger.debug("No Semi")

    if f_upervised:
            logger.debug("Yes Fully")
    else:
        logger.debug("No Fully")
    if self:
        if algo=='noise':
            if args.data_type=='mortgage':
                x_train, x_test, y_train, y_test = dataloader.get_mortgage_noisy_data(args.data_path)
            else:
                x_train, x_test, y_train, y_test = dataloader.get_noisy_data(args.data_path,args.data_type)
        elif algo=='rot':
            if args.data_type=='mortgage':
                x_train, x_test, y_train, y_test = dataloader.get_mortgage_rotation_data(args.data_path)
            else:
                x_train, x_test, y_train, y_test = dataloader.get_rotation_data(args.data_path,args.data_type)
        elif algo=='rotFBnoise':
            x_train, x_test, y_train, y_test = dataloader.rotation_followed_by_noise(args.data_path,args.data_type)
        elif algo=='noiseFBrot':
            x_train, x_test, y_train, y_test = dataloader.noise_followed_by_rotaion(args.data_path,args.data_type)
    else:
        if args.data_type=='mortgage':
            x_train, x_test, y_train, y_test = dataloader.get_mortgage_full(args.data_path)
        elif semi_supervised:
            # x_train, x_test, y_train, y_test = dataloader.get_semi_dataset(args.data_path,args.data_type,args.percent,logger)
            data=pd.read_csv(args.data_path)
            # data = data.drop('Time', axis = 1)
            #     data=(data-data.min())/(data.max()-data.min())
            train, test = train_test_split(data, test_size=0.2, random_state=3, stratify=data[['Class']])
            y_test = test['Class'].values
            test=(test-test.min())/(test.max()-test.min())
            x_test=test.drop('Class',axis=1).values
            train=(train-train.min())/(train.max()-train.min())
            woLabels,wLabels = train_test_split(train, test_size=(args.percent/100), random_state=3, stratify=train[['Class']])
            y_train = wLabels['Class'].values
            x_train = wLabels.drop('Class',axis=1).values
            logger.debug("wLabels:%s"% (wLabels['Class'].value_counts()))
            logger.debug("x_train:%s",x_train.shape)
            logger.debug("y_train:%s",y_train.shape)
            logger.debug("x_test:%s",x_test.shape)
            logger.debug("y_test:%s",y_test.shape)
        else:
            x_train, x_test, y_train, y_test = dataloader.get_full_dataset(args.data_path,args.data_type,logger)


    train_loader, test_loader = utils.get_data_loaders(x_train, y_train, x_test, y_test,device)
    model={}
    optimizer={}
    if self:
        model['SelfSupervised'] = Net()
        # model['SelfSupervised'] = Blocks(x_train.shape[1])
        model['SelfSupervised'].to(device)
        optimizer['SelfSupervised'] = optim.Adam(model['SelfSupervised'].parameters(), lr=0.001)
    else:
        state = torch.load(load_path)
        model['Classifier'] = FinalClassifier()
        model['Classifier'].to(device)
        model['FeatureExtractor']=Net()
        # model['FeatureExtractor'] = Blocks(x_train.shape[1])
        model['FeatureExtractor'].load_state_dict(state['state_dict'])
        model['FeatureExtractor'].to(device)
        optimizer['Classifier'] = optim.Adam(model['Classifier'].parameters(), lr=0.001)
        optimizer['FeatureExtractor'] = optim.Adam(model['FeatureExtractor'].parameters())
        optimizer['FeatureExtractor'].load_state_dict(state['optimizer'])


    criterion = nn.CrossEntropyLoss()


    train_losses,test_losses = [],[]
    for e in tqdm(range(args.epoch)):
        test_losses.append(run_epoch(e,model, optimizer, criterion,train_loader,self,f_upervised,args.block,False))
        train_losses.append(run_epoch(e,model, optimizer, criterion,test_loader,self,f_upervised,args.block))

#    plt.plot(train_losses,'g',test_losses,'r')
#    plt.show()
#    plt.savefig(algo+str(args.epoch)+'.png')
    correct = 0
    total = 0
    Tlabels=[]
    prediction=[]
    with torch.no_grad():
        # if semi_supervised:
        #     x_train, x_test, y_train, y_test = dataloader.get_full_dataset(args.data_path,args.data_type,logger)
        #     train_loader, test_loader = utils.get_data_loaders(x_train, y_train, x_test, y_test,device)
        for data in test_loader:
            images, labels = data
            if self:
                outputs = model['SelfSupervised'](images)
            else:
                outputs = model['FeatureExtractor'](images,self)
                # outputs = model['FeatureExtractor'](images,self,args.block)
                outputs = model['Classifier'](outputs)

            predicted = torch.argmax(outputs, dim=1)
            Tlabels = np.append(Tlabels,labels.cpu().numpy());
            prediction = np.append(prediction,predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.debug('Accuracy of the network: %d %%' % (
        100 * correct / total))
        

    if self:
        state = {
        'epoch': args.epoch,
        'state_dict': model['SelfSupervised'].state_dict(),
        'optimizer': optimizer['SelfSupervised'].state_dict()
        }
        torch.save(state, save_path)
    else:
        tn, fp, fn, tp = metrics.confusion_matrix(Tlabels, prediction).ravel()
        logger.debug("Data==>%s" % (args.data_type))
        logger.debug("Percent==>%s" % (args.percent))
        logger.debug("TN:%f, FP:%f, FN:%f, TP:%f"%(tn,fp,fn,tp))
        logger.debug("Precision:%f"%(metrics.precision_score(Tlabels, prediction)))
        logger.debug("Recall:%f"%(metrics.recall_score(Tlabels, prediction)))
        logger.debug("F1:%f"%(metrics.f1_score(Tlabels, prediction)))
        logger.debug("AUC:%f"%(metrics.roc_auc_score(Tlabels, prediction)))
        logger.debug("Balanced Accuracy:%f"%(metrics.balanced_accuracy_score(Tlabels, prediction)))
        logger.debug("G-Mean:%f"%(geometric_mean_score(Tlabels, prediction)))
        logger.debug("G-Mean-correction:%f"%(geometric_mean_score(Tlabels, prediction, correction=0.001)))
        logger.debug("G-Mean-macro:%f"%(geometric_mean_score(Tlabels, prediction, average='macro')))
        logger.debug("G-Mean-micro:%f"%(geometric_mean_score(Tlabels, prediction, average='micro')))
        logger.debug("G-Mean-weighted:%f"%(geometric_mean_score(Tlabels, prediction, average='weighted')))
        state = {
            'epoch': args.epoch,
            'state_dict': model['Classifier'].state_dict(),
            'optimizer': optimizer['Classifier'].state_dict()
        }
        print(state)
        tn, fp, fn, tp = metrics.confusion_matrix(Tlabels, prediction).ravel()
        print("Data==>%s" % (args.data_type))
        print("Percent==>%s" % (args.percent))
        print("TN:%f, FP:%f, FN:%f, TP:%f"%(tn,fp,fn,tp))
        print("Precision:%f"%(metrics.precision_score(Tlabels, prediction)))
        print("Recall:%f"%(metrics.recall_score(Tlabels, prediction)))
        print("F1:%f"%(metrics.f1_score(Tlabels, prediction)))
        print("AUC:%f"%(metrics.roc_auc_score(Tlabels, prediction)))
        print("Balanced Accuracy:%f"%(metrics.balanced_accuracy_score(Tlabels, prediction)))
        print("G-Mean:%f"%(geometric_mean_score(Tlabels, prediction)))
        print("G-Mean-correction:%f"%(geometric_mean_score(Tlabels, prediction, correction=0.001)))
        print("G-Mean-macro:%f"%(geometric_mean_score(Tlabels, prediction, average='macro')))
        print("G-Mean-micro:%f"%(geometric_mean_score(Tlabels, prediction, average='micro')))
        print("G-Mean-weighted:%f"%(geometric_mean_score(Tlabels, prediction, average='weighted')))
        #print("Precision:%f"%(metrics.precision_score(Tlabels, prediction)))
        #print("Recall:%f"%(metrics.recall_score(Tlabels, prediction)))
        torch.save(state, save_path)
