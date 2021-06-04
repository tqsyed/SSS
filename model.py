
import torch.nn as nn


class FinalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        ## the follwing setup was for eyemovement
        #self.fc1 = nn.Linear(8, 4)
        #self.relu1 = nn.ReLU()
        #self.batchnorm = nn.BatchNorm1d(4)
        #self.out = nn.Linear(4, 8)
        ##---------------##
        self.fc1 = nn.Linear(8, 4)
        self.relu1 = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(4)
        self.out = nn.Linear(4, 2)
        # self.fc1 = nn.Linear(10, 6)
        # self.relu1 = nn.ReLU()
        # self.out = nn.Linear(6, 2)

    def forward(self, input_):
        x = self.fc1(input_)
        x = self.relu1(x)
        x = self.batchnorm(x)
        x = self.out(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ##credit card-----------------------------------------
        self.fc1 = nn.Linear(29, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 12)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(12, 8)
        self.relu3 = nn.ReLU()
        self.out = nn.Linear(8, 4)
        ##8classes
        # self.fc1 = nn.Linear(29, 22)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(22, 15)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(15, 10)
        # self.relu3 = nn.ReLU()
        # self.out = nn.Linear(10, 8)

        ##Mortgage-----------------------------------------
        # self.fc1 = nn.Linear(1473, 750)
        # self.relu1 = nn.ReLU()
        # # self.dout = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(750, 400)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(400, 200)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(200, 100)
        # self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(100, 50)
        # self.relu5 = nn.ReLU()
        # self.fc6 = nn.Linear(50, 22)
        # self.relu6 = nn.ReLU()
        # self.fc7 = nn.Linear(22, 8)
        # self.relu7 = nn.ReLU()
        # self.out = nn.Linear(8, 4)

        ##Redwine-----------------------------------------
        # self.fc1 = nn.Linear(11, 8)
        # self.relu1 = nn.ReLU()
        # self.out = nn.Linear(8, 4)
        ##8classes
        # self.fc1 = nn.Linear(11, 10)
        # self.relu1 = nn.ReLU()
        # self.out = nn.Linear(10, 8)

        #pen_digits-----------------------------------------
        #self.fc1 = nn.Linear(16, 12)
        #self.relu1 = nn.ReLU()
        #self.fc2 = nn.Linear(12, 8)
        #self.relu2 = nn.ReLU()
        #self.out = nn.Linear(8, 4)
        ##8classes
        #self.fc1 = nn.Linear(16, 13)
        #self.relu1 = nn.ReLU()
        #self.fc2 = nn.Linear(13, 10)
        #self.relu2 = nn.ReLU()
        #self.out = nn.Linear(10, 8)

        ##optical_digits-----------------------------------------
        # self.fc1 = nn.Linear(62, 32)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(32, 16)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(16, 8)
        # self.relu3 = nn.ReLU()
        # self.out = nn.Linear(8, 4)
        ##8classes
        # self.fc1 = nn.Linear(62, 42)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(42, 20)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(20, 10)
        # self.relu3 = nn.ReLU()
        # self.out = nn.Linear(10, 8)

        ##web-----------------------------------------
        # self.fc1 = nn.Linear(300, 150)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(150, 75)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(75, 33)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(33, 17)
        # self.relu4 = nn.ReLU()
        # self.fc5 = nn.Linear(17, 8)
        # self.relu5 = nn.ReLU()
        # self.out = nn.Linear(8, 4)

        ##paysim-----------------------------------------
        # self.fc1 = nn.Linear(10, 8)
        # self.relu1 = nn.ReLU()
        # self.out = nn.Linear(8, 4)

        ##mushroomOneHot-----------------------------------------
        # self.fc1 = nn.Linear(116, 60)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(60, 30)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(30, 15)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(15, 8)
        # self.relu4 = nn.ReLU()
        # self.out = nn.Linear(8, 4)
        ##mushroom-----------------------------------------
        # self.fc1 = nn.Linear(22, 16)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(16, 8)
        # self.relu2 = nn.ReLU()
        # self.out = nn.Linear(8, 4)

        ##AdultOneHot-----------------------------------------
        # self.fc1 = nn.Linear(97, 60)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(60, 30)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(30, 15)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(15, 8)
        # self.relu4 = nn.ReLU()
        # self.out = nn.Linear(8, 4)
        ##adult-----------------------------------------
        # self.fc1 = nn.Linear(14, 8)
        # self.relu1 = nn.ReLU()
        # self.out = nn.Linear(8, 4)

        ##adult/Eye Movement-----------------------------------------
        #self.fc1 = nn.Linear(14, 8)
        #self.relu1 = nn.ReLU()
        #self.out = nn.Linear(8, 4)

        ##GiveMeCredit-----------------------------------------
        #self.fc1 = nn.Linear(10, 8)
        #self.relu1 = nn.ReLU()
        #self.out = nn.Linear(8, 4)
        #self.sig = nn.Sigmoid()
        #self.out_act = nn.Softmax(dim=1)

    def forward(self, input_,self_supervised=True):
        x = self.fc1(input_)
        #x = self.relu1(x)##redwine / paysim / adult/GiveMecredit/eye movemnt--------
        #x = self.fc2(x)
        #x = self.relu2(x)##pen_digits / mushromm------
        #x = self.fc3(x)
        x = self.relu3(x)##credit_card/optical_digits-----
        x = self.fc4(x)
        # x = self.relu4(x)##adultOneHot/mushroomOneHot-----
        # x = self.fc5(x)
        # x = self.relu5(x)##web------
        # x = self.fc6(x)
        # x = self.relu6(x)
        # x = self.fc7(x)
        # x = self.relu7(x)##mortgage-------
        if self_supervised:
            x = self.out(x)
        return x


class Blocks(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.b1l1 = nn.Linear(in_channel, 20)
        self.b1r1 = nn.ReLU()
        self.b1l2 = nn.Linear(20, 15)
        self.b1r2 = nn.ReLU()
        self.b1l3 = nn.Linear(15, 10)
        self.b1r3 = nn.ReLU()
        self.b1norm = nn.BatchNorm1d(10)
        ##2nd block
        self.b2l1 = nn.Linear(10, 20)
        self.b2r1 = nn.ReLU()
        self.b2l2 = nn.Linear(20, 20)
        self.b2r2 = nn.ReLU()
        self.b2l3 = nn.Linear(20, 20)
        self.b2r3 = nn.ReLU()
        self.b2norm = nn.BatchNorm1d(20)
        ##3rd block
        self.b3l1 = nn.Linear(20, 20)
        self.b3r1 = nn.ReLU()
        self.b3l2 = nn.Linear(20, 20)
        self.b3r2 = nn.ReLU()
        self.b3l3 = nn.Linear(20, 20)
        self.b3r3 = nn.ReLU()
        self.b3norm = nn.BatchNorm1d(20)
        # ##4th block
        # self.b4l1 = nn.Linear(20, 20)
        # self.b4r1 = nn.ReLU()
        # self.b4l2 = nn.Linear(20, 20)
        # self.b4r2 = nn.ReLU()
        # self.b4l3 = nn.Linear(20, 20)
        # self.b4r3 = nn.ReLU()
        # self.b4norm = nn.BatchNorm1d(20)
        ##out
        self.out = nn.Linear(20, 4)

    def forward(self, input_,self_supervised=True,no_of_block=3):
        x = self.b1l1(input_)
        x = self.b1r1(x)
        x = self.b1l2(x)
        x = self.b1r2(x)
        x = self.b1l3(x)
        x = self.b1r3(x)
        x = self.b1norm(x)
        ##2nd
        if(no_of_block>=2):
            x = self.b2l1(x)
            x = self.b2r1(x)
            x = self.b2l2(x)
            x = self.b2r2(x)
            x = self.b2l3(x)
            x = self.b2r3(x)
            x = self.b2norm(x)
        ##3rd
        if(no_of_block>=3):
            x = self.b3l1(x)
            x = self.b3r1(x)
            x = self.b3l2(x)
            x = self.b3r2(x)
            x = self.b3l3(x)
            x = self.b3r3(x)
            x = self.b3norm(x)


        if self_supervised:
            x = self.out(x)
        return x

