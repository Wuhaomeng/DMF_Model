import numpy as np
import argparse
from DataSet import DataSet, TestSet

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import sys
import os
import heapq
import math
from joblib import cpu_count


def main():
    print('**********')
    parser = argparse.ArgumentParser(description="Options")   #用来读取命令行参数

    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-1m')
    parser.add_argument('-negNum', action='store', dest='negNum', default=1, type=int)   #负样本的数目
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[512, 64])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[1024, 64])
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.0001)
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=30, type=int)
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')  #日志
    parser.add_argument('-topK', action='store', dest='topK', default=10)

    args = parser.parse_args()  #获取解析的参数

    classifier = Model(args)
    classifier.training()

    print('**********')


class UserNet(nn.Module):
    def __init__(self, userLayer, shape):
        super(UserNet, self).__init__()
        self.userLayer = userLayer
        self.shape = shape
        ##定义第一个隐藏层
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features = self.shape[1], out_features= self.userLayer[0]),
            nn.ReLU(),
        )
        hidden2 = []
        for i in range(0, len(self.userLayer)-2):
            hidden2 += [nn.Linear(in_features = self.userLayer[i], out_features= self.userLayer[i+1], bias=True),
                             nn.ReLU()]
        hidden2 += [nn.Linear(in_features = self.userLayer[-2], out_features= self.userLayer[-1], bias=True)]
        self.hidden2 = nn.Sequential(*hidden2)

        for layer in self.hidden1:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight.data, mean=0, std=0.01)

        for layer in self.hidden2:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight.data, mean=0, std=0.01)

        print(self.hidden1[0].weight.data)
        print(self.hidden1[0].weight.data.shape)
        print(self.hidden2[0].weight.data)
        print(self.hidden2[0].weight.data.shape)


    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        return x


class ItemNet(nn.Module):
    def __init__(self, itemLayer, shape):
        super(ItemNet, self).__init__()
        self.itemLayer = itemLayer
        self.shape = shape
        ##定义第一个隐藏层
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features = self.shape[0], out_features= self.itemLayer[0]),
            nn.ReLU(),
        )
        hidden2 = []
        for i in range(0, len(self.itemLayer)-2):
            hidden2 += [nn.Linear(in_features = self.itemLayer[i], out_features= self.itemLayer[i+1], bias=True),
                             nn.ReLU()]
        hidden2 += [nn.Linear(in_features = self.itemLayer[-2], out_features= self.itemLayer[-1], bias=True)]
        self.hidden2 = nn.Sequential(*hidden2)


        for layer in self.hidden1:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight.data, mean=0, std=0.01)

        for layer in self.hidden2:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight.data, mean=0, std=0.01)

        print(self.hidden1[0].weight.data)
        print(self.hidden1[0].weight.data.shape)
        print(self.hidden2[0].weight.data)
        print(self.hidden2[0].weight.data.shape)


    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        return x


class Model:
    def __init__(self, args):
        self.dataName = args.dataName
        self.negNum = args.negNum
        self.dataSet = DataSet(self.dataName, self.negNum)
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate

        self.train = self.dataSet.train
        self.test = self.dataSet.test
        self.userEmbedding = self.dataSet.userEmbedding
        self.itemEmbedding = self.dataSet.itemEmbedding

        self.testNeg = self.dataSet.testNeg
        self.evaluser = self.dataSet.evaluser
        self.evalitem = self.dataSet.evalitem


        self.testSet = TestSet(self.evaluser, self.evalitem, self.userEmbedding, self.itemEmbedding)

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer


        self.lr = args.lr

        self.checkPoint = args.checkPoint

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop

    def training(self):
        usernet = UserNet(self.userLayer, self.shape)
        itemnet = ItemNet(self.itemLayer, self.shape)
        opt_user = optim.Adam(usernet.parameters(), lr=self.lr)
        opt_item = optim.Adam(itemnet.parameters(), lr=self.lr)

        train_loader = DataLoader(
            dataset=self.dataSet, batch_size=self.batchSize, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            dataset=self.testSet, batch_size=100, shuffle=False, num_workers=0
        )

        best_hr = -1
        best_NDCG = -1
        best_epoch = -1

        train_loss_all = []

        for epoch in range(self.maxEpochs):
            # 进度条
            tq = tqdm(train_loader)
            tq.set_description('Epoch{}'.format(epoch))
            #
            loss_list = []
            for step, (b_user, b_item, b_rate) in enumerate(tq):
                output_user = usernet(b_user)
                output_item = itemnet(b_item)

                similarity = torch.cosine_similarity(output_user, output_item, dim=1)
                result = torch.clamp(similarity, min=1e-6)

                train_loss = self.loss(result, b_rate)/self.batchSize
                loss_list.append(train_loss.item())
                tq.set_postfix({'loss':'{:.4f}'.format(train_loss.item())})
                # print(train_loss.item())
                opt_user.zero_grad()
                opt_item.zero_grad()
                train_loss.backward()
                opt_user.step()
                opt_item.step()
                train_loss_all.append(train_loss.item())

            tq.close()
            self.save_model(usernet, './Model/last_usernet.pth')
            self.save_model(itemnet, './Model/last_itemnet.pth')

            #验证*****************************

            hr = []
            NDCG = []
            testUser = self.testNeg[0]
            testItem = self.testNeg[1]

            # 进度条
            tq = tqdm(test_loader)
            tq.set_description('Validation{}:'.format(epoch))

            for step, (b_user, b_item) in enumerate(tq):
                output_user = usernet(b_user)
                output_item = itemnet(b_item)

                similarity = torch.cosine_similarity(output_user, output_item, dim=1)
                result = torch.clamp(similarity, min = 1e-6)
                predict = result.detach().numpy()

                target = testItem[step][0]

                item_score_dict = {}

                for j in range(len(testItem[step])):
                    item = testItem[step][j]
                    item_score_dict[item] = predict[j]

                ranklist = heapq.nlargest(self.topK, item_score_dict, key=item_score_dict.get)

                tmp_hr = self.getHitRatio(ranklist, target)
                tmp_NDCG = self.getNDCG(ranklist, target)
                hr.append(tmp_hr)
                NDCG.append(tmp_NDCG)
                tq.set_postfix({'tmp_hr':'{:.4f}'.format(tmp_hr), 'tmp_NDCG':'{:.4f}'.format(tmp_NDCG)})
            tq.close()

            epoch_hr = np.mean(hr)
            epoch_NDCG = np.mean(NDCG)

            print("Epoch ", epoch, "HR: {}, NDCG: {}".format(epoch_hr, epoch_NDCG))

            if epoch_hr > best_hr or epoch_NDCG > best_NDCG:
                best_hr = epoch_hr
                best_NDCG = epoch_NDCG
                best_epoch = epoch
                self.save_model(usernet, './Model/UserNet')
                self.save_model(itemnet, './Model/ItemNet')

            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")                #几次没有更新最好状态后不再进行训练
                break

        plt.figure()
        plt.plot(train_loss_all, 'r-')
        plt.title('Train loss per interation')
        plt.show()


    def save_model(self, the_model, PATH):
        torch.save(the_model.state_dict(), PATH)

    def loss(self, predict, rate):
        regRate = rate / self.maxRate
        losses = regRate*torch.log(predict) + (1-regRate)*torch.log(1-predict)
        loss = -torch.sum(losses, dim=0)
        return loss


    def getHitRatio(self, ranklist, targetItem):
        for item in ranklist:
            if item == targetItem:
                return 1
        return 0

    def getNDCG(self, ranklist, targetItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == targetItem:
                return math.log(2) / math.log(i+2)
        return 0



if __name__ == '__main__':
    main()