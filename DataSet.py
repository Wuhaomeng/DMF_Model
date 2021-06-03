# -*- Encoding:UTF-8 -*-

import numpy as np
import pandas as pd
import sys
from torch.utils.data import DataLoader,Dataset
import torch

class DataSet(Dataset):
    def __init__(self, fileName, negNum):

        self.data, self.shape = self.getData(fileName)   #shape=[用户，项目]
        self.train, self.test = self.getTrainTest()
        self.userEmbedding = self.getEmbedding()
        self.itemEmbedding = np.transpose(self.userEmbedding)
        self.trainDict = self.getTrainDict()
        self.train_u, self.train_i, self.train_r = self.getInstances(self.train, negNum)
        self.testNeg, self.evaluser, self.evalitem = self.getTestNeg(self.test, 99)

    def __len__(self):

        return len(self.train_u)

    def __getitem__(self, index):

        user_id = self.train_u[index]
        item_id = self.train_i[index]
        rate_id = self.train_r[index]
        # user_onehot = np.zeros([self.shape[0]])
        # item_onehot = np.zeros([self.shape[1]])
        # user_onehot[user_id] = 1
        # item_onehot[item_id] = 1
        return self.userEmbedding[user_id], self.itemEmbedding[item_id], rate_id, # user_onehot, item_onehot


    def getData(self, fileName):
        if fileName == 'ml-1m':
            print("Loading ml-1m data set...")
            data = []
            filePath = './Data/ml-1m/ratings.dat'
            u = 0
            i = 0
            maxr = 0.0
            with open(filePath, 'r') as f:
                for line in f:
                    if line:
                        lines = line[:-1].split("::")  #每行末尾符号是\n
                        user = int(lines[0])
                        movie = int(lines[1])
                        score = float(lines[2])
                        time = int(lines[3])
                        data.append((user, movie, score, time))
                        if user > u:
                            u = user
                        if movie > i:
                            i = movie
                        if score > maxr:
                            maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]
        elif fileName == 'ml-latest-small':
            print("Loading ml-latest-small data set...")
            data = []
            filePath = './Data/ml-latest-small/ratings.csv'
            data_small = pd.read_csv(filePath)
            u = 0
            i = 0
            maxr = 0.0
            for indexs in data_small.index:
                lines = data_small.loc[indexs].values[:]
                user = int(lines[0])
                movie = int(lines[1])
                score = float(lines[2])
                time = int(lines[3])
                data.append((user, movie, score, time))
                if user > u:
                    u = user
                if movie > i:
                    i = movie
                if score > maxr:
                    maxr = score
            self.maxRate = maxr
            print("Loading Success!\n"
                  "Data Info:\n"
                  "\tUser Num: {}\n"
                  "\tItem Num: {}\n"
                  "\tData Size: {}".format(u, i, len(data)))
            return data, [u, i]
        else:
            print("Current data set is not support!")
            sys.exit()


    def getTrainTest(self):
        data = self.data
        data = sorted(data, key=lambda x: (x[0], x[3]))    #sort在原列表上进行操作无返回值，sorted返回新的list
        train = []
        test = []
        for i in range(len(data)-1):
            user = data[i][0]-1
            item = data[i][1]-1
            rate = data[i][2]
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))    #将每个用户的最后一个数据用作测试集

        test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))    #整个数据集最后一个数据
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []      #列表本身允许有重复元素
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        total_user = []
        total_item = []
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]
            i = s[1]
            tmp_user.append(u)
            tmp_item.append(i)
            total_user.append(u)
            total_item.append(i)
            neglist = set()    #创建一个无序不重复元素集，可以删除重复的数据
            neglist.add(i)
            for t in range(negNum):
                j = np.random.randint(self.shape[1])     #产生离散均匀分布的整数
                while (u, j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(u)
                tmp_item.append(j)
                total_user.append(u)
                total_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)], np.array(total_user), np.array(total_item)


class TestSet(Dataset):
    def __init__(self, user, item, UserEmbedding, ItemEmbedding): #shape
        self.evaluser = user
        self.evalitem = item
        self.userEmbedding = UserEmbedding
        self.itemEmbedding = ItemEmbedding
        # self.shape = shape

    def __len__(self):

        return len(self.evaluser)

    def __getitem__(self, index):

        user_id = self.evaluser[index]
        item_id = self.evalitem[index]
        # user_onehot = np.zeros([self.shape[0]])
        # item_onehot = np.zeros([self.shape[1]])
        # user_onehot[user_id] = 1
        # item_onehot[item_id] = 1

        return self.userEmbedding[user_id], self.itemEmbedding[item_id], #user_onehot, item_onehot