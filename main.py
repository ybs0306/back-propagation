#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__  = 'kinoshitakenta'
__email__   = 'ybs0306748@gmail.com'

import sys
import os
from os.path import isdir
import dill     #儲存模型與label map資料用
import numpy as np
import random
import cv2
import csv
from scipy.special import expit     #sigmoid

class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        #給定參數值
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lrate = learning_rate
        
        #給予隨機權重
        self.w_i2h = np.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_h2o = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

        #指定活化函數, sigmoid
        self.activation_function = lambda x: expit(x)
        pass

    
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        #計算input到隱藏層的output
        hidden_inputs = np.dot(self.w_i2h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        #計算隱藏層到輸出的output
        final_inputs = np.dot(self.w_h2o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        #計算誤差值
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.w_h2o.T, output_errors) 
        
        #更新權重
        self.w_h2o += self.lrate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.w_i2h += self.lrate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass


    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.w_i2h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.w_h2o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


#依照產生的label map, 分別對應標籤
def label_map(ls, q):
    for x, y in ls:
        if type(q) is str:
            if x is q:
                return y

        elif type(q) is int:
            if y is q:
                return x
    pass


#回答問題用
def query_ans(q):
    ans = ''
    while ans is not 'y' and ans is not 'n':
        ans = input(q + " (y/n) : ").strip(' ')
        if ans is not 'y' and ans is not 'n':
            print('輸入格式錯誤 請重新輸入')

    if ans == 'y':
        return True
    else:
        return False


def main():
    path = os.getcwd()

    ################# training #################
    #重新訓練模型
    if query_ans('是否重新訓練模型 ?'):

        dataset = []
        index = 0
        print("\n讀取檔案中 ...")

        if os.path.isfile("train_data.csv"):
            qr = query_ans('有偵測到train_data.csv檔案, 要直接讀取嗎 ?')
        else:
            qr = False
        
        
        if qr:
            #從csv讀取training data
            training_data_file = open("train_data.csv", 'r')
            training_data_list = training_data_file.readlines()
            training_data_file.close()

            #建立label map
            a = set()
            for x in training_data_list:
                all_values = x.split(',')
                a.add(all_values[0])
            for x in a:
                dataset.append((x, index))
                index += 1
            pass

        else:
            #從檔案讀取training data
            training_data_list = []
            for dir_label in os.listdir(path + '/training'):
                if isdir(path + '/training/' + dir_label):
                    dataset.append((dir_label, index))
                    index += 1
                    #print(dir_label)
                    for files in os.listdir(path + '/training/' + dir_label):
                        fullpath = path + '/training/' + dir_label + '/' + files

                        image = cv2.imread(fullpath, cv2.IMREAD_GRAYSCALE)
                        img = np.reshape(image,(1, 784)).astype('float32')
                        a = list(list(img)[0])
                        a.insert(0, dir_label)
                        training_data_list.append(a)
            
            random.shuffle(training_data_list)
            #print(dataset)

            #詢問是否寫入csv
            if query_ans('training照片資料是否覆寫寫入train_data.csv ?'):
                with open ('train_data.csv', 'w', newline='') as f:
                    writer = csv.writer(f, delimiter=',')
                    for x in training_data_list:
                        writer.writerow(x)


        #初始化NN數值
        input_nodes = 784
        hidden_nodes = 150
        output_nodes = index
        learning_rate = 0.05

        n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

        epoch = int(input("\n請輸入要多少epoch : "))
        #epochs = 10

        print('\n#########################')
        print(f'input_nodes = {input_nodes}')
        print(f'hidden_nodes = {hidden_nodes}')
        print(f'output_nodes = {output_nodes}')
        print(f'learning_rate = {learning_rate}')
        print(f'epoch = {epoch}')
        print('#########################\n')

        print('----------- 開始training model -----------')
        print(f'training data 共有{len(training_data_list)}筆資料')
        if len(training_data_list) <= 0:
            print('無資料可以訓練')
            sys.exit(0)

        #分割訓練與測試正確率資料, training : test = 19 : 1
        training_range = int(len(training_data_list)*0.95)

        #開始迭代訓練
        for e in range(epoch):
            print('epoch : ' + str(e+1) + '/' + str(epoch))
            #訓練資料
            for record in training_data_list[:training_range]:
                if qr:
                    all_values = record.split(',')
                else:
                    all_values = record

                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                targets = np.zeros(output_nodes) + 0.01

                targets[label_map(dataset, all_values[0])] = 0.99
                n.train(inputs, targets)
                pass
            
            #測試正確率
            scorecard = []
            for record in training_data_list[training_range:]:
                if qr: all_values = record.split(',')
                else: all_values = record
                
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                outputs = n.query(inputs)
                label = label_map(dataset, int(np.argmax(outputs)))

                correct_label = all_values[0]
                if (label == correct_label):
                    scorecard.append(1)
                else:
                    scorecard.append(0)
                pass

            scorecard_array = np.asarray(scorecard)
            print ("正確率 = ", scorecard_array.sum() / scorecard_array.size)


        #儲存NN以及label map至檔案
        f1 = open('model.dat', 'wb')
        dill.dump(n, f1)
        f1.close()
        f2 = open('dataset.dat', 'wb')
        dill.dump(dataset, f2)
        f2.close()

    #不重新訓練模型
    else:
        #檢查檔案存不存在
        try:
            if os.path.exists('model.dat') and os.path.exists('dataset.dat'):
                f1 = open('model.dat', 'rb')
                n = dill.load(f1, "rb")
                f1.close()
                f2 = open('dataset.dat', 'rb')
                dataset = dill.load(f2, "rb")
                f2.close()
            else:
                print('記錄檔不存在, 請重新訓練模型')
                sys.exit(0)

        except Exception as e:
            print('error import file')
            print(e)
            sys.exit(0)


    ################## testing ##################
    print('\n------------ 開始testing data ------------')
    #讀檔存入dict
    test_data_dict = {}
    for dir_label in os.listdir(path + '/test'):
        image = cv2.imread(path + '/test/' + dir_label, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(image,(1, 784)).astype('float32')

        #去掉副檔名.png
        test_data_dict[dir_label[:-4]] = img

    print(f'test data 共有{len(test_data_dict)}筆資料')
    #按照照片名排序
    test_data_dict_list = sorted(test_data_dict.keys())


    #output寫入檔案
    f = open('Answer.txt', 'w')
    for record in test_data_dict_list:
        all_values = test_data_dict[record]
        inputs = (np.asfarray(all_values) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        #print(outputs)

        label = int(np.argmax(outputs))
        label_t = label_map(dataset, label)

        #print(record + ' ' + str(label_t))
        f.write(record + ' ' + str(label_t) + '\n')

    f.close()
    print('測試結果已輸出至Answer.txt\n')


if __name__ == "__main__":

    main()
