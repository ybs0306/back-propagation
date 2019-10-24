#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__  = 'kinoshitakenta'
__email__   = 'ybs0306748@gmail.com'

import sys
import os
from os.path import isdir
import dill
import numpy as np
import random
import cv2
import csv
from scipy.special import expit
#import matplotlib.pyplot as plt

class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.lr = learning_rate
        
        self.w_i2h = np.random.normal(0.0, pow(self.i_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.w_h2o = np.random.normal(0.0, pow(self.h_nodes, -0.5), (self.o_nodes, self.h_nodes))

        self.activation_function = lambda x: expit(x)
        pass

    
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.w_i2h, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.w_h2o, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.w_h2o.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.w_h2o += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.w_i2h += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass


    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.w_i2h, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.w_h2o, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


def label_map(ls, q):
    for x, y in ls:
        if type(q) is str:
            if x is q:
                return y

        elif type(q) is int:
            if y is q:
                return x
    pass


def main(obj):
    
    path = os.getcwd()

    if obj is 'y':

        #training_data_file = open("mnist_train_100.csv", 'r')
        #training_data_file = open("mnist_train.csv", 'r')
        '''
        training_data_file = open("train.csv", 'r')
        training_data_list = training_data_file.readlines()
        training_data_file.close()
        '''

        training_data_list = []
        dataset = []
        index = 0
        print("讀取檔案中 ...")
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

        with open ('train_data.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for x in training_data_list:
                writer.writerow(x)

        # number of input, hidden and output nodes
        input_nodes = 784
        hidden_nodes = 200
        output_nodes = index

        # learning rate
        learning_rate = 0.1
    
        # create instance of neural network
        n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

        # train the neural network

        # epochs is the number of times the training data set is used for training
        epoch = int(input("輸入要多少epoch : "))
        #epochs = 10

        print('\n#########################')
        print(f'input_nodes = {input_nodes}')
        print(f'hidden_nodes = {hidden_nodes}')
        print(f'output_nodes = {output_nodes}')
        print(f'learning_rate = {learning_rate}')
        print(f'epoch = {epoch}')
        print('#########################\n')

        print('開始training model')

        for e in range(epoch):
            print('epoch : ' + str(e+1) + '/' + str(epoch))
            # go through all records in the training data set
            for record in training_data_list:
                # split the record by the ',' commas
                all_values = record#record.split(',')
                # scale and shift the inputs
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
                # create the target output values (all 0.01, except the desired label which is 0.99)
                targets = np.zeros(output_nodes) + 0.01
                # all_values[0] is the target label for this record
                targets[label_map(dataset, all_values[0])] = 0.99
                #targets[int(all_values[0])] = 0.99
                n.train(inputs, targets)
                pass
            pass
        print('')

        f1 = open('model.dat', 'wb')
        dill.dump(n, f1)

    elif obj is 'n':
        try:
            f2 = open('model.dat', 'rb')
            n = dill.load(f2, "rb")
        except:
            print('error import file')
            sys.exit(0)


    # load the test data CSV file into a list
    '''
    test_data_file = open("mnist_test_10.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    '''
    
    print('開始testing data')
    test_data_dict = {}
    for dir_label in os.listdir(path + '/test'):
        #print(dir_label)

        image = cv2.imread(path + '/test/' + dir_label, cv2.IMREAD_GRAYSCALE)
        img = np.reshape(image,(1, 784)).astype('float32')

        test_data_dict[dir_label[:-4]] = img
    

    # scorecard for how well the network performs, initially empty
    #scorecard = []
    test_data_dict_list = sorted(test_data_dict.keys())

    f = open('Answer.txt', 'w')
    # go through all the records in the test data set
    for record in test_data_dict_list:

        all_values = test_data_dict[record]

        inputs = (np.asfarray(all_values) / 255.0 * 0.99) + 0.01

        outputs = n.query(inputs)
        #print(outputs)

        label = int(np.argmax(outputs))
        label_t = label_map(dataset, label)

        #print(record + ' ' + str(label_t))
        f.write(record + ' ' + str(label_t) + '\n')

        '''
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        pass
        '''
    f.close()
    print('測試結果已輸出至Answer.txt')

    # calculate the performance score, the fraction of correct answers
    '''
    scorecard_array = np.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)
    '''

if __name__ == "__main__":

    obj = ""
    while obj is not 'y' and obj is not 'n':
        obj = input("是否重新訓練模型, (y/n) : ").strip(' ')
        if obj is not 'y' and obj is not 'n':
            print('輸入錯誤 請重新輸入')

    print('')
    main(obj)


'''
TODO
加入讀檔 儲存照片轉csv, 儲存label index
偵測有無儲存檔, 偵測label index有沒有不一樣

一開始先分100張 or 500張, 加入訓練時的準確率

修改隱藏層數量
修正參數名稱

'''
