# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 09:39:22 2023

@author: zmzhai
"""


import grid2op
import pandas as pd
import numpy as np
import pickle
import random
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils import class_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV as RSCV
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, LSTM, Dense, Flatten, Conv1D, Input, add
from keras.layers import Dropout, MaxPooling2D, TimeDistributed, MaxPooling1D
from keras.layers import TimeDistributed, Bidirectional
from keras import backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from random import shuffle


class ML:
    def __init__(self, input_dim=[-2], obs_prop=0.1, attack_out='single', norm_multi='softmax'):
        self.input_dim = input_dim
        self.obs_prop = obs_prop
        self.attack_out = attack_out
        self.norm_multi = norm_multi

    def read_data(self, file_path='./data/', file_name='data_case14'):
        pkl_file = open(file_path + file_name + '_train' + '.pkl', 'rb')
        self.data_train = pickle.load(pkl_file)
        pkl_file.close()
        
        pkl_file = open(file_path + file_name + '_val' + '.pkl', 'rb')
        self.data_val = pickle.load(pkl_file)
        pkl_file.close()
        
        pkl_file = open(file_path + file_name + '_test' + '.pkl', 'rb')
        self.data_test = pickle.load(pkl_file)
        pkl_file.close()
        
    def data_process(self, cut_data=1, random_choose=True, centrality=None):
        if cut_data > 0.95:
            cut_data = 0.95
        # suppose that we can only observe one indicator
        # train data
        cut_length = round(np.shape(self.data_train)[0] * cut_data)
        start_length = random.randint(0, int(np.shape(self.data_train)[0]-cut_length-1))

        self.input_train = self.data_train[start_length:cut_length+start_length, :, self.input_dim]
        self.output_train = self.data_train[start_length:cut_length+start_length, :, -1]
        # self.maintain_train = self.data_train[start_length:cut_length+start_length, :, -3]

        cut_length = round(np.shape(self.data_val)[0] * cut_data)
        start_length = random.randint(0, int(np.shape(self.data_val)[0]-cut_length-1))
        self.input_val = self.data_val[start_length:cut_length+start_length, :, self.input_dim]
        self.output_val = self.data_val[start_length:cut_length+start_length, :, -1]
        # self.maintain_val = self.data_val[start_length:cut_length+start_length, :, -3]

        cut_length = round(np.shape(self.data_test)[0] * cut_data)
        start_length = random.randint(0, int(np.shape(self.data_test)[0]-cut_length-1))
        self.input_test = self.data_test[start_length:cut_length+start_length, :, self.input_dim]
        self.output_test = self.data_test[start_length:cut_length+start_length, :, -1]
        # self.maintain_test = self.data_test[start_length:cut_length+start_length, :, -3]
        # get the number of lines and the observed lines
        self.lines_num_full = np.shape(self.input_train)[1]
        self.obs_num = int(round(self.obs_prop * self.lines_num_full))
        
        if random_choose:
            random_x = list(range(self.lines_num_full))
            shuffle(random_x)
            random_x = sorted(random_x[:self.obs_num])
            self.obs_lines = random_x
        else:
            self.obs_lines = list(centrality[:self.obs_num, 0])
            self.obs_lines = [int(i) for i in self.obs_lines]
        # train
        self.input_train = self.input_train[:, self.obs_lines]
        self.input_train = self.input_train[:, :, 0]
        # val
        self.input_val = self.input_val[:, self.obs_lines]
        self.input_val = self.input_val[:, :, 0]
        # test
        self.input_test = self.input_test[:, self.obs_lines]
        self.input_test = self.input_test[:, :, 0]
        
        # delete maintain
        # train_maintain = np.logical_not(np.any(self.maintain_train==1, axis=1))
        # self.input_train = self.input_train[train_maintain]
        # self.output_trian = self.output_train[train_maintain]
        
        # val_maintain = np.logical_not(np.any(self.maintain_val==1, axis=1))
        # self.input_val = self.input_val[val_maintain]
        # self.output_val = self.output_val[val_maintain]
        
        # test_maintain = np.logical_not(np.any(self.maintain_test==1, axis=1))
        # self.input_test = self.input_test[test_maintain]
        # self.output_test = self.output_test[test_maintain]
        
        self.train_length = np.shape(self.input_train)[0]
        self.val_length = np.shape(self.input_val)[0]
        self.test_length = np.shape(self.input_test)[0]

    def normalization(self, add_noise=False):
        # use the same scaler.
        input_train_val_test = np.concatenate((self.input_train, self.input_val, self.input_test))
        scaler = MinMaxScaler()
        input_all = scaler.fit_transform(input_train_val_test)
        
        self.X_norm_train = input_all[:self.train_length, :]
        self.X_norm_val = input_all[self.train_length:self.train_length+self.val_length, :]
        self.X_norm_test = input_all[self.train_length+self.val_length:, :]
        
        # self.X_norm_val_record = self.X_norm_val
        self.X_norm_test_record = self.X_norm_test
        
        if add_noise:
            self.X_norm_train += np.multiply(self.X_norm_train, np.random.normal(0.0, 0.02, size=np.shape(self.X_norm_train)))
        
        if self.attack_out == 'single':
            y_norm_train = np.zeros((np.shape(self.input_train)[0], 1))
            y_norm_val = np.zeros((np.shape(self.input_val)[0], 1))
            y_norm_test = np.zeros((np.shape(self.input_test)[0], 1))
            for t_i in range(np.shape(self.input_train)[0]):
                if 1 in self.output_train[t_i, :]:
                    y_norm_train[t_i] = 1
            for t_i in range(np.shape(self.input_val)[0]):
                if 1 in self.output_val[t_i, :]:
                    y_norm_val[t_i] = 1
            for t_i in range(np.shape(self.input_test)[0]):
                if 1 in self.output_test[t_i, :]:
                    y_norm_test[t_i] = 1
                    
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                y_norm_train = np.zeros((np.shape(self.input_train)[0], self.lines_num_full+1))
                y_norm_val = np.zeros((np.shape(self.input_val)[0], self.lines_num_full+1))
                y_norm_test = np.zeros((np.shape(self.input_test)[0], self.lines_num_full+1))
                for t_i in range(np.shape(self.input_train)[0]):
                    for n_i in range(self.lines_num_full):
                        if self.output_train[t_i, n_i] != 0:
                            y_norm_train[t_i, n_i+1] = 1
                    if 1 in self.output_train[t_i, :]:
                        pass
                    else:
                        y_norm_train[t_i, 0] = 1
                        
                for t_i in range(np.shape(self.input_val)[0]):
                    for n_i in range(self.lines_num_full):
                        if self.output_val[t_i, n_i] != 0:
                            y_norm_val[t_i, n_i+1] = 1
                    if 1 in self.output_val[t_i, :]:
                        pass
                    else:
                        y_norm_val[t_i, 0] = 1
                            
                for t_i in range(np.shape(self.input_test)[0]):
                    for n_i in range(self.lines_num_full):
                        if self.output_test[t_i, n_i] != 0:
                            y_norm_test[t_i, n_i+1] = 1
                    if 1 in self.output_test[t_i, :]:
                        pass
                    else:
                        y_norm_test[t_i, 0] = 1
                            
            elif self.norm_multi == 'sigmoid':
                y_norm_train = np.zeros((np.shape(self.input_train)[0], 1))
                y_norm_val = np.zeros((np.shape(self.input_val)[0], 1))
                y_norm_test = np.zeros((np.shape(self.input_test)[0], 1))
                for t_i in range(np.shape(self.input_train)[0]):
                    for n_i in range(self.lines_num_full):
                        if self.output_train[t_i, n_i] != 0:
                            y_norm_train[t_i] = n_i + 1
                for t_i in range(np.shape(self.input_val)[0]):
                    for n_i in range(self.lines_num_full):
                        if self.output_val[t_i, n_i] != 0:
                            y_norm_val[t_i] = n_i + 1
                for t_i in range(np.shape(self.input_test)[0]):
                    for n_i in range(self.lines_num_full):
                        if self.output_test[t_i, n_i] != 0:
                            y_norm_test[t_i] = n_i + 1
            
        self.y_norm_train = y_norm_train
        self.y_norm_val = y_norm_val
        self.y_norm_test = y_norm_test
        
    def calculate_attack_happens(self):
        if self.attack_out == 'single':
            attack_happens_train = np.count_nonzero(self.y_norm_train == 1)
            attack_happens_prob_train = attack_happens_train / (np.shape(self.y_norm_train)[0])
            attack_happens_val = np.count_nonzero(self.y_norm_val == 1)
            attack_happens_prob_val = attack_happens_val / (np.shape(self.y_norm_val)[0])
            attack_happens_test = np.count_nonzero(self.y_norm_test == 1)
            attack_happens_prob_test = attack_happens_test / (np.shape(self.y_norm_test)[0])
            
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                attack_happens_vector_train = np.zeros((np.shape(self.y_norm_train)[1], 1))
                for n_i in range(0, np.shape(self.y_norm_train)[1]):
                    attack_happens_vector_train[n_i] = np.count_nonzero(self.y_norm_train[:, n_i] == 1)
                attack_happens_prob_train = attack_happens_vector_train / np.shape(self.y_norm_train)[0]
                
                attack_happens_vector_val = np.zeros((np.shape(self.y_norm_val)[1], 1))
                for n_i in range(0, np.shape(self.y_norm_val)[1]):
                    attack_happens_vector_val[n_i] = np.count_nonzero(self.y_norm_val[:, n_i] == 1)
                attack_happens_prob_val = attack_happens_vector_val / np.shape(self.y_norm_val)[0]
                
                attack_happens_vector_test = np.zeros((np.shape(self.y_norm_test)[1], 1))
                for n_i in range(0, np.shape(self.y_norm_test)[1]):
                    attack_happens_vector_test[n_i] = np.count_nonzero(self.y_norm_test[:, n_i] == 1)
                attack_happens_prob_test = attack_happens_vector_test / np.shape(self.y_norm_test)[0]
            elif self.norm_multi == 'sigmoid':
                attack_happens_vector_train = []
                for i in range(self.lines_num_full+1):
                    attack_happens_vector_train.append(np.count_nonzero(self.y_norm_train == i))
                attack_happens_prob_train = attack_happens_vector_train / np.shape(self.y_norm_train)[0]
                
                attack_happens_vector_val = []
                for i in range(self.lines_num_full+1):
                    attack_happens_vector_val.append(np.count_nonzero(self.y_norm_val == i))
                attack_happens_prob_val = attack_happens_vector_val / np.shape(self.y_norm_val)[0]
                
                attack_happens_vector_test = []
                for i in range(self.lines_num_full+1):
                    attack_happens_vector_test.append(np.count_nonzero(self.y_norm_test == i))
                attack_happens_prob_test = attack_happens_vector_test / np.shape(self.y_norm_test)[0]
            
        return attack_happens_prob_train, attack_happens_prob_val, attack_happens_prob_test

    def create_dataset(self):
        pass
        
    def create_dataset_lstm(self, seq_length=8):
        dataX_train, dataY_train = [], []
        dataX_val, dataY_val = [], []
        dataX_test, dataY_test = [], []
        self.seq_length = seq_length
        
        if self.attack_out == 'single':
            for i in range(np.shape(self.y_norm_train)[0]-self.seq_length-1):
                dataX_train.append(self.X_norm_train[i:(i+self.seq_length), :])
                dataY_train.append(self.y_norm_train[i+self.seq_length-1])
            for i in range(np.shape(self.y_norm_val)[0]-self.seq_length-1):
                dataX_val.append(self.X_norm_val[i:(i+self.seq_length), :])
                dataY_val.append(self.y_norm_val[i+self.seq_length-1])
            for i in range(np.shape(self.y_norm_test)[0]-self.seq_length-1):
                dataX_test.append(self.X_norm_test[i:(i+self.seq_length), :])
                dataY_test.append(self.y_norm_test[i+self.seq_length-1])
        
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                for i in range(np.shape(self.y_norm_train)[0]-self.seq_length-1):
                    dataX_train.append(self.X_norm_train[i:(i+self.seq_length), :])
                    dataY_train.append(self.y_norm_train[i+self.seq_length-1, :])
                for i in range(np.shape(self.y_norm_val)[0]-self.seq_length-1):
                    dataX_val.append(self.X_norm_val[i:(i+self.seq_length), :])
                    dataY_val.append(self.y_norm_val[i+self.seq_length-1, :])
                for i in range(np.shape(self.y_norm_test)[0]-self.seq_length-1):
                    dataX_test.append(self.X_norm_test[i:(i+self.seq_length), :])
                    dataY_test.append(self.y_norm_test[i+self.seq_length-1, :])
                    
            elif self.norm_multi == 'sigmoid':
                for i in range(np.shape(self.y_norm_train)[0]-self.seq_length-1):
                    dataX_train.append(self.X_norm_train[i:(i+self.seq_length), :])
                    dataY_train.append(self.y_norm_train[i+self.seq_length-1])
                for i in range(np.shape(self.y_norm_val)[0]-self.seq_length-1):
                    dataX_val.append(self.X_norm_val[i:(i+self.seq_length), :])
                    dataY_val.append(self.y_norm_val[i+self.seq_length-1])
                for i in range(np.shape(self.y_norm_test)[0]-self.seq_length-1):
                    dataX_test.append(self.X_norm_test[i:(i+self.seq_length), :])
                    dataY_test.append(self.y_norm_test[i+self.seq_length-1])
                    
        self.X_norm_train = np.array(dataX_train)
        self.X_norm_val = np.array(dataX_val)
        self.X_norm_test = np.array(dataX_test)
        
        self.y_norm_train = np.array(dataY_train)
        self.y_norm_val = np.array(dataY_val)
        self.y_norm_test = np.array(dataY_test)

    def create_dataset_ngrc(self, seq_length=8):
        dataX_train, dataY_train = [], []
        dataX_val, dataY_val = [], []
        dataX_test, dataY_test = [], []
        self.seq_length = seq_length

        if self.attack_out == 'single':
            for i in range(np.shape(self.y_norm_train)[0]-self.seq_length-1):
                dataX_train.append(self.X_norm_train[i:(i+self.seq_length), :].flatten())
                dataY_train.append(self.y_norm_train[i+self.seq_length-1])
            for i in range(np.shape(self.y_norm_val)[0]-self.seq_length-1):
                dataX_val.append(self.X_norm_val[i:(i+self.seq_length), :].flatten())
                dataY_val.append(self.y_norm_val[i+self.seq_length-1])
            for i in range(np.shape(self.y_norm_test)[0]-self.seq_length-1):
                dataX_test.append(self.X_norm_test[i:(i+self.seq_length), :].flatten())
                dataY_test.append(self.y_norm_test[i+self.seq_length-1])
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                for i in range(np.shape(self.y_norm_train)[0]-self.seq_length-1):
                    dataX_train.append(self.X_norm_train[i:(i+self.seq_length), :].flatten())
                    dataY_train.append(self.y_norm_train[i+self.seq_length-1, :])
                for i in range(np.shape(self.y_norm_val)[0]-self.seq_length-1):
                    dataX_val.append(self.X_norm_val[i:(i+self.seq_length), :].flatten())
                    dataY_val.append(self.y_norm_val[i+self.seq_length-1, :])
                for i in range(np.shape(self.y_norm_test)[0]-self.seq_length-1):
                    dataX_test.append(self.X_norm_test[i:(i+self.seq_length), :].flatten())
                    dataY_test.append(self.y_norm_test[i+self.seq_length-1, :])
                    
            elif self.norm_multi == 'sigmoid':
                for i in range(np.shape(self.y_norm_train)[0]-self.seq_length-1):
                    dataX_train.append(self.X_norm_train[i:(i+self.seq_length), :].flatten())
                    dataY_train.append(self.y_norm_train[i+self.seq_length-1])
                for i in range(np.shape(self.y_norm_val)[0]-self.seq_length-1):
                    dataX_val.append(self.X_norm_val[i:(i+self.seq_length), :].flatten())
                    dataY_val.append(self.y_norm_val[i+self.seq_length-1])
                for i in range(np.shape(self.y_norm_test)[0]-self.seq_length-1):
                    dataX_test.append(self.X_norm_test[i:(i+self.seq_length), :].flatten())
                    dataY_test.append(self.y_norm_test[i+self.seq_length-1])
                    
        self.X_norm_train = np.array(dataX_train)
        self.X_norm_val = np.array(dataX_val)
        self.X_norm_test = np.array(dataX_test)
        
        self.y_norm_train = np.array(dataY_train)
        self.y_norm_val = np.array(dataY_val)
        self.y_norm_test = np.array(dataY_test)

    def lstm_layers(self):
        self.model = Sequential()
        self.model.add(LSTM(128, input_shape=(self.seq_length, self.obs_num), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(64,return_sequences=False))
        self.model.add(Dropout(0.2))
        
        if self.attack_out == 'single':
            self.model.add(Dense(16))
            self.model.add(Dense(1,activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', custom_f1])
            
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                self.model.add(Dense(64))
                self.model.add(Dense(self.lines_num_full+1, activation='softmax'))
                self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', custom_f1])
            elif self.norm_multi == 'sigmoid':
                self.model.add(Dense(32))
                self.model.add(Dense(1,activation='sigmoid'))
                self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', custom_f1])
                
    def fnn_layers(self):
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_shape=(np.shape(self.X_norm_train)[1], )))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64))
        self.model.add(Dropout(0.2))
        if self.attack_out == 'single':
            self.model.add(Dense(16))
            self.model.add(Dense(1,activation='sigmoid'))
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', custom_f1])
        
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                self.model.add(Dense(64))
                self.model.add(Dense(self.lines_num_full+1, activation='softmax'))
                self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy', custom_f1])
            elif self.norm_multi == 'sigmoid':
                self.model.add(Dense(32))
                self.model.add(Dense(1, activation='linear'))
                self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', custom_f1])

    def train_nn(self, epoch=10, batch_size=64, class_weight={0:1, 1:1}):
        
        if self.attack_out == 'single':
            n_splits = 5
            kfold = KFold(n_splits=n_splits, shuffle=False)
            
            fold = 1
            for train_index, val_index in kfold.split(self.X_norm_train):
                X_train_fold, X_val_fold = self.X_norm_train[train_index], self.X_norm_train[val_index]
                y_train_fold, y_val_fold = self.y_norm_train[train_index], self.y_norm_train[val_index]
            
                self.history = self.model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), shuffle=True, epochs=3, batch_size=128, class_weight=class_weight)
            
                fold += 1
            
            self.history = self.model.fit(self.X_norm_train, self.y_norm_train, validation_data=(self.X_norm_val, self.y_norm_val), shuffle=True, epochs=epoch, batch_size=batch_size, class_weight=class_weight)
            
            self.y_pred = self.model.predict(self.X_norm_test)
            
            y_pred_labels = self.y_pred
            y_pred_labels[y_pred_labels >= 0.5] = 1
            y_pred_labels[y_pred_labels < 0.5] = 0
            y_test_labels = self.y_norm_test
            
            f1 = f1_score(y_test_labels, y_pred_labels)
            accuracy = accuracy_score(y_test_labels, y_pred_labels)
            
            self.y_pred_labels = y_pred_labels
            self.y_test_labels = y_test_labels
            
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                n_splits = 5
                kfold = KFold(n_splits=n_splits, shuffle=False)
                
                fold = 1
                for train_index, val_index in kfold.split(self.X_norm_train):
                    X_train_fold, X_val_fold = self.X_norm_train[train_index], self.X_norm_train[val_index]
                    y_train_fold, y_val_fold = self.y_norm_train[train_index], self.y_norm_train[val_index]
                
                    self.history = self.model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), shuffle=True, epochs=3, batch_size=128)
                
                    fold += 1
                
                self.history = self.model.fit(self.X_norm_train, self.y_norm_train, validation_data=(self.X_norm_val, self.y_norm_val), shuffle=True, epochs=epoch, batch_size=batch_size)
                
                self.y_pred = self.model.predict(self.X_norm_test)
                y_pred_labels = self.y_pred
                y_pred_labels[y_pred_labels >= 0.5] = 1
                y_pred_labels[y_pred_labels < 0.5] = 0
                y_test_labels = self.y_norm_test
                
                y_pred_multi, y_test_multi = np.zeros((np.shape(y_pred_labels)[0], 1)), np.zeros((np.shape(y_pred_labels)[0], 1))
                
                for ti in range(np.shape(y_pred_labels)[0]):
                    for ni in range(np.shape(y_pred_labels)[1]):
                        if y_pred_labels[ti, ni] == 1:
                            y_pred_multi[ti] = ni
                        if y_test_labels[ti, ni] == 1:
                            y_test_multi[ti] = ni
                f1 = f1_score(y_test_multi, y_pred_multi, average='weighted')
                accuracy = accuracy_score(y_test_labels, y_pred_labels)
                
                self.y_pred_labels = y_pred_multi
                self.y_test_labels = y_test_multi
                
            elif self.norm_multi == 'sigmoid':
                print('error: do not use sigmoid for lstm!')
                
        return f1, accuracy
        
    
    def train_rf(self, n_estimators=100, max_depth=30):
        self.y_norm_train = self.y_norm_train.ravel()
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
        rf.fit(self.X_norm_train, self.y_norm_train)
        self.y_pred = rf.predict(self.X_norm_test)
        
        y_pred_labels = self.y_pred
        y_pred_labels = [round(i) for i in y_pred_labels]
        y_test_labels = self.y_norm_test
        
        if self.attack_out == 'single':
            f1 = f1_score(y_test_labels, y_pred_labels)
        elif self.attack_out == 'multi':
            f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
            
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
        
        self.y_pred_labels = y_pred_labels
        self.y_test_labels = y_test_labels
    
        return f1, accuracy
    
    def hyper_tunning_rf(self):
        param_choose = {'n_estimators': np.arange(20, 520, 20), 
                        'max_depth': np.arange(1, 30, 2)
                        }
        
        self.model = RSCV(RandomForestClassifier(n_jobs=-1), param_choose, n_iter=20).fit(self.X_norm_val, self.y_norm_val)
        best_rf = self.model.best_estimator_
        
        return best_rf
    
    def train_svm(self):
        clf = svm.SVC(kernel='linear')
        clf.fit(self.X_norm_train, self.y_norm_train)
        self.y_pred = clf.predict(self.X_norm_test)
        
        y_pred_labels = self.y_pred
        y_pred_labels = [round(i) for i in y_pred_labels]
        y_test_labels = self.y_norm_test
        
        if self.attack_out == 'single':
            f1 = f1_score(y_test_labels, y_pred_labels)
        elif self.attack_out == 'multi':
            f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
            
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
    
        return f1, accuracy
    
    def train_knn(self, n_neighbors=3):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(self.X_norm_train, self.y_norm_train)
        self.y_pred = knn.predict(self.X_norm_test)
        
        y_pred_labels = self.y_pred
        y_pred_labels = [round(i) for i in y_pred_labels]
        y_test_labels = self.y_norm_test
        
        if self.attack_out == 'single':
            f1 = f1_score(y_test_labels, y_pred_labels)
        elif self.attack_out == 'multi':
            f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
            
        accuracy = accuracy_score(y_test_labels, y_pred_labels)
    
        return f1, accuracy
    
    
    def naive_method(self, threshold=1.0):
        # make this only on testing data, and write the multi attack scenario
        x_test = self.X_norm_test_record[-np.shape(self.X_norm_test)[0]:, :]
        # x_val = self.X_norm_val_record[-np.shape(self.X_norm_val)[0]:, :]
        
        if self.attack_out == 'single':
            # val_f1_set = []
            # thre_range = np.arange(0.1, 1.1, 0.1)
            # for thre in thre_range:
            #     input_val = np.zeros((np.shape(x_val)[0], 1))
            #     for t_i in range(np.shape(x_val)[0]):
            #         if 0 in x_val[t_i, :] or all(x_val[t_i, :] > thre):
            #             input_val[t_i] = 1
                        
            #     val_f1 = f1_score(self.y_norm_val, input_val)
            #     val_f1_set.append(val_f1)
            
            # max_threshold = thre_range[np.argmax(val_f1_set)]

            input_naive = np.zeros((np.shape(x_test)[0], 1))
            for t_i in range(np.shape(x_test)[0]):
                if 0 in x_test[t_i, :] or all(x_test[t_i, :] > threshold):
                    input_naive[t_i] = 1
                    
            naive_f1 = f1_score(self.y_norm_test, input_naive)
                    
        elif self.attack_out == 'multi':
            if self.norm_multi == 'softmax':
                # val_f1_set = []
                # thre_range = np.arange(0.1, 1.1, 0.1)
                # for thre in thre_range:
                #     input_val = np.zeros((np.shape(x_val)[0], self.lines_num_full+1))
                #     for t_i in range(np.shape(x_val)[0]):
                #         for n_i in range(self.obs_num):
                #             if x_val[t_i, n_i] == 0 or x_val[t_i, n_i] > thre:
                #                 input_val[t_i, self.obs_lines[n_i] + 1] = 1
                #             if 1 in input_val[t_i, :]:
                #                 pass
                #             else:
                #                 input_val[t_i, 0] = 1
                            
                #     val_f1 = f1_score(self.y_norm_val, input_val, average='weighted')
                #     val_f1_set.append(val_f1)
                
                # max_threshold = thre_range[np.argmax(val_f1_set)]
                
                input_naive = np.zeros((np.shape(x_test)[0], self.lines_num_full+1))
                for t_i in range(np.shape(x_test)[0]):
                    for n_i in range(self.obs_num):
                        if x_test[t_i, n_i] == 0 or x_test[t_i, n_i] > threshold:
                            input_naive[t_i, self.obs_lines[n_i] + 1] = 1
                    if 1 in input_naive[t_i, :]:
                        pass
                    else:
                        input_naive[t_i, 0] = 1

                naive_f1 = f1_score(self.y_norm_test, input_naive, average='weighted')
                            
            elif self.norm_multi == 'sigmoid':
                # val_f1_set = []
                # thre_range = np.arange(0.1, 1.1, 0.1)
                # for thre in thre_range:
                #     input_val = np.zeros((np.shape(x_val)[0], 1))
                #     for t_i in range(np.shape(x_val)[0]):
                #         for n_i in range(self.obs_num):
                #             if x_val[t_i, n_i] == 0 or x_val[t_i, n_i] > thre:
                #                 input_val[t_i] = self.obs_lines[n_i]
                            
                #     val_f1 = f1_score(self.y_norm_val, input_val, average='weighted')
                #     val_f1_set.append(val_f1)
                
                # max_threshold = thre_range[np.argmax(val_f1_set)]
                
                input_naive = np.zeros((np.shape(x_test)[0], 1))
                for t_i in range(np.shape(x_test)[0]):
                    for n_i in range(self.obs_num):
                        if x_test[t_i, n_i] == 0 or x_test[t_i, n_i] > threshold:
                            input_naive[t_i] = self.obs_lines[n_i]
                
                naive_f1 = f1_score(self.y_norm_test, input_naive, average='weighted')
                    
                    
        # naive_f1 = f1_score(self.y_norm, input_naive)
        naive_accuracy = accuracy_score(self.y_norm_test, input_naive)
        
        return naive_f1, naive_accuracy

    def plot_accuracy(self):
        classes = unique_labels(self.y_test_labels, self.y_pred_labels)
        classes = [int(i) for i in classes]
            
        cm = confusion_matrix(self.y_test_labels, self.y_pred_labels)
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if self.attack_out == 'single':
            fig, ax = plt.subplots(figsize=(10, 10))
        else:
            fig, ax = plt.subplots(figsize=(20, 20))
        im = ax.imshow(cmn, interpolation='nearest', cmap='coolwarm')
        cb = fig.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title='Normalized Confusion Matrix',
               xlabel='Predicted label',
               ylabel='True label')
        
        thresh = cmn.max() / 2.
        for i in range(cmn.shape[0]):
            for j in range(cmn.shape[1]):
                if cmn[i, j] < 0.01:
                    text = format(int(cmn[i, j]), 'd')
                else:
                    text = format(cmn[i, j], '.2f')
                ax.text(j, i, text,
                        ha="center", va="center",
                        color="white" if cmn[i, j] > thresh else "black")
        plt.show()



def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall


    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))



if __name__ == '__main__':
    print('ML1.0 ...')

    # file_name = 'data_wcci2022'
    file_name = 'data_case14' # two datasets
    seq_length = 5
    # small network observe 0.3, and the large network observe 0.5
    ml = ML(input_dim=[-2], obs_prop=0.3, attack_out='single', norm_multi='softmax')
    ml.read_data(file_path='./data/', file_name=file_name)
    ml.data_process(cut_data=0.9, random_choose=True, centrality=None)
    ml.normalization(add_noise=True)
    
    ml.create_dataset_lstm(seq_length=seq_length)
    ml.lstm_layers()
    # ml.lstm_res_layers()
    # f1_lstm, accuracy_lstm = ml.train_nn(epoch=5, batch_size=64, class_weight={0:1, 1:1})
    # ml.plot_accuracy()
    
    f1_lstm, accuracy_lstm = ml.train_nn(epoch=5, batch_size=64, class_weight={0:1, 1:2})
    
    # # ml.create_dataset_ngrc(seq_length=seq_length)
    # # # ml.create_dataset()
    # # ml.fnn_layers()
    # # f1_fnn, accuracy_fnn = ml.train_nn(epoch=10, batch_size=64, class_weight={0:1, 1:1})
    # # ml.plot_accuracy()
    
    # ml.norm_multi = 'sigmoid'
    # ml.normalization()
    # # # ml.create_dataset_ngrc(seq_length=seq_length)
    # ml.create_dataset()
    # f1_rf, accuracy_rf = ml.train_rf()
    # ml.plot_accuracy()
    # # best_rf = ml.hyper_tunning_rf()
    # print('rf finished')
    
    # # f1_svm, accuracy_svm = ml.train_svm()
    # # ml.plot_accuracy()
    # # print('svm finished')
    
    # f1_knn, accuracy_knn = ml.train_knn()
    ml.plot_accuracy()
    # print('knn finished')
    
    # naive_f1, naive_accuracy = ml.naive_method()

    # attack_happens_prob_train, attack_happens_prob_val, attack_happens_prob_test = ml.calculate_attack_happens()

    # save_file = open('./data_save/' + 'small_multi_cm' + '.pkl', 'wb')
    # pickle.dump(ml.y_test_labels, save_file)
    # pickle.dump(ml.y_pred_labels, save_file)
    # pickle.dump(ml.attack_out, save_file)
    # save_file.close()
    
    # save_file = open('./data_save/' + 'large_single_roc' + '.pkl', 'wb')
    # pickle.dump(ml.X_norm_test, save_file)
    # pickle.dump(ml.y_test_labels, save_file)
    # pickle.dump(ml.y_pred, save_file)
    # pickle.dump(ml.y_pred_labels, save_file)
    # pickle.dump(ml.attack_out, save_file)
    # pickle.dump(ml.obs_lines, save_file)
    # pickle.dump(f1_lstm, save_file)
    # save_file.close()
    
    save_file = open('./data_save/' + 'small_single_change_weight' + '.pkl', 'wb')
    pickle.dump(ml.X_norm_test, save_file)
    pickle.dump(ml.y_test_labels, save_file)
    pickle.dump(ml.y_pred, save_file)
    pickle.dump(ml.y_pred_labels, save_file)
    pickle.dump(ml.attack_out, save_file)
    pickle.dump(ml.obs_lines, save_file)
    pickle.dump(f1_lstm, save_file)
    save_file.close()

















