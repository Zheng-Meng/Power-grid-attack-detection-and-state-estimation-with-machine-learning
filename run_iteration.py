# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 16:35:00 2023

@author: zmzhai
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import time
import ML1_1


def lstm_obs(attack_out='single', file_name='data_case14', cut_data=0.5, 
                          obs_range=np.exp(np.linspace(np.log(0.05), np.log(1), num=10)), 
                          iteration=20, input_dim=[-2], epoch=15, batch_size=64, class_weight={0: 1, 1: 2}, 
                          naive_threshold=0.5, seq_length=8):
    
    f1_lstm, acc_lstm = np.zeros((len(obs_range), iteration)), np.zeros((len(obs_range), iteration))
    # f1_naive, acc_naive = np.zeros((len(obs_range), iteration)), np.zeros((len(obs_range), iteration))
    
    count = 0
    for obs_i in range(len(obs_range)):
        obs_prop = obs_range[obs_i]
        for iter_i in range(iteration):
            ml = ML1_1.ML(input_dim=input_dim, obs_prop=obs_prop, attack_out=attack_out, norm_multi='softmax')
            ml.read_data(file_path='./data/', file_name=file_name)
            ml.data_process(cut_data=cut_data, random_choose=True, centrality=None)
            ml.normalization(add_noise=True)
            
            ml.create_dataset_lstm(seq_length=seq_length)
            ml.lstm_layers()
            f1_1, accuracy_1 = ml.train_nn(epoch=epoch, batch_size=64, class_weight={0:1, 1:1})
            
            # naive_f1, naive_accuracy = ml.naive_method(threshold=naive_threshold)
            
            count += 1
            print('----lstm training---, ', count)
            
            f1_lstm[obs_i, iter_i] = f1_1
            acc_lstm[obs_i, iter_i] = accuracy_1
            # f1_naive[obs_i, iter_i] = naive_f1
            # acc_naive[obs_i, iter_i] = naive_accuracy
    
    if 'case14' in file_name:
        save_file_name = 'lstm_obs_' + 'snet_' + attack_out
    else:
        save_file_name = 'lstm_obs_' + 'lnet_' + attack_out
        
    save_file = open('./data_save/' + save_file_name + '.pkl', 'wb')
    pickle.dump(f1_lstm, save_file)
    pickle.dump(acc_lstm, save_file)
    # pickle.dump(f1_naive, save_file)
    # pickle.dump(acc_naive, save_file)
    pickle.dump(obs_range, save_file)
    save_file.close()
    

def lstm_centrality(attack_out='single', file_name='data_case14', cut_data=0.5, 
                          obs_range=np.exp(np.linspace(np.log(0.05), np.log(1), num=10)), 
                          iteration=20, input_dim=[-2], epoch=15, batch_size=64, class_weight={0: 1, 1: 2}, 
                          naive_threshold=0.5, seq_length=8):
    
    # f1_rf, acc_rf = np.zeros((len(obs_range), iteration)), np.zeros((len(obs_range), iteration))
    f1_central, acc_central = np.zeros((len(obs_range), iteration)), np.zeros((len(obs_range), iteration))
    

    if 'case14' in file_name:
        pkl_file = open('./data/' + 'central_case14' + '.pkl', 'rb')
        centrality = pickle.load(pkl_file)
        pkl_file.close()
    else:
        pkl_file = open('./data/' + 'central_wcci2022' + '.pkl', 'rb')
        centrality = pickle.load(pkl_file)
        pkl_file.close()
    
    count = 0
    for obs_i in range(len(obs_range)):
        obs_prop = obs_range[obs_i]
        for iter_i in range(iteration):
            # centrality
            ml = ML1_1.ML(input_dim=input_dim, obs_prop=obs_prop, attack_out=attack_out, norm_multi='softmax')
            ml.read_data(file_path='./data/', file_name=file_name)
            ml.data_process(cut_data=cut_data, random_choose=False, centrality=centrality)
            ml.normalization(add_noise=True)
            
            ml.create_dataset_lstm(seq_length=seq_length)
            ml.lstm_layers()
            f1_1, accuracy_1 = ml.train_nn(epoch=epoch, batch_size=64, class_weight={0:1, 1:1})
            
            # naive_f1, naive_accuracy = ml.naive_method(threshold=naive_threshold)
            
            count += 1
            print('----lstm training---, ', count)
            
            f1_central[obs_i, iter_i] = f1_1
            acc_central[obs_i, iter_i] = accuracy_1
            
            aaa = 1
    
    if 'case14' in file_name:
        save_file_name = 'lstm_centrality_' + 'snet_' + attack_out
    else:
        save_file_name = 'lstm_centrality_' + 'lnet_' + attack_out
        
    save_file = open('./data_save/' + save_file_name + '.pkl', 'wb')
    pickle.dump(f1_central, save_file)
    pickle.dump(acc_central, save_file)
    pickle.dump(obs_range, save_file)
    save_file.close()
    

def ngrc_time_embedding(attack_out='single', file_name='data_case14', cut_data=0.5, 
                          obs_range=np.exp(np.linspace(np.log(0.05), np.log(1), num=10)), 
                          iteration=20, input_dim=[-2], epoch=15, batch_size=64, class_weight={0: 1, 1: 2}, 
                          naive_threshold=0.5, seq_range=[1, 3, 5]):
    f1_ng, acc_ng = np.zeros((len(obs_range), len(seq_range), iteration)), np.zeros((len(obs_range), len(seq_range), iteration))
    
    count = 0
    for obs_i in range(len(obs_range)):
        obs_prop = obs_range[obs_i]
        for seq_i in range(len(seq_range)):
            seq_length = seq_range[seq_i]
            for iter_i in range(iteration):
                ml = ML1_1.ML(input_dim=input_dim, obs_prop=obs_prop, attack_out=attack_out, norm_multi='softmax')
                ml.read_data(file_path='./data/', file_name=file_name)
                ml.data_process(cut_data=cut_data, random_choose=True, centrality=None)
                ml.normalization(add_noise=True)
                
                ml.create_dataset_ngrc(seq_length=seq_length)
                ml.fnn_layers()
                f1_1, accuracy_1 = ml.train_nn(epoch=10, batch_size=64, class_weight={0:1, 1:1})
                
                # naive_f1, naive_accuracy = ml.naive_method(threshold=naive_threshold)
                
                count += 1
                print('----lstm training---, ', count)
                
                f1_ng[obs_i, seq_i, iter_i] = f1_1
                acc_ng[obs_i, seq_i, iter_i] = accuracy_1
    
    if 'case14' in file_name:
        save_file_name = 'ngrc_obs_' + 'snet_' + attack_out
    else:
        save_file_name = 'ngrc_obs_' + 'lnet_' + attack_out
        
    save_file = open('./data_save/' + save_file_name + '.pkl', 'wb')
    pickle.dump(f1_ng, save_file)
    pickle.dump(acc_ng, save_file)
    pickle.dump(obs_range, save_file)
    pickle.dump(seq_range, save_file)
    save_file.close()
    

if __name__ == '__main__':
    
    attack_out = 'single'
    file_name='data_wcci2022'
    
    cut_data=0.7
    obs_range = np.arange(0.1, 1.1, 0.1)
    iteration=20
    
    input_dim=[-2]
    naive_threshold=0.5
    seq_length=5
    
    lstm_centrality(attack_out=attack_out, file_name=file_name, cut_data=cut_data, 
              obs_range=obs_range, iteration=iteration, input_dim=input_dim, 
              epoch=5, batch_size=64, class_weight={0: 1, 1: 1}, 
              naive_threshold=naive_threshold, seq_length=seq_length)
    
    attack_out = 'multi'
    file_name='data_wcci2022'
    
    lstm_centrality(attack_out=attack_out, file_name=file_name, cut_data=cut_data, 
              obs_range=obs_range, iteration=iteration, input_dim=input_dim, 
              epoch=5, batch_size=64, class_weight={0: 1, 1: 1}, 
              naive_threshold=naive_threshold, seq_length=seq_length)
    
    attack_out='single'
    file_name='data_case14'
    cut_data=0.5
    obs_range = np.exp(np.linspace(np.log(0.05), np.log(1), num=10))
    iteration=20
    
    input_dim=[-2]
    naive_threshold=0.5
    seq_length=5

    lstm_centrality(attack_out=attack_out, file_name=file_name, cut_data=cut_data, 
              obs_range=obs_range, iteration=iteration, input_dim=input_dim, 
              epoch=5, batch_size=64, class_weight={0: 1, 1: 1}, 
              naive_threshold=naive_threshold, seq_length=seq_length)


    attack_out='multi'

    lstm_centrality(attack_out=attack_out, file_name=file_name, cut_data=cut_data, 
              obs_range=obs_range, iteration=iteration, input_dim=input_dim, 
              epoch=5, batch_size=64, class_weight={0: 1, 1: 1}, 
              naive_threshold=naive_threshold, seq_length=seq_length)



















