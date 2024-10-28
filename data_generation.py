# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:21:40 2023

@author: zmzhai
"""

# for generating the data

import grid2op
import pandas as pd
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from grid2op.Action import PowerlineSetAction
from grid2op.Agent import DoNothingAgent, RecoPowerlineAgent
from grid2op.Opponent import RandomLineOpponent, BaseActionBudget
from grid2op.Runner import Runner
from grid2op.Episode import EpisodeReplay
from grid2op.PlotGrid import PlotMatplot


class DataGen:
    def __init__(self, env_name="l2rpn_case14_sandbox", agent_name='donothing', episode_count=10,
                 opponent={"lines_attacked":["1_3_3", "1_4_4", "3_6_15", "9_10_12", "11_12_13", "12_13_14",
         "0_1_0", "0_4_1", "1_2_2", "2_3_5", "3_4_6", "5_10_7", "5_11_8", "5_12_9", "8_9_10", "8_13_11", "3_6_15", "3_8_16", "4_5_17", "6_7_18", "6_8_19"]}):
        self.env_name = env_name
        self.agent_name = agent_name
        self.save_matrix = None
        # self.dead_time = []
        self.opponent = opponent
        self.episode_count = episode_count
        
    def initialize(self, attack_cooldown=12*24, attack_duration=12*4):
        self.env_with_opponent = grid2op.make(self.env_name,
                                opponent_attack_cooldown=attack_cooldown,
                                opponent_attack_duration=attack_duration,
                                opponent_budget_per_ts=0.8,
                                opponent_init_budget=0.,
                                opponent_action_class=PowerlineSetAction,
                                opponent_class=RandomLineOpponent,
                                opponent_budget_class=BaseActionBudget,
                                kwargs_opponent=self.opponent
                                )
        
        self.total_episode = len(self.env_with_opponent.chronics_handler.subpaths)

        self.obs = self.env_with_opponent.reset()
        self.reward = self.env_with_opponent.reward_range[0]
        if self.agent_name == 'donothing':
            self.my_agent = DoNothingAgent(self.env_with_opponent.action_space)
        elif self.agent_name == 'reconnect':
            self.my_agent = RecoPowerlineAgent(self.env_with_opponent.action_space)

        self.time_step = int(0)
        
        self.obs_all, self.info_all = [], []
        self.obs_all.append(self.obs)
        self.info_all.append([])
        
    def run(self, iter_time=10):
        for i in range(self.episode_count):
            print('out_iter:', iter_time, ', in_iter:', i)
            if i % self.total_episode == 0:
                # I shuffle each time i need to
                self.env_with_opponent.chronics_handler.shuffle()
            done = False
            self.obs = self.env_with_opponent.reset()
            
            while True:
                act = self.my_agent.act(self.obs, self.reward, done)
                self.obs, self.reward, done, self.info =self. env_with_opponent.step(act) # implement this action on the powergrid
                
                self.obs_all.append(self.obs)
                self.info_all.append(self.info)
                self.time_step += 1
                
                if done:
                    break

    def record_data(self):
        t = len(self.obs_all)
        n_line = self.env_with_opponent.n_line
        self.record = np.zeros((t, n_line, 13))
        
        for i in range(t):
            obs_i = self.obs_all[i]
            p_or, q_or, v_or, a_or, theta_or = obs_i.p_or, obs_i.q_or, obs_i.v_or, obs_i.a_or, obs_i.theta_or
            p_ex, q_ex, v_ex, a_ex, theta_ex = obs_i.p_ex, obs_i.q_ex, obs_i.v_ex, obs_i.a_ex, obs_i.theta_ex
            maintain = obs_i.duration_next_maintenance
            rho = obs_i.rho

            if i == 0:
                attack_info = None
            else:
                info_i = self.info_all[i]
                attack_info = info_i['opponent_attack_line']
                
            if attack_info is None:
                attack = [0 for _ in range(self.env_with_opponent.n_line)]
            else:
                attack = [int(a) for a in attack_info]
            
            self.record[i, :, 0:5] = np.concatenate((p_or.reshape(-1, 1), q_or.reshape(-1, 1), v_or.reshape(-1, 1), a_or.reshape(-1, 1), theta_or.reshape(-1, 1)), axis=1)
            self.record[i, :, 5:10] = np.concatenate((p_ex.reshape(-1, 1), q_ex.reshape(-1, 1), v_ex.reshape(-1, 1), a_ex.reshape(-1, 1), theta_ex.reshape(-1, 1)), axis=1)
            self.record[i, :, 10] = maintain
            self.record[i, :, 11] = rho
            self.record[i, :, 12] = attack
            # self.record[i, :, 11] = attack_time

    def concat_data(self):
        if self.save_matrix is None:
            self.save_matrix = self.record
        else:
            self.save_matrix = np.concatenate((self.save_matrix, self.record), axis=0)
        # self.dead_time.append(np.shape(self.save_matrix)[0])


    def save_data(self, file_path='./data/', file_name='data_save'):
        save_file = open(file_path + file_name + '.pkl', 'wb')
        pickle.dump(self.save_matrix, save_file)
        # pickle.dump(self.dead_time, save_file)
        save_file.close()
        
    def read_data(self, file_path='./data/', file_name='data_save'):
        pkl_file = open(file_path + file_name + '.pkl', 'rb')
        data1 = pickle.load(pkl_file)
        # data2 = pickle.load(pkl_file)
        pkl_file.close()
        
        return data1
    
    def save_dataframe(self, file_path='./data/', file_name='dataframe'):
        num_lines = self.env_with_opponent.n_line
        list_p_or = list_gen('p_or', num_lines)        
        list_q_or = list_gen('q_or', num_lines)        
        list_v_or = list_gen('v_or', num_lines)        
        list_a_or = list_gen('a_or', num_lines)        
        list_theta_or = list_gen('theta_or', num_lines)        
        list_p_ex = list_gen('p_ex', num_lines)        
        list_q_ex = list_gen('q_ex', num_lines)        
        list_v_ex = list_gen('v_ex', num_lines)        
        list_a_ex = list_gen('a_ex', num_lines)        
        list_theta_ex = list_gen('theta_ex', num_lines)    
        list_maintain = list_gen('maintain', num_lines)
        list_rho = list_gen('rho', num_lines)
        list_attacks = list_gen('attack', num_lines)
        # list_attack = ['attack']
        
        save_matrix_reshape = np.reshape(self.save_matrix, (np.shape(self.save_matrix)[0], -1), order='F')
        
        headers = list_p_or + list_q_or + list_v_or + list_a_or + list_theta_or + \
            list_p_ex + list_q_ex + list_v_ex + list_a_ex + list_theta_ex + list_maintain + list_rho + list_attacks
        save_matrix_reshape = pd.DataFrame(save_matrix_reshape, columns=headers)
        
        save_file = open(file_path + file_name + '.pkl', 'wb')
        pickle.dump(save_matrix_reshape, save_file)
        # pickle.dump(self.dead_time, save_file)
        save_file.close()


def plot_attack(dg, start=0, end=1000):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for i in range(dg.env_with_opponent.n_line):
        ax.plot(range(start, end), dg.save_matrix[start:end, i, 11], label='line_{}'.format(i))
        
    ax.set_xlabel('t')
    ax.set_ylabel('attack (bool)')
    ax.legend(loc='upper left')
    plt.show()
    

def list_gen(name, num_lines):
    l = []
    for i in range(num_lines):
        l.append(f'{name}_{i}')
    return l



if __name__ == '__main__':
    print('data processing 1.0 ...')

    grid2op.change_local_dir('D:\\Users\\admin\\grid2op_env')
    
    # for env_add in ['test', 'val']:
    #     env_name = "l2rpn_case14_sandbox_" + env_add
    #     file_name_1='data_case14_' + env_add
        
    #     env = grid2op.make(env_name)
    #     opponent = {"lines_attacked": env.name_line}
        
    #     agent_name='reconnect'
    
    #     iter_time = 20
    #     episode_count = 20
    #     dg = DataGen(env_name=env_name, agent_name=agent_name, opponent=opponent, episode_count=episode_count)
    #     for i in range(iter_time):
    #         print(i)
    #         attack_cooldown = 12 * random.randint(1, 6)
    #         attack_duration = 12 * random.randint(1, 6)
    #         dg.initialize(attack_cooldown=attack_cooldown, attack_duration=attack_duration)
    #         dg.run(iter_time=i)
    #         dg.record_data()
    #         dg.concat_data()


    #     save_matrix = dg.save_matrix
    #     # save data
    #     dg.save_data(file_name=file_name_1)
    
    
    # wcci2022
    # for env_add in ['train', 'test', 'val']:
    #     env_name = "l2rpn_wcci_2022_" + env_add
    #     file_name_1='data_wcci2022_' + env_add
        
    #     env = grid2op.make(env_name)
    #     opponent = {"lines_attacked": env.name_line}
        
    #     agent_name='reconnect'
        
    #     if env_add == 'train':
    #         iter_time = 50
    #         episode_count = 50
    #     else: 
    #         iter_time = 20
    #         episode_count = 20
    #     dg = DataGen(env_name=env_name, agent_name=agent_name, opponent=opponent, episode_count=episode_count)
    #     for i in range(iter_time):
    #         print(i)
    #         attack_cooldown = 12 * random.randint(1, 6)
    #         attack_duration = 12 * random.randint(1, 6)
    #         dg.initialize(attack_cooldown=attack_cooldown, attack_duration=attack_duration)
    #         dg.run(iter_time=i)
    #         dg.record_data()
    #         dg.concat_data()
    
    
    #     save_matrix = dg.save_matrix
    #     # save data
    #     dg.save_data(file_name=file_name_1)
































