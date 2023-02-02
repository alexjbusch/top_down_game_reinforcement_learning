import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.tetris import Tetris
from collections import deque

from src.top_down import GameState

import time
import random as rand






# original lr = 1e-3

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=5e-2)
    parser.add_argument("--num_decay_epochs", type=float, default=600000)
    parser.add_argument("--num_epochs", type=int, default=3000000)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--replay_memory_size", type=int, default=200,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--model_num", type=str, default="none")
    parser.add_argument("--visualize", type=str, default="True")

    
    args = parser.parse_args()
    return args

    #118750 appears converged


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_angle_dict(env, model):
    if opt.visualize == "False":
        return None


    angle = 0
    angle_dict = {}
    
    if opt.visualize == "Q":
        while angle < 360:
            angle_dict[angle] = model(torch.tensor([[float(angle)]]).cuda())
            angle += 1

        values = [i.item() for i in angle_dict.values()]
        print(values)
        values = NormalizeData(values)
        angle_dict = {i:j for i in angle_dict.keys() for j in values}
        """
        print(angle_dict)
        while True:
            pass
        """
        env.player.angle = rand.randint(0,365)
        env.angle = env.get_angle_actual(env.player.angle)
        return angle_dict
        
    
    while angle < 360:
        env.player.angle = angle
        next_steps = env.get_next_states()
        # Exploration or exploitation

        next_actions, next_states = zip(*next_steps.items())
        #print(next_states)

        next_states = torch.stack(next_states)
        
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        with torch.no_grad():
            predictions = model(next_states)
            action = torch.argmax(predictions).item()



        angle_dict[angle] = action
        angle += 1

    env.player.angle = rand.randint(0,365)
    env.angle = env.get_angle_actual(env.player.angle)



    
    return angle_dict

def visualize(env, angle_dict, opt):
    env.visualize(angle_dict, opt.visualize)
    env.display()


def train(opt):
    angle_dict = {}
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    #env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env = GameState()
    if opt.model_num == "none":
        model = DeepQNetwork()
    else:
        model = torch.load(f"{opt.saved_path}/just_aim_model_{opt.model_num}")


    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()

    

    state = env.reset()
    if torch.cuda.is_available():
        model.cuda()
        state = state.cuda()

    score = 0

    replay_memory = deque(maxlen=opt.replay_memory_size)
    epoch = 0
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        #print(next_states)        
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        model.eval()
        with torch.no_grad():

            predictions = model(next_states)
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()


        next_state = next_states[index, :]
        action = next_actions[index]


        if not angle_dict:
            angle_dict = get_angle_dict(env, model)
            
        reward, done = env.step(action, angle_dict = angle_dict, opt = opt)




        score += reward

        if torch.cuda.is_available():
            next_state = next_state.cuda()




        replay_memory.append([state, reward, next_state, done])






            
        if done:
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            

        if len(replay_memory) < opt.replay_memory_size:
            
            print(f"turns until learning: {(opt.replay_memory_size)-len(replay_memory)}")
            continue

        epoch += 1

        
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()


        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        print(loss)
        loss.backward()
        optimizer.step()

        if not random_action:
            if epoch < opt.num_decay_epochs:
                #print(f"Epoch: {epoch}/{opt.num_epochs}, Action: {action}, Epsilon: {epsilon}")
                pass

        if done:
            
            print(f"Epoch: {epoch}/{opt.num_epochs}, Epsilon: {round(epsilon,3)},  this round score: {score}")

            score = 0
            angle_dict = get_angle_dict(env, model)


        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/just_aim_model_{}".format(opt.saved_path, epoch))
        
    torch.save(model, "{}/final".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
