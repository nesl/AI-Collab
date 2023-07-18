import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from collections import namedtuple, deque
from gym_collab.envs.action import Action
import json
import pdb

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
        
    def save_to_disk(self,file_name):
        out_dict = {"memory":[]}
        open_file = open(file_name,"w")
        for d_idx in range(len(self.memory)):
            memory_element = self.memory[d_idx]._asdict()
            memory_element["state"] = memory_element["state"].tolist()
            memory_element["next_state"] = memory_element["next_state"].tolist()
            out_dict["memory"].append(memory_element)
        json.dump(out_dict,open_file)
        open_file.close()
        
    def load_from_disk(self,file_name,device):
        open_file = open(file_name,"r")
        in_dict = json.load(open_file)
        open_file.close()
        
        for element in in_dict["memory"]:
            if element["action"] == 18:
                continue
            element["state"] = torch.tensor(element["state"], dtype=torch.float32, device=device)
            element["next_state"] = torch.tensor(element["next_state"], dtype=torch.float32, device=device)
            element["reward"] = torch.tensor([element["reward"]], device=device)
            self.memory.append(Transition(element["state"],element["action"],element["next_state"],element["reward"]))
        

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DeepQControl:

    def __init__(self,observation,device, num_steps):

        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.episode_durations = []

        n_actions = Action.drop_object.value+1
        state = np.concatenate((observation['frame'].ravel(),np.array([observation['objects_held']]))) #observation['frame'].ravel()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory_replay = ReplayMemory(10000)
        self.device = device
        self.num_steps = num_steps
        
        self.memory_replay.load_from_disk("memory.json", device)

        
        
        
        
    def start(self,observation):
        state = np.concatenate((observation['frame'].ravel(),np.array([observation['objects_held']]))) #observation['frame'].ravel()
        self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        torch.save(self.target_net.state_dict(), 'target_net') #Save model
        self.last_high_action = self.select_action(self.state,0)
        
        return self.last_high_action
        
    def check_bounds(self, action_index, location, occupancy_map, num_cells):

        if action_index == 0:  # Up
            if location[0] < occupancy_map.shape[0] - 1:
                location[0] += 1*num_cells
        elif action_index == 3:  # Right
            if location[1] > 0:
                location[1] -= 1*num_cells
        elif action_index == 1:  # Down
            if location[0] > 0:
                location[0] -= 1*num_cells
        elif action_index == 2:  # Left
            if location[1] < occupancy_map.shape[1] - 1:
                location[1] += 1*num_cells
        elif action_index == 4:  # Up Right
            if location[0] < occupancy_map.shape[0] - 1 and location[1] > 0:
                location += [1*num_cells, -1*num_cells]
        elif action_index == 5:  # Up Left
            if location[0] < occupancy_map.shape[0] - \
                    1 and location[1] < occupancy_map.shape[1] - 1:
                location += [1*num_cells, 1*num_cells]
        elif action_index == 6:  # Down Right
            if location[0] > 0 and location[1] > 0:
                location += [-1*num_cells, -1*num_cells]
        elif action_index == 7:  # Down Left
            if location[0] > 0 and location[1] < occupancy_map.shape[1] - 1:
                location += [-1*num_cells, 1*num_cells]


        return location
        
    def control(self, reward, terminated, truncated, robotState, action, step_count, ego_location):
    
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(np.concatenate((robotState.latest_map.ravel(),np.array([robotState.object_held]))), dtype=torch.float32, device=self.device).unsqueeze(0)

        # Store the transition in memory

        self.memory_replay.push(self.state, self.last_high_action, next_state, reward)

        # Move to the next state
        self.state = next_state

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        if done or step_count == self.num_steps:
            self.episode_durations.append(step_count)
            return -1
            
        future_action = self.select_action(self.state, step_count)
        #action["action"] = int(input("Next action > "))

        
        self.last_high_action = future_action

        print(Action(future_action), step_count) 
        if future_action <= Action.move_down_left.value:
            new_location = self.check_bounds(future_action, np.array([ego_location[0][0],ego_location[1][0]]), robotState.latest_map, 1)
            print(ego_location, new_location, robotState.latest_map[new_location[0],new_location[1]])
            if robotState.latest_map[new_location[0],new_location[1]] > 0:
                print("can't move")
                future_action = Action.get_occupancy_map.value
                
        return future_action
                
    def select_action(self,state,steps_done):

        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                return int(self.policy_net(state).max(1)[1][0]) #policy_net(state).max(1)[1].view(1, 1)
        else:
            return np.random.randint(0,Action.drop_object.value+1) #env.action_space.sample()["action"] #torch.tensor([[env.action_space.sample()["action"]]], device=device, dtype=torch.long)

    def optimize_model(self):

        if len(self.memory_replay) < self.BATCH_SIZE:
            return

        transitions = self.memory_replay.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))


        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)

        batch_action = []
        for action in batch.action:
            batch_action.append(torch.tensor([[action]], device=self.device, dtype=torch.long))
        
        
        
        action_batch = torch.cat(batch_action)
        reward_batch = torch.cat(batch.reward)



        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


