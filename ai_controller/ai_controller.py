import gym_collab
import gymnasium as gym
import time
import argparse
from collections import defaultdict
import numpy as np
import pdb
import sys

sys.path.append("gym_collab/gym_collab/envs/")
from action import Action

parser = argparse.ArgumentParser(
    description="WebRTC audio / video / data-channels demo"
)
parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
parser.add_argument(
    "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
)
parser.add_argument(
    "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
)
parser.add_argument("--record-to", help="Write received media to a file."),
parser.add_argument("--verbose", "-v", action="count")
parser.add_argument("--use-occupancy", action='store_true', help="Use occupancy maps instead of images")
parser.add_argument("--address", default='https://172.17.15.69:4000', help="Address where our simulation is running")
parser.add_argument("--robot-number", default=1, help="Robot number to control")
parser.add_argument("--view-radius", default=0, help="When using occupancy maps, the view radius")

args = parser.parse_args()


class QLearningAgent:
    def __init__(
        self,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decay)
        
# hyperparameters
learning_rate = 0.01
n_episodes = 100
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = QLearningAgent(
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


'''
ACTIONS

move_up = 0
move_down = 1
move_left = 2
move_right = 3
move_up_right = 4
move_up_left = 5
move_down_right = 6
move_down_left = 7
grab_up = 8
grab_right = 9
grab_down = 10
grab_left = 11
grab_up_right = 12
grab_up_left = 13
grab_down_right = 14
grab_down_left = 15
drop_object = 16

danger_sensing = 17
get_occupancy_map = 18
get_objects_held = 19
check_item = 20
check_robot = 21
get_messages = 22
send_message = 23
request_item_info = 24
request_agent_info = 25

wait = 26

Action space
    {
        "action" : spaces.Discrete(len(self.Action)),
        "item" : spaces.Discrete(self.map_config['num_objects']),
        "robot" : spaces.Discrete(len(self.map_config['all_robots'])+1), #Allow for 0
        "message" : spaces.Text(min_length=0,max_length=100)
    }




Observation space
    {
        "frame" : spaces.Box(low=0, high=5, shape=(map_size, map_size), dtype=int),
        "objects_held" : spaces.Discrete(2),
        "action_status" : spaces.MultiDiscrete([2]*4)
        "item_output" : spaces.Dict(
            {
                "item_weight" : spaces.Discrete(10),
                "item_danger_level" : spaces.Discrete(3),
                "item_location" : spaces.MultiDiscrete([map_size, map_size])
            }
        ),
        "num_items" : spaces.Discrete(self.map_config['num_objects']),
        "neighbors_output" : spaces.Dict(
            {
                "neighbor_type" : spaces.Discrete(2),
                "neighbor_location" : spaces.MultiDiscrete([map_size, map_size])
            }
        
        ),
        "strength" : spaces.Discrete(len(self.map_config['all_robots'])+1), #Strength starts from zero
        "num_messages" : spaces.Discrete(100)
        
        #"objects_danger_level" : spaces.Box(low=1,high=2,shape=(self.map_config['num_objects'],), dtype=int)
    }

'''

    
def print_map(occupancy_map): #Occupancy maps require special printing so that the orientation is correct
    new_occupancy_map = occupancy_map.copy()
    for row_id in range(occupancy_map.shape[0]):
        new_occupancy_map[row_id,:] = occupancy_map[occupancy_map.shape[0]-row_id-1,:]

    new_new_occupancy_map = new_occupancy_map.copy()
    for row_id in range(occupancy_map.shape[1]): 
        new_new_occupancy_map[:,row_id] = new_occupancy_map[:,occupancy_map.shape[1]-row_id-1]
    print(new_new_occupancy_map)


env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=args.use_occupancy, view_radius=args.view_radius, client_number=int(args.robot_number), host=args.host, port=args.port, address=args.address, cert_file=args.cert_file, key_file=args.key_file)

observation, info = env.reset()
#observation, reward, terminated, truncated, info = env.step(17)

done = False

processed_observation = (tuple(map(tuple, observation['frame'])), observation['objects_held'])

print_map(observation["frame"])

next_observation = []

latest_map = observation['frame'].copy()

action_issued = [False,False]
last_action = [0,0]

while not done:

    action = env.action_space.sample()
    
    #Make sure to issue concurrent actions but not of the same type. Else, wait.
    if action["action"] < Action.danger_sensing.value and not action_issued[0]:
        action_issued[0] = True
        last_action[0] = action["action"]
        print("Locomotion", Action(action["action"]))
    elif action["action"] != Action.wait.value and action["action"] >= Action.danger_sensing.value and not action_issued[1]:
        action_issued[1] = True
        last_action[1] = action["action"]
        
        print("Sensing", Action(action["action"]))
    else:
        action["action"] = Action.wait.value


    #action = agent.get_action(processed_observation)
    
    
        
    
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    if reward != 0:
        print('Reward', reward)
        
        
    if next_observation and any(next_observation['action_status']):
        
        
        ego_location = np.where(next_observation['frame'] == 5)
        previous_ego_location = np.where(latest_map == 5)
        latest_map[previous_ego_location[0][0],previous_ego_location[1][0]] = 0
        latest_map[ego_location[0][0],ego_location[1][0]] = 5
        
        if Action(last_action[1]) == Action.get_occupancy_map: #Maintain the state of the occupancy map and update it whenever needed
            
            view_radius = int(args.view_radius)
            
            max_x = ego_location[0][0] + view_radius
            max_y = ego_location[1][0] + view_radius
            min_x = max(ego_location[0][0] - view_radius, 0)
            min_y = max(ego_location[1][0] - view_radius, 0)
            latest_map[min_x:max_x+1,min_y:max_y+1]= next_observation["frame"][min_x:max_x+1,min_y:max_y+1]
            
        print_map(latest_map)
        
        
        print(next_observation['item_output'], next_observation['objects_held'], next_observation['neighbors_output'], next_observation['strength'], next_observation['num_messages'], next_observation['num_items'], next_observation['action_status'], last_action)

        if any(next_observation['action_status'][:2]):
            action_issued[0] = False
        if any(next_observation['action_status'][2:4]):
            action_issued[1] = False
            

    #processed_next_observation = (tuple(map(tuple, next_observation['frame'])), next_observation['objects_held'], next_observation['action_status'])
    
    #agent.update(processed_observation, action, reward, terminated, processed_next_observation)

    #processed_observation = processed_next_observation

    if terminated or truncated:
        done = True


print("Closing environment")
env.close()



