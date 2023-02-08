import gym_collab
import gymnasium as gym
import time
import argparse

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


env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=args.use_occupancy, view_radius=args.view_radius, client_number=int(args.robot_number), host=args.host, port=args.port, address=args.address, cert_file=args.cert_file, key_file=args.key_file)




observation, info = env.reset()
#observation, reward, terminated, truncated, info = env.step(17)


for _ in range(10):

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation["frame"][10:30,15:30])

    if terminated or truncated:
        observation, info = env.reset()


print("Closing environment")
env.close()



