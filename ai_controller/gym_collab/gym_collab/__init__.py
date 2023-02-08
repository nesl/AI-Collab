from gymnasium.envs.registration import register

register(
     id="gym_collab/AICollabWorld-v0",
     entry_point="gym_collab.envs:AICollabEnv",
)
