import os
from stable_baselines3 import PPO
from env_wrapper import DoomEnv
from feature_extractor import CustomCNN
from utilities import test_agent, make_gif


N_ITER = 100
TIMESTEPS_PER_ITER = 2000
ENV_NAME = 'deadly_corridor'
MODEL_NAME = 'PPO_sb3'
INCLUDE_COMPRESSED_GIF = True
GIF_PATH = f'./gifs/{ENV_NAME}'

reward_history = []
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=32),
)

env = DoomEnv(False, frameskips=3, scenario=ENV_NAME)
model = PPO('CnnPolicy', env, ent_coef=0.1, policy_kwargs=policy_kwargs, verbose=1)

if not os.path.exists(GIF_PATH):
    os.makedirs(GIF_PATH)

for it in range(N_ITER):
    model.learn(total_timesteps=TIMESTEPS_PER_ITER, log_interval=None)
    m_reward = test_agent(env, model, n_episodes=50)
    make_gif(model, f'{GIF_PATH}/{MODEL_NAME}_frame_{(it+1) * TIMESTEPS_PER_ITER}.gif',
             iteration=it+1, reward=m_reward,
             include_compressed=INCLUDE_COMPRESSED_GIF)
    print(f'iteration {it + 1}/{N_ITER}, reward = {m_reward:.3f}')
    reward_history.append(m_reward)

env.close()
print('training done!')
model.save('mdl')
