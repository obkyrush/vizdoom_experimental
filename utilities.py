import numpy as np
import cv2
import imageio
import pygifsicle


def test_agent(env, agent, n_episodes=1, deterministic=True):
    total_reward = 0
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _state = agent.predict(state, deterministic=deterministic)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
        total_reward += ep_reward
    return total_reward / n_episodes


def make_gif(agent, filepath, deterministic=True, iteration=None, reward=None, include_compressed=True):
    images = []
    obs = agent.env.reset()
    if include_compressed:
        mode = 'rgb_array_with_compressed'
    else:
        mode = 'rgb_array'
    img = agent.env.render(mode=mode)
    if iteration:
        img = cv2.putText(img=np.copy(img), text=f'iteration {iteration}', org=(20, 15),
                          fontFace=cv2.FONT_HERSHEY_DUPLEX,
                          fontScale=0.7,
                          color=(255, 255, 255),
                          thickness=1)
    if reward:
        img = cv2.putText(img=np.copy(img), text=f'mean reward = {reward:.3f}', org=(20, 40),
                          fontFace=cv2.FONT_HERSHEY_DUPLEX,
                          fontScale=0.7,
                          color=(255, 255, 255),
                          thickness=1)

    done = False
    while not done:
        images.append(img)
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, r, done, _ = agent.env.step(action)
        img = agent.env.render(mode=mode)
        if iteration:
            img = cv2.putText(img=np.copy(img), text=f'iteration {iteration}', org=(20, 15),
                              fontFace=cv2.FONT_HERSHEY_DUPLEX,
                              fontScale=0.7,
                              color=(255, 255, 255),
                              thickness=1)
        if reward:
            img = cv2.putText(img=np.copy(img), text=f'mean reward = {reward:.3f}', org=(20, 40),
                              fontFace=cv2.FONT_HERSHEY_DUPLEX,
                              fontScale=0.7,
                              color=(255, 255, 255),
                              thickness=1)
    images.append(img)
    imageio.mimsave(filepath, [np.array(img) for i, img in enumerate(images) if i % 1 == 0], fps=29)
    pygifsicle.optimize(filepath, colors=128)
