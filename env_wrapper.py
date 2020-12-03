import numpy as np
import gym
import cv2
from gym import spaces
from vizdoom import DoomGame


class DoomEnv(gym.Env):
    """
    Environment wrapper for Vizdoom basic.wad scenario
    """
    def __init__(self, window_visible=False, frameskips=1, scenario='basic'):
        self.scenario = scenario
        self.env = DoomGame()
        self.env.load_config(f'./scenarios/{scenario}.cfg')
        self.env.set_window_visible(window_visible)

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(32, 32, 1), dtype=np.uint8)

        def gen_actions(n):
            res = []
            for i in range(n):
                t = [0] * n
                t[i] = 1
                res.append(t)
            return res

        self._actions = gen_actions(self.action_space.n)

        self._kills_count = 0
        self._num_frameskips = frameskips
        self.env.init()

    def _get_state(self):
        state_ = self.env.get_state().screen_buffer
        state_ = cv2.cvtColor(state_, cv2.COLOR_RGB2GRAY)
        state_ = state_[96:144, :]
        state_ = cv2.resize(state_, (32, 32))
        return state_.reshape(32, 32, 1)

    @staticmethod
    def _get_state_example():
        return np.zeros((32, 32, 1))

    def reset(self):
        self._kills_count = 0
        self.env.new_episode()
        return self._get_state()

    # TODO: stack skipped frames
    def step(self, action: int):
        total_reward_ = 0

        for frame in range(self._num_frameskips):
            reward_ = self.env.make_action(self._actions[action])
            done_ = self.env.is_episode_finished()
            if self.scenario in ['deadly_corridor']:
                if not done_:
                    kills_ = self.env.get_state().game_variables[1]
                    if kills_ != self._kills_count:
                        reward_ += 500
                        self._kills_count = kills_
                reward_ *= 0.001

            total_reward_ += reward_

            if done_:
                break

        total_reward_ = total_reward_ / (frame + 1)

        info_ = {'episode': 'None',
                 'l': 1, 'r': 1}
        if done_:
            state_ = self._get_state_example()
        else:
            state_ = self._get_state()
        return state_, total_reward_, done_, info_

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.env.get_state().screen_buffer
        elif mode == 'rgb_array_with_compressed':
            img1 = self.env.get_state().screen_buffer
            img2 = self._get_state()
            img2 = cv2.resize(img2, (320, 240))
            img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
            return np.column_stack([img1, img2])
        else:
            return NotImplemented

    def close(self):
        self.env.close()
