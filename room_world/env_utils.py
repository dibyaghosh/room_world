from gym.spaces import Dict, Box
import gym
import numpy as np
from multiworld.core.serializable import Serializable


try:
    from rllab.envs.base import Env, Step
    from rllab.misc import logger
    from rllab.envs.gym_env import GymEnv, convert_gym_space
    from rllab.envs.proxy_env import ProxyEnv

    def get_inner_env(env):
        if isinstance(env, ProxyEnv):
            return get_inner_env(env.wrapped_env)
        elif isinstance(env, GymEnv):
            return get_inner_env(env.env)
        elif isinstance(env, gym.Wrapper):
            return get_inner_env(env.env)
        return env


    class GymEnvWrapper(Env, Serializable):
        """
            GYM -> RLLAB Wrapper
        """
        def __init__(self, env):
            Serializable.quick_init(self, locals())
            self.env = env
            self._observation_space = convert_gym_space(env.observation_space)
            self._action_space = convert_gym_space(env.action_space)
            self._horizon = 500
        
        @property
        def observation_space(self):
            return self._observation_space

        @property
        def action_space(self):
            return self._action_space

        @property
        def horizon(self):
            return self._horizon

        def reset(self):
            return self.env.reset()

        def step(self, action):
            next_obs, reward, done, info = self.env.step(action)
            return Step(next_obs, reward, done, **info)

        def render(self):
            return self.env.render()

        def log_diagnostics(self, paths):
            return get_inner_env(self.env).log_diagnostics(paths)

        def get_param_values(self):
            return self.env.get_param_values()
    
        def set_param_values(self, params):
            return self.env.set_param_values(params)

except ImportError:
    print("You do not have RLLAB :(")

class MultiWorldEnvWrapper(gym.Env, Serializable):
    """
        MULTIWORLD -> GYM Wrapper
    """
    def __init__(
            self,
            env,
            obs_keys=None,
            goal_keys=None,
            append_goal_to_obs=True,
    ):
        Serializable.quick_init(self, locals())
        self.env = env

        if obs_keys is None:
            obs_keys = ['observation']
        if goal_keys is None:
            goal_keys = ['desired_goal']
        if append_goal_to_obs:
            obs_keys += goal_keys
        for k in obs_keys:
            assert k in self.env.observation_space.spaces
        
        assert isinstance(self.env.observation_space, Dict)

        self.obs_keys = obs_keys
        self.goal_keys = goal_keys
        # TODO: handle nested dict
        self.observation_space = Box(
            np.hstack([
                self.env.observation_space.spaces[k].low
                for k in obs_keys
            ]),
            np.hstack([
                self.env.observation_space.spaces[k].high
                for k in obs_keys
            ]),
        )
        self.action_space = self.env.action_space


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        flat_obs = np.hstack([obs[k] for k in self.obs_keys])
        return flat_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return np.hstack([obs[k] for k in self.obs_keys])
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def log_diagnostics(self, paths, **kwargs):

        logger2 = kwargs.get('logger', logger)
        stats = self.env.get_diagnostics(paths)
        for k, v in stats.items():
            logger2.record_tabular(k, v)

    @property
    def wrapped_env(self):
        return self.env

    def __getattr__(self, attrname):
        if attrname == '_serializable_initialized':
            return None
        return getattr(self.env, attrname)

    def get_param_values(self):
        if hasattr(self.env, 'get_param_values'):
            return self.env.get_param_values()
        else:
            return dict()

    def set_param_values(self, params):
        if hasattr(self.env, 'set_param_values'):
            return self.env.set_param_values(params)
        else:
            return
