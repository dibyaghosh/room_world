import numpy as np
from room_world.abstractions import RewardAbstraction
import room_world.rooms
from room_world.room_env import RoomEnv
from multiworld.core.serializable import Serializable
from collections import OrderedDict, Sequence
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path

class PMEnv(RoomEnv):   

    FRAME_SKIP = 5
    MAX_PATH_LENGTH = 200

    def __init__(self,
                 room=None, # Specify either room or room type
                 room_type='empty', # Choose from ['empty', 'wall', 'rooms']
                 potential_type="euclidean", # Choose from ['none' (no shaping) ,'shaped' (shortest distance between COMs), 'euclidean' (euclidean distance between states)]
                 shaped=False,
                 *args, **kwargs
                ):
        
        Serializable.quick_init(self, locals())
        self.use_images = False
        room_defaults = dict(
            empty=room_world.rooms.Room('pm', 1.2, 1.2), 
            wall=room_world.rooms.RoomWithWall('pm', 1.2, 1.2),
            rooms=room_world.rooms.FourRoom('pm', 1.2, 1.2),
            long=room_world.rooms.Room('pm', 0.3, 7.2),
        )
        if room is None:
            room = room_defaults[room_type]

        super().__init__(
            room=room,
            potential_type=potential_type,
            shaped=shaped,
            base_reward='com',
            *args, **kwargs
        )
    def preprocess(self, action):
        return action

    def _get_env_obs(self):
        return self.get_body_com("particle")[:2].copy()

    def _get_env_achieved_goal(self, obs):
        return obs

    def viewer_setup(self):

        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -90.0

    def sample_goal_joints(self):
        return np.zeros((0,))

    def get_potential(self, achieved_goal, desired_goal):
        if isinstance(self.potential_type, RewardAbstraction):
            return self.potential_type.reward(achieved_goal, desired_goal)
        elif self.potential_type == 'shaped':
            return -1 * self.room.get_shaped_distance(achieved_goal, desired_goal)
        elif self.potential_type == 'euclidean':
            return -1 * np.linalg.norm(achieved_goal-desired_goal)
        elif self.potential_type == 'none':
            return 0
        else:
            raise NotImplementedError()

    def get_base_reward(self, achieved_goal, desired_goal):
        euclidean_dist = np.linalg.norm(achieved_goal - desired_goal)
        if euclidean_dist < 0.05:
            return 0
        return -1
    
    def _get_info(self, obs):
        current_state = obs['achieved_goal']

        return dict(
            euclidean_distance=np.linalg.norm(current_state-self.goal),
            shaped_distance=self.room.get_shaped_distance(current_state, self.goal),
            position=current_state,
        )

    def _reset_to_xy(self, pos):
        
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos[0:2] = pos - self.baseline_start
        qvel[0:2] = 0

        self.set_state(qpos, qvel)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'euclidean_distance',
            'shaped_distance',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def sample_goals(self, batch_size):
        goals = np.zeros((batch_size, 2))
        goals[:] = self.possible_positions[np.random.choice(len(self.possible_positions), batch_size, replace=True)]
        transformed_goals = self.goal_embedding.get_embeddings(goals)

        return {
            'desired_goal': transformed_goals,
            'state_desired_goal': goals,
        }


if __name__ == "__main__":
    e = PMEnv(room_type='wall')
    for i in range(5):
        for j in range(100):
            e.step(e.action_space.sample())
            e.render()
        e.reset()