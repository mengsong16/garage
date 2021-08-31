"""This module implements a Hindsight Experience Replay (HER).

See: https://arxiv.org/abs/1707.01495.
"""
import copy

import numpy as np

from garage.replay_buffer.path_buffer import PathBuffer

from garage import StepType, TimeStepBatch


class MyHERReplayBuffer(PathBuffer):
    """Replay buffer for HER (Hindsight Experience Replay).

    It constructs hindsight examples using future strategy.

    Args:
        replay_k (int): Number of HER transitions to add for each regular
            Transition. Setting this to 0 means that no HER replays will
            be added.
        reward_fn (callable): Function to re-compute the reward with
            substituted goals.
        capacity_in_transitions (int): total size of transitions in the buffer.
        env_spec (EnvSpec): Environment specification.
    """

    def __init__(self, replay_k, reward_fn, capacity_in_transitions, env_spec):
        self._replay_k = replay_k
        self._reward_fn = reward_fn

        if not float(replay_k).is_integer() or replay_k < 0:
            raise ValueError('replay_k must be an integer and >= 0.')
        super().__init__(capacity_in_transitions, env_spec)

    def _sample_her_goals(self, path, transition_idx):
        """Samples HER goals from the given path.

        Goals are randomly sampled starting from the index after
        transition_idx in the given path.

        Args:
            path (dict[str, np.ndarray]): A dict containing the transition
                keys, where each key contains an ndarray of shape
                :math:`(T, S^*)`.
            transition_idx (int): index of the current transition. Only
                transitions after the current transitions will be randomly
                sampled for HER goals.

        Returns:
            np.ndarray: A numpy array of HER goals with shape
                (replay_k, goal_dim).

        """
        goal_indexes = np.random.randint(transition_idx + 1,
                                         len(path['observations']),
                                         size=self._replay_k)
        return [
            goal['achieved_goal']
            for goal in np.asarray(path['observations'])[goal_indexes]
        ]

    # still have three keys: observation, desired_goal, achieved_goal
    def _flatten_dicts(self, path):
        for key in ['observations', 'next_observations']:
            if not isinstance(path[key], dict):
                path[key] = self._env_spec.observation_space.flatten_n(
                    path[key])
            else:
                path[key] = self._env_spec.observation_space.flatten(path[key])

    def add_episode_batch(self, episodes):
        """Add a EpisodeBatch to the buffer.

        Args:
            episodes (EpisodeBatch): Episodes to add.

        """
        if self._env_spec is None:
            self._env_spec = episodes.env_spec
        env_spec = episodes.env_spec
        obs_space = env_spec.observation_space

        # keep observations as dict
        for eps in episodes.split():
            terminals = np.array([
                step_type == StepType.TERMINAL for step_type in eps.step_types
            ],
                                 dtype=bool)
            path = {
                'observations': eps.observations,
                'next_observations': eps.next_observations,
                'actions': env_spec.action_space.flatten_n(eps.actions),
                'rewards': eps.rewards.reshape(-1, 1),
                'terminals': terminals.reshape(-1, 1),
            }
            self.add_path(path)

    def add_path(self, path):
        """Adds a path to the replay buffer.

        For each transition in the given path except the last one,
        replay_k HER transitions will added to the buffer in addition
        to the one in the path. The last transition is added without
        sampling additional HER goals.

        Args:
            path(dict[str, np.ndarray]): Each key in the dict must map
                to a np.ndarray of shape :math:`(T, S^*)`.

        """
        obs_space = self._env_spec.observation_space
        # make observation dict again
        # have three keys: observation, desired_goal, achieved_goal
        if not isinstance(path['observations'][0], dict):
            # unflatten dicts if they've been flattened
            path['observations'] = obs_space.unflatten_n(path['observations'])
            path['next_observations'] = (obs_space.unflatten_n(
                path['next_observations']))

        # create HER transitions and add them to the buffer
        #print(path['actions'].shape)
        #print(path['observations'].shape)
        #print(path)
        #print('-------------------')
        for idx in range(path['actions'].shape[0] - 1):
            print("-----here-----")

            transition = {key: sample[idx] for key, sample in path.items()}
            her_goals = self._sample_her_goals(path, idx)

            # create replay_k transitions using the HER goals
            for goal in her_goals:
                # goal relabel
                t_new = copy.deepcopy(transition)
                a_g = t_new['next_observations']['achieved_goal']

                t_new['rewards'] = np.array(self._reward_fn(a_g, goal, None))
                t_new['observations']['desired_goal'] = goal
                t_new['next_observations']['desired_goal'] = copy.deepcopy(goal)
                t_new['terminals'] = np.array(False)

                # flatten the transition from dict to numpy array
                self._flatten_dicts(t_new)
                # flatten to one dim vector
                for key in t_new.keys():
                    t_new[key] = t_new[key].reshape(1, -1)

                # Since we're using a PathBuffer, add each new transition
                # as its own path.
                super().add_path(t_new)

        
        # flatten the path from dict to numpy array
        self._flatten_dicts(path)
        super().add_path(path)

    def concat_obs_goal(self, obs_space, obs):
        return obs_space.flatten_with_keys(obs, keys=['observation', 'desired_goal'])

    # for learning
    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        idx = np.random.randint(self._transitions_stored, size=batch_size)

        transitions = {}

        obs_space = self._env_spec.observation_space

        for key, buf_arr in self._buffer.items():
            transitions[key] = buf_arr[idx]

            if key == 'observations':
                #print('------------------------')
                #print(transitions['observations'].shape)

                if not isinstance(transitions['observations'][0], dict):
                    dict_transitions_observations = obs_space.unflatten_n(transitions['observations'])
                else:
                    dict_transitions_observations = transitions['observations']
                
                #print(obs_space)
                concat_transitions_observations = np.array([self.concat_obs_goal(obs_space, trans) for trans in dict_transitions_observations])
                
                #print(concat_transitions_observations.shape)
                #print('------------------------')
                transitions['observations'] = concat_transitions_observations
            elif key == 'next_observations':
                if not isinstance(transitions['next_observations'][0], dict):
                    dict_transitions_next_observations = obs_space.unflatten_n(transitions['next_observations'])
                else:
                    dict_transitions_next_observations = transitions['next_observations']
                
                #print(obs_space)
                concat_transitions_next_observations = np.array([self.concat_obs_goal(obs_space, trans) for trans in dict_transitions_next_observations])
                
                #print(concat_transitions_next_observations.shape)
                #print('------------------------')
                transitions['next_observations'] = concat_transitions_next_observations

        #print(transitions)

        return transitions 

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self.__dict__ = state
