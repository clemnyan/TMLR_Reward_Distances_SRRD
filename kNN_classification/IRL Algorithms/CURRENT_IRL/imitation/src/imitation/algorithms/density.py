"""Density-based baselines for imitation learning.

Each of these algorithms learns a density estimate on some aspect of the demonstrations,
then rewards the agent for following that estimate.
"""

import enum
import itertools
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, cast

import numpy as np
from gymnasium.spaces import utils as space_utils
from sklearn import neighbors, preprocessing
from stable__baselines3.stable__baselines3.common import base_class, vec_env

from imitation.src.imitation.algorithms import base
from imitation.src.imitation.data import rollout, types, wrappers
from imitation.src.imitation.rewards import reward_wrapper
from imitation.src.imitation.util import logger as imit_logger
from imitation.src.imitation.util import util


class DensityType(enum.Enum):
    """Input type the density model should use."""

    STATE_DENSITY = enum.auto()
    """Density on state s."""

    STATE_ACTION_DENSITY = enum.auto()
    """Density on (s,a) pairs."""

    STATE_STATE_DENSITY = enum.auto()
    """Density on (s,s') pairs."""


class DensityAlgorithm(base.DemonstrationAlgorithm):
    """Learns a reward function based on density modeling.

    Specifically, it constructs a non-parametric estimate of `p(s)`, `p(s,a)`, `p(s,s')`
    and then computes a reward using the log of these probabilities.
    """

    is_stationary: bool
    density_type: DensityType
    venv: vec_env.VecEnv
    transitions: Dict[Optional[int], np.ndarray]
    kernel: str
    kernel_bandwidth: float
    standardise: bool

    _scaler: Optional[preprocessing.StandardScaler]
    _density_models: Dict[Optional[int], neighbors.KernelDensity]
    rl_algo: Optional[base_class.BaseAlgorithm]
    buffering_wrapper: wrappers.BufferingWrapper
    venv_wrapped: reward_wrapper.RewardVecEnvWrapper
    wrapper_callback: reward_wrapper.WrappedRewardCallback

    def __init__(
        self,
        *,
        demonstrations: Optional[base.AnyTransitions],
        venv: vec_env.VecEnv,
        rng: np.random.Generator,
        density_type: DensityType = DensityType.STATE_ACTION_DENSITY,
        kernel: str = "gaussian",
        kernel_bandwidth: float = 0.5,
        rl_algo: Optional[base_class.BaseAlgorithm] = None,
        is_stationary: bool = True,
        standardise_inputs: bool = True,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
    ):
        """Builds DensityAlgorithm.

        Args:
            demonstrations: expert demonstration trajectories.
            density_type: type of density to train on: single state, state-action pairs,
                or state-state pairs.
            kernel: kernel to use for density estimation with `sklearn.KernelDensity`.
            kernel_bandwidth: bandwidth of kernel. If `standardise_inputs` is
                true and you are using a Gaussian kernel, then it probably makes sense
                to set this somewhere between 0.1 and 1.
            venv: The environment to learn a reward model in. We don't actually need
                any environment interaction to fit the reward model, but we use this
                to extract the observation and action space, and to train the RL
                algorithm `rl_algo` (if specified).
            rng: random state for sampling from demonstrations.
            rl_algo: An RL algorithm to train on the resulting reward model (optional).
            is_stationary: if True, share same density models for all timesteps;
                if False, use a different density model for each timestep.
                A non-stationary model is particularly likely to be useful when using
                STATE_DENSITY, to encourage agent to imitate entire trajectories, not
                just a few states that have high frequency in the demonstration dataset.
                If non-stationary, demonstrations must be trajectories, not transitions
                (which do not contain timesteps).
            standardise_inputs: if True, then the inputs to the reward model
                will be standardised to have zero mean and unit variance over the
                demonstration trajectories. Otherwise, inputs will be passed to the
                reward model with their ordinary scale.
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
        """
        self.is_stationary = is_stationary
        self.density_type = density_type
        self.venv = venv
        self.transitions = dict()
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.standardise = standardise_inputs
        self._scaler = None
        self._density_models = dict()
        self.rng = rng

        self.rl_algo = rl_algo
        self.buffering_wrapper = wrappers.BufferingWrapper(self.venv)
        self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
            self.buffering_wrapper,
            self,
        )
        self.wrapper_callback = self.venv_wrapped.make_log_callback()

    def _get_demo_from_batch(
        self,
        obs_b: np.ndarray,
        act_b: np.ndarray,
        next_obs_b: Optional[np.ndarray],
    ) -> Dict[Optional[int], List[np.ndarray]]:
        if next_obs_b is None and self.density_type == DensityType.STATE_STATE_DENSITY:
            raise ValueError(
                "STATE_STATE_DENSITY requires next_obs_b "
                "to be provided, but it was None",
            )

        assert act_b.shape[1:] == self.venv.action_space.shape
        assert obs_b.shape[1:] == self.venv.observation_space.shape
        assert len(act_b) == len(obs_b)
        if next_obs_b is not None:
            assert next_obs_b.shape[1:] == self.venv.observation_space.shape
            assert len(next_obs_b) == len(obs_b)

        if next_obs_b is not None:
            next_obs_b_iterator: Iterable = next_obs_b
        else:
            next_obs_b_iterator = itertools.repeat(None)

        transitions: Dict[Optional[int], List[np.ndarray]] = {}
        for obs, act, next_obs in zip(obs_b, act_b, next_obs_b_iterator):
            flat_trans = self._preprocess_transition(obs, act, next_obs)
            transitions.setdefault(None, []).append(flat_trans)
        return transitions

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        """Sets the demonstration data."""
        transitions: Dict[Optional[int], List[np.ndarray]] = {}

        if isinstance(demonstrations, types.TransitionsMinimal):
            next_obs_b = getattr(demonstrations, "next_obs", None)
            transitions.update(
                self._get_demo_from_batch(
                    demonstrations.obs,
                    demonstrations.acts,
                    next_obs_b,
                ),
            )
        elif isinstance(demonstrations, Iterable):
            # Inferring the correct type here is difficult with generics.
            (
                first_item,
                demonstrations,
            ) = util.get_first_iter_element(  # type: ignore[assignment]
                demonstrations,
            )
            if isinstance(first_item, types.Trajectory):
                # we assume that all elements are also types.Trajectory.
                # (this means we have timestamp information)
                # It's not perfectly type safe, but it allows for the flexibility of
                # using iterables, which is useful for large data structures.
                demonstrations = cast(Iterable[types.Trajectory], demonstrations)

                for traj in demonstrations:
                    for i, (obs, act, next_obs) in enumerate(
                        zip(traj.obs[:-1], traj.acts, traj.obs[1:]),
                    ):
                        flat_trans = self._preprocess_transition(obs, act, next_obs)
                        transitions.setdefault(i, []).append(flat_trans)
            elif isinstance(first_item, Mapping):
                # analogous to cast above.
                demonstrations = cast(Iterable[types.TransitionMapping], demonstrations)

                for batch in demonstrations:
                    transitions.update(
                        self._get_demo_from_batch(
                            util.safe_to_numpy(batch["obs"], warn=True),
                            util.safe_to_numpy(batch["acts"], warn=True),
                            util.safe_to_numpy(batch.get("next_obs"), warn=True),
                        ),
                    )
            else:
                raise TypeError(
                    f"Unsupported demonstration type {type(demonstrations)}",
                )
        else:
            raise TypeError(f"Unsupported demonstration type {type(demonstrations)}")

        self.transitions = {k: np.stack(v, axis=0) for k, v in transitions.items()}

        if not self.is_stationary and None in self.transitions:
            raise ValueError(
                "Non-stationary model incompatible with non-trajectory demonstrations.",
            )
        if self.is_stationary:
            self.transitions = {
                None: np.concatenate(list(self.transitions.values()), axis=0),
            }

    def train(self) -> None:
        """Fits the density model to demonstration data `self.transitions`."""
        # if requested, we'll scale demonstration transitions so that they have
        # zero mean and unit variance (i.e. all components are equally important)
        self._scaler = preprocessing.StandardScaler(
            with_mean=self.standardise,
            with_std=self.standardise,
        )
        flattened_dataset = np.concatenate(list(self.transitions.values()), axis=0)
        self._scaler.fit(flattened_dataset)

        # now fit density model
        self._density_models = {
            k: self._fit_density(self._scaler.transform(v))
            for k, v in self.transitions.items()
        }

    def _fit_density(self, transitions: np.ndarray) -> neighbors.KernelDensity:
        density_model = neighbors.KernelDensity(
            kernel=self.kernel,
            bandwidth=self.kernel_bandwidth,
        )
        density_model.fit(transitions)
        return density_model

    def _preprocess_transition(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: Optional[np.ndarray],
    ) -> np.ndarray:

        print("obs1 = ", obs)
        print("act1 = ", act)
        print("n_obs_1 = ", next_obs)
    
        """Compute flattened transition on subset specified by `self.density_type`."""
        if self.density_type == DensityType.STATE_DENSITY:
            flat_observations = space_utils.flatten(self.venv.observation_space, obs)
            if not isinstance(flat_observations, np.ndarray):
                raise ValueError(
                    "The density estimator only supports spaces that "
                    "flatten to a numpy array but the observation space "
                    f"flattens to {type(flat_observations)}",
                )

            return flat_observations
        elif self.density_type == DensityType.STATE_ACTION_DENSITY:
            print("im here mfs")
            flat_observation = space_utils.flatten(self.venv.observation_space, obs)
            flat_action = space_utils.flatten(self.venv.action_space, act)

            print("flat obs = ", flat_observation)
            print("flat_action = ", flat_action)

            if not isinstance(flat_observation, np.ndarray):
                raise ValueError(
                    "The density estimator only supports spaces that "
                    "flatten to a numpy array but the observation space "
                    f"flattens to {type(flat_observation)}",
                )
            if not isinstance(flat_action, np.ndarray):
                raise ValueError(
                    "The density estimator only supports spaces that "
                    "flatten to a numpy array but the action space "
                    f"flattens to {type(flat_action)}",
                )

            return np.concatenate([flat_observation, flat_action])
        elif self.density_type == DensityType.STATE_STATE_DENSITY:
            assert next_obs is not None
            flat_observation = space_utils.flatten(self.venv.observation_space, obs)
            flat_next_observation = space_utils.flatten(
                self.venv.observation_space,
                next_obs,
            )

            if not isinstance(flat_observation, np.ndarray):
                raise ValueError(
                    "The density estimator only supports spaces that "
                    "flatten to a numpy array but the observation space "
                    f"flattens to {type(flat_observation)}",
                )

            assert type(flat_observation) is type(flat_next_observation)

            return np.concatenate([flat_observation, flat_next_observation])
        else:
            raise ValueError(f"Unknown density type {self.density_type}")

    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
        steps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Compute reward from given (s,a,s') transition batch.

        This handles *batches* of observations, since it's designed to work with
        VecEnvs.

        Args:
            state: current batch of observations.
            action: batch of actions that agent took in response to those
                observations.
            next_state: batch of observations encountered after the
                agent took those actions.
            done: is it terminal state?
            steps: What timestep is this from? Used if `self.is_stationary` is false,
                otherwise ignored.

        Returns:
            Array of scalar rewards of the form `r_t(s,a,s') = \log \hat p_t(s,a,s')`
            (one for each environment), where `\log \hat p` is the underlying density
            model (and may be independent of s', a, or t, depending on options passed
            to constructor).

        Raises:
            ValueError: Non-stationary model (`self.is_stationary` false) and `steps`
                is not provided.
        """
        if not self.is_stationary and steps is None:
            raise ValueError("steps must be provided with non-stationary models")

        del done  # TODO(adam): should we handle terminal state specially in any way?

        rew_list = []
        assert len(state) == len(action) and len(state) == len(next_state)
        for idx, (obs, act, next_obs) in enumerate(zip(state, action, next_state)):

            print("obs = ", obs)
            print("act = ", act)
            print("next_obs", next_obs)

            flat_trans = self._preprocess_transition(obs, act, next_obs)
            print("flat_trans = ", flat_trans)
            assert self._scaler is not None
            scaled_padded_trans = self._scaler.transform(flat_trans[np.newaxis])
            if self.is_stationary:
                rew = self._density_models[None].score(scaled_padded_trans)
            else:
                assert steps is not None
                time = steps[idx]
                if time >= len(self._density_models):
                    # Can't do anything sensible here yet. Correct solution is to use
                    # hierarchical model in which we first check whether state is
                    # absorbing, then assign either constant score or a score based on
                    # density.
                    raise ValueError(
                        f"Time {time} out of range (0, {len(self._density_models)}], "
                        "and absorbing states not currently supported",
                    )
                else:
                    time_model = self._density_models[time]
                    rew = time_model.score(scaled_padded_trans)
            rew_list.append(rew)
        rew_array = np.asarray(rew_list, dtype="float32")
        return rew_array

    def train_policy(self, n_timesteps: int = int(1e6), **kwargs: Any) -> None:
        """Train the imitation policy for a given number of timesteps.

        Args:
            n_timesteps: number of timesteps to train the policy for.
            kwargs (dict): extra arguments that will be passed to the `learn()`
                method of the imitation RL model. Refer to Stable Baselines docs for
                details.
        """
        assert self.rl_algo is not None
        self.rl_algo.set_env(self.venv_wrapped)
        self.rl_algo.learn(
            n_timesteps,
            # ensure we can see total steps for all
            # learn() calls, not just for this call
            reset_num_timesteps=False,
            callback=self.wrapper_callback,
            **kwargs,
        )
        trajs, ep_lens = self.buffering_wrapper.pop_trajectories()
        self._check_fixed_horizon(ep_lens)

    def test_policy(self, *, n_trajectories: int = 10, true_reward: bool = True):
        """Test current imitation policy on environment & give some rollout stats.

        Args:
            n_trajectories: number of rolled-out trajectories.
            true_reward: should this use ground truth reward from underlying
                environment (True), or imitation reward (False)?

        Returns:
            dict: rollout statistics collected by
                `imitation.utils.rollout.rollout_stats()`.
        """
        trajs = rollout.generate_trajectories(
            self.rl_algo,
            self.venv if true_reward else self.venv_wrapped,
            sample_until=rollout.make_min_episodes(n_trajectories),
            rng=self.rng,
        )
        # We collect `trajs` above so disregard return value from `pop_trajectories`,
        # but still call it to clear out any saved trajectories.
        self.buffering_wrapper.pop_trajectories()
        self._check_fixed_horizon((len(traj) for traj in trajs))
        reward_stats = rollout.rollout_stats(trajs)
        return reward_stats

    @property
    def policy(self) -> base_class.BasePolicy:
        assert self.rl_algo is not None
        assert self.rl_algo.policy is not None
        return self.rl_algo.policy
