"""Configuration for imitation.src.imitation.scripts.train_adversarial."""

import sacred

from imitation.src.imitation.rewards import reward_nets
from imitation.src.imitation.scripts.ingredients import demonstrations, environment, expert
from imitation.src.imitation.scripts.ingredients import logging as logging_ingredient
from imitation.src.imitation.scripts.ingredients import policy_evaluation, reward, rl

train_adversarial_ex = sacred.Experiment(
    "train_adversarial",
    ingredients=[
        logging_ingredient.logging_ingredient,
        demonstrations.demonstrations_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        expert.expert_ingredient,
        environment.environment_ingredient,
        policy_evaluation.policy_evaluation_ingredient,
    ],
)


@train_adversarial_ex.config
def defaults():
    show_config = False

    total_timesteps = int(1e6)  # Num of environment transitions to sample
    algorithm_kwargs = dict(
        demo_batch_size=1024,  # Number of expert samples per discriminator update
        n_disc_updates_per_round=4,  # Num discriminator updates per generator round
    )
    algorithm_specific = {}  # algorithm_specific[algorithm] is merged with config

    checkpoint_interval = 0  # Num epochs between checkpoints (<0 disables)
    agent_path = None  # Path to load agent from, optional.


@train_adversarial_ex.config
def aliases_default_gen_batch_size(algorithm_kwargs, rl):
    # Setting generator buffer capacity and discriminator batch size to
    # the same number is equivalent to not using a replay buffer at all.
    # "Disabling" the replay buffer seems to improve convergence speed, but may
    # come at a cost of stability.
    algorithm_kwargs["gen_replay_buffer_capacity"] = rl["batch_size"]


# Shared settings

MUJOCO_SHARED_LOCALS = dict(rl=dict(rl_kwargs=dict(ent_coef=0.1)))

ANT_SHARED_LOCALS = dict(
    total_timesteps=int(3e7),
    algorithm_kwargs=dict(shared=dict(demo_batch_size=8192)),
    rl=dict(batch_size=16384),
)


# Classic RL Gym environment named configs


@train_adversarial_ex.named_config
def acrobot():
    environment = dict(gym_id="Acrobot-v1")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def cartpole():
    environment = dict(gym_id="CartPole-v1")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_cartpole():
    environment = dict(gym_id="seals/CartPole-v0")
    total_timesteps = int(1.4e6)


@train_adversarial_ex.named_config
def mountain_car():
    environment = dict(gym_id="MountainCar-v0")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_mountain_car():
    environment = dict(gym_id="seals/MountainCar-v0")


@train_adversarial_ex.named_config
def pendulum():
    environment = dict(gym_id="Pendulum-v1")


# Standard MuJoCo Gym environment named configs


@train_adversarial_ex.named_config
def seals_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    environment = dict(gym_id="seals/Ant-v0")


CHEETAH_SHARED_LOCALS = dict(
    MUJOCO_SHARED_LOCALS,
    rl=dict(batch_size=16384, rl_kwargs=dict(batch_size=1024)),
    algorithm_specific=dict(
        airl=dict(total_timesteps=int(5e6)),
        gail=dict(total_timesteps=int(8e6)),
    ),
    reward=dict(
        algorithm_specific=dict(
            airl=dict(
                net_cls=reward_nets.BasicShapedRewardNet,
                net_kwargs=dict(
                    reward_hid_sizes=(32,),
                    potential_hid_sizes=(32,),
                ),
            ),
        ),
    ),
    algorithm_kwargs=dict(
        # Number of discriminator updates after each round of generator updates
        n_disc_updates_per_round=16,
        # Equivalent to no replay buffer if batch size is the same
        gen_replay_buffer_capacity=16384,
        demo_batch_size=8192,
    ),
)


@train_adversarial_ex.named_config
def half_cheetah():
    locals().update(**CHEETAH_SHARED_LOCALS)
    environment = dict(gym_id="HalfCheetah-v2")


@train_adversarial_ex.named_config
def seals_half_cheetah():
    locals().update(**CHEETAH_SHARED_LOCALS)
    environment = dict(gym_id="seals/HalfCheetah-v0")


@train_adversarial_ex.named_config
def seals_hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="seals/Hopper-v0")


@train_adversarial_ex.named_config
def seals_humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="seals/Humanoid-v0")
    total_timesteps = int(4e6)


@train_adversarial_ex.named_config
def reacher():
    environment = dict(gym_id="Reacher-v2")
    algorithm_kwargs = {"allow_variable_horizon": True}


@train_adversarial_ex.named_config
def seals_swimmer():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="seals/Swimmer-v0")
    total_timesteps = int(2e6)


@train_adversarial_ex.named_config
def seals_walker():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="seals/Walker2d-v0")


# Debug configs


@train_adversarial_ex.named_config
def fast():
    # Minimize the amount of computation. Useful for test cases.

    # Need a minimum of 10 total_timesteps for adversarial training code to pass
    # "any update happened" assertion inside training loop.
    total_timesteps = 10
    algorithm_kwargs = dict(
        demo_batch_size=1,
        n_disc_updates_per_round=4,
    )
