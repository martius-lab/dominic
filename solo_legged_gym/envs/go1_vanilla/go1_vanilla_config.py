from solo_legged_gym.envs import BaseEnvCfg, BaseTrainCfg
import numpy as np


class Go1VanillaEnvCfg(BaseEnvCfg):
    seed = 42

    class env(BaseEnvCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12
        episode_length_s = 20  # episode length in seconds

    class commands(BaseEnvCfg.commands):
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw
        change_commands = False
        change_interval_s = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]

    class init_state(BaseEnvCfg.init_state):
        pos = [0.0, 0.0, 0.40]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(BaseEnvCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {'joint': 20}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        action_scale = 0.25
        clip_actions = 100.
        decimation = 4

    class asset(BaseEnvCfg.asset):
        file = '{root}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand(BaseEnvCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False
        push_interval_s = 15
        max_push_vel_xy = 1.

    class rewards(BaseEnvCfg.rewards):
        class terms:  # [group, sigma]
            lin_vel_x = ["task", np.sqrt(0.05)]
            lin_vel_y = ["task", np.sqrt(0.05)]
            ang_vel_z = ["task", np.sqrt(0.2)]

            base_height = ["tracking", np.sqrt(0.05)]
            lin_vel_z = ["tracking", np.sqrt(0.05)]
            ang_vel_xy = ["tracking", np.sqrt(4.0)]

            action_rate = ["regularizer", np.sqrt(15.0)]

        class scales:
            task = 1.0
            tracking = 0.2
            regularizer = 0.2

        base_height_target = 0.37

    class observations:
        add_noise = False
        noise_level = 1.0  # scales other values
        clip_observations = 100.

        class noise_scales:
            class scales:
                dof_pos = 0.01
                # dof_vel = 0.1
                lin_vel = 0.2
                ang_vel = 0.05
                # gravity = 0.05


class Go1VanillaTrainCfg(BaseTrainCfg):
    runner_class_name = 'OnPolicyRunner'

    class policy(BaseTrainCfg.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm(BaseTrainCfg.algorithm):
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner(BaseTrainCfg.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates
        normalize_observation = True  # it will make the training much faster

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'go1_vanilla'
        run_name = 'test'

        # load
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model

