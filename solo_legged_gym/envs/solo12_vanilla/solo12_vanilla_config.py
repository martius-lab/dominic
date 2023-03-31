from solo_legged_gym.envs import BaseEnvCfg, BaseTrainCfg


class Solo12VanillaEnvCfg(BaseEnvCfg):
    seed = 42

    class env(BaseEnvCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12
        episode_length_s = 20  # episode length in seconds
        play = False

    class commands(BaseEnvCfg.commands):
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw
        change_commands = False
        change_interval_s = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]

    class init_state(BaseEnvCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_HAA": 0.0,
            "HL_HAA": 0.0,
            "FR_HAA": 0.0,
            "HR_HAA": 0.0,
            "FL_HFE": 0.8,
            "HL_HFE": -0.8,
            "FR_HFE": 0.8,
            "HR_HFE": -0.8,
            "FL_KFE": -1.4,
            "HL_KFE": 1.4,
            "FR_KFE": -1.4,
            "HR_KFE": 1.4,
        }

    class control(BaseEnvCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {"HAA": 5.0, "HFE": 5.0, "KFE": 5.0}  # [N*m/rad] # TODO?
        damping = {"HAA": 0.1, "HFE": 0.1, "KFE": 0.1}  # [N*m*s/rad]
        torque_limits = 2.5  # TODO?
        dof_vel_limits = 10.0  # TODO?
        scale_joint_target = 0.25
        clip_joint_target = 100.
        decimation = 4

    class asset(BaseEnvCfg.asset):
        file = '{root}/resources/robots/solo12/urdf/solo12.urdf'
        name = "solo12"
        foot_name = "FOOT"
        terminate_after_contacts_on = ["base", "SHOULDER", "UPPER"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

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
            lin_vel_x = ["task", 0.2]
            lin_vel_y = ["task", 0.2]
            ang_vel_z = ["task", 0.2]

            lin_z = ["pose", 0.1]
            lin_vel_z = ["pose", 1.0]
            ang_xy = ["pose", 0.3]
            ang_vel_xy = ["pose", 3.0]

            joint_targets_rate = ["regularizer", 1.0]
            # stand_still = ["regularizer", 1.0]
            # dof_acc = ["regularizer", 0.1]
            # dof_vel

            # feet_air_time = ["feet", None]

            # collision
            # torques

        class scales:
            task = 1.5
            pose = 0.5
            regularizer = 0.5
            # feet = 0.2

        base_height_target = 0.25

    class viewer(BaseEnvCfg.viewer):
        overview = True

    class observations:
        add_noise = False
        noise_level = 1.0  # scales other values
        clip_observations = 100.

        class noise_scales:
            class scales:
                dof_pos = 0.01
                dof_vel = 0.1
                lin_vel = 0.2
                ang_vel = 0.05
                gravity = 0.05


class Solo12VanillaTrainCfg(BaseTrainCfg):
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
        max_iterations = 1000  # number of policy updates
        normalize_observation = True  # it will make the training much faster

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'solo12_vanilla'
        run_name = 'baseline'

        # load
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model

