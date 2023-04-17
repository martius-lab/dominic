from solo_legged_gym.envs import BaseEnvCfg


class Solo12VanillaEnvCfg(BaseEnvCfg):
    seed = 42

    class env(BaseEnvCfg.env):
        num_envs = 4096
        num_observations = 48
        num_actions = 12
        episode_length_s = 20  # episode length in seconds
        play = False
        debug = False

    class viewer(BaseEnvCfg.viewer):
        overview = True

    class commands(BaseEnvCfg.commands):
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw
        change_commands = False
        change_interval_s = 10.  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]

    class init_state(BaseEnvCfg.init_state):
        pos = [0.0, 0.0, 0.35]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_HAA": 0.05,
            "HL_HAA": 0.05,
            "FR_HAA": -0.05,
            "HR_HAA": -0.05,
            "FL_HFE": 0.6,
            "HL_HFE": -0.6,
            "FR_HFE": 0.6,
            "HR_HFE": -0.6,
            "FL_KFE": -1.4,
            "HL_KFE": 1.4,
            "FR_KFE": -1.4,
            "HR_KFE": 1.4,
        }

    class control(BaseEnvCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {"HAA": 5.0, "HFE": 5.0, "KFE": 5.0}  # [N*m/rad]
        damping = {"HAA": 0.1, "HFE": 0.1, "KFE": 0.1}  # [N*m*s/rad]
        torque_limits = 2.5
        dof_vel_limits = 10.0  # not used anyway...
        scale_joint_target = 0.25
        clip_joint_target = 100.
        decimation = 4

    class asset(BaseEnvCfg.asset):
        file = '{root}/resources/robots/solo12/urdf/solo12.urdf'
        name = "solo12"
        foot_name = "FOOT"
        terminate_after_contacts_on = ["base", "SHOULDER"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand(BaseEnvCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]

        randomize_base_mass = False
        added_mass_range = [-0.5, 0.5]

        push_robots = False
        push_interval_s = 15
        max_push_vel_xyz = 0.5
        max_push_avel_xyz = 0.5

        actuator_lag = False
        randomize_actuator_lag = False
        actuator_lag_steps = 6  # the lag simulated would be actuator_lag_steps * dt * decimation

    class rewards(BaseEnvCfg.rewards):
        class terms:  # [group, sigma]
            lin_vel_x = ["task", 0.2]
            lin_vel_y = ["task", 0.2]
            ang_vel_z = ["task", 0.2]

            lin_z = ["task", 0.4]
            lin_vel_z = ["task", 1.0]
            ang_xy = ["task", 0.6]
            ang_vel_xy = ["task", 3.0]

            joint_default = ["task", 1.5]
            joint_targets_rate = ["task", 0.8]
            stand_still = ["task", 1.0]
            feet_slip = ["task", [0.03, 0.1]]
            # feet_slip_v = ["task", [0.03, 3.0]]
            torques = ["task", 6.0]
            # dof_acc = ["task", 1500]
            # dof_vel = ["task", 80]
            # feet_air_time = ["feet", None]

        class scales:
            task = 1.0

        base_height_target = 0.22

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


class Solo12VanillaTrainCfg:
    algorithm_name = 'PPO'

    class network:
        policy_init_noise_std = 1.0
        policy_hidden_dims = [512, 256, 128]
        policy_activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        value_hidden_dims = [512, 256, 128]
        value_activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # algorithm params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * nsteps / nminibatches
        learning_rate = 1.e-3  # 5.e-4
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
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

        wandb = False

