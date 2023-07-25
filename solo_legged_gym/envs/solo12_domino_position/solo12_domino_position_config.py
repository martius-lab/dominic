from solo_legged_gym.envs import BaseEnvCfg
import numpy as np


class Solo12DOMINOPositionEnvCfg(BaseEnvCfg):
    seed = 42

    class env(BaseEnvCfg.env):
        num_envs = 4096
        num_observations = 30 + 9 * 9 + 12 + 3 + 1  # #states + #height + #actions + #commands + #remaining time
        num_skills = 5  # latent space
        num_actions = 12
        num_features = 7
        episode_length_s = 6  # episode length in seconds
        remaining_check_time_s = 1

        play = False
        plot_target = True

    class terrain:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        mesh_type = 'trimesh'  # plane, heightfield, trimesh

        measure_height = True  # measure the height samples
        measured_points_x = list((np.arange(9) - (9-1) / 2) / 10)
        measured_points_y = list((np.arange(9) - (9-1) / 2) / 10)

        # all below are only used for heightfield and trimesh
        # sub-terrain
        terrain_length = 6.  # [m]
        terrain_width = 6.  # [m]

        init_range = 1.0  # [m]

        num_rows = 20
        num_cols = 7

        border_size = 2  # [m]

        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        slope_threshold = 0.5  # slopes above this threshold will be corrected to vertical surfaces

        train_all_together = 1  # 0: train all together, 1: train curriculum, 2: train jump curriculum

        # choose the type of the terrain, check the params in isaacgym.terrain_utils or utils.terrain
        # pass the params as a dict
        # random_uniform, sloped, pyramid_sloped, discrete_obstacles, wave, stairs, pyramid_stairs,
        # stepping_stones, gap, pit
        type = "special_box"
        params = list(np.arange(7) * 0.05)
        # params = list(np.zeros(5))
        # params = list(np.ones(5) * 0.1)

    class viewer(BaseEnvCfg.viewer):
        overview = True
        ref_pos_b = [1, 1, 0.4]
        record_camera_imgs = True
        overview_pos = [-5, -5, 10]  # [m]
        overview_lookat = [5, 5, 1]  # [m]

    class commands(BaseEnvCfg.commands):
        num_commands = 3  # default: target in x, y, z in base

        class ranges:
            radius = [1.0, 3.0]  # [m]
            direction = [-np.pi, np.pi]  # [rad]
            # yaw = [-np.pi, np.pi]  # [rad]

    class init_state(BaseEnvCfg.init_state):
        pos = [0.0, 0.0, 0.4]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            "FL_HAA": 0.0,
            "HL_HAA": 0.0,
            "FR_HAA": 0.0,
            "HR_HAA": 0.0,
            "FL_HFE": np.pi / 4,
            "HL_HFE": -np.pi / 4,
            "FR_HFE": np.pi / 4,
            "HR_HFE": -np.pi / 4,
            "FL_KFE": -np.pi / 2,
            "HL_KFE": np.pi / 2,
            "FR_KFE": -np.pi / 2,
            "HR_KFE": np.pi / 2,
        }

    class control(BaseEnvCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {"HAA": 5.0, "HFE": 5.0, "KFE": 5.0}  # [N*m/rad]
        damping = {"HAA": 0.1, "HFE": 0.1, "KFE": 0.1}  # [N*m*s/rad]
        torque_limits = 5.0
        # scale_joint_target = [np.pi / 4, np.pi / 4, np.pi / 2,
        #                       np.pi / 4, np.pi / 4, np.pi / 2,
        #                       np.pi / 4, np.pi / 4, np.pi / 2,
        #                       np.pi / 4, np.pi / 4, np.pi / 2]
        scale_joint_target = 0.25
        clip_joint_target = 100.0
        # dof sequence:
        # FL_HAA, FL_HFE, FL_KFE,
        # FR_HAA, FR_HFE, FR_KFE,
        # HL_HAA, HL_HFE, HL_KFE,
        # HR_HAA, HR_HFE, HR_KFE
        decimation = 4

    class asset(BaseEnvCfg.asset):
        file = '{root}/resources/robots/solo12/urdf/solo12.urdf'
        name = "solo12"
        foot_name = "FOOT"
        penalize_contacts_on = ["base", "SHOULDER", "UPPER"]
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class domain_rand(BaseEnvCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.5]

        randomize_base_mass = True
        added_mass_range = [-0.5, 0.5]

        push_robots = True
        push_interval_s = 2
        max_push_vel_xyz = 0.5
        max_push_avel_xyz = 0.5

        actuator_lag = True
        randomize_actuator_lag = False
        actuator_lag_steps = 3  # the lag simulated would be actuator_lag_steps * dt / decimation

    class rewards(BaseEnvCfg.rewards):
        class terms:  # [group, sigma]
            # ang_xy = "[2, 0.1]"
            # ang_vel_xy = "[2, 2.0]"

            # feet_slip = "[2, [0.08, 0.1, 0.2]]"  # target height, sigma, sigma+
            # stall_yaw = "[0, [0.1, 0.1, 0.2]]"  # minimal ang vel, yaw distance, distance, sigma

            # pos = "[1, 0.5]"  # sigma
            # yaw = "[1, 0.5]"  # sigma
            # posl = "[1, 5.0]"  # max error
            posi = "[1, 1.0]"  # scale of the error
            # yawl = f"[1, [{np.pi}, 0.25]]"  # max error
            # pos_yaw = "[1, [0.5, 0.5, 0.25]]"  # sigma

            # lin_acc_z = "[2, 10]"
            # ang_acc_xy = "[2, 20]"

            joint_targets_rate = "[0, 1.5]"
            feet_acc = "[0, [800, 0.9]]"
            contact = "[0, 25]"
            stall_pos = "[0, [0.25, 0.25, 0.1]]"  # minimal vel, distance, sigma

            move_towards = "[2, 0.8]"  # clip/scale
            joint_default = "[2, [2.0, 0.8]]"

            # torques = "[0, 400]"
            # lin_z = "[0, 0.2]"
            # feet_height = "[0, [0.08, 0.2, 0.25]]"  # target height, sigma, pos threshold
            # dof_acc = "[0, 4000]"

            # lin_vel_z = "[0, 0.5]"

            # stand_still = "[0, 0.01]"
            # stand_still_h = "[0, 0.05]"
            # feet_contact_force = "[0, 20.0]"

            # feet_slip_h = "[0, [0.01, 0.01]]"
            # feet_slip_v = "[0, [0.04, 0.8]]"
            # dof_vel = "[0, 50.0]"
            # feet_air_time = "[0, None]"

        num_groups = 3

        base_height_target = 0.25
        base_height_danger = 0.1

    class observations:
        clip_obs = True
        clip_limit = 100.
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            lin_vel = 0.2
            ang_vel = 0.2
            gravity = 0.1
            height_measurements = 0.05


class Solo12DOMINOPositionTrainCfg:
    algorithm_name = 'DOMINO'

    class network:
        init_log_std = 0.5
        drop_out_rate = 0.5
        policy_hidden_dims = [256, 128]
        policy_activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        value_hidden_dims = [256, 128]
        value_activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        succ_feat_hidden_dims = [256, 128]
        succ_feat_activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # algorithm params
        bootstrap_value = False
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.03
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * num_steps / num_minibatches

        policy_lr = 1.e-3  # 5.e-4
        desired_kl = 0.02  # adjust the learning rate automatically
        schedule = 'fixed'  # adaptive, fixed

        value_lr = 1.e-3  # 1.e-3

        fixed_adv_coeff = '[1.5, 1.0, 1.0]'
        intrinsic_adv_coeff = 1.0
        intrinsic_rew_scale = 5.0

        gamma = 0.99  # discount factor
        lam = 0.95  # GAE coeff
        max_grad_norm = 1.

        num_lagrange_steps = 10
        lagrange_lr = 1.e-3
        sigmoid_scale = 1.0  # larger smoother, smaller more like on/off switch?
        clip_lagrange = 'auto_2'  # None, float, 'auto' = 5 / sigmoid_scale, 'auto_a' = a / sigmoid_scale

        alpha = 0.7  # optimality ratio

        avg_values_decay_factor = 0.99
        avg_features_decay_factor = 0.999

        target_dist = 1.0  # l_0 in VDW force
        attractive_power = 3
        repulsive_power = 0
        attractive_coeff = 0

        use_succ_feat = True
        succ_feat_gamma = 0.95
        succ_feat_lr = 1.e-3

        burning_expert_steps = 100

    class runner:
        max_iterations = 1500  # number of policy updates

        num_steps_per_env = 48  # per iteration
        normalize_observation = True  # it will make the training much faster
        normalize_features = True

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'solo12_domino_position'
        run_name = 'diversity4'

        # load
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model

        record_gif = True  # need to enable env.viewer.record_camera_imgs and run with wandb
        record_gif_interval = 50
        record_iters = 10  # should be int * num_st   eps_per_env

        wandb = False  # by default is false, set to true from command line
        wandb_group = 'test'
