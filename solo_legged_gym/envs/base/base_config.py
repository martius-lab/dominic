import inspect


class ABCConfig:
    def __init__(self) -> None:
        """ Initializes all member classes recursively. Ignores all namse starting with '__' (buit-in methods)."""
        self.init_member_classes(self)

    @staticmethod
    def init_member_classes(obj):
        # iterate over all attributes names
        for key in dir(obj):
            # disregard builtin attributes
            # if key.startswith("__"):
            if key == "__class__":
                continue
            # get the corresponding attribute object
            var = getattr(obj, key)
            # check if it the attribute is a class
            if inspect.isclass(var):
                # instantate the class
                i_var = var()
                # set the attribute to the instance instead of the type
                setattr(obj, key, i_var)
                # recursively init members of the attribute
                ABCConfig.init_member_classes(i_var)


class BaseEnvCfg(ABCConfig):
    seed = 1

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class env:
        num_envs = 1  # overwrite in the tasks
        num_observations = None  # overwrite in the tasks
        num_actions = None  # overwrite in the tasks
        env_spacing = 2.  # not used with heightfields/trimeshes
        episode_length_s = 20  # episode length in seconds
        play = False
        debug = False

    class terrain:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class commands:
        num_commands = None  # overwrite in the tasks

    class init_state:
        pos = [0.0, 0.0, 0.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = None

    class control:
        control_type = None  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = None  # [N*m/rad]
        damping = None  # [N*m*s/rad]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 1

    class asset:
        file = None
        name = None
        foot_name = None
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True  # Some .obj meshes must be flipped from y-up to z-up
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = False
        friction_range = [0.9, 1.0]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]

    class rewards:
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100.  # forces above this value are penalized

    # viewer camera:
    class viewer:
        ref_env = 0
        ref_pos_b = [1, 1, 1]
        overview = True
        overview_pos = [-5, -5, 4]  # [m]
        overview_lookat = [50, 50, 2]  # [m]
        camera_horizontal_fov = 75.0
        camera_width = 500
        camera_height = 500
        camera_env_list = [0]
        record_camera_imgs = False


class BaseTrainCfg(ABCConfig):
    runner_class_name = ''
