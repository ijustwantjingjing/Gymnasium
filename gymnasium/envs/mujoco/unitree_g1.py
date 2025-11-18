__credits__ = ["Kallinteris-Andreas"]

import numpy as np
import torch

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import isaac2mujoco

from .unitree_legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from .unitree_g1_config import G1RoughCfg, G1RoughCfgPPO
import os

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}


class UnitreeEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        cfg = G1RoughCfg(),
        train_cfgs = G1RoughCfgPPO(),
        # frame_skip: int = 5,
        # TODO: frame_skip 应该怎么设置? 和self.cfg.control.decimation有关么?
        frame_skip: int = 1,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):


        self.cfg = cfg
        LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        asset_path = self.cfg.asset.mj_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

        # ==================================== task_registry ==================================== 
        # cfg = G1RoughCfg()
        # self.cfg = cfg
        # train_cfgs = G1RoughCfgPPO()

        # set_seed(env_cfg.seed)

        # parse sim params (convert to dict first)
        # sim_params = {"sim": isaac2mujoco.class_to_dict(cfg.sim)}
        self.sim_params = isaac2mujoco.parse_sim_params(self.cfg)
        
        frame_skip = self.cfg.control.decimation

        self.device = 'cpu'


        # TODO:headless什么作用？
        # self.headless = args.headless
        # ==================================== task_registry ==================================== 
        
        utils.EzPickle.__init__(
            self,
            cfg,
            train_cfgs,
            frame_skip,
            default_camera_config,
            **kwargs,
        )

        # ==================================== base_task__init__ ====================================
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # TODO： 这里的作用是什么？需要打开么？
        # optimization flags for pytorch JIT
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_obs, device=self.device, dtype=torch.float)
        # self.rew_buf : float
        self.rew_buf = torch.tensor(0, device=self.device, dtype=torch.float)
        # self.episode_length_buf : int
        self.episode_length_buf = torch.tensor(0, device=self.device, dtype=torch.int)
        print("[DEBUG] episode_length_buf.shape :", self.episode_length_buf.shape)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs
        # ==================================== base_task__init__ ====================================
            
        MujocoEnv.__init__(
            self,
            asset_path,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        # 覆盖 MuJoCo 的 timestep，使之等于 cfg.sim.dt
        # 这样：self.dt = self.model.opt.timestep * self.frame_skip
        #      = cfg.sim.dt * cfg.control.decimation
        self.model.opt.timestep = self.sim_params.dt

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        clip_obs = self.cfg.normalization.clip_observations

        self.observation_space = Box(
            low=-clip_obs,
            high=clip_obs,
            shape=(self.num_obs,),
            dtype=np.float32,
        )


        # ==================================== base_task__init__ ====================================
        self.extras = {}

        # create envs, sim and viewer
        self.create_sim()

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # TODO： 增加headless的逻辑
        # if running with a viewer, set up keyboard shortcuts and camera
        # if self.headless == False:
        #     # subscribe to keyboard shortcuts
        #     self.viewer = self.gym.create_viewer(
        #         self.sim, gymapi.CameraProperties())
        #     self.gym.subscribe_viewer_keyboard_event(
        #         self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        #     self.gym.subscribe_viewer_keyboard_event(
        #         self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
        # ==================================== base_task__init__ ==================================== 

        # ==================================== legged_robot__init__ ==================================== 
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training
        """
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)

        # TODO： 增加headless的逻辑
        # if not self.headless:
        #     self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)

        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        # ==================================== legged_robot__init__ ====================================
        

        # TODO: 这里直接加compute_observations会报错,因为phase变量在_post_physics_step_callback中调用, _post_physics_step_callback在step调用
        # 在 __init__ 里，完成 create_sim / _parse_cfg / _init_buffers 之后加：
        # with torch.no_grad():
        #     self.compute_observations()
        #     assert self.obs_buf.shape[-1] == self.num_obs, \
        #         f"cfg.env.num_observations={self.num_obs}, " \
        #         f"but compute_observations() gives {self.obs_buf.shape[-1]}"

    
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self._create_envs()


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # mjmodel和data的创建应该回由gymnasium完成，在init的时候传入了路径
        # asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        # self.model = mujoco.MjModel.from_xml_path(asset_path)
        # self.data = mujoco.MjData(self.model)

        self.num_dof = isaac2mujoco.get_asset_dof_count(self.model, self.data)

        self.num_bodies = isaac2mujoco.get_asset_rigid_body_count(self.model, self.data)

        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = isaac2mujoco.get_asset_dof_properties(self.model, self.data, self.num_dof)

        # save body names from the asset
        body_names = isaac2mujoco.get_asset_rigid_body_names(self.model, self.data)
        self.dof_names = isaac2mujoco.get_asset_dof_names(self.model, self.data)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = isaac2mujoco.to_torch(base_init_state_list, device=self.device, requires_grad=False)

        self.custom_origins = False
        self.env_origins = torch.zeros(3, device=self.device, requires_grad=False)

        # create env instance
        # TODO: 修改_process_rigid_shape_props
        # rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        # rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
        # self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
        
        self._process_dof_props()
        # TODO：将修改后的dof_pos_limits 应用回mujoco
        # self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

        # TODO: 修改_process_rigid_body_props
        # body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
        # body_props = self._process_rigid_body_props(body_props, i)
        # self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = isaac2mujoco.find_actor_rigid_body_handle(self.model, self.data, feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = isaac2mujoco.find_actor_rigid_body_handle(self.model, self.data, penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = isaac2mujoco.find_actor_rigid_body_handle(self.model, self.data, termination_contact_names[i])
    
    # TODO: 修改_process_rigid_shape_props
    # def _process_rigid_shape_props(self, props, env_id):
    #     """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
    #         Called During environment creation.
    #         Base behavior: randomizes the friction of each environment

    #     Args:
    #         props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
    #         env_id (int): Environment id

    #     Returns:
    #         [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
    #     """
    #     if self.cfg.domain_rand.randomize_friction:
    #         if env_id==0:
    #             # prepare friction randomization
    #             friction_range = self.cfg.domain_rand.friction_range
    #             num_buckets = 64
    #             bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
    #             friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
    #             self.friction_coeffs = friction_buckets[bucket_ids]

    #         for s in range(len(props)):
    #             props[s].friction = self.friction_coeffs[env_id]
    #     return props

    def _process_dof_props(self):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF
        """
        for i in range(self.num_dof):
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

    # TODO: 修改_process_rigid_body_props
    # def _process_rigid_body_props(self, props, env_id):
    #     # if env_id==0:
    #     #     sum = 0
    #     #     for i, p in enumerate(props):
    #     #         sum += p.mass
    #     #         print(f"Mass of body {i}: {p.mass} (before randomization)")
    #     #     print(f"Total mass {sum} (before randomization)")
    #     # randomize base mass
    #     if self.cfg.domain_rand.randomize_base_mass:
    #         rng = self.cfg.domain_rand.added_mass_range
    #         props[0].mass += np.random.uniform(rng[0], rng[1])
    #     return props
    

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # create some wrapper tensors for different slices
        self.root_states = isaac2mujoco.acquire_actor_root_state_tensor(self.model, self.data)
        self.dof_pos, self.dof_vel = isaac2mujoco.acquire_dof_state_tensor(self.model, self.data)
        self.contact_forces = isaac2mujoco.acquire_net_contact_force_tensor(self.model, self.data)

        self.base_quat = self.root_states[3:7]
        # root_states已经将mujoco的wxyz形式转换为了xyzw
        self.rpy = isaac2mujoco.get_euler_xyz(self.base_quat)
        self.base_pos = self.root_states[0:3]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = isaac2mujoco.to_torch(isaac2mujoco.get_axis_params(-1., self.up_axis_idx), device=self.device)
        self.forward_vec = isaac2mujoco.to_torch([1., 0., 0.], device=self.device)
        self.torques = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[7:13])
        self.commands = torch.zeros(self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = isaac2mujoco.quat_rotate_inverse(self.base_quat, self.root_states[7:10])
        self.base_ang_vel = isaac2mujoco.quat_rotate_inverse(self.base_quat, self.root_states[10:13])
        self.projected_gravity = isaac2mujoco.quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

        # 这里不需要unsqueeze，因为dof_pos已经去除了env维度
        # self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.feet_num = len(self.feet_indices)
        
        self.rigid_body_states_view = isaac2mujoco.acquire_rigid_body_state_tensor(self.model, self.data)
        self.feet_state = self.rigid_body_states_view[self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :3]
        self.feet_vel = self.feet_state[:, 7:10]

    # ============================================== legged_robot ==============================================
    # def _get_noise_scale_vec(self, cfg):
    #     """ Sets a vector used to scale the noise added to the observations.
    #         [NOTE]: Must be adapted when changing the observations structure

    #     Args:
    #         cfg (Dict): Environment config file

    #     Returns:
    #         [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
    #     """
    #     noise_vec = torch.zeros_like(self.obs_buf[0])
    #     self.add_noise = self.cfg.noise.add_noise
    #     noise_scales = self.cfg.noise.noise_scales
    #     noise_level = self.cfg.noise.noise_level
    #     noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
    #     noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
    #     noise_vec[6:9] = noise_scales.gravity * noise_level
    #     noise_vec[9:12] = 0. # commands
    #     noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
    #     noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
    #     noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

    #     return noise_vec
    # ============================================== legged_robot ==============================================


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
        return noise_vec

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = { name: torch.zeros((), dtype=torch.float, device=self.device)
            for name in self.reward_scales.keys() }

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        # 这里不需要unsqueeze，因为dof_pos已经去除了env维度
        # self.default_dof_pos = self.default_dof_pos.unsqueeze(0)


    def _get_obs(self):

        self.compute_observations()
        print("[DEBUG] self.obs_buf:", self.obs_buf)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        # TODO: unitree 原版中返回值有privileged_obs_buf，什么作用？这里怎么提供返回值？
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # 将pytorch变量转换到numpy输出
        observation = self.obs_buf.detach().cpu().numpy()
        print("[DEBUG] observation:", observation)

        return observation

    def step(self, action):
        clip_actions = self.cfg.normalization.clip_actions

        # gymnasium的action是numpy，但是下面的运算是torch，需要转换
        action = torch.from_numpy(action).float().to(self.device)

        self.actions = torch.clip(action, -clip_actions, clip_actions).to(self.device)

        self.torques = self._compute_torques(self.actions).view(self.torques.shape)
        
        # gymnasium 函数
        self.do_simulation(self.torques, self.frame_skip)
        
        # ==================================== self.post_physics_step() ==================================== 
        # 将post_physics_step拆开，observation、reward等部分分别放在_get_obs、_get_rew
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos = self.root_states[0:3].clone()
        self.base_quat = self.root_states[3:7].clone()
        self.rpy = isaac2mujoco.get_euler_xyz(self.base_quat)
        self.base_lin_vel = isaac2mujoco.quat_rotate_inverse(self.base_quat, self.root_states[7:10])
        self.base_ang_vel = isaac2mujoco.quat_rotate_inverse(self.base_quat, self.root_states[10:13])
        self.projected_gravity = isaac2mujoco.quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        # self.check_termination()
        # self.compute_reward()

        # 修改为调用gymnasium提供的接口
        terminated = self.check_termination()
        reward, reward_info = self._get_rew()
        
        info = {**reward_info}

        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        observation = self._get_obs()

        self.last_actions = self.actions.clone()
        self.last_dof_vel = self.dof_vel.clone()
        self.last_root_vel = self.root_states[7:13].clone()
        # ==================================== self.post_physics_step() ==================================== 
        # obs裁减的部分挪动到_get_obs中

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        need_push = self.episode_length_buf % int(self.cfg.domain_rand.push_interval) == 0
        if need_push == False:
            return
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[7:9] = isaac2mujoco.torch_rand_float(-max_vel, max_vel, device=self.device) # lin vel x/y
        
        isaac2mujoco.set_actor_root_state_tensor_indexed(self.model, self.data, self.root_states)


    def _get_rew(self):

        reward, reward_info = self.compute_reward()

        return reward, reward_info


    def reset_model(self):

        self.reset_idx()

        # gymnasium函数
        # self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
    
    def reset_idx(self):
        # reset robot states
        self._reset_dofs()
        self._reset_root_states()

        self._resample_commands()

        # reset buffers
        # 必须使用张量形式，因为用到了pytorch的cat等函数，要求传入的变量必须是张量
        self.actions[:] = 0.
        self.last_actions[:] = 0.
        self.last_dof_vel[:] = 0.
        self.feet_air_time[:] = 0.
        self.episode_length_buf.fill_(0) # 0-d tensor
        self.reset_buf = 1

        # TODO: 写入extras, 现在不写入是因为在g1原版中 reset_idx 是在 check_termination 之后执行的(都在step中).
        # TODO: 但是gymnasium自己管理reset_model, 所以出现了没有调用check_termination直接调用reset_idx的情况, 获取不到self.time_out_buf. 需要确定gymnasium函数的调用时机
        # fill extras
        # self.extras["episode"] = {}
        # for key in self.episode_sums.keys():
        #     self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key]) / self.max_episode_length_s
        #     self.episode_sums[key] = 0.
        # if self.cfg.commands.curriculum:
        #     self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # # send timeout info to the algorithm
        # if self.cfg.env.send_timeouts:
        #     self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        """
        self.dof_pos = self.default_dof_pos * isaac2mujoco.torch_rand_float(0.5, 1.5, device=self.device)
        # self.dof_vel = 0.
        self.dof_vel = torch.zeros(self.num_dof, dtype=torch.float, device=self.device)
        isaac2mujoco.set_dof_state_tensor_indexed(self.model, self.data, self.dof_pos, self.dof_vel)

    
    def _reset_root_states(self):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        """
        # base position
        if self.custom_origins:
            self.root_states = self.base_init_state
            self.root_states[:3] += self.env_origins
            self.root_states[:2] += isaac2mujoco.torch_rand_float(-1., 1., device=self.device) # xy position within 1m of the center
        else:
            self.root_states = self.base_init_state
            self.root_states[:3] += self.env_origins
        # base velocities
        self.root_states[7:13] = isaac2mujoco.torch_rand_float(-0.5, 0.5, device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        
        isaac2mujoco.set_actor_root_state_tensor_indexed(self.model, self.data, self.root_states)


    def _resample_commands(self):
        """ Randommly select commands of some environments
        """
        self.commands[0] = isaac2mujoco.torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], device=self.device)
        self.commands[1] = isaac2mujoco.torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], device=self.device)
        if self.cfg.commands.heading_command:
            self.commands[3] = isaac2mujoco.torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], device=self.device)
        else:
            self.commands[2] = isaac2mujoco.torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], device=self.device)
            
        # set small commands to zero
        self.commands[:2] *= float(torch.norm(self.commands[:2]) > 0.2)


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # =========================================== g1_env ==========================================
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :3]
        self.feet_vel = self.feet_state[:, 7:10]

        period = 0.8
        offset = 0.5
        # episode_length_buf shape (1,)
        self.phase = (self.episode_length_buf * self.dt) % period / period
        # self.episode_length_buf 是 0d 张量，因为要用来做 | 运算，这里保存为1D
        self.phase = self.phase.unsqueeze(0) # 0-d -> 1-d

        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        # cat 后shape变为 (2,)
        self.leg_phase = torch.cat([self.phase_left, self.phase_right], dim=-1)
        print("[DEBUG] self.leg_phase.shape:", self.leg_phase.shape)
        # =========================================== g1_env ==========================================

        need_resample = bool(self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
        if need_resample:
            self._resample_commands()
        if self.cfg.commands.heading_command:
            forward = isaac2mujoco.quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[1], forward[0])
            self.commands[2] = torch.clip(0.5*isaac2mujoco.wrap_to_pi(self.commands[3] - heading), -1., 1.)


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[self.termination_contact_indices, :], dim=-1) > 1., dim=0)
        print("[DEBUG] self.reset_buf.shape:", self.reset_buf.shape)

        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[1])>1.0, torch.abs(self.rpy[0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        print("[DEBUG] self.time_out_buf.shape:", self.time_out_buf.shape)
        self.reset_buf |= self.time_out_buf

        print("[DEBUG] self.reset_buf:", self.reset_buf)
        return self.reset_buf


    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf = 0.
        reward_info = {}

        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

            key = f"reward_{name}"
            reward_info[key] = rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf = torch.clip(self.rew_buf, min=0.)

        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

            key = "_reward_termination"
            reward_info[key] = rew
        
        print("[DEBUG] self.rew_buf:", self.rew_buf)
        print("[DEBUG] reward_info:", reward_info)
        return self.rew_buf, reward_info
    
    # ============================================== legged_robot ==============================================
    # def compute_observations(self):
    #     """ Computes observations
    #     """
    #     self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
    #                                 self.base_ang_vel  * self.obs_scales.ang_vel,
    #                                 self.projected_gravity,
    #                                 self.commands[:3] * self.commands_scale,
    #                                 (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                 self.dof_vel * self.obs_scales.dof_vel,
    #                                 self.actions
    #                                 ),dim=-1)
    #     # add perceptive inputs if not blind
    #     # add noise if needed
    #     if self.add_noise:
    #         self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
    #     return self.obs_buf
    # ============================================== legged_robot ==============================================

    def compute_observations(self):
        """ Computes observations
        """
        # TODO begin: 这里也有gymnasium调用时机的问题, phase是在step中计算的, 但是单独整理到obs计算后不一定会先执行step, 所以就没有self.phase
        # 需要确定gymnasium函数的调用时机，然后重构代码结构
        # 这里先简单重新计算phase
        period = 0.8
        self.phase = (self.episode_length_buf * self.dt) % period / period
        # self.episode_length_buf 是 0d 张量，因为要用来做 | 运算，这里保存为1D
        self.phase = self.phase.unsqueeze(0) # 0-d -> 1-d
        # TODO end
        
        sin_phase = torch.sin(2 * np.pi * self.phase )
        cos_phase = torch.cos(2 * np.pi * self.phase )

        parts = [
            self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase
        ]

        names = [
            "base_ang_vel",
            "projected_gravity",
            "commands",
            "dof_pos",
            "dof_vel",
            "actions",
            "sin_phase",
            "cos_phase"
        ]

        for n, t in zip(names, parts):
            print(f"[DEBUG] {n}: shape={t.shape}, dim={t.dim()}")

        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _parse_cfg(self, cfg):
        # dt在init中覆盖gymnasium的原本设置
        # self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = isaac2mujoco.class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = isaac2mujoco.class_to_dict(self.cfg.commands.ranges)
     

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)



    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:2]), dim=0)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:2]), dim=0)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[2]
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=0)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=0)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=0)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=0)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[self.penalised_contact_indices, :], dim=-1) > 0.1), dim=0)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=0)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=0)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=0)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:2] - self.base_lin_vel[:2]), dim=0)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[2] - self.base_ang_vel[2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=0) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:2], dim=0) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[self.feet_indices, :2], dim=1) >\
             5 *torch.abs(self.contact_forces[self.feet_indices, 2]), dim=0)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=0) * (torch.norm(self.commands[:2], dim=0) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=0)

    # =========================================== g1_env ==========================================
    def _reward_contact(self):
        res = torch.tensor(0, device=self.device, dtype=torch.float) # 0-D tensor
        for i in range(self.feet_num):
            is_stance = self.leg_phase[i] < 0.55
            contact = self.contact_forces[self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[self.feet_indices, :3], dim=1) > 1.
        pos_error = torch.square(self.feet_pos[:, 2] - 0.08) * ~contact
        return torch.sum(pos_error, dim=(0))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[self.feet_indices, :3], dim=1) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :3])
        return torch.sum(penalize, dim=(0,1))
    
    def _reward_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[[1,2,7,8]]), dim=0)
    # =========================================== g1_env ==========================================
