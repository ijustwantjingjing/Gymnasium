import numpy as np
import os

import torch
from torch import Tensor

import mujoco
import mujoco.viewer  # 需 mujoco>=3.x
from types import SimpleNamespace

# TODO：_process_rigid_shape_props

# TODO：_process_rigid_body_props

def simulate(model: mujoco.MjModel, data: mujoco.MjData, n_frames):
    mujoco.mj_step(model, data, nstep=n_frames)
    
    # As of MuJoCo 2.0, force-related quantities like cacc are not computed
    # unless there's a force sensor in the model.
    # See https://github.com/openai/gym/issues/1541
    # mujoco.mj_rnePostConstraint(model, data)


def find_actor_rigid_body_handle(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    indice = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    print("[DEBUG Warpper] name:", name, "-> indice:", indice)
    return indice


def get_asset_dof_properties(model: mujoco.MjModel, data: mujoco.MjData, num_dof: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [num_dof, 2] 位置上下限
    dof_pos_limits = torch.zeros(num_dof, 2, dtype=torch.float, device="cpu", requires_grad=False)
    # 注意，这里要求每个DOF对应一个joint（必须是hinge/slide，只有1个自由度）
    jnt_range = model.jnt_range          # shape: [njnt, 2]

    # TODO: [num_dof] 速度上限, MuJoCo 无法获得这个参数
    dof_vel_limits = torch.zeros(num_dof, dtype=torch.float, device="cpu", requires_grad=False)

    # TODO: [num_dof] 力矩上限，Mjcf中获取的数值全为0
    torque_limits = torch.zeros(num_dof, dtype=torch.float, device="cpu", requires_grad=False)
    act_forcerange = model.actuator_forcerange  # [nu, 2]
    act_ctrlrange  = model.actuator_ctrlrange   # [nu, 2]

    for i in range(num_dof):
        # 因为joint包括root的free joint,所以需要+1. 但如果数据的来源是actuator 就不需要(actuator不包含root)
        jnt_id = i + 1
        low, high = jnt_range[jnt_id]

        # 如果 low==high==0，说明没设限制，可以给一个很大的虚拟范围：
        if low == 0.0 and high == 0.0:
            low, high = -1e6, 1e6

        dof_pos_limits[i, 0] = low
        dof_pos_limits[i, 1] = high

    # TODO： 速度上限：MuJoCo 不存这个，只能你自己指定。这里直接传isaac中log出来的数值
    isaac_vel_limits = torch.tensor([32., 20., 32., 20., 37., 37., 32., 20., 32., 20., 37., 37.], dtype=torch.float, device="cpu")
    assert isaac_vel_limits.shape[0] == num_dof
    dof_vel_limits[:] = isaac_vel_limits

    # TODO: 力矩上限：从 actuator 中读. 目前mjcf获取数据不正确,还是手动填入isaac中的数值
    isaac_torque_limits = torch.tensor([88., 139., 88., 139., 50., 50., 88., 139., 88., 139., 50., 50.], dtype=torch.float, device="cpu")
    assert isaac_torque_limits.shape[0] == num_dof
    torque_limits[:] = isaac_torque_limits

    # 如果将来你把 MJCF 的 forcerange 配好了，可以改成：
    # act_forcerange = model.actuator_forcerange  # [nu, 2]
    # for i in range(num_dof):
    #     fmin, fmax = act_forcerange[i]
    #     self.torque_limits[i] = max(abs(fmin), abs(fmax))

    print("[DEBUG Warpper] mujoco dof_pos_limits =", dof_pos_limits)
    print("[DEBUG Warpper] mujoco dof_vel_limits =", dof_vel_limits)
    print("[DEBUG Warpper] mujoco torque_limits  =", torque_limits)

    return dof_pos_limits, dof_vel_limits, torque_limits


def get_asset_dof_names(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    # dof_names 需要排除root的free joint
    dof_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(1, model.njnt)
    ]
    print("[DEBUG Warpper] dof_names = ",dof_names)
    return dof_names


def get_asset_rigid_body_names(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    body_names = [model.body(i).name for i in range(model.nbody)]
    print("[DEBUG Warpper] body_names = ",body_names)
    return body_names


def get_asset_dof_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    # 这里必须-6,因为 num_dof得排除freejoint的6个自由度
    num_dof = int(model.nv - 6)
    print("[DEBUG Warpper] num_dof = ",num_dof)
    return num_dof


def get_asset_rigid_body_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    num_bodies = model.nbody
    print("[DEBUG Warpper] num_bodies = ",num_bodies)
    return num_bodies


def acquire_dof_state_tensor(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[torch.Tensor, torch.Tensor]:
    # 这里必须排除freejoint,因为后续的 default_dof_pos 是按照名字遍历的, 默认每个自由度为1
    qj_np  = data.qpos[7:].copy()
    dqj_np = data.qvel[6:].copy()

    # torch.as_tensor() 除非 dtype/device 不同，不然不会复制数据。torch.tensor()则一定会复制数据
    dof_pos = torch.as_tensor(qj_np,  dtype=torch.float32, device="cpu")
    print("[DEBUG Warpper] dof_pos.shape = ",dof_pos.shape)
    print("[DEBUG Warpper] dof_pos = ",dof_pos)

    dof_vel = torch.as_tensor(dqj_np, dtype=torch.float32, device="cpu")
    print("[DEBUG Warpper] dof_vel.shape = ",dof_vel.shape)
    print("[DEBUG Warpper] dof_vel = ",dof_vel)

    return dof_pos, dof_vel

def acquire_rigid_body_state_tensor(model: mujoco.MjModel, data: mujoco.MjData) -> torch.Tensor:
    # 不创建self.rigid_body_states(num_envs * num_bodies, 13), 直接创建rigid_body_states_view (num_envs, num_bodies, 13)
    # 预分配与 Isaac 同 shape 的 PyTorch 缓冲区（后续原地写入，保持视图有效）
    rigid_body_states_view = torch.empty((model.nbody, 13), device='cpu', dtype=torch.float32, requires_grad=False)

    # --- 位置与姿态（世界系） ---
    # data.xpos, data.xquat 是 numpy，但只作为数据来源；转换到 torch（保持在你指定的 device 上）
    xpos  = torch.as_tensor(data.xpos,  dtype=torch.float32, device='cpu')   # (nbody, 3)
    xquat = torch.as_tensor(data.xquat, dtype=torch.float32, device='cpu')   # (nbody, 4)

    rigid_body_states_view[:, 0:3] = xpos
    rigid_body_states_view[:, 3:7] = xquat

    # --- 线速度与角速度（世界朝向，body-centered） ---
    # 只能用 MuJoCo C API 写入一个 numpy 缓冲区，再一次性搬到 torch。
    # 根据文档：mj_objectVelocity 返回 [lin(0:3), ang(3:6)]，且 flg_local=0 表示与世界系对齐。
    # 获取每个body的速度(世界坐标系)
    tmp = np.zeros((model.nbody, 6), dtype=np.float64 if data.xpos.dtype == np.float64 else np.float32)
    for b in range(model.nbody):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, tmp[b], 0)

    lin = torch.as_tensor(tmp[:, 0:3], dtype=torch.float32, device='cpu')   # 线速度
    ang = torch.as_tensor(tmp[:, 3:6], dtype=torch.float32, device='cpu')   # 角速度

    rigid_body_states_view[:, 7:10]  = lin
    rigid_body_states_view[:, 10:13] = ang
    print("[DEBUG Warpper] rigid_body_states_view.shape = ",rigid_body_states_view.shape)
    print("[DEBUG Warpper] rigid_body_states_view = ",rigid_body_states_view)

    return rigid_body_states_view


def acquire_actor_root_state_tensor(model: mujoco.MjModel, data: mujoco.MjData) -> torch.Tensor:
    # root直接认为序号为0
    # jid  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, rootname)

    qadr = model.jnt_qposadr[0]  
    vadr = model.jnt_dofadr[0]

    # 更高效的张量构造（避免频繁 torch.tensor），一次性用 NumPy → Torch
    # 不使用copy的话在torch.from_numpy时会与mujoco共享内存，下一次mujoco更新的时候pytorch的变量也会更新
    qpos = data.qpos[qadr:qadr+7].copy()
    qvel = data.qvel[vadr:vadr+6].copy()
    
    # 直接拼成 numpy，再一次转 torch
    root_np = np.empty((13,), dtype=np.float32)
    root_np[0:3]  = qpos[0:3]          # px,py,pz
    root_np[3:7]  = [qpos[4], qpos[5], qpos[6], qpos[3]]  # qx,qy,qz,qw
    root_np[7:13] = qvel               # vx,vy,vz,wx,wy,wz
    print("[DEBUG Warpper] root_np.shape = ",root_np.shape)
    print("[DEBUG Warpper] root_np = ",root_np)

    return torch.from_numpy(root_np)


def acquire_net_contact_force_tensor(model: mujoco.MjModel, data: mujoco.MjData) -> torch.Tensor:
    # MuJoCo 在每个仿真步后会给出每个 body 的外力与外矩阵列：data.cfrc_ext[body_id]，这是一个 6 维向量 [torque(3), force(3)]，定义在所谓 “c-frame”（原点在子树质心，方向与世界坐标系对齐）。因此后三个分量可直接当作“世界系力” 使用。(MuJoCo Documentation)
    # 注意：cfrc_ext 是“外力”合力，通常包括接触力；若你场景里还施加了其它外力（例如 xfrc_applied），它们也会算进去。所以这是“近似地等同于 Isaac 的 net contact force”。如果你需要严格只统计接触产生的力，请用方式 B。
    # MuJoCo: data.cfrc_ext 形状 (nbody, 6) = [torque(3), force(3)]
    forces_world_np = data.cfrc_ext[:, 3:6] # (nbody, 3)
    contact_forces = torch.as_tensor(forces_world_np, dtype=torch.float32, device="cpu")
    
    print("[DEBUG Warpper] contact_forces.shape = ",contact_forces.shape)
    print("[DEBUG Warpper] rootcontact_forces_np = ",contact_forces)

    return contact_forces


def set_dof_actuation_force_tensor(model: mujoco.MjModel, data: mujoco.MjData, torques: Tensor):
    data.ctrl[:] = torques


# 设置dof的pos和vel，不包括根节点的freejoint
def set_dof_state_tensor_indexed(model: mujoco.MjModel, data: mujoco.MjData, dof_pos: torch.Tensor, dof_vel: torch.Tensor):
    print("[DEBUG Warpper] [input] dof_pos.shape :", dof_pos.shape)
    print("[DEBUG Warpper] [input] dof_pos :", dof_pos)
    print("[DEBUG Warpper] [input] dof_vel.shape :", dof_vel.shape)
    print("[DEBUG Warpper] [input] dof_vel :", dof_vel)

    nq = model.nq
    nv = model.nv
    # freejoint
    base_qpos_offset = 7
    base_qvel_offset = 6

    dof_pos_np = dof_pos.numpy()
    dof_vel_np = dof_vel.numpy()

    # 关节（纯 joint）段： [base_offset : 该 env 末尾)
    data.qpos[base_qpos_offset : nq] = dof_pos_np

    data.qvel[base_qvel_offset : nv] = dof_vel_np

    # 让 MuJoCo 重新计算状态的派生量（约束、传感器等）
    mujoco.mj_forward(model, data)


# 写根节点的数据
def set_actor_root_state_tensor_indexed(model: mujoco.MjModel, data: mujoco.MjData, root_states: torch.Tensor):
    """
    root_state: 形状 (13,) 的 torch 张量
    [0:3]  pos(x,y,z)
    [3:7]  quat(x,y,z,w)   # Isaac Gym 顺序
    [7:10] lin_vel(x,y,z)
    [10:13] ang_vel(x,y,z)
    """
    print("[DEBUG Warpper] [input] root_states.shape :", root_states.shape)
    print("[DEBUG Warpper] [input] root_states :", root_states)

    # torch -> numpy（无梯度）
    root_state = root_states.detach().cpu().numpy()

    # Isaac: pos(x,y,z), quat(x,y,z,w) -> MuJoCo: quat(w,x,y,z), pos(x,y,z)
    pos_xyz   = root_state[0:3]
    quat_xyzw = root_state[3:7]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
    ang = root_state[10:13]
    lin = root_state[7:10]
    print("[DEBUG Warpper] [input] root_state pos:", pos_xyz)
    print("[DEBUG Warpper] [input] root_state quat_wxyz:", quat_wxyz)
    print("[DEBUG Warpper] [input] root_state ang:", ang)
    print("[DEBUG Warpper] [input] root_state lin:", lin)

    # 写 qvel（顺序：ang_vel(3) + lin_vel(3)）
    data.qvel[0:6] = np.concatenate([ang, lin], axis=0)
    # 写 qpos（顺序：quat(wxyz) + pos(xyz)）
    data.qpos[0:7] = np.concatenate([quat_wxyz, pos_xyz], axis=0)

    # 前向推进
    # mujoco.mj_normalizeQuat(model, data)
    mujoco.mj_forward(model, data)


def parse_sim_params(cfg):
    """
    Replacements of IsaacGym SimParams for Mujoco-based envs.
    Just convert cfg.sim into an object with attributes.
    """

    sim_cfg = cfg.sim   # 这是你 config 里的 class sim

    # 将 cfg.sim 转成一个 namespace 对象
    sim_params = SimpleNamespace()

    sim_params.dt = sim_cfg.dt
    sim_params.substeps = sim_cfg.substeps
    sim_params.gravity = sim_cfg.gravity
    sim_params.up_axis = sim_cfg.up_axis

    return sim_params


def torch_rand_float(lower, upper, device):
    # 返回一个 0 维 tensor（标量）
    return (upper - lower) * torch.rand((), device=device) + lower

def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)

def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

# def get_euler_xyz(q):
#     qx, qy, qz, qw = 0, 1, 2, 3
#     # roll (x-axis rotation)
#     sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
#     cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
#                 q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
#     roll = torch.atan2(sinr_cosp, cosr_cosp)

#     # pitch (y-axis rotation)
#     sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
#     pitch = torch.where(
#         torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

#     # yaw (z-axis rotation)
#     siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
#     cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
#                 q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
#     yaw = torch.atan2(siny_cosp, cosy_cosp)

#     return torch.stack((roll, pitch, yaw), dim=-1)

def get_euler_xyz(q):
    q = torch.as_tensor(q, dtype=torch.float32)

    qx, qy, qz, qw = 0, 1, 2, 3

    # roll
    sinr_cosp = 2.0 * (q[qw] * q[qx] + q[qy] * q[qz])
    cosr_cosp = q[qw] * q[qw] - q[qx] * q[qx] - q[qy] * q[qy] + q[qz] * q[qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch
    sinp = 2.0 * (q[qw] * q[qy] - q[qz] * q[qx])
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    # yaw
    siny_cosp = 2.0 * (q[qw] * q[qz] + q[qx] * q[qy])
    cosy_cosp = q[qw] * q[qw] + q[qx] * q[qx] - q[qy] * q[qy] - q[qz] * q[qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw))


def to_torch(x, dtype=torch.float, device='cpu', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

def get_axis_params(value, axis_idx, x_value=0., dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))

# def quat_rotate_inverse(q, v):
#     shape = q.shape
#     q_w = q[:, -1]
#     q_vec = q[:, :3]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * \
#         torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
#             shape[0], 3, 1)).squeeze(-1) * 2.0
#     return a - b + c

def quat_rotate_inverse(q, v):
    # 转成 tensor，确保类型一致
    q = torch.as_tensor(q, dtype=torch.float32)
    v = torch.as_tensor(v, dtype=torch.float32)

    # q: [x, y, z, w]
    q_vec = q[:3]   # (3,)
    q_w = q[3]      # 标量

    # 按公式：v' = (2 w^2 - 1) v - 2 w (q × v) + 2 (q · v) q
    a = v * (2.0 * q_w ** 2 - 1.0)              # (3,)
    b = torch.cross(q_vec, v) * q_w * 2.0       # (3,)
    c = q_vec * (torch.dot(q_vec, v)) * 2.0     # (3,)

    return a - b + c                            # (3,)
