import numpy as np
import scipy
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = np.empty((0,3), dtype=np.float64)
    parent = []

    file = open(bvh_file_path, 'r')
    idx = -1
    stack = []
    start = 0
    while True:
        line = file.readline()        

        if line == "MOTION":
            break

        line = line.split() 

        if line[0] == "ROOT":
            joint_name.append(line[1])
            joint_parent.append(-1)
        elif line[0] == '{':
            start = 1
            stack.append('{')
            idx += 1
            parent.append(idx)
        elif line[0] == "OFFSET":
            rot = R.from_euler('XYZ',line[1:],degrees=True)
            rot = rot.as_euler('XYZ',degrees=True)
            rot = [np.float64(rot)]
            joint_offset = np.append(joint_offset, rot, axis=0)
        elif line[0] == "JOINT":
            joint_name.append(line[1])
            joint_parent.append(parent[-1])
        elif line[0] == "End":
            joint_name.append(joint_name[-1] + "_end")
            joint_parent.append(parent[-1])
        elif line[0] == '}':
            stack.pop(-1)
            parent.pop(-1)

        if len(stack) == 0 and start:
            break

    # joint_name = None
    # joint_parent = None
    # joint_offset = None
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    joint_positions = np.empty((0,3), dtype=np.float64)
    joint_orientations = np.empty((0,4), dtype=np.float64)

    motion = motion_data[frame_id]
    mn = 0
    for j in range(len(joint_name)):
        parent = joint_parent[j]
        offset = joint_offset[j]

        if j==0: # root
                pos = motion[:3]
                pos = [np.float64(pos)]
                rot = R.from_euler('XYZ', motion[3:6], degrees=True)
                rot = rot.as_quat()
                rot = [np.float64(rot)]
                joint_positions = np.append(joint_positions, pos, axis=0)
                joint_orientations = np.append(joint_orientations, rot, axis=0)
                mn += 1
                
        else:
                # current motion rotation
                if not joint_name[j].endswith("end"):
                    rot = R.from_euler('XYZ', motion[5+mn:5+mn+3], degrees=True)
                    mn += 3
                else:
                    rot = R.from_euler('XYZ', [0,0,0], degrees=True)
                p_rot = R.from_quat(joint_orientations[parent])
                Q = p_rot * rot
                Q = Q.as_quat()
                Q = [np.float64(Q)]

                rotmtx = p_rot.as_matrix()
                pos = np.dot(rotmtx, offset)
                pos = joint_positions[parent] + pos
                pos = [np.float64(pos)]
                
                joint_positions = np.append(joint_positions, pos, axis=0)
                joint_orientations = np.append(joint_orientations, Q, axis=0)      
            
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    # find index for each joint in A
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion_data = load_motion_data(A_pose_bvh_path)
    mn = 0
    ls = 0
    rs = 0
    index = {}
    for i in range(len(A_joint_name)):
        if A_joint_name[i] == "lShoulder":
            ls = mn
        if A_joint_name[i] == "rShoulder":
            rs = mn
        
        if A_joint_name[i] == "RootJoint":
            index["RootJoint"] = mn
            mn += 6
        else:
            if not A_joint_name[i].endswith("end"):
                index[A_joint_name[i]] = mn
                mn += 3

    # find index list refer to T
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    idx = 0
    idx_ls = []
    for name in T_joint_name:
        if name == "RootJoint":
            for i in range(6):
                idx_ls.append(i)
            idx += 6
        else:
            if not name.endswith('end'):
                old = index[name]
                if old  != idx:
                    idx_ls.append(old)
                    idx_ls.append(old+1)
                    idx_ls.append(old+2)
                else:
                    idx_ls.append(idx)
                    idx_ls.append(idx+1)
                    idx_ls.append(idx+2)
                idx += 3

    # change order
    frame_id = 0
    while frame_id < len(A_motion_data):
        A_motion_data[frame_id][ls+2] -= 45
        A_motion_data[frame_id][rs+2] += 45
        A_motion_data[frame_id] = A_motion_data[frame_id][idx_ls]
        # print(frame_id)
        frame_id += 1
    return A_motion_data



