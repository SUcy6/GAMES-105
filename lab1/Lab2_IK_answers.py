import numpy as np
from scipy.spatial.transform import Rotation as R

def touch(end, target):
    print(end)
    t1 = (abs(end[0]-target[0]) <= 0.01)
    t2 = (abs(end[1]-target[1]) <= 0.01)
    t3 = (abs(end[2]-target[2]) <= 0.01)
    print(t1, t2, t3)
            
    return t1*t2*t3

def get_pose(id, positions):
    pos = []
    for i in range(3):
        pos.append(positions[id][i])
    
    return pos

def get_offset(names, parents, positions):
    length = len(names)
    joint_offset = np.zeros((length, 3), dtype=np.float64)
    for i in range(length):
        parent = parents[i]
        if parent != -1:
            joint_offset[i] = positions[i] - positions[parent]
    return joint_offset

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position
    root_joint = meta_data.root_joint
    end_joint = meta_data.end_joint
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_offset = get_offset(joint_name, joint_parent, joint_initial_position)
    
    end_id = path[-1]
    end_pos = get_pose(end_id, joint_positions)
    
    while not touch(end_pos, target_pose.tolist()):
        for j in range(len(path)-2,-1,-1):
            # print(path[j])
            pos1 = get_pose(path[j], joint_positions)
            v1 = np.array(pos1) - np.array(target_pose.tolist())
            v2 = np.array(pos1) - np.array(end_pos)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            v1 = v1/norm1
            v2 = v2/norm2
            angle = np.arccos(np.dot(v1, v2))
            axis = np.cross(v1, v2)
            axis = axis/np.linalg.norm(axis)
            rot = R.from_rotvec(-angle * axis)
            crot = R.from_quat(joint_orientations[path[j]])
            nrot = rot * crot

            for c in range(j+1, len(path)):
                if c == j+1:
                    off = joint_offset[path[c]]
                    pos = joint_positions[path[c-1]]
                    newpos = pos + np.dot(nrot.as_matrix(), off)
                    joint_positions[path[c]] = newpos
                else:
                    off = joint_offset[path[c]]
                    pos = joint_positions[path[c-1]]
                    q = R.from_quat(joint_orientations[path[c-1]])
                    newpos = pos + np.dot(q.as_matrix(), off)
                    joint_positions[path[c]] = newpos

            nrot = nrot.as_quat()
            joint_orientations[path[j]] = nrot
            end_pos = get_pose(end_id, joint_positions)
        
    # update whole body
    for c in range(len(joint_name)):
        if c != 0:
            off = joint_offset[c]
            pos = joint_positions[joint_parent[c]]
            q = R.from_quat(joint_orientations[joint_parent[c]])
            newpos = pos + np.dot(q.as_matrix(), off)
            joint_positions[c] = newpos

    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations