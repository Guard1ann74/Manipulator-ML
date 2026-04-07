from Agilebot.IR.A.arm import Arm
from Agilebot.IR.A.status_code import StatusCodeEnum
from Agilebot.IR.A.sdk_types import RobotStatusEnum
from Agilebot.IR.A.sdk_classes import MotionPose
from Agilebot.IR.A.common.const import const
import time
import serial


# 初始化Arm类
arm = Arm()
# 连接控制器
ret = arm.connect("10.27.1.254")
# 检查是否连接成功
assert ret == StatusCodeEnum.OK

# 初始化位姿
a = 1
while a > 0:
    motion_pose = MotionPose()
    motion_pose.pt = const.JOINT
    motion_pose.joint.j1 = 0
    motion_pose.joint.j2 = 84.549
    motion_pose.joint.j3 = -120.821
    motion_pose.joint.j4 = 134.360
    motion_pose.joint.j5 = 268.441
    motion_pose.joint.j6 = -168.557
    ret = arm.motion.move_to_pose(motion_pose, const.MOVE_JOINT)

'''
    motion_pose = MotionPose()
    motion_pose.pt = const.JOINT
    motion_pose.joint.j1 = -290.568
    motion_pose.joint.j2 = 84.549
    motion_pose.joint.j3 = -120.821
    motion_pose.joint.j4 = 134.360
    motion_pose.joint.j5 = 268.441
    motion_pose.joint.j6 = 168.557
    ret = arm.motion.move_to_pose(motion_pose, const.MOVE_JOINT)

    motion_pose = MotionPose()
    motion_pose.pt = const.JOINT
    motion_pose.joint.j1 = -274.577
    motion_pose.joint.j2 = 99.232
    motion_pose.joint.j3 = -78.322
    motion_pose.joint.j4 = 67.975
    motion_pose.joint.j5 = 268.443
    motion_pose.joint.j6 = 168.556
    ret = arm.motion.move_to_pose(motion_pose, const.MOVE_JOINT)
'''

"""
while arm.get_robot_status()[1] != RobotStatusEnum.ROBOT_IDLE:
    pass

motion_pose.joint.j1 = -265.845
motion_pose.joint.j2 = 68.540
motion_pose.joint.j3 = -109.087
motion_pose.joint.j4 = 133.130
motion_pose.joint.j5 = 268.903
motion_pose.joint.j6 = 168.995
ret = arm.motion.move_to_pose(motion_pose, const.MOVE_JOINT)

while arm.get_robot_status()[1] != RobotStatusEnum.ROBOT_IDLE:
    pass

motion_pose.joint.j1 = -274.568
motion_pose.joint.j2 = 53.549
motion_pose.joint.j3 = -116.821
motion_pose.joint.j4 = 154.360
motion_pose.joint.j5 = 268.441
motion_pose.joint.j6 = 168.557

ret = arm.motion.move_to_pose(motion_pose, const.MOVE_JOINT)

while arm.get_robot_status()[1] != RobotStatusEnum.ROBOT_IDLE:
    pass

motion_pose.joint.j1 = 89.442
motion_pose.joint.j2 = 81.396
motion_pose.joint.j3 = -146.242
motion_pose.joint.j4 = 56.717
motion_pose.joint.j5 = -291.680
motion_pose.joint.j6 = -255.199
ret = arm.motion.move_to_pose(motion_pose, const.MOVE_JOINT)
"""

arm.disconnect()