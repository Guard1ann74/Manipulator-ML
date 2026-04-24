from Agilebot.IR.A.arm import Arm
from Agilebot.IR.A.status_code import StatusCodeEnum
from Agilebot.IR.A.sdk_types import RobotStatusEnum
from Agilebot.IR.A.sdk_classes import MotionPose
from Agilebot.IR.A.common.const import const
import time


def init_arm():
    arm = Arm()
    ret = arm.connect("10.27.1.254")
    if ret != StatusCodeEnum.OK:
        print(f"Ошибка подключения: {ret}")
        return None
    print("Подключение успешно")
    return arm


def disconnect_arm(arm):
    arm.disconnect()
    print("Манипулятор отключён")


def wait_for_idle(arm, timeout=30):
    start_time = time.time()
    while True:
        try:
            status = arm.get_robot_status()
            if status[1] == RobotStatusEnum.ROBOT_IDLE:
                break
        except Exception as e:
            print(f"Ошибка при проверке статуса: {e}")

        if time.time() - start_time > timeout:
            print(f"Таймаут ожидания манипулятора ({timeout} сек.)")
            break
        time.sleep(0.5)


def move_to_joints(arm, joints, timeout=10):
    motion_pose = MotionPose()
    motion_pose.pt = const.JOINT
    motion_pose.joint.j1 = joints[0]
    motion_pose.joint.j2 = joints[1]
    motion_pose.joint.j3 = joints[2]
    motion_pose.joint.j4 = joints[3]
    motion_pose.joint.j5 = joints[4]
    motion_pose.joint.j6 = joints[5]

    ret = arm.motion.move_to_pose(motion_pose, const.MOVE_JOINT)
    if ret != StatusCodeEnum.OK:
        print(f"Ошибка движения: {ret}")
        return False

    wait_for_idle(arm, timeout)
    return True

# Начальная позиция
HOME_POSE = [94.097, 55.639, -102.516, 48.190, 92.608, -181.231]

POSITION_FORWARD = [94.097, 30.489, -59.866, 30.227, 91.608, -181.231]
POSITION_BACKWARD = [94.097, 68.639, -138.516, 71.190, 92.608, -181.231]
POSITION_LEFT = [123.097, 55.142, -102.516, 48.190, 92.608, -181.231]
POSITION_RIGHT = [53.097, 55.639, -102.516, 48.190, 92.608, -181.231]


def move_forward(arm):
    print("Выполняется движение: ВПЕРЁД")
    success = move_to_joints(arm, POSITION_FORWARD)
    if success:
        print("Движение вперёд выполнено")
    else:
        print("Ошибка при движении вперёд")


def move_backward(arm):
    print("Выполняется движение: НАЗАД")
    success = move_to_joints(arm, POSITION_BACKWARD)
    if success:
        print("Движение назад выполнено")
    else:
        print("Ошибка при движении назад")


def move_left(arm):
    print("Выполняется движение: ВЛЕВО")
    success = move_to_joints(arm, POSITION_LEFT)
    if success:
        print("Движение влево выполнено")
    else:
        print("Ошибка при движении влево")


def move_right(arm):
    print("Выполняется движение: ВПРАВО")
    success = move_to_joints(arm, POSITION_RIGHT)
    if success:
        print("Движение вправо выполнено")
    else:
        print("Ошибка при движении вправо")


def move_home(arm):
    print("Возврат в начальную позицию")
    success = move_to_joints(arm, HOME_POSE)
    if success:
        print("Возврат выполнен")
    else:
        print("Ошибка при возврате")