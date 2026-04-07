from Agilebot.IR.A.arm import Arm
from Agilebot.IR.A.status_code import StatusCodeEnum
from Agilebot.IR.A.sdk_types import PoseType, RobotStatusEnum, ServoStatusEnum
from Agilebot.IR.A.sdk_classes import MotionPose
from Agilebot.IR.A.sdk_classes import Posture
import time

def wait_until_settled(arm: Arm, *, poll_s: float = 0.05, timeout_s: float = 30.0) -> None:
    t0 = time.monotonic()

    while True:
        robot_status, ret = arm.get_robot_status()
        if ret != StatusCodeEnum.OK:
            raise RuntimeError(f"get_robot_status failed: {ret}")

        servo_status, ret = arm.get_servo_status()
        if ret != StatusCodeEnum.OK:
            raise RuntimeError(f"get_servo_status failed: {ret}")

        robot_idle = robot_status == RobotStatusEnum.ROBOT_IDLE
        servo_idle = servo_status == ServoStatusEnum.SERVO_IDLE

        if robot_idle and servo_idle:
            return

        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(
                f"Robot did not settle in {timeout_s:.1f}s. "
                f"robot_status={robot_status}, servo_status={servo_status}"
            )

        time.sleep(poll_s)

def main():
    # 初始化Arm类
    arm = Arm()
    # 连接控制器
    ret = arm.connect("10.27.1.254")
    # ret = arm.connect("192.168.1.201")

    # 检查是否连接成功
    assert ret == StatusCodeEnum.OK, f"connection error: {ret}"

    ret = arm.motion.set_UF(0)
    assert ret == StatusCodeEnum.OK

    ret = arm.motion.set_TF(0)
    assert ret == StatusCodeEnum.OK

    # 初始化位姿
    X_OFFSET = -100.0

    motion_pose = MotionPose()
    motion_pose.pt = PoseType.CART
    motion_pose.cartData.position.x = -108.0 + X_OFFSET
    motion_pose.cartData.position.y = 716.0
    motion_pose.cartData.position.z = 720.0
    motion_pose.cartData.position.a = 160.0
    motion_pose.cartData.position.b = 60.0
    motion_pose.cartData.position.c = 160.0

    # motion_pose.cartData.posture = Posture()
    # motion_pose.cartData.posture.arm_up_down = 1  # 0 = локоть вниз, 1 = вверх
    # motion_pose.cartData.posture.wrist_flip = 0  # 0 = без переворота запястья
    # motion_pose.cartData.posture.arm_back_front = 0  # для 5-осевого может не использоваться
    # # turnCircle – массив целых чисел, обычно [0,0,0,0,0,0]

    ret = arm.motion.move_joint(motion_pose, vel=0.5, acc=0.2)
    assert ret == StatusCodeEnum.OK, f"move_joint error: {ret}"
    wait_until_settled(arm)
    arm.disconnect()



if __name__ == '__main__':
    main()
