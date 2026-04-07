from pyDHgripper import AG95
import time

gripper = AG95(port='COM4')

def set_abs_rot(val : int) -> None:
    """Установка угла поворота гриппера с блокировкой других команд до конца прокручивания гриппера"""
    gripper.set_abs_rot(val=val)
    while gripper.read_state() != 1:
        time.sleep(0.05)


def main():
    while gripper.read_state() != 1:
        time.sleep(0.05)
    while gripper.read_state() != 1:
        time.sleep(0.05)

    gripper.set_pos(val=100)

    gripper.set_pos(val=100)
    gripper.set_pos(val=1000)
    return


if __name__ == "__main__":
    main()