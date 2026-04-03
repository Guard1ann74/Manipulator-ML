from pyDHgripper.pyDHgripper import RGD
import time


gripper = RGD(port='COM3')

def set_abs_rot(val : int) -> None:
    """Установка угла поворота гриппера с блокировкой других команд до конца прокручивания гриппера"""
    gripper.set_abs_rot(val=val)
    while gripper.read_rot_state() != 1:
        time.sleep(0.05)


def main():
    while gripper.read_state() != 1:
        time.sleep(0.05)
        print("2")

    set_abs_rot(90)
    gripper.set_pos()



def united():
    a = 1
    while a>0:
        gripper.set_pos(val=0)
        set_abs_rot(50)
        set_abs_rot(0)
        set_abs_rot(180)
if __name__ == "__main__":
    main()

