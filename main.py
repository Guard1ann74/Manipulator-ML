from model_2.predict_nn import check_answer
from manipulator_control import (
    move_forward, move_backward, move_left, move_right,
    move_home, init_arm, disconnect_arm
)

print("Система запущена")

# Инициализация манипулятора
arm = init_arm()

print("\nПеремещение манипулятора в начальную точку")
move_home(arm)
print("Манипулятор в начальной точк\n")


def main():
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in ('exit', 'выход', 'quit'):
            print("Выход из системы")
            break

        status, message, command_label = check_answer(user_input)
        print(message)

        if status == 1:  # команда распознана
            # Выполняем соответствующее движение
            if command_label == 1:  # forward (вперёд)
                move_forward(arm)
            elif command_label == 2:  # backward (назад)
                move_backward(arm)
            elif command_label == 3:  # left (влево)
                move_left(arm)
            elif command_label == 4:  # right (вправо)
                move_right(arm)

            # Возвращаемся в начальную точку после выполнения команды
            print("Возврат в начальную точку")
            move_home(arm)
            print("Готов к следующей команде")

    # Отключаем манипулятор
    disconnect_arm(arm)


if __name__ == "__main__":
    main()