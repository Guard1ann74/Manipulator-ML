import ctypes
import numpy as np
import cv2
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from Camera.MvCameraControl_class import MvCamera
from Camera.CameraParams_header import *

MV_GIGE_DEVICE = 1
MV_TRIGGER_MODE_OFF = 0
MV_ACQ_MODE_CONTINUOUS = 2
MV_ACCESS_Exclusive = 1

PixelType_Gvsp_BGR8_Packed = 0x02180014
PixelType_Gvsp_RGB8_Packed = 0x0218001F

def set_camera_rgb(cam):
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    cam.MV_CC_SetEnumValue("AcquisitionMode", MV_ACQ_MODE_CONTINUOUS)
    try:
        cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
        return "RGB8"
    except Exception as e:
        print("Камера не поддерживает RGB формат")
        return None


def process_frame_rgb(img_array, width, height, pixel_type):
    try:
        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            frame = img_array.reshape((height, width, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        elif pixel_type == PixelType_Gvsp_BGR8_Packed: # Если камера все же в BGR
            frame = img_array.reshape((height, width, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        else:
            print(f"Неизвестный формат: {hex(pixel_type)}")
            return None
    except Exception as e:
        print(f"Ошибка обработки: {e}")
        return None


def main():
    print("\n1. Поиск камеры")
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, deviceList)

    if ret != 0 or deviceList.nDeviceNum == 0:
        print("Камера не найдена")
        return

    print("\n2. Подключение")
    cam = MvCamera()
    device_info = ctypes.cast(deviceList.pDeviceInfo[0],
                              ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(device_info)
    if ret != 0:
        return

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"Ошибка открытия: {ret}")
        cam.MV_CC_DestroyHandle()
        return

    print("Камера подключена")
    # Настройка RGB
    format_name = set_camera_rgb(cam)

    enum_value = MVCC_ENUMVALUE()
    ret = cam.MV_CC_GetEnumValue("PixelFormat", enum_value)
    if ret == 0:
        current_format = enum_value.nCurValue
        print(f"\nТекущий формат камеры: {hex(current_format)}")
        if current_format == PixelType_Gvsp_RGB8_Packed:
            print("Камера работает в RGB")
        elif current_format == PixelType_Gvsp_BGR8_Packed:
            print("Камера работает в BGR")

    # Запуск захвата
    print("\nЗапуск захвата")
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"Ошибка запуска: {ret}")
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        return

    print("✓ Захват запущен")
    print("\nУправление:")
    print("  'q' - выход")
    print("  's' - сохранить кадр")
    print("  'f' - показать информацию о формате")

    # Окно
    cv2.namedWindow('Hikrobot Camera (RGB Mode)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hikrobot Camera (RGB Mode)', 800, 600)

    frame_count = 0
    save_count = 0
    last_frame = None

    try:
        while True:
            st_frame = MV_FRAME_OUT()
            ctypes.memset(ctypes.byref(st_frame), 0, ctypes.sizeof(st_frame))
            ret = cam.MV_CC_GetImageBuffer(st_frame, 500)

            if ret == 0:
                frame_count += 1

                height = st_frame.stFrameInfo.nHeight
                width = st_frame.stFrameInfo.nWidth
                pixel_type = st_frame.stFrameInfo.enPixelType
                buffer_size = st_frame.stFrameInfo.nFrameLen

                # Копируем данные
                data_buffer = (ctypes.c_ubyte * buffer_size)()
                ctypes.memmove(data_buffer, st_frame.pBufAddr, buffer_size)
                img_array = np.frombuffer(data_buffer, dtype=np.uint8)

                # Обрабатываем в RGB режиме
                frame = process_frame_rgb(img_array, width, height, pixel_type)

                if frame is not None:
                    last_frame = frame.copy()

                    # Информация на кадре
                    format_text = "RGB Mode" if pixel_type == PixelType_Gvsp_RGB8_Packed else "BGR->RGB"
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Size: {width}x{height}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, format_text, (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow('Hikrobot Camera (RGB Mode)', frame)

                    if frame_count % 30 == 0:
                        print(f"✓ Кадр {frame_count}: {width}x{height}")

                cam.MV_CC_FreeImageBuffer(st_frame)

            # Клавиши
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Выход")
                break

    except KeyboardInterrupt:
        print("Остановка")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        print("\nОстановка захвата")
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()