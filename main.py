#添加模块搜索路径
import sys
sys.path.append('/sdcard')

from libs.PipeLine import PipeLine
from apps.ai_face.face_recog_core.face_recognition import FaceRecognition
from uart_protocol import *
import ulab.numpy as np
import os, gc, time

# ========== 初始化 ==========
pl = PipeLine()
pl.create()

anchors = np.fromfile("/sdcard/utils/prior_data_320.bin", dtype=np.float)
anchors = anchors.reshape((4200, 4))

fr = FaceRecognition(
    face_det_kmodel="/sdcard/kmodel/face_detection_320.kmodel",
    face_reg_kmodel="/sdcard/kmodel/face_recognition.kmodel",
    det_input_size=[320, 320],
    reg_input_size=[112, 112],
    database_dir="/data/face_database/",
    anchors=anchors,
    confidence_threshold=0.5,
    nms_threshold=0.2,
    face_recognition_threshold=0.65,
    rgb888p_size=pl.rgb888p_size,
    display_size=pl.display_size
)
gc.collect()

# 初始化串口
uart = init_uart()
print("UART 初始化完成")

# ========== 加载数据库 ==========
db_data = []
db_name = []

db_dir = "/data/face_database/"
try:
    files = os.listdir(db_dir)
    for f in files:
        if f.endswith(".bin"):
            fp = open(db_dir + f, "rb")
            feature = np.frombuffer(fp.read(), dtype=np.float)
            db_data.append(feature)
            db_name.append(f.split(".")[0])
            fp.close()

except Exception as e:
    print("数据库为空:", e)

print("已加载人脸：", db_name)
gc.collect()

# ========== 状态控制 ==========
mode = 0  # 0=待机(只检测不识别), 1=识别模式
last_send_name = ""
last_send_time = 0

# ========== 主循环 ==========
while True:
    os.exitpoint()

    # 1. 检查 STM32 命令
    result = read_frame(uart, timeout_ms=5)
    if result:
        cmd, data = result
        if cmd == CMD_START_RECOG:
            mode = 1
            last_send_name = ""
            print("STM32 要求开始识别")
        elif cmd == CMD_START_ENROLL:
            mode = 2
            print("STM32 要求开始录入")
        elif cmd == CMD_DELETE_FACE:
            # 删除指定人脸
            if len(data) > 0:
                name_len = data[0]
                del_name = data[1:1+name_len].decode()
                del_path = db_dir + del_name + ".bin"
                try:
                    os.remove(del_path)
                    # 从内存列表也删
                    if del_name in db_name:
                        idx = db_name.index(del_name)
                        db_name.pop(idx)
                        db_data.pop(idx)
                    send_frame(uart, CMD_DELETE_RESULT, bytes([1]))
                    print("已删除:", del_name)
                except:
                    send_frame(uart, CMD_DELETE_RESULT, bytes([0]))

    # 2. 获取画面
    img = pl.get_frame()
    pl.osd_img.clear()

    if mode == 0:
        # 待机模式：只检测画框，不识别
        det_boxes, _ = fr.face_det.run(img)
        for box in det_boxes:
            scale_x = pl.display_size[0] / 320
            scale_y = pl.display_size[1] / 320
            x = int(box[0] * scale_x)
            y = int(box[1] * scale_y)
            w = int(box[2] * scale_x)
            h = int(box[3] * scale_y)
            pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 255, 0, 0), thickness=2)
        pl.osd_img.draw_string_advanced(10, 10, 25, "Standby", color=(255, 255, 0, 0))

    elif mode == 1:
        # 识别模式
        det_boxes, landms = fr.face_det.run(img)

        if len(det_boxes) > 0:
            for i in range(len(det_boxes)):
                box = det_boxes[i]
                landm = landms[i]

                # 提取特征
                fr.face_reg.config_preprocess(landm)
                feature = fr.face_reg.run(img)

                # 比对数据库
                best_name = "unknown"
                best_score = 0.0
                for j in range(len(db_data)):
                    score = np.dot(feature, db_data[j]) / 2 + 0.5
                    if score > best_score:
                        best_score = score
                        best_name = db_name[j]

                # 画框
                scale_x = pl.display_size[0] / 320
                scale_y = pl.display_size[1] / 320
                x = int(box[0] * scale_x)
                y = int(box[1] * scale_y)
                w = int(box[2] * scale_x)
                h = int(box[3] * scale_y)

                if best_score > fr.face_recognition_threshold:
                    color = (255, 0, 0, 255)
                    label = best_name + " " + str(round(best_score, 2))

                    # 识别成功，通知 STM32（同一人 3 秒内不重复发）
                    now = time.ticks_ms()
                    if best_name != last_send_name or time.ticks_diff(now, last_send_time) > 3000:
                        name_bytes = best_name.encode()
                        send_data = bytes([1, len(name_bytes)]) + name_bytes + bytes([int(best_score * 100)])
                        send_frame(uart, CMD_RECOG_RESULT, send_data)
                        last_send_name = best_name
                        last_send_time = now
                        print("识别成功，已通知 STM32:", best_name)
                else:
                    color = (255, 255, 0, 0)
                    label = "unknown " + str(round(best_score, 2))

                pl.osd_img.draw_rectangle(x, y, w, h, color=color, thickness=2)
                pl.osd_img.draw_string_advanced(x, y - 30, 30, label, color=color)

        pl.osd_img.draw_string_advanced(10, 10, 25, "Recognizing...", color=(255, 0, 0, 255))
    elif mode == 2:
            # 录入模式：检测人脸 → 提取特征 → 保存为 user1.bin
            det_boxes, landms = fr.face_det.run(img)

            if len(det_boxes) > 0:
                box = det_boxes[0]
                landm = landms[0]

                # 画框
                scale_x = pl.display_size[0] / 320
                scale_y = pl.display_size[1] / 320
                x = int(box[0] * scale_x)
                y = int(box[1] * scale_y)
                w = int(box[2] * scale_x)
                h = int(box[3] * scale_y)
                pl.osd_img.draw_rectangle(x, y, w, h, color=(0, 255, 0, 0), thickness=2)
                pl.osd_img.draw_string_advanced(x, y - 30, 30, "Enrolling...", color=(0, 255, 0, 0))

                # 提取特征
                fr.face_reg.config_preprocess(landm)
                feature = fr.face_reg.run(img)

                # 保存为 user1.bin
                save_path = db_dir + "user1.bin"
                fp = open(save_path, "wb")
                fp.write(feature.tobytes())
                fp.close()

                # 加入内存列表（先删旧的再加新的）
                if "user1" in db_name:
                    idx = db_name.index("user1")
                    db_data[idx] = feature
                else:
                    db_data.append(feature)
                    db_name.append("user1")

                # 通知 STM32 录入成功
                send_data = bytes([1, 5]) + b"user1"
                send_frame(uart, CMD_ENROLL_RESULT, send_data)
                print("录入成功: user1")

                # 录完回到待机
                mode = 0
            else:
                pl.osd_img.draw_string_advanced(10, 10, 25, "No Face!", color=(255, 0, 0, 255))

    pl.show_image()
    gc.collect()
