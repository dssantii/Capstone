from picamera2 import Picamera2
import cv2
import numpy as np
import time
import sys
import pytesseract
from tflite_runtime.interpreter import Interpreter

# load the Yahboom car driver
sys.path.append("/home/dannymyland/Downloads/Yahboom_project/Raspbot/2.Hardware Control course/02.Drive motor")
import YB_Pcb_Car

# MAX RUNTIME SAFETY
MAX_RUNTIME_SECONDS = 60
program_start_time = time.time()

# load the model
MODEL_PATH = "New_model.tflite"
LABELS = ["Stop Sign", "Speed Limit Sign"]

# detection settings
GENERAL_CONFIDENCE_THRESHOLD = 0.20
STOP_SIGN_CONFIDENCE_THRESHOLD = 0.75
SPEED_SIGN_CONFIDENCE_THRESHOLD = 0.75
NMS_THRESHOLD = 0.45

# stop sign thresholds
STOP_BOX_AREA_RATIO_THRESHOLD = 0.06
RESET_STOP_AREA_RATIO_THRESHOLD = 0.01

# speed sign thresholds
SPEED_BOX_AREA_RATIO_THRESHOLD = 0.06
RESET_SPEED_AREA_RATIO_THRESHOLD = 0.03

# performance settings
SKIP_FRAMES = 35
frame_count = 0

# driving settings
DRIVE_SPEED = 35
STOP_HOLD_TIME = 2.5
IGNORE_STOP_AFTER_GO = 4.0

# valid speeds we will accept from OCR
VALID_SPEEDS = {10, 15, 20, 25, 30, 35}

# only use OCR once per speed sign event
speed_sign_handled = False

# Tesseract config: digits only
TESS_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"

# box colors
STOP_COLOR = (0, 0, 255)
SPEED_COLOR = (0, 255, 255)
OTHER_COLOR = (0, 255, 0)

# remember last detections
last_best_label = "none"
last_best_conf = 0.0
last_boxes = []
last_stop_area_ratio = 0.0
last_speed_area_ratio = 0.0
last_speed_text = "none"
stop_sign_visible_now = False
speed_sign_visible_now = False

# car state
car_state = "DRIVING"
stopped_at_time = None
current_drive_speed = DRIVE_SPEED

# stop sign latch
stop_sign_handled = False
ignore_stop_until_time = 0.0

# setup car
car = YB_Pcb_Car.YB_Pcb_Car()

# load TFLite model
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = int(input_details[0]["shape"][1])
input_width = int(input_details[0]["shape"][2])

# start camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

time.sleep(1)
print("RUNNING STOP + OCR SPEED SIGN DRIVE TEST")

# start driving
car.Car_Run(current_drive_speed, current_drive_speed)

def read_speed_from_crop(crop_bgr):
    try:
        crop_h, crop_w = crop_bgr.shape[:2]
        if crop_h < 10 or crop_w < 10:
            return None, ""

        # use lower-middle region where the speed number usually is
        number_crop = crop_bgr[int(crop_h * 0.45):, int(crop_w * 0.1):int(crop_w * 0.9)]

        if number_crop.size == 0:
            return None, ""

        crop_gray = cv2.cvtColor(number_crop, cv2.COLOR_BGR2GRAY)
        crop_gray = cv2.resize(crop_gray, (0, 0), fx=2, fy=2)
        crop_gray = cv2.GaussianBlur(crop_gray, (3, 3), 0)
        _, crop_thresh = cv2.threshold(
            crop_gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        raw_text = pytesseract.image_to_string(
            crop_thresh, config=TESS_CONFIG
        ).strip()

        # fallback to full crop
        if not raw_text:
            raw_text = pytesseract.image_to_string(
                crop_bgr, config=TESS_CONFIG
            ).strip()

        digits = ''.join(filter(str.isdigit, raw_text))
        if not digits:
            return None, raw_text

        number = int(digits)
        if number in VALID_SPEEDS:
            return number, raw_text

        return None, raw_text

    except Exception:
        return None, ""

try:
    while True:
        current_time = time.time()

        # runtime check
        elapsed_time = current_time - program_start_time
        if elapsed_time >= MAX_RUNTIME_SECONDS:
            print("MAX RUNTIME REACHED - STOPPING PROGRAM")
            break

        # get frame from camera
        frame = picam2.capture_array()

        # fix color format
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # rotate image right side up
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        # copy frame for drawing
        display_frame = frame.copy()
        frame_h, frame_w = display_frame.shape[:2]
        screen_area = frame_h * frame_w

        frame_count += 1

        # only run the model every few frames
        if frame_count % SKIP_FRAMES == 0:
            img = cv2.resize(frame, (input_width, input_height))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()

            raw_output = interpreter.get_tensor(output_details[0]["index"])[0]
            predictions = raw_output.T

            boxes = []
            scores = []
            class_ids = []

            # decode YOLO output
            for pred in predictions:
                cx, cy, w, h, score0, score1 = pred

                class_scores = [score0, score1]
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])

                if confidence < GENERAL_CONFIDENCE_THRESHOLD:
                    continue

                x1 = float(cx - w / 2)
                y1 = float(cy - h / 2)

                # handle either normalized coords or model-space coords
                if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:
                    x1 = x1 * frame_w
                    y1 = y1 * frame_h
                    w_scaled = float(w) * frame_w
                    h_scaled = float(h) * frame_h
                else:
                    x1 = x1 * frame_w / input_width
                    y1 = y1 * frame_h / input_height
                    w_scaled = float(w) * frame_w / input_width
                    h_scaled = float(h) * frame_h / input_height

                x1 = int(x1)
                y1 = int(y1)
                w_scaled = int(w_scaled)
                h_scaled = int(h_scaled)

                if w_scaled <= 0 or h_scaled <= 0:
                    continue

                boxes.append([x1, y1, w_scaled, h_scaled])
                scores.append(confidence)
                class_ids.append(class_id)

            current_boxes = []
            best_label = "none"
            best_conf = 0.0
            largest_stop_area_ratio = 0.0
            largest_speed_area_ratio = 0.0
            stop_visible = False
            speed_visible = False
            speed_sign_box = None

            # apply NMS
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, GENERAL_CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

                if len(indices) > 0:
                    for idx in indices.flatten():
                        x, y, w, h = boxes[idx]
                        conf = scores[idx]
                        class_id = class_ids[idx]

                        label = LABELS[class_id] if class_id < len(LABELS) else f"class_{class_id}"

                        x1 = x
                        y1 = y
                        x2 = x + w
                        y2 = y + h

                        if x2 <= x1 or y2 <= y1:
                            continue

                        x1 = max(0, min(frame_w - 1, x1))
                        y1 = max(0, min(frame_h - 1, y1))
                        x2 = max(0, min(frame_w - 1, x2))
                        y2 = max(0, min(frame_h - 1, y2))

                        box_area = max(0, x2 - x1) * max(0, y2 - y1)
                        box_area_ratio = box_area / screen_area

                        current_boxes.append((x1, y1, x2, y2, label, conf, class_id, box_area_ratio))

                        if conf > best_conf:
                            best_conf = conf
                            best_label = label

                        if label == "Stop Sign" and conf >= STOP_SIGN_CONFIDENCE_THRESHOLD:
                            stop_visible = True
                            if box_area_ratio > largest_stop_area_ratio:
                                largest_stop_area_ratio = box_area_ratio

                        if label == "Speed Limit Sign" and conf >= SPEED_SIGN_CONFIDENCE_THRESHOLD:
                            speed_visible = True
                            if box_area_ratio > largest_speed_area_ratio:
                                largest_speed_area_ratio = box_area_ratio
                                speed_sign_box = (x1, y1, x2, y2)

            last_boxes = current_boxes
            last_best_label = best_label
            last_best_conf = best_conf
            last_stop_area_ratio = largest_stop_area_ratio
            last_speed_area_ratio = largest_speed_area_ratio
            stop_sign_visible_now = stop_visible
            speed_sign_visible_now = speed_visible

            # reset stop latch when stop sign is gone or far away again
            if (not stop_sign_visible_now) or (last_stop_area_ratio < RESET_STOP_AREA_RATIO_THRESHOLD):
                stop_sign_handled = False

            # reset speed latch when speed sign is gone or far away again
            if (not speed_sign_visible_now) or (last_speed_area_ratio < RESET_SPEED_AREA_RATIO_THRESHOLD):
                speed_sign_handled = False

            # trigger stop logic
            if (
                car_state == "DRIVING"
                and current_time >= ignore_stop_until_time
                and not stop_sign_handled
                and stop_sign_visible_now
                and last_stop_area_ratio >= STOP_BOX_AREA_RATIO_THRESHOLD
            ):
                car.Car_Stop()
                car_state = "STOPPED"
                stopped_at_time = current_time
                stop_sign_handled = True
                print("STOPPING AT STOP SIGN")

            # trigger OCR speed change logic
            if (
                car_state == "DRIVING"
                and not speed_sign_handled
                and speed_sign_visible_now
                and last_speed_area_ratio >= SPEED_BOX_AREA_RATIO_THRESHOLD
                and speed_sign_box is not None
            ):
                x1, y1, x2, y2 = speed_sign_box

                # small padding like your Colab notebook
                pad_x = int((x2 - x1) * 0.09)
                pad_y = int((y2 - y1) * 0.09)

                x1p = max(0, x1 - pad_x)
                y1p = max(0, y1 - pad_y)
                x2p = min(frame_w, x2 + pad_x)
                y2p = min(frame_h, y2 + pad_y)

                crop = frame[y1p:y2p, x1p:x2p]
                number, raw_text = read_speed_from_crop(crop)

                if number is not None:
                    current_drive_speed = number
                    car.Car_Run(current_drive_speed, current_drive_speed)
                    last_speed_text = f"{number} mph"
                    speed_sign_handled = True
                    print(f"SPEED SIGN OCR -> {number} mph (raw: '{raw_text}')")
                else:
                    last_speed_text = f"unreadable ({raw_text})"
                    print(f"SPEED SIGN OCR FAILED (raw: '{raw_text}')")

            print(
                "Best prediction:",
                last_best_label,
                "Confidence:",
                round(last_best_conf, 3),
                "Stop area:",
                round(last_stop_area_ratio, 3),
                "Speed area:",
                round(last_speed_area_ratio, 3),
                "Current speed:",
                current_drive_speed,
                "OCR:",
                last_speed_text
            )

        # if stopped, hold for 3 seconds, then go again
        if car_state == "STOPPED":
            if stopped_at_time is not None and (current_time - stopped_at_time) >= STOP_HOLD_TIME:
                car.Car_Run(current_drive_speed, current_drive_speed)
                car_state = "DRIVING"
                stopped_at_time = None
                ignore_stop_until_time = current_time + IGNORE_STOP_AFTER_GO
                print("3-SECOND STOP DONE - RESUMING DRIVE")

        # draw boxes every frame
        object_count = 0

        for xmin, ymin, xmax, ymax, classname, conf, classidx, box_area_ratio in last_boxes:
            if classname == "Stop Sign":
                color = STOP_COLOR
            elif classname == "Speed Limit Sign":
                color = SPEED_COLOR
            else:
                color = OTHER_COLOR

            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 5)

            label_text = f"{classname}: {int(conf * 100)}% area:{box_area_ratio:.3f}"

            label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            label_y = max(ymin, label_size[1] + 12)

            cv2.rectangle(
                display_frame,
                (xmin, label_y - label_size[1] - 12),
                (xmin + label_size[0] + 6, label_y + base_line - 8),
                color,
                cv2.FILLED
            )

            cv2.putText(
                display_frame,
                label_text,
                (xmin + 3, label_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2
            )

            object_count += 1

        # on-screen status
        cv2.putText(display_frame, f"Best: {last_best_label} ({last_best_conf:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.putText(display_frame, f"Objects: {object_count}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_frame, f"STATE: {car_state}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_frame, f"SPEED: {current_drive_speed}",
                    (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_frame, f"OCR: {last_speed_text}",
                    (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        time_left = max(0, MAX_RUNTIME_SECONDS - elapsed_time)
        cv2.putText(
            display_frame,
            f"TIME LEFT: {time_left:.1f}s",
            (10, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Stop + OCR Speed Sign Drive Test", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    car.Car_Stop()
    cv2.destroyAllWindows()
    picam2.stop()