import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time
import threading

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.capture.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame
            time.sleep(0.001)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join()
        self.capture.release()

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.6, min_tracking_confidence=0.6)

is_holding = False
prev_hand_state = "0_FIST"
prev_vol_dist = 0
vol_action_text = ""

smooth_x, smooth_y = 0, 0
prev_x, prev_y = 0, 0
smoothing_factor = 0.2

frame_count = 0
last_face_landmarks = None

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return rotated

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, bg_channels = background.shape
    ov_h, ov_w, ov_channels = overlay.shape

    if x >= bg_w or y >= bg_h: return background
    if x + ov_w < 0 or y + ov_h < 0: return background

    if x < 0:
        overlay = overlay[:, -x:]; ov_w = overlay.shape[1]; x = 0
    if y < 0:
        overlay = overlay[-y:, :]; ov_h = overlay.shape[0]; y = 0
    if x + ov_w > bg_w:
        overlay = overlay[:, :bg_w - x]; ov_w = overlay.shape[1]
    if y + ov_h > bg_h:
        overlay = overlay[:bg_h - y, :]; ov_h = overlay.shape[0]

    if ov_channels == 4:
        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0
        background_part = background[y:y+ov_h, x:x+ov_w]
        if background_part.shape[0] != ov_h or background_part.shape[1] != ov_w:
            return background
        background[y:y+ov_h, x:x+ov_w] = (1.0 - mask) * background_part + mask * overlay_image
    else:
        background[y:y+ov_h, x:x+ov_w] = overlay
    return background

def get_finger_state(hand_landmarks):
    thumb_dist = math.hypot(hand_landmarks.landmark[4].x - hand_landmarks.landmark[17].x, hand_landmarks.landmark[4].y - hand_landmarks.landmark[17].y)
    thumb_mcp_dist = math.hypot(hand_landmarks.landmark[2].x - hand_landmarks.landmark[17].x, hand_landmarks.landmark[2].y - hand_landmarks.landmark[17].y)
    thumb_ext = thumb_dist > thumb_mcp_dist * 1.2

    index_ext = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
    middle_ext = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
    ring_ext = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
    pinky_ext = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

    if index_ext and middle_ext and ring_ext and pinky_ext:
        return "PALM_MOVE"
    elif index_ext and thumb_ext and not middle_ext:
        return "C_VOLUME"
    elif index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "2_HOLD"
    elif index_ext and not middle_ext and not thumb_ext:
        return "1_CLICK"
    else:
        return "0_FIST"

mask_image_path = "pngegg.png"
mask = cv2.imread(mask_image_path, cv2.IMREAD_UNCHANGED)

if mask is None:
    print(f"Error: Tidak bisa memuat gambar dari {mask_image_path}.")
    exit()

cap = ThreadedCamera(0)
time.sleep(1.0)

print("q")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_count += 1

    if frame_count % 3 == 0:
        face_results = face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            last_face_landmarks = face_results.multi_face_landmarks[0]

    if last_face_landmarks:
        forehead = last_face_landmarks.landmark[10]
        chin = last_face_landmarks.landmark[152]
        left_cheek = last_face_landmarks.landmark[234]
        right_cheek = last_face_landmarks.landmark[454]
        left_eye = last_face_landmarks.landmark[33]
        right_eye = last_face_landmarks.landmark[263]

        forehead_x, forehead_y = int(forehead.x * w), int(forehead.y * h)
        chin_x, chin_y = int(chin.x * w), int(chin.y * h)
        left_x = int(left_cheek.x * w)
        right_x = int(right_cheek.x * w)

        face_width = right_x - left_x
        face_height = chin_y - forehead_y

        if face_width > 0 and face_height > 0:
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            angle = math.degrees(math.atan2(dy, dx))

            mask_width = int(face_width * 1.4)
            mask_height = int(face_height * 1.6)

            if mask_width > 0 and mask_height > 0:
                resized_mask = cv2.resize(mask, (mask_width, mask_height), interpolation=cv2.INTER_AREA)
                rotated_mask = rotate_image(resized_mask, angle)

                pos_x = int((left_x + right_x) / 2 - mask_width / 2)
                pos_y = int(forehead_y - face_height * 0.2)

                frame = overlay_transparent(frame, rotated_mask, pos_x, pos_y)

    hand_results = hands.process(rgb_frame)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            hand_state = get_finger_state(hand_landmarks)
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            box_start = (int(x_min * w) - 15, int(y_min * h) - 15)
            box_end = (int(x_max * w) + 15, int(y_max * h) + 15)
            
            if hand_state == "PALM_MOVE":
                box_color = (0, 255, 0)
            elif hand_state == "1_CLICK":
                box_color = (0, 255, 255)
            elif hand_state == "2_HOLD":
                box_color = (255, 0, 0)
            elif hand_state == "C_VOLUME":
                box_color = (0, 165, 255)
            else:
                box_color = (0, 0, 255)
                
            cv2.rectangle(frame, box_start, box_end, box_color, 2)

            if hand_state == "C_VOLUME":
                dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) * 100
                
                if prev_vol_dist != 0:
                    delta = dist - prev_vol_dist
                    if delta > 1.0:
                        pyautogui.press('volumeup')
                        vol_action_text = "VOL UP"
                    elif delta < -1.0:
                        pyautogui.press('volumedown')
                        vol_action_text = "VOL DOWN"
                
                prev_vol_dist = dist
                cv2.line(frame, (int(thumb_tip.x*w), int(thumb_tip.y*h)), (int(index_tip.x*w), int(index_tip.y*h)), (0, 255, 255), 3)
                
                if is_holding:
                    pyautogui.mouseUp()
                    is_holding = False

            else:
                prev_vol_dist = 0
                vol_action_text = ""
                
                if hand_state in ("1_CLICK", "2_HOLD", "PALM_MOVE"):
                    target_x = np.interp(index_tip.x, (0.2, 0.8), (0, screen_w))
                    target_y = np.interp(index_tip.y, (0.2, 0.8), (0, screen_h))
                    
                    smooth_x = prev_x + (target_x - prev_x) * smoothing_factor
                    smooth_y = prev_y + (target_y - prev_y) * smoothing_factor
                    prev_x, prev_y = smooth_x, smooth_y
                    
                    pyautogui.moveTo(smooth_x, smooth_y)

                    if hand_state == "2_HOLD":
                        if not is_holding:
                            pyautogui.mouseDown()
                            is_holding = True
                    else:
                        if is_holding:
                            pyautogui.mouseUp()
                            is_holding = False

                    if hand_state == "1_CLICK" and prev_hand_state != "1_CLICK":
                        pyautogui.click()
                else:
                    if is_holding:
                        pyautogui.mouseUp()
                        is_holding = False

            prev_hand_state = hand_state

    status_text = "IDLE"
    if is_holding:
        status_text = "If"
    elif prev_hand_state == "1_CLICK":
        status_text = "Else If #1"
    elif prev_hand_state == "PALM_MOVE":
        status_text = "Else If #2"
    elif prev_hand_state == "C_VOLUME":
        status_text = f"VOLUME (Else If #c) - {vol_action_text}"

    cv2.putText(frame, f"State: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(':3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
