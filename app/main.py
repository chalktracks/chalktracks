from maix import camera, display, image, nn, app, uart, time, sys, fs
from perception_utils import SignObservationTracker, ChalkTracker, ImageLogger
import numpy as np
from collections import defaultdict
import time


FORWARD_SPEED = 0.13
SPEED_SIGN_MULTIPLIER = 2.0  # how much faster to go when speed sign observed
MAX_ACCEL = FORWARD_SPEED / 0.5  # reach max speed in 0.5s
LATERAL_GAIN = -5.0
MAX_TURN_SPEED = 6
IMAGE_LOGGING_FPS = 1.0
IMAGE_LOGGING_ENABLED = False
MODEL_FILE = "/root/models/chalk/giga-camel-20250911.mud"

STOP_SIGN_HOLD_PERIOD = 4. # s
U_TURN_HOLD_PERIOD = 1.5 # s
SPEED_HOLD_PERIOD = 3. # s
CHALK_HOLD_PERIOD = 0.5 # s - how long to track last seen chalk pos when not observed in current frame

def sign(x):
    if x < 0:
        return -1.0
    else:
        return +1.0

def limit_signed_val(val:float, max_val:float):
    """
    Limit given value to +/- max_val
    max_val must be provided as non-negative number
    """
    assert max_val >= 0, "Error - logic assumes max is positive - received negative max"
    if abs(val) > max_val:
        return max_val * sign(val)
    else:
        return val
    
class AccelLimiter:
    def __init__(
        self,
        accel_limit:float=0.8  # m/s^2
    ):
        self._current_speed = 0
        self._accel_limit = accel_limit
        self._last_update_time = time.time()
    
    def request_speed(self, vel:float):
        current_time = time.time()
        dt = current_time - self._last_update_time
        max_dv = dt * self._accel_limit
        requested_dv = vel - self._current_speed
        dv = limit_signed_val(requested_dv, max_dv)
        self._current_speed += dv
        return self._current_speed
            


class RobotInterface:
    def __init__(self):
        device = "/dev/ttyS0"
        self._serial = uart.UART(device, 9600)
    
    def drive(self, vel:float, ang_vel:float):
        data = f"{vel:.2f},{ang_vel:.2f}\n".encode()
        self._serial.write(data)


detector = nn.YOLO11(model=MODEL_FILE)
robot = RobotInterface()

chalk_tracker = ChalkTracker(
    detector.input_width(),
    detector.input_height(),
    height_portion=0.5,   #  middle
    horz_scale=3.0,
    detector=detector,
)

detection_trackers = {
    sign_type: SignObservationTracker()
    for sign_type in detector.labels
}
# set hold times for detections
detection_trackers["sign_stop"].set_hold_period(STOP_SIGN_HOLD_PERIOD)
detection_trackers["sign_u_turn"].set_hold_period(U_TURN_HOLD_PERIOD)
detection_trackers["sign_speed"].set_hold_period(SPEED_HOLD_PERIOD)
detection_trackers["chalk"].set_hold_period(CHALK_HOLD_PERIOD)

accel_limiter = AccelLimiter(MAX_ACCEL)

print(f"{detector.input_width()=}")
cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format())
dis = display.Display()
next_save_time = time.time()
image_logger = ImageLogger(IMAGE_LOGGING_FPS, IMAGE_LOGGING_ENABLED)

label_to_color = defaultdict(
    lambda : image.COLOR_GREEN,
    {
        "chalk":image.COLOR_BLUE,
        "sign_stop":image.COLOR_RED,
        "sign_turn":image.COLOR_YELLOW,
    }
)

while not app.need_exit():
    t = time.time()

    img = cam.read()
    image_logger.process_image(img)
    objs = detector.detect(img, conf_th = 0.2, iou_th = 0.2)

    # convert from list of objs to dict of {label:[list of objs with that label]}
    objs_by_label = defaultdict(list)
    for obj in objs:
        label = detector.labels[obj.class_id]
        objs_by_label[label].append(obj)

    # now update the trackers
    for label in detector.labels:
        detection_trackers[label].update(objs_by_label[label], t)

    # draw objects on the image for live view
    for obj in objs:
        label = detector.labels[obj.class_id]

        if label == "chalk":
            detector.draw_seg_mask(img, obj.x, obj.y, obj.seg_mask, threshold=127)
        else:
            color = label_to_color[label]
            img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = color)
            msg = f'{label}: {obj.score:.2f}'
            img.draw_string(obj.x, obj.y, msg, color = color)

    chalk_objs = detection_trackers["chalk"].get_objects()
    chalk_x = chalk_tracker.get_chalk_pos(chalk_objs, img)

    # handle stop/go interaction
    # in particular, when stop is active but not observed (ie seen recently but not right now)
    # allow a current "go sign" observation to set the stop observation as inactive.
    # This allows the robot to resume motion as soon as the go sign is observed (when stop sign no longer observed)
    def handle_stop_go_interaction(stop_tracker:SignObservationTracker, go_tracker:SignObservationTracker):
        stop_detected = stop_tracker.is_detected()
        stop_active = stop_tracker.is_active()
        go_detected = go_tracker.is_detected()
        if (not stop_detected) and stop_active and go_detected:
            stop_tracker.reset()

    handle_stop_go_interaction(detection_trackers["sign_stop"], detection_trackers["sign_go"])

    # compute what speed to use for forward speed, if robot is in a forward moving state
    forward_speed = FORWARD_SPEED if not detection_trackers["sign_speed"].is_active() else FORWARD_SPEED * SPEED_SIGN_MULTIPLIER

    # determine turn speed to use if in turning state
    turn_sign_speed = MAX_TURN_SPEED * 0.75

    # now determine output speeds based on sign states

    if detection_trackers["sign_stop"].is_active():
        forward_speed = 0
        turn_speed = 0

    elif detection_trackers["sign_turn_left"].is_active():
        forward_speed = 0
        turn_speed = turn_sign_speed

    elif detection_trackers["sign_turn_right"].is_active():
        forward_speed = 0
        turn_speed = -turn_sign_speed

    elif detection_trackers["sign_u_turn"].is_active():
        forward_speed = 0
        turn_speed = turn_sign_speed

    elif detection_trackers["chalk"].is_active():
        forward_speed = forward_speed
        turn_speed = limit_signed_val(chalk_x * LATERAL_GAIN, MAX_TURN_SPEED)

    elif detection_trackers["sign_go"].is_active():
        # if see go sign, but nothing else, just drive forward
        forward_speed = forward_speed
        turn_speed = 0

    else:
        forward_speed = 0
        turn_speed = 0
    
    # apply speed filters
    forward_speed = accel_limiter.request_speed(forward_speed)  # apply accel limit

    # send filtered commands to robot
    robot.drive(forward_speed,turn_speed)

    # print(f"{forward_speed=}")
    # print(f"{turn_speed=}")
    # for label, tracker in detection_trackers.items():
    #     print(f"{label} is active: {tracker.is_active()}")

    dis.show(img)

