import time
import random
import string
import sys
import numpy as np
from maix import image, time, fs, nn


class SignObservationTracker:
    """
    Contains some postprocessing logic on detected objects
    and tracks time since last observation.

    Expected usage:
    For a single frame, call update with list of objects in that frame
    Then can call is_detected, is_active etc for current state

    """
    def __init__(self, detection_height_threshold:float=50, hold_period:float=0):
        self._hold_period = hold_period  # how long to hold observation as "active" after observation
        self._detection_height_threshold = detection_height_threshold
        self.reset()
    
    def update(self, objects:list, time:float):
        self._objects = objects
        any_detected = any(self._obj_is_detected(obj) for obj in self._objects)
        if any_detected:
            self._last_observation_time = time
            self._detected = True
        else:
            self._detected = False
    
    def reset(self):
        self._last_observation_time = -float("inf")
        self._objects = []
        self._detected = False

    def is_detected(self) -> bool:
        """
        Return true if object detected in last update
        """
        return self._detected
    
    def is_active(self) -> bool:
        """
        Return true if object detected within hold period
        """
        return self.is_detected() or (self.time_since_observed() < self._hold_period)

    def get_objects(self):
        return self._objects
    
    def time_since_observed(self) -> float:
        return time.time() - self._last_observation_time
    
    def set_hold_period(self, hold_period:float):
        self._hold_period = hold_period
    
    def _obj_is_detected(self, obj):
        obj_point = (obj.x + obj.w//2, obj.y + obj.h//2) 
        return (obj_point[1] > self._detection_height_threshold)




    
    
def create_distance_map(image_width: int, image_height: int, target_x: int, target_y: int, horz_scale:float) -> np.ndarray:
    """
    GEMINI-GENERATED
    Creates a NumPy array (representing an image) where each pixel's value
    is its Euclidean distance from a specified target pixel.

    Args:
        image_width (int): The width of the desired image.
        image_height (int): The height of the desired image.
        target_x (int): The x-coordinate (column) of the target pixel.
        target_y (int): The y-coordinate (row) of the target pixel.
        horz_scale (float) : scaling factor to apply different weight to vert vs horz distance

    Returns:
        np.ndarray: A 2D NumPy array of shape (image_height, image_width)
                    containing the Euclidean distances. The data type will be float.
    """
    # Create coordinate grids for x and y
    # np.arange(image_width) creates [0, 1, ..., image_width-1]
    # np.arange(image_height) creates [0, 1, ..., image_height-1]
    # np.meshgrid creates two 2D arrays:
    # X_coords: A 2D array where each row is [0, 1, ..., image_width-1]
    # Y_coords: A 2D array where each column is [0, 1, ..., image_height-1] (transposed)
    X_coords, Y_coords = np.meshgrid(np.arange(image_width), np.arange(image_height))

    # Calculate the squared difference for x-coordinates from the target_x
    # (X_coords - target_x) creates an array where each element is (current_x - target_x)
    # **2 squares each element
    dx_squared = (X_coords - target_x) ** 2

    # Calculate the squared difference for y-coordinates from the target_y
    # (Y_coords - target_y) creates an array where each element is (current_y - target_y)
    # **2 squares each element
    dy_squared = (Y_coords - target_y) ** 2

    # Calculate the Euclidean distance: sqrt(dx^2 + dy^2)
    # np.sqrt takes the square root of each element in the sum
    #  
    distance_map = np.sqrt(dx_squared + dy_squared*horz_scale)

    # normalize to 0-255
    return (distance_map / distance_map.max() * 255).astype(np.uint8)

class ChalkTracker:
    def __init__(self, img_width, img_height, height_portion, horz_scale, detector):

        # distance map is used to find chalk point closest to a given center point on the image
        # (centered horizontally, vertical position controled by height_portion. 1=botton of image, 0=top of image)
        self.img_width = img_width
        self.img_height = img_height

        # horz scale is used to search more laterally than vertically for the closest chalk point
        self.horz_scale = horz_scale

        # Search point:
        # Traker looks for the closest chalk point to where it was last observed
        # This is intened to improve single line tracking when multiple are present
        self._default_search_point = (img_width/2, img_height*height_portion)
        self._reset_search_point()

        self._last_detected_x = self.search_point[0]

        # need to hold onto the detector just for calling 'draw_seg_mask'
        self._detector = detector




    def get_chalk_pos(self, objs, img=None):
        """
        get chalk x-value from image center from detection objects
        if img provded, draw chalk point on it
        if chalk not detected, returns last detected position
        """
        if not objs:
            self._reset_search_point()
            return self._last_detected_x
        
        chalk_mask = image.Image(self.img_width, self.img_height, image.Format.FMT_GRAYSCALE)
        for obj in objs:
            self._detector.draw_seg_mask(chalk_mask, obj.x, obj.y, obj.seg_mask, threshold=127)
        chalk_mask_np = image.image2cv(chalk_mask, ensure_bgr=False).squeeze() > 0
        # print(f"{chalk_mask_np.sum()=}") # todo: consider using this sum to filter 

        # not sure how efficient it is to compute this on every frame
        distance_map = create_distance_map(
            self.img_width,
            self.img_height,
            self.search_point[0],
            self.search_point[1],
            self.horz_scale,
        )

        chalk_distance = distance_map * chalk_mask_np
        chalk_distance[~chalk_mask_np] = 255
        chalk_pixel = np.unravel_index(np.argmin(chalk_distance), chalk_distance.shape)

        # keep search height fixed, update search lateral value with detection
        self.search_point = (int(chalk_pixel[1]), self.search_point[1])
        chalk_x = (chalk_pixel[1] - chalk_mask_np.shape[1]/2) / (chalk_mask_np.shape[1]/2)
        if img:
            img.draw_keypoints([chalk_pixel[1], chalk_pixel[0]], image.Color.from_rgb(255, 0, 0), size=1, thickness=3)
        self._last_detected_x = chalk_x
        return chalk_x
    
    def _reset_search_point(self):
        """
        On init or loss of tracking reset search point to default
        """
        self.search_point = self._default_search_point

def get_disk_use_percent() -> float:
    du = sys.disk_usage()
    du_percent = (du['used'] / du['total'])*100
    return du_percent

class ImageLogger:
    def __init__(self, logging_frequency:float, enabled:bool = True):
        rand_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        run_name = f"run_{int(time.time())}_{rand_str}" # append random string in case of no clock available
        outdir = "/root/runs/"+run_name
        self._outdir = outdir
        self._logging_period = 1./logging_frequency
        if enabled:
            fs.mkdir(outdir, recursive=True)
            assert fs.isdir(outdir)
            self._next_save_time = time.time()
            print(f"Image logger enabled and saving to {outdir}")
            print(f"Initial disk usage: {get_disk_use_percent():.2f}%")
        else:
            self._next_save_time = float("inf")
            print("Image logger disabled")

    def process_image(self, image:image.Image):
        """
        Receive image. If time to save, save to disk
        """
        if time.time() >= self._next_save_time:
            # disable logging if disk usage too high

            if get_disk_use_percent() > 90:
                print("WARN: disk use past 90%, disabling image logger")
                self._next_save_time = float("inf")
                return
            
            img_name = f"image_{str(time.time_ms()).zfill(15)}.png"
            outpath = self._outdir + "/" + img_name
            img.save(outpath)
            self._next_save_time += self._logging_period