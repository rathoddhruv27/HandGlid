import cv2
import logging
from config.settings import ConfigManager

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.cap = None
        self.is_running = False

    def start(self):
        """Initializes the camera connection."""
        camera_index = self.config.get("camera_index", 0)
        width = self.config.get("camera_width", 1280)
        height = self.config.get("camera_height", 720)

        logger.info(f"Starting camera index {camera_index} at {width}x{height}...")
        self.cap = cv2.VideoCapture(camera_index)

        if not self.is_available():
            logger.error(f"Failed to open camera index {camera_index}. Please check connection.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.is_running = True
        logger.info("Camera initialized successfully.")
        return True

    def read_frame(self):
        """Reads a frame from the camera, applying necessary transformations."""
        if not self.is_running or not self.cap:
            return False, None

        success, frame = self.cap.read()
        if not success:
            logger.warning("Failed to grab frame from camera.")
            return False, None

        if self.config.get("mirror_camera", True):
            frame = cv2.flip(frame, 1)

        return True, frame

    def is_available(self):
        """Checks if the camera is opened and responsive."""
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        """Safely releases the camera resource."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.is_running = False
        logger.info("Camera released safely.")
