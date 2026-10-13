import cv2
import sys
import tkinter as tk
from utils.logger import setup_logger
from config.settings import ConfigManager
from camera.camera_manager import CameraManager
from gui.main_window import HandGlidGUI

def main():
    logger = setup_logger()
    logger.info("Application started")

    config = ConfigManager()
    camera = CameraManager(config)

    if not camera.start():
        logger.error("Exiting due to camera initialization failure.")
        sys.exit(1)

    try:
        app = HandGlidGUI(camera, config=config)
        app.mainloop()
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
    finally:
        if camera.is_running:
            camera.release()
        logger.info("Application stopped")

if __name__ == "__main__":
    main()
