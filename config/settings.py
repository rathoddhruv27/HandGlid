import json
import os
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'config.json')

DEFAULT_CONFIG = {
    "camera_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "cursor_sensitivity": 1.0,
    "smoothing_factor": 0.7,
    "pinch_threshold": 0.05,
    "scroll_sensitivity": 1.0,
    "gesture_cooldown": 0.3,
    "mirror_camera": True
}

class ConfigManager:
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        self.config_path = config_path
        self.config = {}
        self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}. Loading defaults.")
            self.config = DEFAULT_CONFIG.copy()
            self.save_config()
            return

        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load config file: {e}. Falling back to defaults.")
            self.config = DEFAULT_CONFIG.copy()

    def save_config(self):
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info("Configuration saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save config file: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()
