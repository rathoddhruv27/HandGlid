import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import logging

logger = logging.getLogger(__name__)

# Set appearance mode and color theme
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class HandGlidGUI(ctk.CTk):
    def __init__(self, camera):
        super().__init__()
        
        self.camera = camera
        
        # Configure window
        self.title("HandGlid")
        self.geometry("1100x700")
        
        # Configure grid layout (1x2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- Sidebar Frame ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="HandGlid", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.tracking_switch = ctk.CTkSwitch(self.sidebar_frame, text="Tracking Active", font=ctk.CTkFont(size=14))
        self.tracking_switch.grid(row=1, column=0, padx=20, pady=20, sticky="w")
        self.tracking_switch.select() # Default to active
        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=3, column=0, padx=20, pady=(10, 10), sticky="w")
        
        self.quit_button = ctk.CTkButton(self.sidebar_frame, text="Quit Application", command=self.on_closing, fg_color="#D32F2F", hover_color="#B71C1C")
        self.quit_button.grid(row=5, column=0, padx=20, pady=20, sticky="s")
        
        # --- Main Video Frame ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        self.status_var = ctk.StringVar(value="Status: Running")
        self.status_bar = ctk.CTkLabel(self.main_frame, textvariable=self.status_var, text_color="gray", font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=0, pady=(0, 10), sticky="s")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start video loop
        self.update_video()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def update_video(self):
        """Fetches a frame from the camera and updates the Tkinter label."""
        if self.camera and self.camera.is_running:
            success, frame = self.camera.read_frame()
            if success:
                # Convert OpenCV BGR to RGB
                cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Dynamically calculate size based on the available space
                target_w = self.main_frame.winfo_width()
                target_h = self.main_frame.winfo_height() - 40 # Subtract status bar approx height
                
                if target_w > 10 and target_h > 10:
                    # Maintain aspect ratio
                    h, w, _ = frame.shape
                    scale = min(target_w / w, target_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    
                    cv_img = cv2.resize(cv_img, (new_w, new_h))
                    pil_img = Image.fromarray(cv_img)
                    self.photo = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))
                    
                    # Update label
                    self.video_label.configure(image=self.photo)
        
        # Schedule next update in 30ms (~33 FPS)
        self.after(30, self.update_video)

    def on_closing(self):
        """Handle window close event safely."""
        logger.info("Closing GUI...")
        if self.camera:
            self.camera.release()
        self.destroy()
