import os
import time
import logging
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk

from config.settings import ConfigManager, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Appearance configuration
ctk.set_appearance_mode("Dark")  # "Dark", "Light", "System"
ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"


class HandGlidGUI(ctk.CTk):
    """
    Advanced desktop GUI for HandGlid - AI Air Mouse & Hand Gesture Tracking.
    Provides real-time camera preview, HUD overlays, sensitivity tuning,
    gesture cheat-sheets, system telemetry, and camera device controls.
    """

    def __init__(self, camera, config: ConfigManager = None):
        super().__init__()

        self.camera = camera
        self.config = config if config is not None else getattr(camera, 'config', ConfigManager())

        # State management
        self.is_tracking_active = True
        self.is_feed_paused = False
        self.active_mode = "Mouse"
        self.current_gesture = "Index Point (Glide)"

        # Overlay toggles
        self.show_tracking_box = True
        self.show_hud_overlay = True
        self.show_center_guide = False

        # Telemetry & performance metrics
        self.fps = 0.0
        self.frame_count = 0
        self.last_frame_time = time.perf_counter()
        self.latency_ms = 0.0
        self.last_fps_update = time.perf_counter()
        self.frame_times = []
        self._toast_timer_id = None
        self.latest_raw_frame = None

        # Window configuration
        self.title("HandGlid - AI Air Mouse & Gesture Controller")
        self.geometry("1260x800")
        self.minsize(1080, 680)

        # 2-column layout: Column 0 = Sidebar (355px), Column 1 = Main Viewport (expanding)
        self.grid_columnconfigure(0, weight=0, minsize=355)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Build UI components
        self._build_sidebar()
        self._build_main_viewport()

        # Keyboard shortcuts
        self._bind_shortcuts()

        # Window close handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initial logging
        self.log_event("HandGlid GUI initialized successfully.")
        self.log_event(f"Loaded config: sensitivity={self.config.get('cursor_sensitivity', 1.0)}x, smoothing={self.config.get('smoothing_factor', 0.7):.2f}")

        # Start live video loop
        self.update_video()

    # =========================================================================
    # UI BUILDERS
    # =========================================================================

    def _build_sidebar(self):
        """Builds the left sidebar with header, master controls, tabs, and footer."""
        self.sidebar_frame = ctk.CTkFrame(self, width=355, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)

        self.sidebar_frame.grid_rowconfigure(2, weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        # --- 1. Branding Header ---
        self.header_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, padx=18, pady=(16, 10), sticky="ew")

        title_row = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        title_row.pack(fill="x", anchor="w")

        self.logo_label = ctk.CTkLabel(
            title_row,
            text="🖐 HandGlid",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.pack(side="left")

        self.badge_label = ctk.CTkLabel(
            title_row,
            text="PRO",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#2563EB", "#1D4ED8"),
            text_color="#FFFFFF",
            corner_radius=6,
            padx=8,
            pady=2
        )
        self.badge_label.pack(side="left", padx=(10, 0))

        self.subtitle_label = ctk.CTkLabel(
            self.header_frame,
            text="AI Hand Tracking & Air Mouse",
            font=ctk.CTkFont(size=12),
            text_color=("gray50", "gray70")
        )
        self.subtitle_label.pack(anchor="w", pady=(2, 0))

        # --- 2. Master Engine Control Card ---
        self.engine_card = ctk.CTkFrame(
            self.sidebar_frame,
            corner_radius=10,
            border_width=1,
            border_color=("#E2E8F0", "#334155"),
            fg_color=("#F8FAFC", "#1E293B")
        )
        self.engine_card.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="ew")

        # Engine Status Row
        card_status_row = ctk.CTkFrame(self.engine_card, fg_color="transparent")
        card_status_row.pack(fill="x", padx=12, pady=(10, 4))

        self.engine_title = ctk.CTkLabel(
            card_status_row,
            text="Tracking Engine",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.engine_title.pack(side="left")

        self.status_pill = ctk.CTkLabel(
            card_status_row,
            text="● Active",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#10B981"
        )
        self.status_pill.pack(side="right")

        # Master Toggle Switch
        self.master_switch = ctk.CTkSwitch(
            self.engine_card,
            text="Enable Air Mouse",
            font=ctk.CTkFont(size=13),
            command=self.on_toggle_tracking
        )
        self.master_switch.pack(anchor="w", padx=12, pady=(4, 8))
        self.master_switch.select()

        # Mode Segmented Button
        self.mode_label = ctk.CTkLabel(
            self.engine_card,
            text="Operational Mode:",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray70")
        )
        self.mode_label.pack(anchor="w", padx=12, pady=(2, 2))

        self.mode_segmented = ctk.CTkSegmentedButton(
            self.engine_card,
            values=["Mouse", "Scroll", "Present"],
            command=self.on_change_mode,
            font=ctk.CTkFont(size=12)
        )
        self.mode_segmented.pack(fill="x", padx=12, pady=(0, 10))
        self.mode_segmented.set("Mouse")

        # --- 3. Navigation Tabview ---
        self.tabview = ctk.CTkTabview(self.sidebar_frame, corner_radius=10)
        self.tabview.grid(row=2, column=0, padx=16, pady=(0, 10), sticky="nsew")

        # Configure tab bar font for clean rendering
        self.tab_controls = self.tabview.add("Controls")
        self.tab_tuning = self.tabview.add("Tuning")
        self.tab_gestures = self.tabview.add("Gestures")
        self.tab_stats = self.tabview.add("Stats")

        try:
            self.tabview._segmented_button.configure(font=ctk.CTkFont(size=12, weight="bold"))
        except Exception:
            pass

        self._build_tab_controls()
        self._build_tab_tuning()
        self._build_tab_gestures()
        self._build_tab_stats()

        # --- 4. Sidebar Footer ---
        self.footer_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.footer_frame.grid(row=3, column=0, padx=16, pady=(0, 14), sticky="ew")

        # Appearance Mode Selector
        theme_row = ctk.CTkFrame(self.footer_frame, fg_color="transparent")
        theme_row.pack(fill="x", pady=(0, 8))

        theme_lbl = ctk.CTkLabel(theme_row, text="Theme:", font=ctk.CTkFont(size=12))
        theme_lbl.pack(side="left")

        self.appearance_menu = ctk.CTkOptionMenu(
            theme_row,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode_event,
            width=140,
            font=ctk.CTkFont(size=12)
        )
        self.appearance_menu.pack(side="right")
        self.appearance_menu.set("Dark")

        # Quit Button
        self.quit_button = ctk.CTkButton(
            self.footer_frame,
            text="✕  Quit Application",
            command=self.on_closing,
            fg_color="#EF4444",
            hover_color="#DC2626",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=34
        )
        self.quit_button.pack(fill="x")

    def _build_tab_controls(self):
        """Builds Controls tab: camera source, overlays, snapshot."""
        parent = self.tab_controls

        # Camera Source Group
        cam_group = ctk.CTkFrame(parent, fg_color="transparent")
        cam_group.pack(fill="x", pady=(4, 10))

        cam_lbl = ctk.CTkLabel(cam_group, text="Camera Device:", font=ctk.CTkFont(size=12, weight="bold"))
        cam_lbl.pack(anchor="w", pady=(0, 4))

        cam_select_row = ctk.CTkFrame(cam_group, fg_color="transparent")
        cam_select_row.pack(fill="x")

        curr_cam = self.config.get("camera_index", 0)
        self.cam_menu = ctk.CTkOptionMenu(
            cam_select_row,
            values=["Camera 0 (Default)", "Camera 1", "Camera 2"],
            command=self.on_select_camera,
            font=ctk.CTkFont(size=12)
        )
        self.cam_menu.set(f"Camera {curr_cam}" + (" (Default)" if curr_cam == 0 else ""))
        self.cam_menu.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_reconnect = ctk.CTkButton(
            cam_select_row,
            text="🔄",
            width=36,
            command=self.reconnect_camera,
            font=ctk.CTkFont(size=14)
        )
        self.btn_reconnect.pack(side="right")

        # Video Stream Actions
        actions_row = ctk.CTkFrame(parent, fg_color="transparent")
        actions_row.pack(fill="x", pady=(0, 12))

        self.btn_pause = ctk.CTkButton(
            actions_row,
            text="⏸ Pause Feed",
            command=self.toggle_stream_pause,
            font=ctk.CTkFont(size=12),
            fg_color=("#475569", "#334155"),
            hover_color=("#334155", "#1E293B")
        )
        self.btn_pause.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_snapshot = ctk.CTkButton(
            actions_row,
            text="📸 Snapshot",
            command=self.take_snapshot,
            font=ctk.CTkFont(size=12),
            fg_color=("#0284C7", "#0369A1"),
            hover_color=("#0369A1", "#075985")
        )
        self.btn_snapshot.pack(side="right", fill="x", expand=True)

        # Display Overlays Group
        overlay_lbl = ctk.CTkLabel(parent, text="Visual Overlays:", font=ctk.CTkFont(size=12, weight="bold"))
        overlay_lbl.pack(anchor="w", pady=(4, 6))

        # Mirror Camera Switch
        is_mirror = self.config.get("mirror_camera", True)
        self.mirror_switch = ctk.CTkSwitch(
            parent,
            text="Mirror Camera Feed",
            font=ctk.CTkFont(size=12),
            command=self.toggle_mirror
        )
        self.mirror_switch.pack(anchor="w", pady=4)
        if is_mirror:
            self.mirror_switch.select()

        # Tracking Box Switch
        self.box_switch = ctk.CTkSwitch(
            parent,
            text="Active Tracking Zone",
            font=ctk.CTkFont(size=12),
            command=self.toggle_tracking_box
        )
        self.box_switch.pack(anchor="w", pady=4)
        self.box_switch.select()

        # HUD Overlay Switch
        self.hud_switch = ctk.CTkSwitch(
            parent,
            text="Telemetry HUD Overlay",
            font=ctk.CTkFont(size=12),
            command=self.toggle_hud_overlay
        )
        self.hud_switch.pack(anchor="w", pady=4)
        self.hud_switch.select()

        # Center Crosshair Switch
        self.crosshair_switch = ctk.CTkSwitch(
            parent,
            text="Center Crosshair Guide",
            font=ctk.CTkFont(size=12),
            command=self.toggle_center_guide
        )
        self.crosshair_switch.pack(anchor="w", pady=4)

    def _build_tab_tuning(self):
        """Builds Tuning tab with sliders for sensitivity, smoothing, pinch, and scroll."""
        scroll_frame = ctk.CTkScrollableFrame(self.tab_tuning, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True)

        # 1. Cursor Sensitivity
        sens = float(self.config.get("cursor_sensitivity", 1.0))
        self.slider_sens, self.lbl_sens_val = self._create_slider_row(
            scroll_frame,
            title="Cursor Sensitivity",
            desc="Cursor displacement speed multiplier",
            min_val=0.2,
            max_val=3.0,
            init_val=sens,
            fmt="{:.2f}x",
            callback=lambda val: self._on_slider_change("cursor_sensitivity", val, self.lbl_sens_val, "{:.2f}x")
        )

        # 2. Smoothing Factor
        smooth = float(self.config.get("smoothing_factor", 0.7))
        self.slider_smooth, self.lbl_smooth_val = self._create_slider_row(
            scroll_frame,
            title="Motion Smoothing",
            desc="Higher values reduce jitter (0.1 to 0.95)",
            min_val=0.1,
            max_val=0.95,
            init_val=smooth,
            fmt="{:.2f}",
            callback=lambda val: self._on_slider_change("smoothing_factor", val, self.lbl_smooth_val, "{:.2f}")
        )

        # 3. Pinch Threshold
        pinch = float(self.config.get("pinch_threshold", 0.05))
        self.slider_pinch, self.lbl_pinch_val = self._create_slider_row(
            scroll_frame,
            title="Click Detection (Pinch)",
            desc="Thumb-index contact distance threshold",
            min_val=0.01,
            max_val=0.15,
            init_val=pinch,
            fmt="{:.3f}",
            callback=lambda val: self._on_slider_change("pinch_threshold", val, self.lbl_pinch_val, "{:.3f}")
        )

        # 4. Scroll Sensitivity
        scroll = float(self.config.get("scroll_sensitivity", 1.0))
        self.slider_scroll, self.lbl_scroll_val = self._create_slider_row(
            scroll_frame,
            title="Scroll Speed",
            desc="Velocity multiplier during scroll gesture",
            min_val=0.2,
            max_val=4.0,
            init_val=scroll,
            fmt="{:.2f}x",
            callback=lambda val: self._on_slider_change("scroll_sensitivity", val, self.lbl_scroll_val, "{:.2f}x")
        )

        # 5. Gesture Cooldown
        cooldown = float(self.config.get("gesture_cooldown", 0.3))
        self.slider_cooldown, self.lbl_cooldown_val = self._create_slider_row(
            scroll_frame,
            title="Click Cooldown",
            desc="Delay between consecutive click triggers",
            min_val=0.05,
            max_val=1.0,
            init_val=cooldown,
            fmt="{:.2f}s",
            callback=lambda val: self._on_slider_change("gesture_cooldown", val, self.lbl_cooldown_val, "{:.2f}s")
        )

        # Action Buttons
        btn_box = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        btn_box.pack(fill="x", pady=(12, 6))

        self.btn_save_config = ctk.CTkButton(
            btn_box,
            text="💾 Save Config",
            command=self.on_save_config,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#16A34A", "#15803D"),
            hover_color=("#15803D", "#166534")
        )
        self.btn_save_config.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_reset_config = ctk.CTkButton(
            btn_box,
            text="↺ Reset",
            command=self.on_reset_defaults,
            font=ctk.CTkFont(size=12),
            fg_color=("#475569", "#334155"),
            hover_color=("#334155", "#1E293B")
        )
        self.btn_reset_config.pack(side="right", fill="x", expand=True)

    def _create_slider_row(self, parent, title, desc, min_val, max_val, init_val, fmt, callback):
        """Helper to create a slider with title, description, and dynamic value badge."""
        card = ctk.CTkFrame(parent, fg_color=("gray90", "#1E293B"), corner_radius=8)
        card.pack(fill="x", pady=4, padx=2)

        header = ctk.CTkFrame(card, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 0))

        title_lbl = ctk.CTkLabel(header, text=title, font=ctk.CTkFont(size=12, weight="bold"))
        title_lbl.pack(side="left")

        val_lbl = ctk.CTkLabel(
            header,
            text=fmt.format(init_val),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#2563EB", "#60A5FA")
        )
        val_lbl.pack(side="right")

        slider = ctk.CTkSlider(
            card,
            from_=min_val,
            to=max_val,
            command=callback
        )
        slider.set(init_val)
        slider.pack(fill="x", padx=10, pady=(4, 2))

        desc_lbl = ctk.CTkLabel(
            card,
            text=desc,
            font=ctk.CTkFont(size=10),
            text_color=("gray50", "gray70")
        )
        desc_lbl.pack(anchor="w", padx=10, pady=(0, 6))

        return slider, val_lbl

    def _build_tab_gestures(self):
        """Builds Gestures cheat-sheet tab with visual cards."""
        scroll_frame = ctk.CTkScrollableFrame(self.tab_gestures, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True)

        gestures = [
            ("Point & Glide", "Move cursor smoothly across the screen using index finger position."),
            ("Pinch (Thumb + Index)", "Triggers Left Mouse Click. Hold pinch to drag and drop."),
            ("Two Fingers Up", "Triggers Right Mouse Click / Context Menu."),
            ("Open Palm Motion", "Enables Vertical Scrolling (move hand up/down)."),
            ("Closed Fist", "Puts tracking on Standby to prevent unintended clicks.")
        ]

        for name, detail in gestures:
            card = ctk.CTkFrame(scroll_frame, fg_color=("gray90", "#1E293B"), corner_radius=8)
            card.pack(fill="x", pady=4, padx=2)

            name_lbl = ctk.CTkLabel(card, text=name, font=ctk.CTkFont(size=12, weight="bold"))
            name_lbl.pack(anchor="w", padx=10, pady=(6, 2))

            desc_lbl = ctk.CTkLabel(
                card,
                text=detail,
                font=ctk.CTkFont(size=11),
                text_color=("gray40", "gray70"),
                wraplength=250,
                justify="left"
            )
            desc_lbl.pack(anchor="w", padx=10, pady=(0, 6))

        # Real-time gesture status box
        status_box = ctk.CTkFrame(scroll_frame, fg_color=("#DBEAFE", "#1E3A8A"), corner_radius=8)
        status_box.pack(fill="x", pady=(10, 4), padx=2)

        st_title = ctk.CTkLabel(
            status_box,
            text="Detected Gesture",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=("#1E40AF", "#93C5FD")
        )
        st_title.pack(anchor="w", padx=10, pady=(6, 0))

        self.lbl_active_gesture_card = ctk.CTkLabel(
            status_box,
            text=self.current_gesture,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#1E3A8A", "#FFFFFF")
        )
        self.lbl_active_gesture_card.pack(anchor="w", padx=10, pady=(2, 6))

    def _build_tab_stats(self):
        """Builds Stats & Diagnostics tab with telemetry metrics and event log."""
        parent = self.tab_stats

        # 2x2 Telemetry Grid
        grid_frame = ctk.CTkFrame(parent, fg_color="transparent")
        grid_frame.pack(fill="x", pady=(4, 8))
        grid_frame.grid_columnconfigure((0, 1), weight=1)

        self.card_fps, self.lbl_fps_val = self._create_stat_tile(grid_frame, 0, 0, "Video FPS", "0 FPS")
        self.card_lat, self.lbl_lat_val = self._create_stat_tile(grid_frame, 0, 1, "Frame Latency", "0 ms")
        self.card_res, self.lbl_res_val = self._create_stat_tile(grid_frame, 1, 0, "Resolution", "1280x720")
        self.card_cnt, self.lbl_cnt_val = self._create_stat_tile(grid_frame, 1, 1, "Frames", "0")

        # Event Activity Log
        log_lbl = ctk.CTkLabel(parent, text="Activity Log:", font=ctk.CTkFont(size=12, weight="bold"))
        log_lbl.pack(anchor="w", pady=(6, 2))

        self.log_textbox = ctk.CTkTextbox(
            parent,
            font=ctk.CTkFont(family="monospace", size=10),
            height=130,
            corner_radius=8
        )
        self.log_textbox.pack(fill="both", expand=True, pady=(0, 4))
        self.log_textbox.configure(state="disabled")

    def _create_stat_tile(self, parent, r, c, label, val):
        """Helper to create a telemetry metric card."""
        frame = ctk.CTkFrame(parent, fg_color=("gray90", "#1E293B"), corner_radius=8)
        frame.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

        lbl = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=10), text_color=("gray50", "gray70"))
        lbl.pack(anchor="w", padx=8, pady=(4, 0))

        val_lbl = ctk.CTkLabel(frame, text=val, font=ctk.CTkFont(size=14, weight="bold"))
        val_lbl.pack(anchor="w", padx=8, pady=(0, 4))
        return frame, val_lbl

    def _build_main_viewport(self):
        """Builds the right main panel: HUD header, video canvas, and bottom status bar."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=12)
        self.main_frame.grid(row=0, column=1, padx=(10, 18), pady=16, sticky="nsew")

        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # --- Top Bar: Stream Info & Chips ---
        self.top_bar = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.top_bar.grid(row=0, column=0, padx=14, pady=(10, 6), sticky="ew")

        # Left: Live Pill & Camera Info
        top_left = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        top_left.pack(side="left")

        self.live_badge = ctk.CTkLabel(
            top_left,
            text="● LIVE",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=("#DCFCE7", "#064E3B"),
            text_color="#10B981",
            corner_radius=6,
            padx=8,
            pady=2
        )
        self.live_badge.pack(side="left", padx=(0, 8))

        curr_cam = self.config.get("camera_index", 0)
        res_w, res_h = self._get_stream_resolution()
        self.cam_info_lbl = ctk.CTkLabel(
            top_left,
            text=f"Camera {curr_cam} • {res_w}×{res_h}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=("gray30", "gray80")
        )
        self.cam_info_lbl.pack(side="left")

        # Right: HUD Chips
        top_right = ctk.CTkFrame(self.top_bar, fg_color="transparent")
        top_right.pack(side="right")

        self.chip_fps = self._create_hud_chip(top_right, "FPS: 30")
        self.chip_lat = self._create_hud_chip(top_right, "12 ms")
        self.chip_mode = self._create_hud_chip(top_right, "Mode: Mouse")
        self.chip_gesture = self._create_hud_chip(top_right, "Gesture: Ready")

        # --- Center: Video Container Frame ---
        self.video_container = ctk.CTkFrame(
            self.main_frame,
            corner_radius=10,
            fg_color=("#0F172A", "#0B0F17")
        )
        self.video_container.grid(row=1, column=0, padx=14, pady=4, sticky="nsew")
        self.video_container.grid_rowconfigure(0, weight=1)
        self.video_container.grid_columnconfigure(0, weight=1)

        # The actual video display label
        self.video_label = ctk.CTkLabel(self.video_container, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # Stream Paused Overlay (initially hidden)
        self.pause_overlay = ctk.CTkFrame(
            self.video_container,
            fg_color=("#1E293B", "#0F172A"),
            corner_radius=12,
            border_width=1,
            border_color="#334155"
        )
        pause_icon = ctk.CTkLabel(self.pause_overlay, text="⏸️", font=ctk.CTkFont(size=40))
        pause_icon.pack(pady=(20, 6))

        pause_title = ctk.CTkLabel(
            self.pause_overlay,
            text="Camera Stream Paused",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        pause_title.pack(pady=(0, 4))

        pause_sub = ctk.CTkLabel(
            self.pause_overlay,
            text="Tracking is on hold. Press Space or click below to resume.",
            font=ctk.CTkFont(size=12),
            text_color="gray70"
        )
        pause_sub.pack(pady=(0, 16))

        btn_resume = ctk.CTkButton(
            self.pause_overlay,
            text="▶ Resume Stream",
            command=self.toggle_stream_pause,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669"
        )
        btn_resume.pack(pady=(0, 20))

        # --- Bottom Bar: Status, Toast, and Hotkeys ---
        self.bottom_bar = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.bottom_bar.grid(row=2, column=0, padx=14, pady=(6, 10), sticky="ew")

        self.status_var = ctk.StringVar(value="🟢 System Status: Active • Tracking Hand")
        self.status_bar_label = ctk.CTkLabel(
            self.bottom_bar,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray70")
        )
        self.status_bar_label.pack(side="left")

        # Toast notification label (center)
        self.toast_var = ctk.StringVar(value="")
        self.toast_label = ctk.CTkLabel(
            self.bottom_bar,
            textvariable=self.toast_var,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#10B981"
        )
        self.toast_label.pack(side="left", padx=(20, 0))

        # Hotkeys cheat sheet (right)
        self.hotkey_hint = ctk.CTkLabel(
            self.bottom_bar,
            text="Shortcuts: [Space] Pause  [M] Mirror  [S] Snapshot  [Esc] Quit",
            font=ctk.CTkFont(size=11),
            text_color=("gray50", "gray60")
        )
        self.hotkey_hint.pack(side="right")

    def _create_hud_chip(self, parent, text):
        """Helper to create a small HUD chip on top-right of main viewport."""
        chip = ctk.CTkLabel(
            parent,
            text=text,
            font=ctk.CTkFont(size=11),
            fg_color=("gray85", "#1E293B"),
            corner_radius=6,
            padx=8,
            pady=2
        )
        chip.pack(side="left", padx=4)
        return chip

    def _bind_shortcuts(self):
        """Binds keyboard shortcuts for quick control."""
        self.bind("<space>", lambda e: self.toggle_stream_pause())
        self.bind("<p>", lambda e: self.toggle_stream_pause())
        self.bind("<P>", lambda e: self.toggle_stream_pause())
        self.bind("<m>", lambda e: self.toggle_mirror())
        self.bind("<M>", lambda e: self.toggle_mirror())
        self.bind("<s>", lambda e: self.take_snapshot())
        self.bind("<S>", lambda e: self.take_snapshot())
        self.bind("<Escape>", lambda e: self.on_closing())

    # =========================================================================
    # VIDEO PROCESSING & OVERLAYS
    # =========================================================================

    def update_video(self):
        """Fetches a frame from the camera, applies overlays, and updates GUI."""
        if self.camera and self.camera.is_running and not self.is_feed_paused:
            now = time.perf_counter()
            dt = now - self.last_frame_time
            self.last_frame_time = now

            # Frame rate smoothing
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * inst_fps if self.fps > 0 else inst_fps

            t_start = time.perf_counter()
            success, frame = self.camera.read_frame()
            self.latency_ms = (time.perf_counter() - t_start) * 1000.0

            if success and frame is not None:
                self.frame_count += 1
                self.latest_raw_frame = frame.copy()

                # Draw overlays if enabled
                annotated_frame = self._draw_overlays(frame)

                # Convert OpenCV BGR to RGB
                cv_img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Dynamically fit within container while maintaining aspect ratio
                target_w = max(100, self.video_container.winfo_width())
                target_h = max(100, self.video_container.winfo_height())

                h, w, _ = cv_img.shape
                scale = min(target_w / w, target_h / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))

                cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                pil_img = Image.fromarray(cv_img)
                self.photo = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))

                self.video_label.configure(image=self.photo)

            # Periodically update HUD readouts (every 100ms)
            if now - self.last_fps_update > 0.1:
                self._update_telemetry_ui()
                self.last_fps_update = now

        # Schedule next frame (~30 FPS)
        self.after(30, self.update_video)

    def _draw_overlays(self, frame):
        """Draws tracking zone brackets, center crosshair, and HUD onto OpenCV frame."""
        h, w, _ = frame.shape
        img = frame.copy()

        # 1. Active Tracking Zone Brackets
        if self.show_tracking_box:
            # 70% active area
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            x1, y1 = margin_x, margin_y
            x2, y2 = w - margin_x, h - margin_y

            color = (255, 215, 0) if self.is_tracking_active else (80, 160, 240)  # Cyan if active, Amber if standby
            corner_len = min(40, int(w * 0.06))
            thick = 2

            # Top-Left corner
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thick)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thick)
            # Top-Right corner
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thick)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thick)
            # Bottom-Left corner
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thick)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thick)
            # Bottom-Right corner
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thick)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thick)

            # Subtle boundary line
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)

            # Label neatly inside the tracking zone
            tag_text = "ACTIVE TRACKING ZONE" if self.is_tracking_active else "TRACKING PAUSED (STANDBY)"
            cv2.putText(
                img,
                tag_text,
                (x1 + 10, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                color,
                1,
                cv2.LINE_AA
            )

        # 2. Center Crosshair Guide
        if self.show_center_guide:
            cx, cy = w // 2, h // 2
            guide_color = (0, 255, 255)
            cv2.line(img, (cx - 15, cy), (cx + 15, cy), guide_color, 1)
            cv2.line(img, (cx, cy - 15), (cx, cy + 15), guide_color, 1)
            cv2.circle(img, (cx, cy), 3, guide_color, -1)

        # 3. Semi-transparent On-Frame HUD Pill
        if self.show_hud_overlay:
            hud_text = f"HandGlid Engine  |  FPS: {self.fps:.0f}  |  {self.active_mode}"
            (tw, th), _ = cv2.getTextSize(hud_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            px, py = 20, 18
            pw, ph = tw + 36, th + 16

            if py + ph < h and px + pw < w:
                sub = img[py:py+ph, px:px+pw]
                dark_bg = np.full_like(sub, (20, 24, 33))
                cv2.addWeighted(sub, 0.35, dark_bg, 0.65, 0, sub)
                img[py:py+ph, px:px+pw] = sub
                cv2.rectangle(img, (px, py), (px+pw, py+ph), (70, 80, 100), 1)

                dot_color = (0, 220, 120) if self.is_tracking_active else (0, 180, 240)
                cv2.circle(img, (px + 12, py + ph // 2), 4, dot_color, -1)
                cv2.putText(
                    img,
                    hud_text,
                    (px + 24, py + ph // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (240, 245, 255),
                    1,
                    cv2.LINE_AA
                )

        return img

    def _update_telemetry_ui(self):
        """Updates text and badges for FPS, latency, and system status."""
        fps_text = f"FPS: {self.fps:.0f}"
        lat_text = f"{self.latency_ms:.0f} ms"

        self.chip_fps.configure(text=fps_text)
        self.chip_lat.configure(text=lat_text)
        self.lbl_fps_val.configure(text=f"{self.fps:.1f} FPS")
        self.lbl_lat_val.configure(text=f"{self.latency_ms:.1f} ms")
        self.lbl_cnt_val.configure(text=f"{self.frame_count:,}")

        res_w, res_h = self._get_stream_resolution()
        self.lbl_res_val.configure(text=f"{res_w}×{res_h}")
        curr_cam = self.config.get("camera_index", 0)
        self.cam_info_lbl.configure(text=f"Camera {curr_cam} • {res_w}×{res_h}")

    def _get_stream_resolution(self):
        """Retrieves stream resolution from camera or config."""
        if hasattr(self.camera, 'get_resolution'):
            return self.camera.get_resolution()
        return self.config.get("camera_width", 1280), self.config.get("camera_height", 720)

    # =========================================================================
    # EVENT HANDLERS & USER ACTIONS
    # =========================================================================

    def on_toggle_tracking(self):
        """Toggles the master air-mouse tracking engine."""
        self.is_tracking_active = self.master_switch.get() == 1
        if self.is_tracking_active:
            self.status_pill.configure(text="● Active", text_color="#10B981")
            self.status_var.set("🟢 System Status: Active • Tracking Hand")
            self.current_gesture = f"{self.active_mode} Active"
            self.lbl_active_gesture_card.configure(text=self.current_gesture)
            self.show_toast("Air Mouse Tracking Activated")
            self.log_event("Air Mouse tracking resumed.")
        else:
            self.status_pill.configure(text="● Standby", text_color="#F59E0B")
            self.status_var.set("🟡 System Status: Standby • Tracking Suspended")
            self.current_gesture = "Standby (Paused)"
            self.lbl_active_gesture_card.configure(text=self.current_gesture)
            self.show_toast("Tracking Suspended (Standby)")
            self.log_event("Air Mouse tracking put on Standby.")

    def on_change_mode(self, mode: str):
        """Handles switching between Mouse, Scroll, and Present modes."""
        self.active_mode = mode
        self.chip_mode.configure(text=f"Mode: {mode}")
        if self.is_tracking_active:
            if mode == "Mouse":
                self.current_gesture = "Index Point (Glide & Click)"
            elif mode == "Scroll":
                self.current_gesture = "Open Palm (Vertical Scroll)"
            else:
                self.current_gesture = "Two Fingers (Slide Navigation)"
            self.lbl_active_gesture_card.configure(text=self.current_gesture)
        self.log_event(f"Operational mode switched to '{mode}'.")
        self.show_toast(f"Switched to {mode} Mode")

    def toggle_stream_pause(self):
        """Pauses or resumes the live video preview."""
        self.is_feed_paused = not self.is_feed_paused
        if self.is_feed_paused:
            self.btn_pause.configure(text="▶ Resume Feed")
            self.live_badge.configure(text="⏸ PAUSED", text_color="#F59E0B", fg_color=("#FEF3C7", "#78350F"))
            self.pause_overlay.place(relx=0.5, rely=0.5, anchor="center")
            self.show_toast("Video stream paused")
            self.log_event("Camera preview paused.")
        else:
            self.btn_pause.configure(text="⏸ Pause Feed")
            self.live_badge.configure(text="● LIVE", text_color="#10B981", fg_color=("#DCFCE7", "#064E3B"))
            self.pause_overlay.place_forget()
            self.show_toast("Video stream resumed")
            self.log_event("Camera preview resumed.")

    def toggle_mirror(self):
        """Toggles horizontal camera flip in config and UI."""
        new_val = not self.config.get("mirror_camera", True)
        self.config.set("mirror_camera", new_val)
        if hasattr(self, 'mirror_switch'):
            if new_val:
                self.mirror_switch.select()
            else:
                self.mirror_switch.deselect()
        self.show_toast(f"Mirror Camera: {'ON' if new_val else 'OFF'}")
        self.log_event(f"Mirror camera set to {new_val}.")

    def toggle_tracking_box(self):
        """Toggles drawing of active tracking bounds."""
        self.show_tracking_box = self.box_switch.get() == 1
        self.log_event(f"Active tracking zone display: {self.show_tracking_box}.")

    def toggle_hud_overlay(self):
        """Toggles telemetry text watermark on the video feed."""
        self.show_hud_overlay = self.hud_switch.get() == 1
        self.log_event(f"Telemetry HUD overlay: {self.show_hud_overlay}.")

    def toggle_center_guide(self):
        """Toggles center crosshair guide on video feed."""
        self.show_center_guide = self.crosshair_switch.get() == 1
        self.log_event(f"Center crosshair guide: {self.show_center_guide}.")

    def take_snapshot(self):
        """Captures and saves current frame to screenshots directory."""
        if self.latest_raw_frame is None:
            self.show_toast("No frame available to capture.")
            return

        shots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "screenshots")
        os.makedirs(shots_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"handglid_snapshot_{timestamp}.png"
        filepath = os.path.join(shots_dir, filename)

        cv2.imwrite(filepath, self.latest_raw_frame)
        self.show_toast(f"📸 Snapshot saved to screenshots/{filename}")
        self.log_event(f"Saved snapshot to {filepath}")

    def on_select_camera(self, choice: str):
        """Switches the camera index."""
        try:
            index = int(choice.split()[1])
            if index != self.config.get("camera_index", 0):
                self.reconnect_camera(new_index=index)
        except Exception as e:
            logger.error(f"Failed to parse camera selection '{choice}': {e}")

    def reconnect_camera(self, new_index: int = None):
        """Safely restarts the camera stream."""
        if new_index is not None:
            target_idx = new_index
        else:
            target_idx = self.config.get("camera_index", 0)

        self.log_event(f"Reconnecting to Camera index {target_idx}...")
        self.show_toast(f"Reconnecting Camera {target_idx}...")

        if hasattr(self.camera, 'restart'):
            success = self.camera.restart(camera_index=target_idx)
        else:
            if self.camera:
                self.camera.release()
            self.config.set("camera_index", target_idx)
            success = self.camera.start()

        if success:
            res_w, res_h = self._get_stream_resolution()
            self.cam_info_lbl.configure(text=f"Camera {target_idx} • {res_w}×{res_h}")
            self.show_toast("Camera connected successfully!")
            self.log_event(f"Camera {target_idx} connected ({res_w}×{res_h}).")
        else:
            self.show_toast("⚠️ Failed to open camera device.")
            self.log_event(f"Error: Could not open camera {target_idx}.")

    def _on_slider_change(self, key, value, label_widget, fmt):
        """Updates live slider label and memory config."""
        label_widget.configure(text=fmt.format(value))
        self.config.config[key] = round(value, 4)

    def on_save_config(self):
        """Saves current tuning parameters to config.json."""
        self.config.save_config()
        self.show_toast("✓ Settings saved to config.json!")
        self.log_event("Configuration saved to disk.")

    def on_reset_defaults(self):
        """Resets sliders and settings to default values."""
        for key, val in DEFAULT_CONFIG.items():
            self.config.set(key, val)

        self.slider_sens.set(DEFAULT_CONFIG["cursor_sensitivity"])
        self.lbl_sens_val.configure(text=f"{DEFAULT_CONFIG['cursor_sensitivity']:.2f}x")

        self.slider_smooth.set(DEFAULT_CONFIG["smoothing_factor"])
        self.lbl_smooth_val.configure(text=f"{DEFAULT_CONFIG['smoothing_factor']:.2f}")

        self.slider_pinch.set(DEFAULT_CONFIG["pinch_threshold"])
        self.lbl_pinch_val.configure(text=f"{DEFAULT_CONFIG['pinch_threshold']:.3f}")

        self.slider_scroll.set(DEFAULT_CONFIG["scroll_sensitivity"])
        self.lbl_scroll_val.configure(text=f"{DEFAULT_CONFIG['scroll_sensitivity']:.2f}x")

        self.slider_cooldown.set(DEFAULT_CONFIG["gesture_cooldown"])
        self.lbl_cooldown_val.configure(text=f"{DEFAULT_CONFIG['gesture_cooldown']:.2f}s")

        self.show_toast("↺ Settings reset to factory defaults.")
        self.log_event("Settings reset to defaults.")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        """Switches CustomTkinter appearance mode."""
        ctk.set_appearance_mode(new_appearance_mode)
        self.log_event(f"Appearance mode changed to {new_appearance_mode}.")

    def show_toast(self, message: str, duration_ms: int = 3000):
        """Displays a temporary notification in the bottom bar."""
        self.toast_var.set(message)
        if self._toast_timer_id is not None:
            self.after_cancel(self._toast_timer_id)
        self._toast_timer_id = self.after(duration_ms, lambda: self.toast_var.set(""))

    def log_event(self, message: str):
        """Appends a timestamped message to the activity log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}\n"
        logger.info(f"GUI: {message}")

        if hasattr(self, 'log_textbox'):
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", entry)
            self.log_textbox.see("end")
            self.log_textbox.configure(state="disabled")

    def on_closing(self):
        """Handle window close event safely."""
        logger.info("Closing GUI...")
        if self.camera:
            self.camera.release()
        self.destroy()
