import time
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# ---------------------------------------------------------------------------
# PPE Model Class Map — best.pt (YOLOv11 8-Class Model)
# Exact model classes:
# 0: person, 1: helmet, 2: no_helmet, 3: vest, 4: no_vest,
# 5: gloves, 6: eye_protection, 7: footwear
# ---------------------------------------------------------------------------
PPE_CLASSES: Dict[str, int] = {
    "person":         0,
    "helmet":         1,
    "no_helmet":      2,
    "vest":           3,
    "no_vest":        4,
    "gloves":         5,
    "eye_protection": 6,
    "footwear":       7,
}

# Inverse lookup: class_id -> class_name
PPE_CLASS_NAMES: Dict[int, str] = {
    0: "person",
    1: "helmet",
    2: "no_helmet",
    3: "vest",
    4: "no_vest",
    5: "gloves",
    6: "eye_protection",
    7: "footwear",
}

# Standardized display specifications for each model class (BGR colors)
CLASS_DISPLAY_CONFIG: Dict[int, Dict] = {
    0: {"label": "PERSON",         "color": (30, 214, 112), "is_violation": False},  # Emerald Green
    1: {"label": "HELMET",         "color": (30, 214, 112), "is_violation": False},  # Emerald Green
    2: {"label": "NO HELMET",      "color": (0, 0, 255),    "is_violation": True},   # Pure Red
    3: {"label": "VEST",           "color": (30, 214, 112), "is_violation": False},  # Emerald Green
    4: {"label": "NO VEST",        "color": (0, 0, 255),    "is_violation": True},   # Pure Red
    5: {"label": "GLOVES",         "color": (255, 191, 0),  "is_violation": False},  # Cyan/Teal
    6: {"label": "EYE PROTECTION", "color": (230, 150, 30), "is_violation": False},  # Sky Blue
    7: {"label": "FOOTWEAR",       "color": (30, 214, 112), "is_violation": False},  # Emerald Green
}

PPE_DETECT_CLASSES: List[int] = [0, 1, 2, 3, 4, 5, 6, 7]

DIRECT_VIOLATION_CLASSES: Dict[int, str] = {
    2: "missing_helmet",
    4: "missing_vest",
}


def item_belongs_to_person(item_box: List[float], person_box: List[float]) -> bool:
    """
    Checks if an item detection belongs to a person.
    Expands the person bounding box vertically and horizontally to account for:
    - Helmets worn on top of the head (above py1)
    - Footwear worn at the base of feet (at or below py2)
    - Outstretched arms / gloves (outside px1, px2)
    - Direct bounding box intersection
    """
    ix1, iy1, ix2, iy2 = item_box
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)

    # Expanded person bounds:
    # 25% height above head for helmets
    # 15% height below feet for footwear
    # 15% width on either side for hands/tools
    epx1 = px1 - 0.15 * pw
    epx2 = px2 + 0.15 * pw
    epy1 = py1 - 0.25 * ph
    epy2 = py2 + 0.15 * ph

    # 1. Centroid check within expanded bounds
    icx = (ix1 + ix2) / 2.0
    icy = (iy1 + iy2) / 2.0
    if (epx1 <= icx <= epx2) and (epy1 <= icy <= epy2):
        return True

    # 2. Box intersection check
    ix_min = max(ix1, px1)
    iy_min = max(iy1, py1)
    ix_max = min(ix2, px2)
    iy_max = min(iy2, py2)
    inter_w = max(0.0, ix_max - ix_min)
    inter_h = max(0.0, iy_max - iy_min)
    inter_area = inter_w * inter_h
    item_area = max(1.0, (ix2 - ix1) * (iy2 - iy1))

    # If >15% of item overlaps person box
    if (inter_area / item_area) >= 0.15:
        return True

    return False


def check_vest_compliance(
    person_box: List[float],
    vest_detections: List[Tuple],
    conf_threshold: float = 0.35,
) -> bool:
    """
    Check if a person is wearing a vest.
    """
    for vest_box, vest_conf in vest_detections:
        if vest_conf < conf_threshold:
            continue
        if item_belongs_to_person(vest_box, person_box):
            return True
    return False


def _item_on_person(
    person_box: List[float],
    item_detections: List[Tuple],
    conf_threshold: float = 0.15,
) -> bool:
    """
    Check if any positive PPE item detection belongs to the person.
    """
    for item_box, item_conf in item_detections:
        if item_conf < conf_threshold:
            continue
        if item_belongs_to_person(item_box, person_box):
            return True
    return False


def check_head_helmet_status(
    frame: np.ndarray,
    person_box: List[float],
    helmets: List,
    no_helmets: List,
) -> Tuple[bool, List[float], float, str, bool]:
    """
    Accurately determines helmet and face protection compliance:
    1. Returns (True, box, conf, label, has_visor=True/False) if:
       - YOLO explicitly detected a helmet, OR
       - The head region exhibits protective helmet shell characteristics (curved dome glare,
         smooth convex cranium, safety/dark shell) and YOLO did not detect exposed hair/bare head.
    2. Returns (False, box, conf, label, False) if:
       - YOLO explicitly detected no_helmet (exposed hair/scalp/face), OR
       - The head is bare without any protective headgear.
    """
    px1, py1, px2, py2 = person_box
    pw = max(1.0, px2 - px1)
    ph = max(1.0, py2 - py1)
    default_head_box = [px1 + 0.15 * pw, py1, px2 - 0.15 * pw, py1 + 0.35 * ph]

    # 1. Direct positive helmet detection from neural network
    if helmets:
        b, c, _, _ = helmets[0]
        return True, b, c, f"HELMET ✔ {c:.2f}", False

    # 2. Direct violation detection from neural network (bare head / hair detected)
    if no_helmets:
        b, c, _, _ = no_helmets[0]
        if c >= 0.20:
            return False, b, c, f"NO HELMET ✘ {c:.2f}", False

    # 3. Model did not output explicit no_helmet.
    # Examine head region for protective shell (motorcycle helmet, welding helmet, dark hard hat)
    hx1 = max(0, int(px1 + 0.12 * pw))
    hx2 = min(frame.shape[1], int(px2 - 0.12 * pw))
    hy1 = max(0, int(py1))
    hy2 = min(frame.shape[0], int(py1 + 0.38 * ph))
    head_box = [hx1, hy1, hx2, hy2]

    if (hx2 - hx1) < 15 or (hy2 - hy1) < 15:
        return False, default_head_box, 0.50, "NO HELMET ✘", False

    head_crop = frame[hy1:hy2, hx1:hx2]
    if head_crop.size == 0:
        return False, default_head_box, 0.50, "NO HELMET ✘", False

    gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
    top_crown = gray[:int(head_crop.shape[0] * 0.45), :]
    if top_crown.size == 0:
        return False, default_head_box, 0.50, "NO HELMET ✘", False

    max_glare = int(np.max(top_crown))
    mean_val = float(np.mean(top_crown))
    has_dome_glare = (max_glare > 175 and (max_glare - mean_val) > 40)

    # Edge density: hair has high fibrous strand density; helmet shell is smooth and curved
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edge_density = np.count_nonzero(edges) / max(1, gray.size)

    # Dark / safety tinted helmet shell or visor
    dark_ratio = float(np.mean(gray < 75))

    # A helmet exhibits specular glare on its dome, or smooth curvature with dark/tinted shell
    if has_dome_glare or dark_ratio > 0.35 or edge_density < 0.10:
        has_visor = (dark_ratio > 0.25 or has_dome_glare)
        return True, head_box, 0.88, "HELMET ✔ 0.88", has_visor
    else:
        return False, head_box, 0.70, "NO HELMET ✘", False


class Zone:
    def __init__(self, name: str, points: List[Dict[str, float]], zone_type: str = "warning", dwell_time: float = 0.0):
        self.name = name
        self.points = points  # List of {x, y} in 0-1 range
        self.type = zone_type  # "danger", "warning", etc.
        self.dwell_time = dwell_time
        self.occupied_since: Optional[float] = None
        self.is_breached = False
        self.is_pending = False
        self.alert_triggered = False
        self.last_alert_time: float = 0.0

    def contains(self, x: float, y: float, width: int, height: int) -> bool:
        if not self.points or len(self.points) < 3:
            return False
        poly = np.array([(p['x'] * width, p['y'] * height) for p in self.points], dtype=np.int32)
        return cv2.pointPolygonTest(poly, (int(x), int(y)), False) >= 0

    @property
    def is_cooldown(self) -> bool:
        """Returns True if the zone is in its alert cooldown window (5s to 10s)."""
        return (time.time() - self.last_alert_time) < 10.0


class IntrusionDetector:
    def __init__(self, confidence: float = 0.25, alert_cooldown: int = 10, model=None):
        if model is not None:
            self.model = model
        else:
            backend_dir = Path(__file__).resolve().parent.parent
            pt_file = backend_dir / "best.pt"
            if pt_file.exists():
                self.model = YOLO(str(pt_file), task="detect")
            else:
                self.model = YOLO("best.pt", task="detect")

        # Auto-select best available device: CUDA GPU > CPU
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Pre-warm the model with a dummy frame so first-frame latency is zero
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, conf=0.25, verbose=False, imgsz=320, device=self._device)
        except Exception:
            pass

        self.confidence = confidence
        # Global alert cooldown for fencing — minimum 3s for rapid zone re-alerting
        self.alert_cooldown = max(3.0, float(alert_cooldown if alert_cooldown else 10.0))
        # PPE-specific cooldown — strictly 5-10s interval per user specification
        self.ppe_alert_cooldown = max(5.0, min(10.0, float(alert_cooldown if alert_cooldown else 7.0)))
        self.last_alert_time = 0.0
        self.last_ppe_alert_time = 0.0

        # PPE Alert Dispatch Toggle:
        # If False: "Monitor Only" — draws bounding boxes on screen, dispatches NO alerts
        # If True:  "Alert Mode"  — saves to DB and sends email/telegram/call at 5-10s intervals
        self.ppe_alerts_enabled: bool = os.getenv("PPE_ALERTS_ENABLED", "false").lower() in ("true", "1", "yes")
        self.frame_height: int = 480
        self.frame_width: int = 640

        # Multi-camera zones: { "cam_01": [Zone, ...], "cam_02": [] }
        self.zones_map: Dict[str, List[Zone]] = {}
        self.active_camera_id = "cam_01"

        self.last_results = []
        self.inference_time = 0.0
        self.fps = 0.0
        self.last_annotated_frame = None
        self.intrusion_active = False
        self.current_status = "safe"
        self.intruder_count = 0

        # PPE compliance state — populated each frame by compute_ppe_compliance()
        self.all_detections: List[Tuple] = []   # (xyxy, conf, class_id, class_name)
        self.compliance_results: List[Dict] = []
        self.vest_iou_threshold: float = 0.2
        self.vest_conf_threshold: float = 0.35
        self.ppe_violation_conf_threshold: float = confidence
        self._init_class_mappings()

    @staticmethod
    def draw_badge_box(
        frame: np.ndarray,
        box: List[float],
        label: str,
        color: Tuple[int, int, int],
        thickness: int = 2,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        position: str = "top_left",
    ):
        """Draws a clean, industrial bounding box with an anti-aliased, collision-free header badge."""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
        pad_x, pad_y = 6, 4
        badge_w = tw + (pad_x * 2)
        badge_h = th + (pad_y * 2)

        if position == "top_left":
            bx1 = x1
            by2 = y1 if y1 >= badge_h + 4 else min(H, y1 + badge_h + 2)
            by1 = max(0, by2 - badge_h)
            bx2 = min(W, bx1 + badge_w)
        elif position == "top_right":
            bx2 = x2
            bx1 = max(0, bx2 - badge_w)
            by2 = y1 if y1 >= badge_h + 4 else min(H, y1 + badge_h + 2)
            by1 = max(0, by2 - badge_h)
        elif position == "inside_top":
            bx1 = x1 + 2
            bx2 = min(W, bx1 + badge_w)
            by1 = y1 + 2
            by2 = min(H, by1 + badge_h)
        elif position == "inside_top_right":
            bx2 = x2 - 2
            bx1 = max(0, bx2 - badge_w)
            by1 = y1 + 2
            by2 = min(H, by1 + badge_h)
        elif position == "inside_bottom":
            bx1 = x1 + 2
            bx2 = min(W, bx1 + badge_w)
            by2 = y2 - 2
            by1 = max(0, by2 - badge_h)
        else:
            bx1 = x1
            by2 = y1 if y1 >= badge_h + 4 else min(H, y1 + badge_h + 2)
            by1 = max(0, by2 - badge_h)
            bx2 = min(W, bx1 + badge_w)

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(frame, label, (bx1 + pad_x, by2 - pad_y), font, font_scale, text_color, font_thick, cv2.LINE_AA)

    def _init_class_mappings(self):
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, list):
            self.class_names = {i: str(n) for i, n in enumerate(names)}
        else:
            self.class_names = dict(PPE_CLASS_NAMES)

        name_to_id = {v.lower().strip().replace("-", "_").replace(" ", "_"): k for k, v in self.class_names.items()}

        def find_id(*candidates):
            for c in candidates:
                c_clean = c.lower().strip().replace("-", "_").replace(" ", "_")
                if c_clean in name_to_id:
                    return name_to_id[c_clean]
            for c in candidates:
                c_clean = c.lower().strip().replace("-", "_").replace(" ", "_")
                for name, cid in name_to_id.items():
                    if c_clean in name:
                        return cid
            return None

        self.person_class_id = find_id("person") if find_id("person") is not None else 0
        self.helmet_class_id = find_id("helmet", "hard_hat", "hardhat") if find_id("helmet") is not None else 1
        self.no_helmet_class_id = find_id("no_helmet", "no_hardhat") if find_id("no_helmet") is not None else 2
        self.vest_class_id = find_id("vest", "safety_vest") if find_id("vest") is not None else 3
        self.no_vest_class_id = find_id("no_vest", "without_vest") if find_id("no_vest") is not None else 4
        self.gloves_class_id = find_id("gloves", "glove", "safety_gloves") if find_id("gloves") is not None else 5
        self.eye_protection_class_id = find_id("eye_protection", "goggles", "glasses", "safety_glasses") if find_id("eye_protection") is not None else 6
        self.footwear_class_id = find_id("footwear", "boots", "shoes", "boot", "safety_boots") if find_id("footwear") is not None else 7

        # Aliases for backward compatibility
        self.goggles_class_id = self.eye_protection_class_id
        self.no_goggles_class_id = None
        self.boots_class_id = self.footwear_class_id
        self.no_boots_class_id = None
        self.no_gloves_class_id = None

        self.direct_violation_classes = {}
        if self.no_helmet_class_id is not None:
            self.direct_violation_classes[self.no_helmet_class_id] = "missing_helmet"
        if self.no_vest_class_id is not None:
            self.direct_violation_classes[self.no_vest_class_id] = "missing_vest"

    @property
    def zones(self) -> List[Zone]:
        return self.zones_map.get(self.active_camera_id, [])

    def set_zones(self, zones_data: List[Dict], camera_id: str = None) -> List[Zone]:
        target_cam = camera_id or self.active_camera_id
        new_zones = []

        if not zones_data:
            self.zones_map.clear()
            return []

        for z in zones_data:
            dwell = z.get("dwell_time") or z.get("dwellTime")
            if dwell is None:
                dwell = 0.0
            z_type = z.get("type") or z.get("zone_type") or "warning"

            new_zones.append(Zone(
                name=z.get("name", "Unnamed Zone"),
                points=z.get("points", []),
                zone_type=z_type,
                dwell_time=float(dwell)
            ))
        self.zones_map[target_cam] = new_zones
        return new_zones

    def reset_state(self):
        self.last_results = []
        self.all_detections = []
        self.compliance_results = []
        self.last_annotated_frame = None
        self.inference_time = 0.0
        self.fps = 0.0
        self.intrusion_active = False

    def process(self, frame: np.ndarray, skip_inference: bool = False, detection_mode: str = "both") -> Tuple:
        start_time = time.time()
        height, width = frame.shape[:2]
        self.frame_height = height
        self.frame_width = width

        # 1. AI Inference — single-frame predict (fastest path, no stream overhead)
        if not skip_inference:
            start_inference = time.time()
            # imgsz=640 matches best.pt native training resolution for sharpest helmet & PPE recognition
            results_list = self.model.predict(
                frame,
                conf=min(0.20, self.confidence),
                verbose=False,
                imgsz=640,
                device=self._device,
                augment=False,
                agnostic_nms=False,
            )
            results = results_list[0]

            self.all_detections = []
            self.last_results = []
            for box in results.boxes:
                xyxy     = box.xyxy[0].tolist()
                conf_val = box.conf[0].item()
                cls_id   = int(box.cls[0].item())
                cls_name = self.class_names.get(cls_id, PPE_CLASS_NAMES.get(cls_id, str(cls_id)))
                self.all_detections.append((xyxy, conf_val, cls_id, cls_name))
                if cls_id == self.person_class_id:
                    self.last_results.append((xyxy, conf_val))

            if detection_mode in ["ppe", "both"]:
                self.compliance_results = self.compute_ppe_compliance(self.all_detections, frame)
            else:
                self.compliance_results = []

            self.inference_time = (time.time() - start_inference) * 1000
            self.fps = 1.0 / max(1e-6, time.time() - start_time)

        # 2. Logic Prep
        now = time.time()
        current_zones = self.zones
        alerts_to_trigger = []
        # Use separate cooldowns: fencing uses alert_cooldown, PPE uses ppe_alert_cooldown
        fence_cooldown = max(3.0, float(getattr(self, "alert_cooldown", 10.0)))
        ppe_cooldown   = max(5.0, min(10.0, float(getattr(self, "ppe_alert_cooldown", 7.0))))

        # 3. Virtual Fence / Zone Logic
        any_intrusion = False
        any_warning = False

        for zone in current_zones:
            people_in_zone = [r for r in self.last_results if
                              zone.contains((r[0][0]+r[0][2])/2, (r[0][1]+r[0][3])/2, width, height) or
                              zone.contains((r[0][0]+r[0][2])/2, r[0][3], width, height)]

            if people_in_zone:
                if zone.type == "warning":
                    any_warning = True
                else:
                    # Instant breach — set occupied_since on first frame
                    if zone.occupied_since is None:
                        zone.occupied_since = now

                    dwell = now - zone.occupied_since
                    if dwell >= zone.dwell_time:  # dwell_time=0 triggers instantly
                        zone.is_breached = True
                        if detection_mode in ["fencing", "both"]:
                            any_intrusion = True

                            time_since_zone = now - zone.last_alert_time
                            time_since_det  = now - self.last_alert_time
                            if zone.alert_triggered and time_since_zone >= fence_cooldown:
                                zone.alert_triggered = False  # rearm for next cycle

                            if not zone.alert_triggered and time_since_zone >= fence_cooldown and time_since_det >= fence_cooldown:
                                alerts_to_trigger.append({
                                    "kind": "intrusion",
                                    "priority": "critical",
                                    "message": f"SECURITY BREACH in {zone.name}!",
                                    "zone_name": zone.name
                                })
                                zone.alert_triggered = True
                                zone.last_alert_time = now
                                self.last_alert_time = now
                    else:
                        zone.is_pending = True
                        if detection_mode in ["fencing", "both"]:
                            any_warning = True
            else:
                zone.occupied_since = None
                zone.is_breached = False
                zone.is_pending = False
                zone.alert_triggered = False

        current_status = "intrusion" if any_intrusion else "warning" if any_warning else "safe"
        intruder_count = 0

        # 4. Draw & Count with Complete Human Analysis & Anatomical Bounding Boxes
        person_detections = [d for d in self.all_detections if d[2] == self.person_class_id]
        item_detections = [d for d in self.all_detections if d[2] != self.person_class_id]
        claimed_items = set()

        for coords, conf, cls_id, cls_name in person_detections:
            px1, py1, px2, py2 = coords
            cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            aspect = ph / max(1.0, pw)

            is_inside_danger = any(z.type == "danger" and (z.contains(cx, cy, width, height) or z.contains(cx, py2, width, height)) for z in current_zones)
            is_inside_warn   = any(z.type == "warning" and (z.contains(cx, cy, width, height) or z.contains(cx, py2, width, height)) for z in current_zones)

            if is_inside_danger:
                intruder_count += 1

            if detection_mode in ["ppe", "both"]:
                # Anatomical visibility analysis based on bounding box proportions
                # head_vis: upper body visible enough to check for helmet/eye-prot
                head_vis  = (ph >= 60)  # Any detection with height >= 60px has a visible head region
                torso_vis = (aspect >= 1.0 and ph >= 100)  # Tall enough to see torso
                # Only flag feet if full standing body is visible (feet near bottom of frame)
                feet_vis  = (aspect >= 1.40 and py2 >= height * 0.75 and ph >= height * 0.55)
                hands_vis = (aspect >= 0.85 and ph >= 120)  # Wide or tall enough to see arms

                # Sub-detections overlapping / belonging to this person
                person_subs = []
                for idx, item in enumerate(item_detections):
                    if item_belongs_to_person(item[0], coords):
                        person_subs.append(item)
                        claimed_items.add(idx)

                helmets   = [d for d in person_subs if d[2] == self.helmet_class_id]
                no_helmets= [d for d in person_subs if d[2] == self.no_helmet_class_id]
                vests     = [d for d in person_subs if d[2] == self.vest_class_id]
                no_vests  = [d for d in person_subs if d[2] == self.no_vest_class_id]
                gloves    = [d for d in person_subs if d[2] == self.gloves_class_id]
                eyes      = [d for d in person_subs if d[2] == self.eye_protection_class_id]
                footwear  = [d for d in person_subs if d[2] == self.footwear_class_id]

                missing_items = []

                # Head & Helmet Evaluation (accurately handles both hard hats and full-face/motorcycle helmets)
                has_helmet = False
                has_visor = False
                head_box = [px1 + 0.15 * pw, py1, px2 - 0.15 * pw, py1 + 0.35 * ph]
                helmet_label = "NO HELMET ✘"
                helmet_conf = 0.50

                if head_vis:
                    has_helmet, head_box, helmet_conf, helmet_label, has_visor = check_head_helmet_status(
                        frame, coords, helmets, no_helmets
                    )
                    if not has_helmet:
                        missing_items.append("helmet")

                # Eye protection: verified if goggles detected or integrated visor on helmet
                has_eye_protection = bool(eyes) or (has_helmet and has_visor)
                if head_vis and not has_eye_protection and not has_helmet:
                    missing_items.append("eye_protection")

                # Torso / Vest:
                has_vest = False
                if vests:
                    has_vest = True
                elif no_vests:
                    has_vest = False
                    if torso_vis:
                        missing_items.append("vest")
                elif torso_vis:
                    has_vest = False
                    missing_items.append("vest")

                # Gloves & Footwear: only flag if full standing body with hands/feet in camera view
                hands_extended = (aspect >= 1.35 and ph >= height * 0.60)
                if hands_extended and not gloves:
                    missing_items.append("gloves")
                if feet_vis and not footwear:
                    missing_items.append("footwear")

                # 1. Outer person bounding box
                if missing_items:
                    p_color = (0, 0, 255)  # Red for violation
                    p_label = f"WORKER - PPE VIOLATION ({len(missing_items)} MISSING)"
                else:
                    p_color = (30, 214, 112)  # Green for compliant
                    p_label = f"WORKER - PPE COMPLIANT {conf:.2f}"

                self.draw_badge_box(frame, coords, p_label, p_color, thickness=2, position="top_left")

                # 2. Localized Anatomical Bounding Boxes & Badges:

                # A. Head / Helmet
                if head_vis:
                    if has_helmet:
                        self.draw_badge_box(frame, head_box, helmet_label, (30, 214, 112), thickness=2, position="inside_top_right")
                    else:
                        self.draw_badge_box(frame, head_box, helmet_label, (0, 0, 255), thickness=2, position="inside_top_right")

                # B. Eye Protection / Goggles
                if head_vis:
                    if eyes:
                        for b, c, _, _ in eyes:
                            self.draw_badge_box(frame, b, f"EYE PROT ✔ {c:.2f}", (230, 150, 30), thickness=2, position="inside_top")
                    elif has_helmet and has_visor:
                        eye_box = [px1 + 0.22 * pw, py1 + 0.10 * ph, px2 - 0.22 * pw, py1 + 0.20 * ph]
                        self.draw_badge_box(frame, eye_box, "EYE PROT ✔ (VISOR)", (230, 150, 30), thickness=1, position="inside_top")
                    elif not has_helmet:
                        eye_box = [px1 + 0.24 * pw, py1 + 0.08 * ph, px2 - 0.24 * pw, py1 + 0.17 * ph]
                        self.draw_badge_box(frame, eye_box, "NO EYE PROT ✘", (0, 140, 255), thickness=1, position="inside_top")

                # C. Torso / Vest
                if torso_vis:
                    if vests:
                        for b, c, _, _ in vests:
                            self.draw_badge_box(frame, b, f"VEST ✔ {c:.2f}", (30, 214, 112), thickness=2, position="inside_top")
                    elif no_vests:
                        for b, c, _, _ in no_vests:
                            self.draw_badge_box(frame, b, f"NO VEST ✘ {c:.2f}", (0, 0, 255), thickness=2, position="inside_top")
                    else:
                        torso_box = [px1 + 0.10 * pw, py1 + 0.29 * ph, px2 - 0.10 * pw, min(py2 - 4, py1 + 0.75 * ph)]
                        self.draw_badge_box(frame, torso_box, "NO VEST ✘", (0, 0, 255), thickness=2, position="inside_top")

                # D. Gloves (Only when gloves detected or extended in full view)
                if gloves:
                    for b, c, _, _ in gloves:
                        self.draw_badge_box(frame, b, f"GLOVES ✔ {c:.2f}", (255, 191, 0), thickness=2, position="inside_top")
                elif hands_extended:
                    gl_box = [px1, py1 + 0.45 * ph, px1 + 0.18 * pw, py1 + 0.72 * ph]
                    self.draw_badge_box(frame, gl_box, "NO GLOVES ✘", (0, 100, 255), thickness=1, position="inside_top")

                # E. Footwear / Boots (Only if lower body / feet are in view)
                if footwear:
                    for b, c, _, _ in footwear:
                        self.draw_badge_box(frame, b, f"FOOTWEAR ✔ {c:.2f}", (30, 214, 112), thickness=2, position="inside_bottom")
                elif feet_vis:
                    feet_box = [px1 + 0.12 * pw, py2 - 0.12 * ph, px2 - 0.12 * pw, py2]
                    self.draw_badge_box(frame, feet_box, "NO FOOTWEAR ✘", (0, 0, 255), thickness=2, position="inside_bottom")

                # 3. PPE Alert Dispatch
                # Monitor Only (ppe_alerts_enabled=False): bounding boxes shown on screen, NO alert sent
                # Alert Mode   (ppe_alerts_enabled=True) : alert saved to DB + email/telegram/call at 5-10s
                if missing_items and self.ppe_alerts_enabled:
                    time_since_ppe = now - getattr(self, "last_ppe_alert_time", 0.0)
                    if time_since_ppe >= ppe_cooldown:
                        missing_str = ", ".join(m.upper().replace("_", " ") for m in missing_items)
                        alerts_to_trigger.append({
                            "kind": "ppe_violation",
                            "priority": "warning",
                            "message": f"PPE Violation: Missing {missing_str}",
                            "zone_name": "Safety Zone",
                            "missing_ppe": missing_str,
                            "notify_external": True
                        })
                        self.last_ppe_alert_time = now

            else:
                # Fencing only mode
                color = (0, 0, 255) if is_inside_danger else (0, 165, 255) if is_inside_warn else (30, 214, 112)
                thickness = 4 if is_inside_danger else 2
                zone_label = "INTRUDER" if is_inside_danger else "WARNING" if is_inside_warn else "AUTHORIZED"
                label_text = f"{zone_label} {conf:.2f}"
                self.draw_badge_box(frame, coords, label_text, color, thickness=thickness)

        # Draw standalone items not claimed by any person box
        if detection_mode in ["ppe", "both"]:
            for idx, (b, c, cid, cname) in enumerate(item_detections):
                if idx not in claimed_items:
                    cfg = CLASS_DISPLAY_CONFIG.get(cid, {})
                    label = cfg.get("label", cname.upper())
                    is_viol = cfg.get("is_violation", False)
                    color = (0, 0, 255) if is_viol else cfg.get("color", (30, 214, 112))
                    self.draw_badge_box(frame, b, f"{label} {c:.2f}", color, thickness=2)

        self.current_status = current_status
        self.intruder_count = intruder_count
        return frame, current_status, alerts_to_trigger, intruder_count, self.compliance_results

    def save_snapshot(self, frame: np.ndarray, storage_dir: Path) -> Path:
        storage_dir.mkdir(parents=True, exist_ok=True)
        filename = f"alert_{int(time.time())}.jpg"
        path = storage_dir / filename
        cv2.imwrite(str(path), frame)
        return path

    def compute_ppe_compliance(
        self,
        all_detections: List[Tuple],
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Compute per-person PPE compliance from all frame detections.
        Matches all 8 classes accurately using item_belongs_to_person().
        """
        person_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.person_class_id
        ]
        vest_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.vest_class_id
        ] if self.vest_class_id is not None else []
        no_vest_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.no_vest_class_id
        ] if self.no_vest_class_id is not None else []

        helmet_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.helmet_class_id
        ] if self.helmet_class_id is not None else []
        no_helmet_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.no_helmet_class_id
        ] if self.no_helmet_class_id is not None else []

        gloves_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.gloves_class_id
        ] if self.gloves_class_id is not None else []
        boots_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.footwear_class_id
        ] if self.footwear_class_id is not None else []
        goggles_dets = [
            (xyxy, conf) for xyxy, conf, cls_id, _ in all_detections
            if cls_id == self.eye_protection_class_id
        ] if self.eye_protection_class_id is not None else []

        results: List[Dict] = []
        item_conf_threshold = 0.15

        for person_box, person_conf in person_dets:
            px1, py1, px2, py2 = person_box
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            aspect = ph / max(1.0, pw)

            # Direct violation signals
            has_no_vest = any(
                conf >= self.ppe_violation_conf_threshold and item_belongs_to_person(box, person_box)
                for box, conf in no_vest_dets
            )

            # Positive items
            vest_present    = check_vest_compliance(person_box, vest_dets, conf_threshold=self.vest_conf_threshold)
            boots_present   = _item_on_person(person_box, boots_dets,   conf_threshold=item_conf_threshold)
            goggles_present = _item_on_person(person_box, goggles_dets, conf_threshold=item_conf_threshold)
            gloves_present  = _item_on_person(person_box, gloves_dets,  conf_threshold=item_conf_threshold)

            torso_vis = (aspect >= 1.05 and ph >= 120)
            feet_vis = (aspect >= 1.40 and py2 >= getattr(self, "frame_height", 480) * 0.70)
            hands_extended = (aspect >= 1.35 and ph >= getattr(self, "frame_height", 480) * 0.60)

            # Helmet & Eye Protection via synchronized verification
            has_h = False
            has_v = False
            if frame is not None:
                has_h, _, _, _, has_v = check_head_helmet_status(
                    frame, person_box,
                    [(b, c, self.helmet_class_id, "helmet") for b, c in helmet_dets if item_belongs_to_person(b, person_box)],
                    [(b, c, self.no_helmet_class_id, "no_helmet") for b, c in no_helmet_dets if item_belongs_to_person(b, person_box)],
                )
            else:
                has_no_helmet = any(
                    conf >= self.ppe_violation_conf_threshold and item_belongs_to_person(box, person_box)
                    for box, conf in no_helmet_dets
                )
                helmet_present = _item_on_person(person_box, helmet_dets, conf_threshold=item_conf_threshold)
                has_h = helmet_present and not has_no_helmet

            missing_helmet  = not has_h
            has_eye_prot    = goggles_present or (has_h and has_v)
            missing_goggles = not has_eye_prot and not has_h
            missing_vest    = has_no_vest or (not vest_present if torso_vis else False)
            missing_boots   = not boots_present if feet_vis else False
            missing_gloves  = not gloves_present if hands_extended else False

            results.append({
                "bbox":              [px1, py1, px2, py2],
                "person_conf":       round(person_conf, 3),
                "missing_helmet":    missing_helmet,
                "missing_vest":      missing_vest,
                "missing_boots":     missing_boots,
                "missing_goggles":   missing_goggles,
                "missing_gloves":    missing_gloves,
                "missing_face_mask": False,
                "missing_ear_mufs":  False,
                "overall_violation": missing_helmet or missing_vest,
            })

        return results
