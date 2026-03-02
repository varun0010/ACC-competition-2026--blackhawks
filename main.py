"""
Black Hawks — Master Controller  (Final Integration)
=====================================================
Architecture:
  Layer 1 — PATH:   GPS + EKF + Stanley controller → global route following
  Layer 2 — LANE:   Dual-channel pipeline + EKF polynomial smoother → lane keeping
  Layer 3 — ML:     YOLOv8 every 5 frames — stop sign, pedestrian, crosswalk,
                    traffic lights (with red→green 10s immunity cooldown)
  Layer 4 — BLEND:  Adaptive Stanley/Lane steer blend with conflict resolution

ML behaviour:
  • Stop sign      — right-half ROI, depth gate, 3-frame streak → stop 5s, cooldown 8s
  • Pedestrian     — right-half ROI (same as stop sign), stop while visible, cooldown 5s
  • Crosswalk      — bottom-25% ROI, slow to 0.10 m/s while detected
  • Red/Yellow TL  — stop immediately (centre-biased TL selector)
  • Green TL       — clears stop sign/ped lock + 10s immunity (no re-stop on same light)

Steer blend (per-frame adaptive):
  • Lane + GPS OK     → 70% Stanley + 30% Lane
  • Lane + GPS stale  → 30% Stanley + 70% Lane
  • Conflict >15 deg  → 90% Stanley + 10% Lane
  • No lane           → 100% Stanley

Throttle hierarchy (highest priority first):
  1. ML STOP (red light / stop sign / pedestrian) → 0.0
  2. Paused                                       → 0.0
  3. Crosswalk detected                           → cap 0.032
  4. Sharp curve (R < 25m)                        → cap 0.030
  5. GPS corner >20 deg                           → cap 0.038
  6. SpeedPID targeting V_REF

Controls: Q = quit    P = pause/resume
Hardcoded: QCar2, csi[3]
No QVL required.
"""

from __future__ import annotations
import sys, os, time, math, signal, warnings
from collections import deque
from typing import Optional, Tuple, List
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

# =========================================================================
# PATH SETUP
# =========================================================================
PAL_PATH = (
    r"C:\Users\Admin\Desktop\Q-car"
    r"\Quanser_Academic_Resources-dev-windows"
    r"\Quanser_Academic_Resources-dev-windows"
    r"\0_libraries\python"
)
if PAL_PATH not in sys.path:
    sys.path.insert(0, PAL_PATH)

from pal.products.qcar import QCar, QCarGPS, QCarCameras

REALSENSE_AVAILABLE = False
try:
    from pal.products.qcar import QCarRealSense
    REALSENSE_AVAILABLE = True
    print("RealSense available.")
except ImportError:
    print("RealSense not found — depth fallback (area-based).")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 available.")
except ImportError:
    print("YOLO not available.")

# =========================================================================
# WAYPOINTS
# =========================================================================
INITIAL_POSE = [-1.205, -0.830, math.radians(-44.7)]

WAYPOINTS_XY = [
    (-0.716900, -1.122700),(-0.592100, -1.140700),(-0.441700, -1.139600),
    (-0.269300, -1.124400),(-0.111600, -1.111900),(+0.054300, -1.110900),
    (+0.216100, -1.116500),(+0.375100, -1.124300),(+0.433800, -1.125000),
    (+0.584700, -1.130200),(+0.749500, -1.133100),(+0.930800, -1.132700),
    (+1.079400, -1.126700),(+1.243700, -1.097400),(+1.299200, -1.083500),
    (+1.449600, -1.029500),(+1.601000, -0.963600),(+1.757200, -0.885200),
    (+1.873400, -0.808900),(+2.018400, -0.658500),(+2.092000, -0.540000),
    (+2.120000, -0.490000),(+2.159200, -0.350900),(+2.185900, -0.199200),
    (+2.206700, -0.041600),(+2.224100, +0.131400),(+2.227800, +0.184200),
    (+2.238700, +0.366600),(+2.245300, +0.536400),(+2.244500, +0.586200),
    (+2.247200, +0.768300),(+2.248400, +0.963300),(+2.249400, +1.132100),
    (+2.250200, +1.294900),(+2.250300, +1.470000),(+2.248200, +1.519900),
    (+2.249400, +1.706200),(+2.249900, +1.891500),(+2.250100, +2.054800),
    (+2.249400, +2.107800),(+2.248000, +2.279700),(+2.245200, +2.458900),
    (+2.242000, +2.630300),(+2.240800, +2.683800),(+2.237100, +2.870800),
    (+2.233700, +3.035300),(+2.229900, +3.213000),(+2.225900, +3.384000),
    (+2.220200, +3.559400),(+2.205300, +3.716900),(+2.173600, +3.868500),
    (+2.127300, +3.995700),(+2.058900, +4.116300),(+1.942200, +4.242300),
    (+1.834800, +4.327900),(+1.715700, +4.392300),(+1.590100, +4.437100),
    (+1.462000, +4.459100),(+1.330800, +4.460400),(+1.183500, +4.451600),
    (+1.022500, +4.439200),(+0.958900, +4.430600),(+0.805300, +4.421300),
    (+0.644900, +4.416000),(+0.479300, +4.414300),(+0.303300, +4.414500),
    (+0.251200, +4.411000),(+0.077400, +4.413600),(-0.112700, +4.415300),
    (-0.288800, +4.417400),(-0.342600, +4.414700),(-0.519300, +4.417900),
    (-0.688300, +4.420000),(-0.857900, +4.420000),(-1.014200, +4.409200),
    (-1.149300, +4.380700),(-1.289300, +4.323100),(-1.334400, +4.303500),
    (-1.503900, +4.201700),(-1.619600, +4.123900),(-1.734800, +4.025500),
    (-1.815400, +3.930600),(-1.905200, +3.769900),(-1.948200, +3.644500),
    (-1.965000, +3.515800),(-1.971000, +3.351400),(-1.971100, +3.184100),
    (-1.968600, +3.137500),(-1.964800, +2.949000),(-1.960900, +2.778600),
    (-1.956200, +2.598600),(-1.953000, +2.549000),(-1.947700, +2.360500),
    (-1.943100, +2.198100),(-1.937900, +2.149300),(-1.933200, +1.961400),
    (-1.927800, +1.772600),(-1.918600, +1.594100),(-1.900200, +1.448500),
    (-1.855300, +1.289300),(-1.771400, +1.131600),(-1.688500, +1.025200),
    (-1.590800, +0.936700),(-1.460300, +0.851700),(-1.297000, +0.793000),
    (-1.176300, +0.768700),(-1.038300, +0.764300),(-0.981000, +0.765700),
    (-0.832000, +0.778100),(-0.667800, +0.793700),(-0.505400, +0.809000),
    (-0.452700, +0.817500),(-0.257500, +0.833300),(-0.095800, +0.844500),
    (+0.073400, +0.853900),(+0.249400, +0.862600),(+0.298700, +0.867800),
    (+0.481600, +0.875600),(+0.667800, +0.882300),(+0.818700, +0.882200),
    (+0.872300, +0.883400),(+1.062400, +0.872200),(+1.313400, +0.846300),
    (+1.459600, +0.812200),(+1.574000, +0.769300),(+1.694100, +0.696200),
    (+1.797200, +0.610400),(+1.896300, +0.487300),(+1.963900, +0.365500),
    (+2.007900, +0.229500),(+2.025000, +0.122800),(+2.019100, -0.044200),
    (+1.989700, -0.177400),(+1.943100, -0.290200),(+1.860600, -0.427400),
    (+1.763000, -0.563600),(+1.665400, -0.671800),(+1.622200, -0.709900),
    (+1.421700, -0.835800),(+1.286200, -0.880300),(+1.150400, -0.896000),
    (+1.104000, -0.897000),(+0.911000, -0.874000),(+0.772600, -0.843100),
    (+0.567900, -0.756900),(+0.458300, -0.681600),(+0.365400, -0.593300),
    (+0.279000, -0.476100),(+0.217600, -0.357200),(+0.172500, -0.212700),
    (+0.156300, -0.099200),(+0.155200, +0.040200),(+0.161600, +0.211000),
    (+0.161700, +0.346100),(+0.147700, +0.488200),(+0.080400, +0.667700),
    (+0.021900, +0.774800),(-0.060700, +0.882100),(-0.170100, +0.981100),
    (-0.283900, +1.056400),(-0.419400, +1.115800),(-0.596600, +1.148600),
    (-0.724900, +1.151600),(-0.857000, +1.129300),(-0.986000, +1.087100),
    (-1.108800, +1.020800),(-1.222800, +0.926100),(-1.307700, +0.824400),
    (-1.338100, +0.779800),(-1.409500, +0.638400),(-1.478400, +0.461100),
    (-1.524000, +0.316000),(-1.549200, +0.183600),(-1.554200, +0.047800),
    (-1.545400, -0.112900),(-1.525400, -0.287600),(-1.513200, -0.347000),
    (-1.450300, -0.577900),(-1.384500, -0.701200),(-1.352100, -0.748600),
    (-1.270800, -0.841700),(-1.255700, -0.861500),
]
WP   = np.array(WAYPOINTS_XY, dtype=np.float64)
N_WP = len(WP)

# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        TUNING BLOCK                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================

# ── Camera ────────────────────────────────────────────────────────────────
FRONT_CSI_IDX    = 3
FRAME_W, FRAME_H = 640, 480

# ── Speed ─────────────────────────────────────────────────────────────────
V_REF             = 0.20    # cruise m/s
V_REF_SLOW        = 0.12    # corners m/s
THROTTLE_MIN      = 0.020
THROTTLE_MAX      = 0.055
THR_CRAWL         = 0.0     # ML stop

# ── Speed PID ─────────────────────────────────────────────────────────────
SPD_KP, SPD_KI, SPD_KD = 0.8, 0.5, 0.005

# ── Stanley ───────────────────────────────────────────────────────────────
K_STANLEY        = 1.5
MAX_STEER        = math.pi / 6   # 30 deg
REACH_RADIUS     = 0.25
SLOW_RADIUS      = 0.80
CORNER_THRESH    = 20.0
LOOKAHEAD_WPS    = 4

# ── EKF (GPS position) ────────────────────────────────────────────────────
Q_XY, Q_YAW      = 0.0001, 0.001
R_GPS             = 0.05 ** 2
GPS_TIMEOUT       = 2.0

# ── Steer blend ───────────────────────────────────────────────────────────
STANLEY_WEIGHT    = 0.70
LANE_WEIGHT       = 0.30
CONFLICT_THRESH   = math.radians(15)
STEER_ALPHA       = 0.40     # exponential smoothing

# ── Lane pipeline ─────────────────────────────────────────────────────────
ROI_TOP_FRAC      = 0.52
ROI_TL_X_FRAC     = 0.08
ROI_TR_X_FRAC     = 0.78
ROI_BL_X_FRAC     = 0.02
ROI_BR_X_FRAC     = 0.98
WARP_L            = 0.12
WARP_R            = 0.88

YELLOW_OFFSET_PX       = 55
YELLOW_OFFSET_CURVE_PX = 30
HALF_LANE_PX           = 120
XM_PER_PIX             = 3.7 / 820
YM_PER_PIX             = 30.0 / 410
CURVE_RADIUS_THRESHOLD = 25.0
MIN_LANE_WIDTH_PX      = 100
MAX_LANE_WIDTH_PX      = 700

# ── Sliding window ────────────────────────────────────────────────────────
SW_NWIN          = 12
SW_MARGIN_BASE   = 90
SW_MINPIX        = 30
MIN_FIT_PIXELS   = 150

# ── EKF polynomial smoother ───────────────────────────────────────────────
PROC_NOISE        = 0.08
MEAS_NOISE        = 0.40

# ── Colour thresholds ─────────────────────────────────────────────────────
YELLOW_LOW   = np.array([ 18,  80,  80], dtype=np.uint8)
YELLOW_HIGH  = np.array([ 38, 255, 255], dtype=np.uint8)
WHITE_LOW    = np.array([  0,   0, 200], dtype=np.uint8)
WHITE_HIGH   = np.array([180,  40, 255], dtype=np.uint8)
S_THRESH     = (100, 255)
SX_THRESH    = ( 20, 100)
L_THRESH     = (200, 255)
B_THRESH     = (145, 200)
LDW_THRESH   = 50

# ── ML / YOLO ─────────────────────────────────────────────────────────────
YOLO_FRAME_SKIP       = 5
YOLO_CONF             = 0.45

# Stop sign
STOP_DEPTH_MAX        = 2.0      # metres
STOP_AREA_FALLBACK    = 3000     # px² when no depth sensor
STOP_MIN_CONF         = 0.60
STOP_STREAK_REQ       = 3
STOP_HOLD_S           = 5.0
STOP_COOLDOWN_S       = 8.0

# Traffic light
TL_MIN_CONF           = 0.50
TL_MIN_AREA           = 300
TL_RIGHT_EXCL         = 0.80     # ignore TL detections right of this fraction
TL_GREEN_AFTER_RED_COOLDOWN = 10.0   # seconds to ignore ALL red lights after a red→green cycle

# ── Pedestrian detection ──────────────────────────────────────────────────
# ROI: right half of frame, full height (mirrors stop sign side)
PED_ROI_X1_FRAC       = 0.50    # left edge of pedestrian ROI (right half of frame)
PED_ROI_X2_FRAC       = 1.00
PED_ROI_Y1_FRAC       = 0.00
PED_ROI_Y2_FRAC       = 1.00
PED_MIN_CONF          = 0.50
PED_MIN_AREA          = 800      # px² — ignore tiny distant blobs
PED_STOP_HOLD_S       = 3.0      # hold stop while pedestrian visible
PED_COOLDOWN_S        = 5.0

# ── Crosswalk ─────────────────────────────────────────────────────────────
# ROI: full width, bottom 25% of frame only
CW_ROI_X1_FRAC        = 0.00
CW_ROI_X2_FRAC        = 1.00
CW_ROI_Y1_FRAC        = 0.75    # bottom quarter
CW_ROI_Y2_FRAC        = 1.00
CW_MIN_CONF           = 0.45
CW_MIN_AREA           = 500
CW_SLOW_SPEED         = 0.10    # m/s target while crosswalk visible
CW_SLOW_THROTTLE_CAP  = 0.032   # hard throttle cap during crosswalk

# Encoder (dead-reckoning between GPS fixes)
METRES_PER_TICK       = 5.11e-7
ENCODER_OVERFLOW      = 2**31 - 1
GYRO_BIAS             = 0.003243
GYRO_NOISE_FLOOR      = 0.02

# Map display
MAP_W = MAP_H = 500
WORLD_XMIN, WORLD_XMAX = -3.0, 3.5
WORLD_YMIN, WORLD_YMAX = -2.0, 5.5

# =========================================================================
# HELPERS
# =========================================================================
def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))

def w2p(x: float, y: float):
    u = int((x - WORLD_XMIN) / (WORLD_XMAX - WORLD_XMIN) * MAP_W)
    v = int((1 - (y - WORLD_YMIN) / (WORLD_YMAX - WORLD_YMIN)) * MAP_H)
    return max(0, min(MAP_W-1, u)), max(0, min(MAP_H-1, v))

def eval_poly(c: np.ndarray, y) -> np.ndarray:
    return c[0]*y**2 + c[1]*y + c[2]

def box_in_roi(x1: int, y1: int, x2: int, y2: int,
               fw: int, fh: int,
               x1f: float, y1f: float, x2f: float, y2f: float) -> bool:
    """Return True if box centre falls inside the fractional ROI."""
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (fw*x1f <= cx <= fw*x2f) and (fh*y1f <= cy <= fh*y2f)

def draw_roi_outline(frame: np.ndarray,
                     x1f: float, y1f: float, x2f: float, y2f: float,
                     color: Tuple[int,int,int], label: str = "") -> None:
    """Draw a dashed-style ROI rectangle with optional label on frame."""
    h, w = frame.shape[:2]
    rx1, ry1 = int(w*x1f), int(h*y1f)
    rx2, ry2 = int(w*x2f), int(h*y2f)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 1)
    if label:
        cv2.putText(frame, label, (rx1+4, ry1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

# =========================================================================
# LAYER 1 — POSITION EKF  [x, y, yaw]
# =========================================================================
class PoseEKF:
    def __init__(self, x0, y0, yaw0):
        self.mu = np.array([x0, y0, yaw0], dtype=np.float64)
        self.P  = np.eye(3) * 0.1
        self.Q  = np.diag([Q_XY, Q_XY, Q_YAW])

    def predict(self, dist_m: float, gyro: float, dt: float):
        x, y, yaw = self.mu
        nyaw = wrap(yaw + gyro * dt)
        F = np.array([
            [1, 0, -dist_m * math.sin(nyaw)],
            [0, 1,  dist_m * math.cos(nyaw)],
            [0, 0,  1]
        ], dtype=np.float64)
        self.mu = np.array([x + dist_m*math.cos(nyaw),
                            y + dist_m*math.sin(nyaw), nyaw])
        self.P  = F @ self.P @ F.T + self.Q

    def update_gps(self, gx: float, gy: float, gyaw: float):
        H = np.eye(3)
        R = np.diag([R_GPS, R_GPS, R_GPS * 4])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innov    = np.array([gx, gy, gyaw]) - self.mu
        innov[2] = wrap(innov[2])
        self.mu += K @ innov;  self.mu[2] = wrap(self.mu[2])
        self.P   = (np.eye(3) - K @ H) @ self.P

    @property
    def x(self):   return float(self.mu[0])
    @property
    def y(self):   return float(self.mu[1])
    @property
    def yaw(self): return float(self.mu[2])


# =========================================================================
# LAYER 1 — SPEED PID
# =========================================================================
class SpeedPID:
    def __init__(self):
        self.ei = 0.0;  self.prev_e = 0.0

    def update(self, v: float, v_ref: float, dt: float) -> float:
        e        = v_ref - v
        self.ei += e * dt
        de       = (e - self.prev_e) / max(dt, 1e-4)
        self.prev_e = e
        raw = SPD_KP*e + SPD_KI*self.ei + SPD_KD*de
        return float(np.clip(raw, THROTTLE_MIN, THROTTLE_MAX)) if v_ref > 0.01 else 0.0

    def reset(self): self.ei = 0.0; self.prev_e = 0.0


# =========================================================================
# LAYER 1 — STANLEY CONTROLLER
# =========================================================================
class Stanley:
    def __init__(self):
        self.wpi = 0;  self.completed = False;  self._adv = 0

    def _advance(self, p: np.ndarray):
        while self.wpi < N_WP - 1:
            seg  = WP[self.wpi+1] - WP[self.wpi]
            slen = np.linalg.norm(seg) + 1e-9
            s    = float(np.dot(p - WP[self.wpi], seg / slen))
            if s >= slen or np.linalg.norm(WP[self.wpi+1] - p) < REACH_RADIUS:
                self.wpi += 1;  self._adv += 1
            else:
                break
        if self.wpi >= N_WP - 1 and self._adv > N_WP // 2:
            self.completed = True

    def upcoming_corner_deg(self) -> float:
        mx = 0.0
        for i in range(self.wpi, min(self.wpi + LOOKAHEAD_WPS, N_WP - 2)):
            a1 = math.atan2(WP[i+1][1]-WP[i][1],   WP[i+1][0]-WP[i][0])
            j  = min(i+2, N_WP-1)
            a2 = math.atan2(WP[j][1]-WP[i+1][1],   WP[j][0]-WP[i+1][0])
            mx = max(mx, abs(math.degrees(wrap(a2-a1))))
        return mx

    def dist_to_next(self, p: np.ndarray) -> float:
        return float(np.linalg.norm(WP[self.wpi] - p)) if self.wpi < N_WP else 999.0

    def steer(self, p: np.ndarray, th: float, speed: float) -> float:
        p = np.asarray(p, dtype=np.float64)
        self._advance(p)
        if self.completed: return 0.0
        i       = min(self.wpi, N_WP - 2)
        seg     = WP[i+1] - WP[i]
        slen    = np.linalg.norm(seg) + 1e-9
        suv     = seg / slen
        tangent = math.atan2(suv[1], suv[0])
        ep      = WP[i] + suv * float(np.dot(p - WP[i], suv))
        ct      = ep - p
        dirn    = wrap(math.atan2(ct[1], ct[0]) - tangent)
        ect     = float(np.linalg.norm(ct)) * math.copysign(1, dirn)
        psi     = wrap(tangent - th)
        raw     = wrap(psi + math.atan2(K_STANLEY * ect, max(speed, 0.1)))
        return float(np.clip(raw, -MAX_STEER, MAX_STEER))


# =========================================================================
# LAYER 2 — POLYNOMIAL EKF SMOOTHER
# =========================================================================
class PolyEKF:
    """EKF over 2nd-degree polynomial coeffs [a, b, c]. Smooths noisy fits."""
    def __init__(self):
        self.x = None
        self.P = np.eye(3) * 10.0
        self.Q = np.eye(3) * PROC_NOISE
        self.R = np.eye(3) * MEAS_NOISE

    def update(self, z: np.ndarray) -> np.ndarray:
        if self.x is None:
            self.x = z.copy(); return self.x
        xp = self.x.copy();  Pp = self.P + self.Q
        K  = Pp @ np.linalg.inv(Pp + self.R)
        self.x = xp + K @ (z - xp)
        self.P = (np.eye(3) - K) @ Pp
        return self.x

    def predict_only(self) -> Optional[np.ndarray]:
        if self.x is None: return None
        self.P += self.Q; return self.x

    def reset(self):
        self.x = None; self.P = np.eye(3) * 10.0


# =========================================================================
# LAYER 2 — LANE LINE STATE
# =========================================================================
class LaneLine:
    MAX_LOST = 8

    def __init__(self):
        self.ekf        = PolyEKF()
        self.detected   = False
        self.best_fit   : Optional[np.ndarray] = None
        self.frames_lost = 0

    def update(self, raw: np.ndarray):
        self.best_fit    = self.ekf.update(raw)
        self.detected    = True
        self.frames_lost = 0

    def predict(self) -> Optional[np.ndarray]:
        self.frames_lost += 1
        if self.frames_lost > self.MAX_LOST:
            self.reset(); return None
        return self.ekf.predict_only()

    def get_fit(self) -> Optional[np.ndarray]:
        return self.best_fit

    def reset(self):
        self.ekf.reset(); self.detected=False
        self.best_fit=None; self.frames_lost=0


# =========================================================================
# LAYER 2 — WARP CACHE
# =========================================================================
_WARP_CACHE: dict = {}

def get_warp(h: int, w: int):
    if (h, w) not in _WARP_CACHE:
        src = np.float32([
            [w*ROI_TL_X_FRAC, h*ROI_TOP_FRAC], [w*ROI_TR_X_FRAC, h*ROI_TOP_FRAC],
            [w*ROI_BR_X_FRAC, h-1],             [w*ROI_BL_X_FRAC, h-1],
        ])
        dst = np.float32([
            [w*WARP_L, 0], [w*WARP_R, 0],
            [w*WARP_R, h-1], [w*WARP_L, h-1],
        ])
        M    = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        _WARP_CACHE[(h, w)] = (M, Minv, src.astype(np.int32))
    return _WARP_CACHE[(h, w)]


# =========================================================================
# LAYER 2 — BINARY MASKS
# =========================================================================
def build_masks(bgr: np.ndarray):
    """Returns (mask_yellow, mask_white, combined_gradient) all uint8 0/1."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    my  = (cv2.inRange(hsv, YELLOW_LOW,  YELLOW_HIGH) > 0).astype(np.uint8)
    mw  = (cv2.inRange(hsv, WHITE_LOW,   WHITE_HIGH)  > 0).astype(np.uint8)
    k3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    my  = cv2.morphologyEx(my, cv2.MORPH_CLOSE, k3)
    mw  = cv2.morphologyEx(mw, cv2.MORPH_CLOSE, k3)

    hls   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    lab   = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch  = hls[:,:,1];  s_ch = hls[:,:,2];  b_ch = lab[:,:,2]
    sx    = np.abs(cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3))
    mx    = sx.max()
    sc    = (sx/mx*255).astype(np.uint8) if mx > 0 else np.zeros_like(sx, dtype=np.uint8)

    sx_b  = ((sc  >=SX_THRESH[0])&(sc  <=SX_THRESH[1])).astype(np.uint8)
    s_b   = ((s_ch>=S_THRESH[0] )&(s_ch<=S_THRESH[1] )).astype(np.uint8)
    l_b   = ((l_ch>=L_THRESH[0] )&(l_ch<=L_THRESH[1] )).astype(np.uint8)
    b_b   = ((b_ch>=B_THRESH[0] )&(b_ch<=B_THRESH[1] )).astype(np.uint8)
    comb  = np.zeros_like(sx_b)
    comb[((s_b&sx_b)==1)|(l_b==1)|(b_b==1)] = 1
    k5    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    comb  = cv2.morphologyEx(comb, cv2.MORPH_CLOSE, k5)
    comb  = cv2.morphologyEx(comb, cv2.MORPH_OPEN,  k5)
    return my, mw, comb


# =========================================================================
# LAYER 2 — SLIDING WINDOW
# =========================================================================
def sliding_window(binary: np.ndarray, x_start: int,
                   prior: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    h, w   = binary.shape
    win_h  = max(h // SW_NWIN, 1)
    nz     = binary.nonzero()
    nzy, nzx = np.array(nz[0]), np.array(nz[1])
    cur    = x_start
    inds   = []
    for wi in range(SW_NWIN):
        yl = h - (wi+1)*win_h;  yh = h - wi*win_h;  yc = (yl+yh)//2
        if prior is not None:
            cur = int(np.clip(eval_poly(prior, yc), 0, w-1))
        margin = int(SW_MARGIN_BASE * (1.0 + 0.6*wi/SW_NWIN))
        xl = max(0, cur-margin);  xr = min(w, cur+margin)
        good = ((nzy>=yl)&(nzy<yh)&(nzx>=xl)&(nzx<xr)).nonzero()[0]
        inds.append(good)
        if len(good) > SW_MINPIX: cur = int(np.mean(nzx[good]))
    all_i = np.concatenate(inds) if inds else np.array([], dtype=np.int32)
    return nzx[all_i], nzy[all_i]


def hpeak(binary: np.ndarray, lo: int, hi: int) -> int:
    h = np.sum(binary[binary.shape[0]//2:, lo:hi], axis=0).astype(float)
    h = uniform_filter1d(h, size=40)
    return int(np.argmax(h)) + lo


def fit_ekf(px: np.ndarray, py: np.ndarray,
            lane: LaneLine, bh: int) -> Optional[np.ndarray]:
    if len(px) >= MIN_FIT_PIXELS:
        try:
            raw = np.polyfit(py, px, 2)
            lane.update(raw)
            return lane.get_fit()
        except Exception:
            pass
    return lane.predict()


def curvature(coeffs: np.ndarray, bh: int) -> Tuple[float, str]:
    try:
        ploty  = np.linspace(0, bh-1, bh)
        rc     = np.polyfit(ploty*YM_PER_PIX, eval_poly(coeffs,ploty)*XM_PER_PIX, 2)
        y_eval = (bh-1)*YM_PER_PIX
        r      = ((1+(2*rc[0]*y_eval+rc[1])**2)**1.5) / max(abs(2*rc[0]), 1e-6)
        bot    = eval_poly(coeffs, bh-1);  top = eval_poly(coeffs, 0)
        d      = "Right" if bot-top>30 else "Left" if top-bot>30 else "Straight"
        return float(r), d
    except Exception:
        return 9999.0, "Straight"


# =========================================================================
# LAYER 2 — LANE PIPELINE (returns steer error + annotated frame)
# =========================================================================
def run_lane_pipeline(frame: np.ndarray,
                      left_lane: LaneLine,
                      right_lane: LaneLine
                      ) -> Tuple[float, float, str, np.ndarray, str]:
    """
    Returns: (steer_error [-1,1], curve_radius, curve_dir, ann_frame, mode_str)
    """
    h, w  = frame.shape[:2]
    ploty = np.linspace(0, h-1, h)

    try:
        M, Minv, roi_pts = get_warp(h, w)
        mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_pts], 255)
        warped = cv2.warpPerspective(
            cv2.bitwise_and(frame, frame, mask=mask), M, (w, h))
    except Exception:
        return 0.0, 9999.0, "Straight", frame.copy(), "WARP_ERR"

    mask_y, mask_w, comb = build_masks(warped)

    # Peak initialisation with EKF warm-start
    lf_p = left_lane.get_fit()
    rf_p = right_lane.get_fit()
    y_pk = (int(np.clip(eval_poly(lf_p, h-1), 0, w-1))
            if lf_p is not None else hpeak(mask_y, 0, int(w*0.60)))
    w_pk = (int(np.clip(eval_poly(rf_p, h-1), 0, w-1))
            if rf_p is not None else max(hpeak(mask_w, int(w*0.50), w), w//2))

    lpx, lpy = sliding_window(mask_y, y_pk, lf_p)
    rpx, rpy = sliding_window(mask_w, w_pk, rf_p)
    if len(rpx) < MIN_FIT_PIXELS:           # gradient fallback for right
        rpx2, rpy2 = sliding_window(comb, w_pk, rf_p)
        if len(rpx2) > len(rpx): rpx, rpy = rpx2, rpy2

    lf = fit_ekf(lpx, lpy, left_lane,  h)
    rf = fit_ekf(rpx, rpy, right_lane, h)

    # Lane-width sanity gate
    if lf is not None and rf is not None:
        width = eval_poly(rf, h-1) - eval_poly(lf, h-1)
        if not (MIN_LANE_WIDTH_PX < width < MAX_LANE_WIDTH_PX):
            if len(lpx) >= len(rpx): right_lane.reset(); rf = None
            else:                     left_lane.reset();  lf = None

    # Centre calculation
    if lf is not None and rf is not None:
        centre_x = (eval_poly(lf,h-1) + eval_poly(rf,h-1)) / 2.0
        lx_arr   = eval_poly(lf, ploty);  rx_arr = eval_poly(rf, ploty)
        crad, cd = curvature(lf, h);      mode   = "BOTH"
    elif lf is not None:
        crad, cd = curvature(lf, h)
        t        = max(0.0, min(1.0, crad/CURVE_RADIUS_THRESHOLD))
        offset   = (YELLOW_OFFSET_CURVE_PX + t*(YELLOW_OFFSET_PX-YELLOW_OFFSET_CURVE_PX)
                    if crad < CURVE_RADIUS_THRESHOLD else YELLOW_OFFSET_PX)
        centre_x = eval_poly(lf, h-1) + offset
        lx_arr   = eval_poly(lf, ploty);  rx_arr = None;  mode = "YELLOW"
    elif rf is not None:
        crad, cd = curvature(rf, h)
        centre_x = eval_poly(rf, h-1) - HALF_LANE_PX
        lx_arr   = None;  rx_arr = eval_poly(rf, ploty);  mode = "R_WHITE"
    else:
        left_lane.reset(); right_lane.reset()
        centre_x = w / 2.0
        lx_arr = rx_arr = None
        crad, cd = 9999.0, "Straight";  mode = "NO LANE"

    dev_px     = (w/2.0) - centre_x
    steer_err  = float(np.clip(dev_px / HALF_LANE_PX, -1.0, 1.0))

    # ── Annotation ────────────────────────────────────────────────────────
    ann = frame.copy()
    try:
        wz = np.zeros((h, w, 3), dtype=np.uint8)
        if lx_arr is not None and rx_arr is not None:
            ptsl = np.array([np.transpose(np.vstack([lx_arr.clip(0,w-1),ploty]))],dtype=np.int32)
            ptsr = np.array([np.flipud(np.transpose(np.vstack([rx_arr.clip(0,w-1),ploty])))],dtype=np.int32)
            cv2.fillPoly(wz,[np.hstack((ptsl,ptsr))],(0,160,0))
        elif lx_arr is not None:
            cv2.polylines(wz,
                [np.array([np.transpose(np.vstack([lx_arr.clip(0,w-1),ploty]))],dtype=np.int32)],
                False,(0,200,255),4)
        ca   = np.full_like(ploty, centre_x)
        cv2.polylines(wz,
            [np.array([np.transpose(np.vstack([ca.clip(0,w-1),ploty]))],dtype=np.int32)],
            False,(0,255,255),3)
        uw   = cv2.warpPerspective(wz, Minv, (w, h))
        ann  = cv2.addWeighted(ann, 1.0, uw, 0.38, 0)
    except Exception: pass

    # ROI outline
    try: cv2.polylines(ann,[roi_pts.reshape(1,-1,2)],True,(0,220,80),2)
    except Exception: pass

    # LDW
    if abs(dev_px) > LDW_THRESH:
        sd = "RIGHT" if dev_px > 0 else "LEFT"
        cv2.rectangle(ann,(10,85),(480,140),(0,0,100),-1)
        cv2.putText(ann,f"LDW: DRIFT {sd}",(18,128),
                    cv2.FONT_HERSHEY_SIMPLEX,0.80,(0,0,255),2)
    else:
        cv2.rectangle(ann,(10,85),(340,140),(0,60,0),-1)
        cv2.putText(ann,"LDW: Lane OK",(18,128),
                    cv2.FONT_HERSHEY_SIMPLEX,0.80,(0,255,0),2)

    return steer_err, crad, cd, ann, mode


# =========================================================================
# LAYER 3 — DEPTH SENSOR
# =========================================================================
class DepthSensor:
    def __init__(self):
        self.rs  = None
        self.ok  = False
        if not REALSENSE_AVAILABLE: return
        try:
            self.rs = QCarRealSense(mode='RGB, Depth')
            self.ok = True
        except Exception as e:
            print(f"RealSense init failed: {e}")

    def depth_in_box(self, x1,y1,x2,y2) -> Optional[float]:
        if not self.ok or self.rs is None: return None
        try:
            self.rs.read_depth()
            df = self.rs.imageBufferDepth
            if df is None or df.size == 0: return None
            roi   = df[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            valid = roi[(roi>0.05)&(roi<10.0)]
            return float(np.median(valid)) if valid.size >= 10 else None
        except Exception: return None


# =========================================================================
# LAYER 3 — TRAFFIC LIGHT SELECTOR
# =========================================================================
def select_tl(detections: List[dict], fw: int) -> Optional[str]:
    """Centre-biased: exclude rightmost, pick closest to frame centre."""
    cands = [d for d in detections
             if d['cx'] < fw*TL_RIGHT_EXCL
             and d['area'] >= TL_MIN_AREA
             and d['conf'] >= TL_MIN_CONF]
    if not cands: return None
    return min(cands, key=lambda d: abs(d['cx'] - fw/2.0))['label']


# =========================================================================
# LAYER 4 — MAP CANVAS
# =========================================================================
def make_map() -> np.ndarray:
    c = np.zeros((MAP_H, MAP_W, 3), dtype=np.uint8)
    for xg in range(int(WORLD_XMIN)-1, int(WORLD_XMAX)+2):
        u,_ = w2p(xg,0); cv2.line(c,(u,0),(u,MAP_H),(30,30,30),1)
    for yg in range(int(WORLD_YMIN)-1, int(WORLD_YMAX)+2):
        _,v = w2p(0,yg); cv2.line(c,(0,v),(MAP_W,v),(30,30,30),1)
    for i in range(1, N_WP):
        cv2.line(c, w2p(*WP[i-1]), w2p(*WP[i]), (0,140,255), 1)
    cv2.circle(c, w2p(*WP[0]),  8, (0,255,0), 2)
    cv2.circle(c, w2p(*WP[-1]), 8, (0,0,255), 2)
    return c


# =========================================================================
# SIGNAL HANDLER
# =========================================================================
_car_ref = None
KILL     = False
def _stop(sig=None,_=None):
    global KILL, _car_ref
    KILL = True
    if _car_ref:
        try: _car_ref.write(0.0, 0.0, np.zeros(8,dtype=int))
        except: pass
signal.signal(signal.SIGINT,  _stop)
signal.signal(signal.SIGTERM, _stop)


# =========================================================================
# MAIN
# =========================================================================
def main():
    global _car_ref, KILL

    print("\n  Black Hawks — Master Controller")
    print(f"  {N_WP} waypoints   QCar2   csi[{FRONT_CSI_IDX}]")
    print("  Q=quit   P=pause/resume\n")

    # ── YOLO + depth ──────────────────────────────────────────────────────
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO('best.pt')
            print(f"  YOLO: {list(yolo_model.names.values())}")
        except Exception as e:
            print(f"  YOLO load failed: {e}")
    depth_sensor = DepthSensor()

    # ── Controllers ───────────────────────────────────────────────────────
    pose_ekf   = PoseEKF(*INITIAL_POSE)
    speed_pid  = SpeedPID()
    stanley    = Stanley()
    left_lane  = LaneLine()
    right_lane = LaneLine()

    # ── State ─────────────────────────────────────────────────────────────
    prev_enc          = None
    last_gps_t        = time.time()
    gps_fresh         = False
    last_time         = time.time()
    paused            = False
    prev_steer        = 0.0
    lane_integral     = 0.0
    lane_prev_err     = 0.0
    yolo_frame_ctr    = 0
    frame_idx         = 0

    # ML state machine
    stop_active       = False
    stop_start_t      = 0.0
    stop_cooldown_end = 0.0
    stop_streak       = 0
    red_active        = False
    green_active      = False
    ml_brake          = False
    ml_status         = "PATH CLEAR"
    ml_color          = (0,255,0)

    # Pedestrian state
    ped_active        = False
    ped_start_t       = 0.0
    ped_cooldown_end  = 0.0

    # Crosswalk state
    cw_detected       = False   # refreshed every YOLO frame

    # Traffic light: red→green immunity (10 s after a completed red→green cycle)
    tl_immunity_end   = 0.0     # time until TL red is ignored

    # Persistent bounding boxes for display (updated every YOLO frame, shown every frame)
    yolo_boxes: List[dict] = []   # list of {x1,y1,x2,y2,label,color,conf}

    base_map  = make_map()
    traj_map  = np.zeros((MAP_H,MAP_W,3), dtype=np.uint8)

    with QCar(readMode=0) as car, \
         QCarCameras(enableFront=True,enableBack=True,
                     enableLeft=True,enableRight=True) as cams, \
         QCarGPS(initialPose=INITIAL_POSE) as gps:

        _car_ref = car

        # First GPS fix
        cams.readAll(); car.read()
        for _ in range(25):
            if gps.readGPS():
                pose_ekf.update_gps(float(gps.position[0]),
                                    float(gps.position[1]),
                                    float(gps.orientation[2]))
                gps_fresh = True; last_gps_t = time.time()
                print(f"  First fix: ({pose_ekf.x:+.4f},{pose_ekf.y:+.4f})")
                break
            time.sleep(0.05)

        try: prev_enc = int(car.motorEncoder[0])
        except: prev_enc = 0

        try:
            while not KILL:
                now  = time.time()
                dt   = max(now - last_time, 1e-4)
                last_time = now

                cams.readAll(); car.read()

                # ── Sensors ───────────────────────────────────────────────
                try:    raw_gyro = float(car.gyroscope[2])
                except: raw_gyro = 0.0
                try:    raw_enc = int(car.motorEncoder[0])
                except: raw_enc = prev_enc
                try:    v_meas = abs(float(car.motorTach))
                except: v_meas = 0.0

                dticks = raw_enc - prev_enc
                if dticks >  ENCODER_OVERFLOW//2: dticks -= ENCODER_OVERFLOW
                if dticks < -ENCODER_OVERFLOW//2: dticks += ENCODER_OVERFLOW
                prev_enc = raw_enc
                dist_m   = dticks * METRES_PER_TICK
                gyro_c   = raw_gyro - GYRO_BIAS
                if abs(gyro_c) < GYRO_NOISE_FLOOR: gyro_c = 0.0

                # ── Pose EKF ──────────────────────────────────────────────
                pose_ekf.predict(dist_m, gyro_c, dt)
                if gps.readGPS():
                    pose_ekf.update_gps(float(gps.position[0]),
                                        float(gps.position[1]),
                                        float(gps.orientation[2]))
                    gps_fresh = True; last_gps_t = now
                gps_age   = now - last_gps_t
                gps_fresh = gps_age < GPS_TIMEOUT
                ex, ey, eyaw = pose_ekf.x, pose_ekf.y, pose_ekf.yaw
                p_front = np.array([ex+0.2*math.cos(eyaw), ey+0.2*math.sin(eyaw)])

                # ── Stanley ───────────────────────────────────────────────
                stanley_steer = stanley.steer(p_front, eyaw, v_meas)
                corner_deg    = stanley.upcoming_corner_deg()
                dist_next     = stanley.dist_to_next(p_front)
                v_ref = (V_REF_SLOW if corner_deg > CORNER_THRESH or
                         dist_next < SLOW_RADIUS else V_REF)

                # ══════════════════════════════════════════════════════════
                # LAYER 3 — ML PERCEPTION (every YOLO_FRAME_SKIP frames)
                # ══════════════════════════════════════════════════════════
                csi = cams.csi[FRONT_CSI_IDX]
                frame_raw = None
                if csi is not None and csi.imageData is not None:
                    frame_raw = cv2.resize(csi.imageData.copy(), (FRAME_W, FRAME_H))

                yolo_frame_ctr += 1
                if yolo_model is not None and frame_raw is not None and \
                        (yolo_frame_ctr % YOLO_FRAME_SKIP == 0):
                    try:
                        res          = yolo_model.predict(source=frame_raw,
                                                          conf=YOLO_CONF, verbose=False)
                        red_active   = False
                        green_active = False
                        stop_raw     = False
                        ped_raw      = False
                        cw_raw       = False
                        tl_dets: List[dict] = []
                        yolo_boxes   = []   # reset draw list each YOLO frame

                        for r in res:
                            for box in r.boxes:
                                lbl  = yolo_model.names[int(box.cls[0])].lower()
                                conf = float(box.conf[0])
                                x1,y1,x2,y2 = map(int, box.xyxy[0])
                                cx   = (x1+x2)//2;  cy = (y1+y2)//2
                                area = (x2-x1)*(y2-y1)

                                # ── Traffic lights ───────────────────────
                                if any(k in lbl for k in ('red','yellow','green')):
                                    key = ('red' if 'red' in lbl else
                                           'yellow' if 'yellow' in lbl else 'green')
                                    tl_col = ((0,0,255) if key=='red' else
                                              (0,200,255) if key=='yellow' else (0,255,0))
                                    tl_dets.append({'label':key,'cx':cx,'cy':cy,
                                                    'area':area,'conf':conf,
                                                    'box':(x1,y1,x2,y2)})
                                    yolo_boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                                                       'label':f"TL:{key} {conf:.2f}",
                                                       'color':tl_col})

                                # ── Stop sign (right-half ROI) ────────────
                                elif 'stop' in lbl and conf >= STOP_MIN_CONF:
                                    in_roi = box_in_roi(x1,y1,x2,y2,FRAME_W,FRAME_H,
                                                        0.50,0.00,1.00,1.00)
                                    if in_roi:
                                        dm = depth_sensor.depth_in_box(x1,y1,x2,y2)
                                        close = (dm < STOP_DEPTH_MAX if dm is not None
                                                 else area > STOP_AREA_FALLBACK)
                                        if close:
                                            stop_raw = True
                                    yolo_boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                                                       'label':f"STOP {conf:.2f}",
                                                       'color':(0,0,255)})

                                # ── Pedestrian (right-half ROI) ───────────
                                elif 'pedestrian' in lbl or 'person' in lbl:
                                    if conf >= PED_MIN_CONF and area >= PED_MIN_AREA:
                                        in_roi = box_in_roi(x1,y1,x2,y2,FRAME_W,FRAME_H,
                                                            PED_ROI_X1_FRAC,PED_ROI_Y1_FRAC,
                                                            PED_ROI_X2_FRAC,PED_ROI_Y2_FRAC)
                                        if in_roi:
                                            ped_raw = True
                                    yolo_boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                                                       'label':f"PED {conf:.2f}",
                                                       'color':(0,165,255)})

                                # ── Crosswalk (bottom-25% ROI) ────────────
                                elif 'crosswalk' in lbl or 'cross' in lbl or 'zebra' in lbl:
                                    if conf >= CW_MIN_CONF and area >= CW_MIN_AREA:
                                        in_roi = box_in_roi(x1,y1,x2,y2,FRAME_W,FRAME_H,
                                                            CW_ROI_X1_FRAC,CW_ROI_Y1_FRAC,
                                                            CW_ROI_X2_FRAC,CW_ROI_Y2_FRAC)
                                        if in_roi:
                                            cw_raw = True
                                    yolo_boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                                                       'label':f"XWALK {conf:.2f}",
                                                       'color':(255,255,0)})

                                # ── Other detections — draw but no action ─
                                else:
                                    yolo_boxes.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                                                       'label':f"{lbl} {conf:.2f}",
                                                       'color':(180,180,180)})

                        # ── Traffic light: centre-biased selection ────────
                        chosen = select_tl(tl_dets, FRAME_W)
                        tl_in_immunity = (now < tl_immunity_end)

                        if chosen in ('red','yellow') and not tl_in_immunity:
                            red_active   = True;  green_active = False
                        elif chosen == 'green':
                            # Green after red → start immunity window
                            if red_active:
                                tl_immunity_end = now + TL_GREEN_AFTER_RED_COOLDOWN
                                print(f"  [ML] Red→Green: TL immunity {TL_GREEN_AFTER_RED_COOLDOWN:.0f}s")
                            green_active = True;  red_active = False

                        # ── Stop sign streak ──────────────────────────────
                        in_stop_cd = now < stop_cooldown_end
                        if stop_raw and not stop_active and not in_stop_cd:
                            stop_streak += 1
                        else:
                            stop_streak = 0
                        if stop_streak >= STOP_STREAK_REQ and not stop_active:
                            stop_active  = True
                            stop_start_t = now
                            stop_streak  = 0
                            print(f"  [ML] STOP SIGN triggered  t={now:.1f}")

                        # ── Pedestrian activation ─────────────────────────
                        in_ped_cd = now < ped_cooldown_end
                        if ped_raw and not in_ped_cd:
                            if not ped_active:
                                ped_active  = True
                                ped_start_t = now
                                print(f"  [ML] PEDESTRIAN detected  t={now:.1f}")
                        else:
                            if ped_active and not ped_raw:
                                # Pedestrian left ROI — start cooldown
                                ped_active        = False
                                ped_cooldown_end  = now + PED_COOLDOWN_S
                                print(f"  [ML] Pedestrian cleared  cooldown {PED_COOLDOWN_S:.0f}s")

                        # ── Crosswalk: direct flag (no hold needed) ───────
                        cw_detected = cw_raw

                    except Exception as e:
                        print(f"  [YOLO err] {e}")

                # ── ML state machine (every frame) ────────────────────────
                ml_brake  = False
                ml_slow   = False   # crosswalk slow-down flag
                ml_status = "PATH CLEAR";  ml_color = (0,255,0)

                # 1. Red / yellow light (highest priority, respects immunity)
                if red_active:
                    ml_brake  = True
                    ml_status = "RED LIGHT — STOP";  ml_color = (0,0,255)

                # 2. Green light clears stop sign lock + shows immunity countdown
                if green_active and stop_active:
                    stop_active = False
                    print("  [ML] Green → stop lock cleared")

                if green_active and not red_active and not stop_active:
                    remaining_imm = max(0.0, tl_immunity_end - now)
                    if remaining_imm > 0:
                        ml_status = f"GREEN — GO (TL immune {remaining_imm:.0f}s)"
                    else:
                        ml_status = "GREEN LIGHT — GO"
                    ml_color = (0,255,0)

                # 3. Stop sign hold
                if stop_active:
                    elapsed = now - stop_start_t
                    if elapsed < STOP_HOLD_S:
                        ml_brake  = True
                        ml_status = f"STOP SIGN — {STOP_HOLD_S-elapsed:.1f}s"
                        ml_color  = (0,140,255)
                    else:
                        stop_active       = False
                        stop_cooldown_end = now + STOP_COOLDOWN_S
                        print(f"  [ML] Stop cleared — cooldown {STOP_COOLDOWN_S:.0f}s")

                if not stop_active and now < stop_cooldown_end:
                    if ml_status == "PATH CLEAR":
                        ml_status = f"STOP COOLDOWN {stop_cooldown_end-now:.1f}s"
                        ml_color  = (0,200,200)

                # 4. Pedestrian hold (stop while ped in ROI, PED_STOP_HOLD_S minimum)
                if ped_active:
                    elapsed_ped = now - ped_start_t
                    if elapsed_ped < PED_STOP_HOLD_S or True:  # hold while still detected
                        ml_brake  = True
                        ml_status = "PEDESTRIAN — STOP";  ml_color = (0,100,255)

                if not ped_active and now < ped_cooldown_end:
                    if ml_status == "PATH CLEAR":
                        ml_status = f"PED COOLDOWN {ped_cooldown_end-now:.1f}s"
                        ml_color  = (100,200,255)

                # 5. Crosswalk — slow down (does NOT stop, just caps throttle)
                if cw_detected and not ml_brake:
                    ml_slow   = True
                    if ml_status == "PATH CLEAR":
                        ml_status = "CROSSWALK — SLOW";  ml_color = (0,220,220)

                # ══════════════════════════════════════════════════════════
                # LAYER 2 — LANE PIPELINE
                # ══════════════════════════════════════════════════════════
                lane_steer    = 0.0
                lane_detected = False
                lane_mode     = "NO LANE"
                crad, cdir    = 9999.0, "Straight"
                ann           = None

                if frame_raw is not None:
                    try:
                        steer_err, crad, cdir, ann, lane_mode = run_lane_pipeline(
                            frame_raw, left_lane, right_lane)

                        if lane_mode != "NO LANE":
                            lane_detected  = True
                            deriv          = (steer_err - lane_prev_err) / dt
                            lane_integral += steer_err * dt
                            lane_integral  = float(np.clip(lane_integral, -0.35, 0.35))
                            lane_steer     = float(np.clip(
                                0.90*steer_err + 0.004*lane_integral + 0.22*deriv,
                                -MAX_STEER, MAX_STEER))
                            lane_prev_err  = steer_err
                        else:
                            lane_integral = 0.0; lane_prev_err = 0.0

                    except Exception:
                        pass

                # ══════════════════════════════════════════════════════════
                # LAYER 4 — BLEND + THROTTLE
                # ══════════════════════════════════════════════════════════
                throttle = 0.0;  delta = 0.0;  source_txt = ""

                if ml_brake:
                    throttle  = THR_CRAWL
                    delta     = 0.0
                    lane_integral = 0.0
                    speed_pid.reset()
                    source_txt = ml_status

                elif paused or stanley.completed:
                    throttle = 0.0;  delta = 0.0
                    source_txt = "PAUSED" if paused else "DONE"

                else:
                    # Steer blend with conflict detection
                    conflict = (lane_detected and
                                abs(stanley_steer - lane_steer) > CONFLICT_THRESH)
                    if not lane_detected:
                        sw, lw = 1.00, 0.00;  source_txt = "STANLEY only"
                    elif conflict:
                        sw, lw = 0.90, 0.10
                        source_txt = (f"CONFLICT "
                                      f"S:{math.degrees(stanley_steer):+.0f}° "
                                      f"L:{math.degrees(lane_steer):+.0f}°")
                    elif not gps_fresh:
                        sw, lw = 0.30, 0.70
                        source_txt = f"LANE-DOM GPS stale {gps_age:.1f}s"
                    else:
                        sw, lw = STANLEY_WEIGHT, LANE_WEIGHT
                        source_txt = (f"BLEND "
                                      f"S:{math.degrees(stanley_steer):+.0f}° "
                                      f"L:{math.degrees(lane_steer):+.0f}°")

                    blended = float(np.clip(
                        sw*stanley_steer + lw*lane_steer, -MAX_STEER, MAX_STEER))
                    delta      = STEER_ALPHA*prev_steer + (1-STEER_ALPHA)*blended
                    prev_steer = delta

                    # Throttle — hierarchy: crosswalk > sharp curve > corner > normal
                    base_thr = speed_pid.update(v_meas, v_ref, dt)
                    is_sharp = (crad < CURVE_RADIUS_THRESHOLD) and lane_detected
                    if   ml_slow:                    throttle = min(base_thr, CW_SLOW_THROTTLE_CAP)
                    elif is_sharp:                   throttle = min(base_thr, 0.030)
                    elif corner_deg > CORNER_THRESH:  throttle = min(base_thr, 0.038)
                    else:                             throttle = base_thr

                car.write(float(throttle), float(delta), np.zeros(8, dtype=int))

                # ── MAP ───────────────────────────────────────────────────
                eu, ev = w2p(ex, ey)
                cv2.circle(traj_map,(eu,ev),2,
                           (0,220,255) if gps_fresh else (0,100,180),-1)
                dm = cv2.addWeighted(base_map,1.0,traj_map,1.0,0)
                cv2.circle(dm,(eu,ev),7,(0,255,0),-1)
                fu,fv = w2p(ex+0.35*math.cos(eyaw),ey+0.35*math.sin(eyaw))
                cv2.arrowedLine(dm,(eu,ev),(fu,fv),(0,255,0),2,tipLength=0.4)
                if stanley.wpi < N_WP:
                    tw,tv = w2p(*WP[stanley.wpi])
                    cv2.circle(dm,(tw,tv),6,(255,100,0),-1)
                prog = stanley.wpi/max(N_WP-1,1)
                cv2.rectangle(dm,(4,MAP_H-14),(int(4+prog*(MAP_W-8)),MAP_H-4),(0,180,100),-1)

                status_tag = ("STOP" if ml_brake else "PAUSED" if paused else
                              "DONE" if stanley.completed else "RUN")
                map_color  = (0,0,255) if ml_brake else (0,255,255)
                for txt,col,pos in [
                    (f"[{status_tag}] WP:{stanley.wpi}/{N_WP} "
                     f"GPS:{'OK' if gps_fresh else 'STALE'}",
                     map_color,(4,18)),
                    (f"EKF({ex:+.2f},{ey:+.2f}) {math.degrees(eyaw):+.0f}°",
                     (0,200,255),(4,34)),
                    (f"S:{math.degrees(stanley_steer):+.0f}° "
                     f"L:{math.degrees(lane_steer):+.0f}° "
                     f"→{math.degrees(delta):+.0f}°",
                     (200,255,100),(4,50)),
                    (source_txt,(255,200,0),(4,66)),
                    (f"ML: {ml_status}",ml_color,(4,82)),
                ]:
                    cv2.putText(dm,txt,pos,cv2.FONT_HERSHEY_SIMPLEX,0.34,col,1)
                cv2.imshow("Path Map", dm)

                # ── CAMERA ────────────────────────────────────────────────
                if ann is not None:
                    h_a, w_a = ann.shape[:2]

                    # ── Draw persistent YOLO bounding boxes ───────────────
                    for det in yolo_boxes:
                        bx1,by1,bx2,by2 = det['x1'],det['y1'],det['x2'],det['y2']
                        bcol = det['color']
                        blbl = det['label']
                        cv2.rectangle(ann, (bx1,by1), (bx2,by2), bcol, 2)
                        # Label background pill
                        (tw,th), _ = cv2.getTextSize(blbl,
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
                        lx1,ly1 = bx1, max(by1-th-6, 0)
                        lx2,ly2 = bx1+tw+6, max(by1, th+6)
                        cv2.rectangle(ann, (lx1,ly1), (lx2,ly2), bcol, -1)
                        cv2.putText(ann, blbl, (lx1+3, ly2-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 1)

                    # ── Draw ROI zones (thin coloured outlines) ───────────
                    # Stop sign ROI (right half) — red dashed outline
                    draw_roi_outline(ann, 0.50,0.00,1.00,1.00,
                                     (0,0,200), "STOP/PED ROI")
                    # Pedestrian ROI is identical — no double draw needed
                    # Crosswalk ROI (bottom 25%) — yellow outline
                    draw_roi_outline(ann, CW_ROI_X1_FRAC, CW_ROI_Y1_FRAC,
                                     CW_ROI_X2_FRAC, CW_ROI_Y2_FRAC,
                                     (0,220,220), "XWALK ROI")

                    # ── ML banner (top) ───────────────────────────────────
                    cv2.rectangle(ann,(5,5),(w_a-5,75),(0,0,0),-1)
                    cv2.putText(ann, f"ML: {ml_status}",
                        (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, ml_color, 2)
                    cv2.putText(ann,
                        f"LANE:{lane_mode}  {cdir}  R={crad:.0f}m  "
                        f"WP:{stanley.wpi}/{N_WP}  streak:{stop_streak}",
                        (12,62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1)

                    # ── Bottom HUD ────────────────────────────────────────
                    cv2.rectangle(ann,(0,h_a-30),(w_a,h_a),(0,0,0),-1)
                    cv2.putText(ann,
                        f"Thr:{throttle:.4f}  Steer:{math.degrees(delta):+.1f}°  "
                        f"v:{v_meas:.3f}  {source_txt}",
                        (8,h_a-8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255,220,60), 1)

                    cv2.imshow("Black Hawks | Master", ann)

                print(f"  [{frame_idx:05d}] "
                      f"EKF({ex:+.3f},{ey:+.3f}) "
                      f"WP={stanley.wpi} "
                      f"S={math.degrees(stanley_steer):+.0f}° "
                      f"L={math.degrees(lane_steer):+.0f}° "
                      f"→{math.degrees(delta):+.0f}° "
                      f"thr={throttle:.4f} "
                      f"ML={ml_status[:12]}",
                      end='\r')

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): print("\n  Q."); break
                elif key == ord('p'):
                    paused = not paused
                    if paused: speed_pid.reset()
                    print(f"\n  [{'PAUSED' if paused else 'RESUMED'}]")

                if stanley.completed:
                    print(f"\n  [DONE] {N_WP} waypoints complete."); break
                frame_idx += 1

        except Exception as e:
            print(f"\n  [Exception] {e}")
            import traceback; traceback.print_exc()

        finally:
            car.write(0.0, 0.0, np.zeros(8,dtype=int))

    cv2.destroyAllWindows()
    print(f"\n  Shutdown. frames={frame_idx}")

if __name__ == "__main__":
    main()