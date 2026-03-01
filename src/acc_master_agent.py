"""
Black Hawks — Combined ADAS v5 + ML Master Agent  (PC #2)
==========================================================
Architecture:
  • Expert lane pipeline  — dual-channel (yellow centre / white edge), EKF-smoothed
    polynomial history, curvature-adaptive PID, sliding-window with prior warm-start.
  • ML perception         — YOLOv8 every 5th frame, RealSense depth gating, ROI
    filtering, 3-frame streak confirmation for stop sign.
  • Traffic light logic   — centre-biased multi-signal selection (excludes rightmost).
  • Stop sign state machine — 5 s hold → 8 s cooldown (YOLO still runs, sign ignored).
  • Depth gating          — RealSense median depth inside bbox must be < STOP_DEPTH_MAX.

Tuning constants are in the TUNING BLOCK section only — do not edit elsewhere.

Controls: q in OpenCV window to quit.
"""

from __future__ import annotations
import sys, os, time, signal as _signal, warnings

from collections import deque
from typing import Optional, Tuple, List

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

# =========================================================================
# PATH SETUP  (PC #2 locked)
# =========================================================================
PAL_PATH = (
    r"C:\Users\91778\Downloads\Q-car\Q-car"
    r"\Quanser_Academic_Resources-dev-windows"
    r"\Quanser_Academic_Resources-dev-windows"
    r"\0_libraries\python"
)
if PAL_PATH not in sys.path:
    sys.path.append(PAL_PATH)

try:
    from pal.products.qcar import QCar, QCarCameras
except ImportError as e:
    print(f"PAL import error: {e}"); sys.exit(1)

# RealSense depth
REALSENSE_AVAILABLE = False
try:
    # Official Quanser wrapper — uses Quanser's hardware lock, not raw pyrealsense2
    from pal.products.qcar import QCarRealSense
    REALSENSE_AVAILABLE = True
    print("QCar RealSense API available.")
except ImportError as e:
    print(f"QCarRealSense not found: {e} — depth gating disabled (area fallback).")

YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 available.")
except ImportError:
    print("YOLO not found — perception disabled.")


# =========================================================================
# ╔══════════════════════════════════════════════════════════════════════╗
# ║                        TUNING BLOCK                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
# =========================================================================

# ── Camera ────────────────────────────────────────────────────────────────
FRONT_CSI_IDX   = 3
FRAME_W, FRAME_H = 640, 480        # normalised inference resolution

# ── Throttle schedule ─────────────────────────────────────────────────────
BASE_THROTTLE   = 0.036
THR_FAST        = BASE_THROTTLE * 1.15   # straight, lane centred
THR_MID         = BASE_THROTTLE * 0.90   # mild curve
THR_SLOW        = BASE_THROTTLE * 0.58   # sharp curve / degraded lane
THR_BLIND       = BASE_THROTTLE * 0.22   # no lane detected — safety creep
THR_CRAWL       = 0.0                    # full stop (red light / stop sign)

# ── PID ───────────────────────────────────────────────────────────────────
KP              = 0.92
KI              = 0.004
KD              = 0.24
MAX_STEER       = 0.50
STEER_ALPHA     = 0.38    # exponential smoothing (lower = smoother, more lag)
INTEGRAL_CLAMP  = 0.35

# ── Bird's-eye warp ───────────────────────────────────────────────────────
ROI_TOP_FRAC    = 0.52   # captures lower-third yellow line (was 0.48)
ROI_TL_X_FRAC   = 0.08   # mild inset top-left
ROI_TR_X_FRAC   = 0.78   # mild inset top-right
ROI_BL_X_FRAC   = 0.02   # slight inset only — left white edge must be visible
ROI_BR_X_FRAC   = 0.98   # near full width — right white edge must be visible
WARP_L          = 0.12
WARP_R          = 0.88

# ── Lane geometry ─────────────────────────────────────────────────────────
YELLOW_OFFSET_PX        = 55      # px offset right of yellow centre line
YELLOW_OFFSET_CURVE_PX  = 30      # tighter offset in sharp curves
HALF_LANE_PX            = 120     # half lane width in warped px
XM_PER_PIX              = 3.7 / 820
YM_PER_PIX              = 30.0 / 410
CURVE_RADIUS_THRESHOLD  = 25.0    # m — below this = "sharp curve"
MIN_LANE_WIDTH_PX       = 100
MAX_LANE_WIDTH_PX       = 700

# ── Sliding window ────────────────────────────────────────────────────────
SW_NWIN         = 12
SW_MARGIN_BASE  = 90
SW_MINPIX       = 30
MIN_FIT_PIXELS  = 150

# ── EKF polynomial smoother ───────────────────────────────────────────────
# State = [a, b, c] of y = a*x^2 + b*x + c
# Increase PROC_NOISE for faster adaptation, decrease for smoother output.
HISTORY_LEN     = 5
PROC_NOISE      = 0.08    # process noise scalar (Q = I * PROC_NOISE)
MEAS_NOISE      = 0.40    # measurement noise scalar (R = I * MEAS_NOISE)

# ── Colour thresholds ─────────────────────────────────────────────────────
YELLOW_LOW      = np.array([ 18,  80,  80], dtype=np.uint8)
YELLOW_HIGH     = np.array([ 38, 255, 255], dtype=np.uint8)
WHITE_LOW       = np.array([  0,   0, 200], dtype=np.uint8)
WHITE_HIGH      = np.array([180,  40, 255], dtype=np.uint8)
S_THRESH        = (100, 255)
SX_THRESH       = ( 20, 100)
L_THRESH        = (200, 255)
B_THRESH        = (145, 200)

# ── LDW ───────────────────────────────────────────────────────────────────
LDW_THRESHOLD_PX = 50

# ── YOLO ──────────────────────────────────────────────────────────────────
YOLO_FRAME_SKIP     = 5
YOLO_CONF_THRESHOLD = 0.45

# ── Stop sign ─────────────────────────────────────────────────────────────
STOP_DEPTH_MAX          = 2.0     # metres — react only when closer than this
STOP_DEPTH_FALLBACK_AREA= 3000    # px² fallback if RealSense unavailable
STOP_SIGN_MIN_CONF      = 0.60
STOP_SIGN_STREAK_REQ    = 3       # consecutive YOLO frames before triggering
STOP_HOLD_DURATION      = 5.0     # seconds car holds at stop
STOP_COOLDOWN_DURATION  = 8.0     # seconds sign is ignored after stop clears

# ── Traffic light ─────────────────────────────────────────────────────────
TL_MIN_CONF             = 0.50
TL_MIN_AREA             = 300     # px² — reject tiny ghost detections
# Centre-biased selection: only consider boxes whose centre cx < FRAME_W * TL_RIGHT_EXCLUDE
TL_RIGHT_EXCLUDE_FRAC   = 0.80    # anything right of 80% of frame width is ignored


# =========================================================================
# GLOBALS
# =========================================================================
KILL = False
def _sig_handler(*_):
    global KILL; KILL = True
_signal.signal(_signal.SIGINT,  _sig_handler)
_signal.signal(_signal.SIGTERM, _sig_handler)


# =========================================================================
# ── REALSENSE DEPTH HELPER ────────────────────────────────────────────────
# =========================================================================
class DepthSensor:
    """
    Quanser-native RealSense wrapper using QCarRealSense API.
    Uses Quanser's hardware lock — never calls pyrealsense2 directly.
    Falls back to bbox-area proxy if unavailable (simulator mode).
    """
    def __init__(self):
        self.rs_cam  = None
        self.enabled = False
        if not REALSENSE_AVAILABLE:
            return
        try:
            self.rs_cam  = QCarRealSense(mode='RGB, Depth')
            self.enabled = True
            print("QCarRealSense initialised (RGB + Depth).")
        except Exception as e:
            print(f"QCarRealSense init failed: {e} — area fallback active.")

    def median_depth_in_box(self, x1: int, y1: int, x2: int, y2: int) -> Optional[float]:
        """
        Read latest depth frame via Quanser API and return median depth (m)
        inside the bounding box. Returns None if unavailable.
        """
        if not self.enabled or self.rs_cam is None:
            return None
        try:
            self.rs_cam.read_depth()
            depth_frame = self.rs_cam.imageBufferDepth  # 2D float array, values in metres
            if depth_frame is None or depth_frame.size == 0:
                return None
            roi   = depth_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            valid = roi[(roi > 0.05) & (roi < 10.0)]   # filter invalid/saturated readings
            if valid.size < 10:
                return None
            return float(np.median(valid))
        except Exception:
            return None

    def stop(self):
        # QCarRealSense is managed by Quanser's background server — no explicit stop needed
        self.rs_cam  = None
        self.enabled = False


# =========================================================================
# ── EKF POLYNOMIAL SMOOTHER ───────────────────────────────────────────────
# =========================================================================
class PolyEKF:
    """
    Scalar Extended Kalman Filter over a 2nd-degree polynomial [a, b, c].
    State  x  = [a, b, c]
    Model:  x_{k+1} = x_k  (constant-polynomial assumption)
    Measurement: z = polyfit result from the lane detector
    """
    def __init__(self):
        self.x  = None                          # state [a, b, c]
        self.P  = np.eye(3) * 10.0             # large initial covariance
        self.Q  = np.eye(3) * PROC_NOISE
        self.R  = np.eye(3) * MEAS_NOISE

    def update(self, measurement: np.ndarray) -> np.ndarray:
        if self.x is None:
            self.x = measurement.copy()
            return self.x
        # Predict
        x_pred = self.x.copy()
        P_pred = self.P + self.Q
        # Update (H = I for direct measurement)
        S      = P_pred + self.R
        K      = P_pred @ np.linalg.inv(S)
        self.x = x_pred + K @ (measurement - x_pred)
        self.P = (np.eye(3) - K) @ P_pred
        return self.x

    def predict_only(self) -> Optional[np.ndarray]:
        """Called on frames where no measurement available — returns prior."""
        if self.x is None:
            return None
        self.P = self.P + self.Q   # inflate uncertainty
        return self.x

    def reset(self):
        self.x = None
        self.P = np.eye(3) * 10.0


# =========================================================================
# ── LANE LINE STATE ───────────────────────────────────────────────────────
# =========================================================================
class LaneLine:
    def __init__(self):
        self.ekf       = PolyEKF()
        self.detected  = False
        self.best_fit  : Optional[np.ndarray] = None
        self.frames_lost = 0
        self.MAX_LOST  = 8    # frames before full reset

    def update(self, raw_fit: np.ndarray):
        self.best_fit   = self.ekf.update(raw_fit)
        self.detected   = True
        self.frames_lost = 0

    def predict(self) -> Optional[np.ndarray]:
        """Use EKF prior when detector fails this frame."""
        self.frames_lost += 1
        if self.frames_lost > self.MAX_LOST:
            self.reset()
            return None
        return self.ekf.predict_only()

    def get_fit(self) -> Optional[np.ndarray]:
        return self.best_fit

    def reset(self):
        self.ekf.reset()
        self.detected    = False
        self.best_fit    = None
        self.frames_lost = 0


# =========================================================================
# ── WARP & ROI ────────────────────────────────────────────────────────────
# =========================================================================
def build_warp_matrices(h: int, w: int):
    src = np.float32([
        [w * ROI_TL_X_FRAC, h * ROI_TOP_FRAC],
        [w * ROI_TR_X_FRAC, h * ROI_TOP_FRAC],
        [w * ROI_BR_X_FRAC, h - 1],
        [w * ROI_BL_X_FRAC, h - 1],
    ])
    dst = np.float32([
        [w * WARP_L, 0],
        [w * WARP_R, 0],
        [w * WARP_R, h - 1],
        [w * WARP_L, h - 1],
    ])
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv, src.astype(np.int32)

_WARP_CACHE: dict = {}

def get_warp(h: int, w: int):
    key = (h, w)
    if key not in _WARP_CACHE:
        _WARP_CACHE[key] = build_warp_matrices(h, w)
    return _WARP_CACHE[key]


def apply_roi_warp(frame: np.ndarray):
    h, w = frame.shape[:2]
    M, Minv, roi_pts = get_warp(h, w)
    # Mask everything outside trapezoid
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_pts], 255)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    warped = cv2.warpPerspective(masked, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped, Minv, roi_pts


# =========================================================================
# ── BINARY MASKS ──────────────────────────────────────────────────────────
# =========================================================================
def build_binary_channels(bgr: np.ndarray):
    """
    Returns:
        mask_yellow  — yellow centre line pixels
        mask_white   — white edge line pixels
        combined     — HLS/LAB gradient fusion for fallback
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, YELLOW_LOW,  YELLOW_HIGH)
    mask_white  = cv2.inRange(hsv, WHITE_LOW,   WHITE_HIGH)

    # Morphological clean
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, k3)
    mask_white  = cv2.morphologyEx(mask_white,  cv2.MORPH_CLOSE, k3)

    # Gradient-based fallback channel (HLS + LAB)
    hls  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ch = hls[:, :, 1]
    s_ch = hls[:, :, 2]
    b_ch = lab[:, :, 2]

    sx    = cv2.Sobel(l_ch, cv2.CV_64F, 1, 0, ksize=3)
    abs_s = np.abs(sx)
    mx    = abs_s.max()
    scaled = (abs_s / mx * 255).astype(np.uint8) if mx > 0 else np.zeros_like(abs_s, dtype=np.uint8)

    sx_bin = ((scaled >= SX_THRESH[0]) & (scaled <= SX_THRESH[1])).astype(np.uint8)
    s_bin  = ((s_ch   >= S_THRESH[0])  & (s_ch   <= S_THRESH[1])).astype(np.uint8)
    l_bin  = ((l_ch   >= L_THRESH[0])  & (l_ch   <= L_THRESH[1])).astype(np.uint8)
    b_bin  = ((b_ch   >= B_THRESH[0])  & (b_ch   <= B_THRESH[1])).astype(np.uint8)

    combined = np.zeros_like(sx_bin)
    combined[((s_bin & sx_bin) == 1) | (l_bin == 1) | (b_bin == 1)] = 1

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k5)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k5)

    return (mask_yellow > 0).astype(np.uint8), (mask_white > 0).astype(np.uint8), combined


# =========================================================================
# ── SLIDING WINDOW DETECTION ──────────────────────────────────────────────
# =========================================================================
def histogram_peak(binary: np.ndarray, x_lo: int, x_hi: int) -> int:
    hist = np.sum(binary[binary.shape[0] // 2:, x_lo:x_hi], axis=0).astype(float)
    hist = uniform_filter1d(hist, size=40)
    return int(np.argmax(hist)) + x_lo

def sliding_window(
    binary: np.ndarray,
    x_start: int,
    prior_fit: Optional[np.ndarray],
    label: str = ""
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expert sliding window:
      - If prior_fit available, centres each window on the predicted x (warm-start).
      - Adaptive margin: wider at top where uncertainty is greater.
      - Returns (px_x, px_y, debug_vis_slice)
    """
    h, w   = binary.shape
    win_h  = h // SW_NWIN
    nz     = binary.nonzero()
    nzy    = np.array(nz[0])
    nzx    = np.array(nz[1])

    lx_cur = x_start
    inds   : List[np.ndarray] = []
    debug  = np.dstack([binary * 128] * 3).astype(np.uint8)

    for win_i in range(SW_NWIN):
        yl = h - (win_i + 1) * win_h
        yh = h - win_i * win_h
        yc = (yl + yh) // 2

        # Warm-start from prior EKF fit
        if prior_fit is not None:
            pred = int(np.clip(
                prior_fit[0]*yc**2 + prior_fit[1]*yc + prior_fit[2],
                0, w - 1
            ))
            lx_cur = pred

        # Adaptive margin: wider near top of warp (higher win_i)
        margin = int(SW_MARGIN_BASE * (1.0 + 0.6 * win_i / SW_NWIN))
        xl = max(0, lx_cur - margin)
        xr = min(w, lx_cur + margin)

        good = ((nzy >= yl) & (nzy < yh) & (nzx >= xl) & (nzx < xr)).nonzero()[0]
        inds.append(good)

        if len(good) > SW_MINPIX:
            lx_cur = int(np.mean(nzx[good]))

        cv2.rectangle(debug, (xl, yl), (xr, yh), (0, 255, 0), 1)

    all_inds = np.concatenate(inds) if inds else np.array([], dtype=np.int32)
    return nzx[all_inds], nzy[all_inds], debug


# =========================================================================
# ── POLYNOMIAL FIT WITH EKF ───────────────────────────────────────────────
# =========================================================================
def eval_poly(coeffs: np.ndarray, y) -> np.ndarray:
    return coeffs[0] * y**2 + coeffs[1] * y + coeffs[2]

def fit_with_ekf(
    px: np.ndarray, py: np.ndarray,
    lane: LaneLine, bh: int
) -> Optional[np.ndarray]:
    """Fit polynomial, feed through EKF, return smoothed coeffs."""
    if len(px) >= MIN_FIT_PIXELS:
        try:
            raw = np.polyfit(py, px, 2)
            return lane.update(raw) or lane.get_fit()
        except Exception:
            pass
    # No measurement — use EKF prediction
    return lane.predict()


# =========================================================================
# ── CURVATURE ─────────────────────────────────────────────────────────────
# =========================================================================
def curvature(coeffs: np.ndarray, bh: int) -> Tuple[float, str]:
    try:
        ploty  = np.linspace(0, bh - 1, bh)
        real_c = np.polyfit(
            ploty * YM_PER_PIX,
            eval_poly(coeffs, ploty) * XM_PER_PIX, 2
        )
        y_eval = (bh - 1) * YM_PER_PIX
        r = ((1 + (2*real_c[0]*y_eval + real_c[1])**2)**1.5) / max(abs(2*real_c[0]), 1e-6)
        bot = eval_poly(coeffs, bh - 1)
        top = eval_poly(coeffs, 0)
        if   bot - top >  30: direction = "Right Curve"
        elif top - bot >  30: direction = "Left Curve"
        else:                 direction = "Straight"
        return float(r), direction
    except Exception:
        return 9999.0, "Straight"


# =========================================================================
# ── OVERLAY DRAWING ───────────────────────────────────────────────────────
# =========================================================================
def draw_lane_overlay(
    original: np.ndarray,
    Minv: np.ndarray,
    bh: int, bw: int,
    ploty: np.ndarray,
    left_fitx: Optional[np.ndarray],
    right_fitx: Optional[np.ndarray],
    centre_x: float,
):
    wz  = np.zeros((bh, bw, 3), dtype=np.uint8)
    try:
        if left_fitx is not None and right_fitx is not None:
            ptsl = np.array([np.transpose(np.vstack([
                left_fitx.clip(0, bw-1), ploty]))], dtype=np.int32)
            ptsr = np.array([np.flipud(np.transpose(np.vstack([
                right_fitx.clip(0, bw-1), ploty])))], dtype=np.int32)
            cv2.fillPoly(wz, [np.hstack((ptsl, ptsr))], (0, 160, 0))
        elif left_fitx is not None:
            pts = np.array([np.transpose(np.vstack([
                left_fitx.clip(0, bw-1), ploty]))], dtype=np.int32)
            cv2.polylines(wz, pts, False, (0, 200, 255), 4)
        c_arr = np.full_like(ploty, centre_x)
        ptsc  = np.array([np.transpose(np.vstack([
            c_arr.clip(0, bw-1), ploty]))], dtype=np.int32)
        cv2.polylines(wz, ptsc, False, (0, 255, 255), 3)
        unwarped = cv2.warpPerspective(wz, Minv, (original.shape[1], original.shape[0]))
        return cv2.addWeighted(original, 1.0, unwarped, 0.38, 0)
    except Exception:
        return original


# =========================================================================
# ── FULL LANE PIPELINE ────────────────────────────────────────────────────
# =========================================================================
def run_lane_pipeline(
    frame: np.ndarray,
    left_lane: LaneLine,
    right_lane: LaneLine,
) -> Tuple[float, float, str, np.ndarray, str]:
    """
    Returns:
        steer_error   — normalised [-1, 1]
        curve_radius  — metres
        curve_dir     — "Straight" / "Left Curve" / "Right Curve"
        ann_frame     — annotated frame
        mode          — debug string
    """
    h, w  = frame.shape[:2]
    ploty = np.linspace(0, h - 1, h)

    try:
        warped, Minv, roi_pts = apply_roi_warp(frame)
    except Exception:
        return 0.0, 9999.0, "Straight", frame, "WARP_ERR"

    mask_y, mask_w, combined = build_binary_channels(warped)

    # ── Peak initialisation ────────────────────────────────────────────────
    mid = w // 2
    # Yellow centre line: search left 60% (it sits left of car centre)
    y_peak = histogram_peak(mask_y, 0, int(w * 0.60))
    # Right white edge: search right 50% only
    # Left white edge is a fallback only — always prefer right white for centre calc
    w_peak = histogram_peak(mask_w, int(w * 0.50), w)

    # Override peak with EKF prior if available (warm-start)
    lf_prior = left_lane.get_fit()
    rf_prior = right_lane.get_fit()

    if lf_prior is not None:
        y_peak = int(np.clip(eval_poly(lf_prior, h-1), 0, w-1))
    if rf_prior is not None:
        w_peak = int(np.clip(eval_poly(rf_prior, h-1), 0, w-1))

    # ── Sliding windows ───────────────────────────────────────────────────
    lpx, lpy, _ = sliding_window(mask_y, y_peak, lf_prior, "Y")
    # Primary white = RIGHT edge line (must be in right half of warped frame)
    w_peak = max(w_peak, w // 2)
    rpx, rpy, _ = sliding_window(mask_w, w_peak, rf_prior, "W")
    # Secondary white fallback = LEFT edge (only used if right white fails AND yellow fails)
    lw_peak = histogram_peak(mask_w, 0, w // 2)
    lwpx, lwpy, _ = sliding_window(mask_w, lw_peak, None, "LW")

    # Fallback: if white channel fails use combined gradient
    if len(rpx) < MIN_FIT_PIXELS:
        rpx2, rpy2, _ = sliding_window(combined, w_peak, rf_prior, "Wcomb")
        if len(rpx2) > len(rpx):
            rpx, rpy = rpx2, rpy2

    # ── EKF-smoothed polynomial fit ───────────────────────────────────────
    lf = fit_with_ekf(lpx, lpy, left_lane, h)
    rf = fit_with_ekf(rpx, rpy, right_lane, h)

    # ── Lane width sanity gate ────────────────────────────────────────────
    if lf is not None and rf is not None:
        width_px = eval_poly(rf, h-1) - eval_poly(lf, h-1)
        if not (MIN_LANE_WIDTH_PX < width_px < MAX_LANE_WIDTH_PX):
            # Width implausible — drop the less-confident line
            if len(lpx) >= len(rpx):
                right_lane.reset(); rf = None
            else:
                left_lane.reset();  lf = None

    # ── Centre calculation + mode ─────────────────────────────────────────
    # ── Centre priority: BOTH > YELLOW_ONLY > RIGHT_WHITE > LEFT_WHITE_FALLBACK ──
    lw_fit = fit_with_ekf(lwpx, lwpy, LaneLine(), h) if 'lwpx' in dir() else None

    if lf is not None and rf is not None:
        # Best case: yellow centre + right white edge → true lane centre
        centre_x   = (eval_poly(lf, h-1) + eval_poly(rf, h-1)) / 2.0
        lx_arr     = eval_poly(lf, ploty)
        rx_arr     = eval_poly(rf, ploty)
        crad, cdir = curvature(lf, h)
        mode       = "BOTH"

    elif lf is not None:
        # Yellow only → offset right by lane-width estimate
        crad, cdir = curvature(lf, h)
        offset = (YELLOW_OFFSET_CURVE_PX +
                  max(0.0, min(1.0, crad / CURVE_RADIUS_THRESHOLD)) *
                  (YELLOW_OFFSET_PX - YELLOW_OFFSET_CURVE_PX)
                  if crad < CURVE_RADIUS_THRESHOLD
                  else YELLOW_OFFSET_PX)
        centre_x = eval_poly(lf, h-1) + offset
        lx_arr   = eval_poly(lf, ploty)
        rx_arr   = None
        mode     = "YELLOW"

    elif rf is not None:
        # Right white only → offset left by half lane
        crad, cdir = curvature(rf, h)
        centre_x = eval_poly(rf, h-1) - HALF_LANE_PX
        lx_arr   = None
        rx_arr   = eval_poly(rf, ploty)
        mode     = "R_WHITE"

    elif lw_fit is not None:
        # Left white fallback → offset RIGHT by full lane width (dangerous — low confidence)
        crad, cdir = curvature(lw_fit, h)
        centre_x = eval_poly(lw_fit, h-1) + int(HALF_LANE_PX * 2.0)
        lx_arr   = eval_poly(lw_fit, ploty)
        rx_arr   = None
        mode     = "L_WHITE_FALLBACK"

    else:
        crad, cdir = 9999.0, "Straight"
        centre_x   = w / 2.0
        lx_arr     = None
        rx_arr     = None
        mode       = "NO LANE"

    # ── Error (deviation from centre, normalised) ─────────────────────────
    deviation_px = (w / 2.0) - centre_x
    steer_error  = float(np.clip(deviation_px / HALF_LANE_PX, -1.0, 1.0))

    # ── Annotation ───────────────────────────────────────────────────────
    ann = draw_lane_overlay(frame, Minv, h, w, ploty, lx_arr, rx_arr, centre_x)

    # ROI outline
    _, _, roi_pts = get_warp(h, w)
    cv2.polylines(ann, [roi_pts.reshape(1, -1, 2)], True, (0, 220, 80), 2)

    # LDW banner
    if abs(deviation_px) > LDW_THRESHOLD_PX:
        side = "RIGHT" if deviation_px > 0 else "LEFT"
        cv2.rectangle(ann, (10, 85), (480, 140), (0, 0, 100), -1)
        cv2.putText(ann, f"LDW: DRIFT {side}", (18, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2)
    else:
        cv2.rectangle(ann, (10, 85), (340, 140), (0, 60, 0), -1)
        cv2.putText(ann, "LDW: Lane OK", (18, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 0), 2)

    return steer_error, crad, cdir, ann, mode


# =========================================================================
# ── TRAFFIC LIGHT SELECTION ───────────────────────────────────────────────
# =========================================================================
def select_traffic_light(
    detections: List[dict],
    frame_w: int
) -> Optional[str]:
    """
    Centre-biased selection:
      1. Exclude any box whose centre cx >= frame_w * TL_RIGHT_EXCLUDE_FRAC.
      2. Among remaining, pick the one whose cx is closest to frame centre.
    Returns label string or None.
    """
    candidates = [
        d for d in detections
        if d['cx'] < frame_w * TL_RIGHT_EXCLUDE_FRAC
        and d['area'] >= TL_MIN_AREA
        and d['conf'] >= TL_MIN_CONF
    ]
    if not candidates:
        return None
    frame_centre = frame_w / 2.0
    best = min(candidates, key=lambda d: abs(d['cx'] - frame_centre))
    return best['label']


# =========================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────
# =========================================================================
def main():
    global KILL

    # ── Depth sensor ──────────────────────────────────────────────────────
    depth_sensor = DepthSensor()

    # ── YOLO ──────────────────────────────────────────────────────────────
    yolo_model = None
    if YOLO_AVAILABLE:
        try:
            yolo_model = YOLO('best.pt')
            print(f"YOLO loaded. Classes: {yolo_model.names}")
        except Exception as e:
            print(f"YOLO load failed: {e}")

    # ── Lane state ────────────────────────────────────────────────────────
    left_lane  = LaneLine()
    right_lane = LaneLine()

    # ── PID state ─────────────────────────────────────────────────────────
    integral   = 0.0
    prev_err   = 0.0
    prev_steer = 0.0
    t_prev     = time.time()

    # ── YOLO frame counter ────────────────────────────────────────────────
    yolo_frame_ctr = 0

    # ── ML state machine ──────────────────────────────────────────────────
    # Stop sign
    stop_active         = False
    stop_start_time     = 0.0
    stop_cooldown_until = 0.0
    stop_streak         = 0

    # Traffic light (persistent across YOLO frames)
    red_active   = False
    green_active = False

    # ── HUD state (persists between YOLO frames) ──────────────────────────
    ml_status   = "PATH CLEAR"
    ml_color    = (0, 255, 0)
    vision_brake_ml = False   # ML-only brake (separate from ADAS)

    print("\n  Black Hawks Master Agent v5 — PC #2")
    print("  Press Q in OpenCV window to quit.\n")

    with QCar(readMode=0) as car, \
         QCarCameras(enableFront=True, enableBack=True,
                     enableLeft=True, enableRight=True) as cams:

        while not KILL:
            t_now  = time.time()
            dt     = max(t_now - t_prev, 1e-4)
            t_prev = t_now

            cams.readAll()
            car.read()

            csi = cams.csi[FRONT_CSI_IDX]
            if csi is None or csi.imageData is None:
                car.write(0.0, 0.0, np.zeros(8, dtype=int))
                time.sleep(0.02)
                continue

            raw = csi.imageData.copy()
            frame = cv2.resize(raw, (FRAME_W, FRAME_H))
            h_f, w_f = frame.shape[:2]
            yolo_frame_ctr += 1

            # ==============================================================
            # 1. YOLO PERCEPTION  (every YOLO_FRAME_SKIP frames)
            # ==============================================================
            if yolo_model is not None and (yolo_frame_ctr % YOLO_FRAME_SKIP == 0):
                try:
                    results = yolo_model.predict(
                        source=frame,
                        conf=YOLO_CONF_THRESHOLD,
                        verbose=False
                    )

                    # Reset light state each YOLO cycle
                    red_active   = False
                    green_active = False
                    stop_raw     = False

                    tl_detections: List[dict] = []

                    for r in results:
                        for box in r.boxes:
                            lbl   = yolo_model.names[int(box.cls[0])].lower()
                            conf  = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cx    = (x1 + x2) // 2
                            cy    = (y1 + y2) // 2
                            area  = (x2 - x1) * (y2 - y1)

                            # ── Traffic lights ──────────────────────────
                            if any(k in lbl for k in ('red', 'yellow', 'green')):
                                tl_key = (
                                    'red'    if 'red'    in lbl else
                                    'yellow' if 'yellow' in lbl else
                                    'green'
                                )
                                tl_detections.append({
                                    'label': tl_key,
                                    'cx': cx, 'cy': cy,
                                    'area': area, 'conf': conf,
                                    'box': (x1, y1, x2, y2)
                                })

                            # ── Stop sign ────────────────────────────────
                            elif 'stop' in lbl:
                                if conf < STOP_SIGN_MIN_CONF:
                                    continue

                                # Depth gate
                                depth_m = depth_sensor.median_depth_in_box(x1, y1, x2, y2)
                                if depth_m is not None:
                                    close_enough = depth_m < STOP_DEPTH_MAX
                                else:
                                    # Fallback: area-based proximity
                                    close_enough = area > STOP_DEPTH_FALLBACK_AREA

                                if close_enough:
                                    stop_raw = True

                            # ── Yield / roundabout — explicit blacklist ──
                            elif 'yield' in lbl or 'roundabout' in lbl:
                                continue

                    # ── Traffic light: centre-biased selection ───────────
                    chosen_tl = select_traffic_light(tl_detections, w_f)
                    if chosen_tl in ('red', 'yellow'):
                        red_active   = True
                        green_active = False
                    elif chosen_tl == 'green':
                        green_active = True
                        red_active   = False

                    # ── Stop sign streak counter ─────────────────────────
                    if stop_raw and not stop_active and t_now > stop_cooldown_until:
                        stop_streak += 1
                    else:
                        stop_streak = 0

                    # Trigger state machine when streak reached
                    if stop_streak >= STOP_SIGN_STREAK_REQ and not stop_active:
                        stop_active      = True
                        stop_start_time  = t_now
                        stop_streak      = 0
                        print(f"[STATE] CRUISE -> STOPPED  t={t_now:.1f}")

                except Exception as e:
                    print(f"[YOLO error] {e}")

            # ==============================================================
            # 2. ML STATE MACHINE EVALUATION  (every frame)
            # ==============================================================
            vision_brake_ml = False
            ml_status       = "PATH CLEAR"
            ml_color        = (0, 255, 0)

            # Red / yellow light — highest priority
            if red_active:
                vision_brake_ml = True
                ml_status = "RED LIGHT — STOP"
                ml_color  = (0, 0, 255)

            # Green light clears stop sign lock
            if green_active and stop_active:
                stop_active = False
                print("[STATE] Green light — stop sign lock cleared.")

            # Stop sign hold
            if stop_active:
                elapsed = t_now - stop_start_time
                if elapsed < STOP_HOLD_DURATION:
                    vision_brake_ml = True
                    remaining = STOP_HOLD_DURATION - elapsed
                    ml_status = f"STOP SIGN — {remaining:.1f}s"
                    ml_color  = (0, 140, 255)
                else:
                    stop_active         = False
                    stop_cooldown_until = t_now + STOP_COOLDOWN_DURATION
                    print(f"[STATE] STOPPED -> CRUISE  cooldown until {stop_cooldown_until:.1f}")

            # Cooldown HUD indicator
            if not stop_active and t_now < stop_cooldown_until:
                remaining_cd = stop_cooldown_until - t_now
                if ml_status == "PATH CLEAR":
                    ml_status = f"COOLDOWN {remaining_cd:.1f}s"
                    ml_color  = (0, 200, 200)

            # Green light info (no brake override)
            if green_active and not red_active and not stop_active:
                ml_status = "GREEN LIGHT — GO"
                ml_color  = (0, 255, 0)

            # ==============================================================
            # 3. LANE PIPELINE
            # ==============================================================
            steer_error, crad, cdir, ann, lane_mode = run_lane_pipeline(
                frame, left_lane, right_lane
            )

            # ==============================================================
            # 4. PID + ADAPTIVE THROTTLE
            # ==============================================================
            deriv     = (steer_error - prev_err) / dt
            integral += steer_error * dt
            integral  = float(np.clip(integral, -INTEGRAL_CLAMP, INTEGRAL_CLAMP))

            raw_steer  = float(np.clip(
                KP * steer_error + KI * integral + KD * deriv,
                -MAX_STEER, MAX_STEER
            ))
            steer      = STEER_ALPHA * prev_steer + (1.0 - STEER_ALPHA) * raw_steer
            prev_err   = steer_error
            prev_steer = steer

            steer_norm    = abs(steer) / MAX_STEER
            is_sharp      = 0 < crad < CURVE_RADIUS_THRESHOLD * 1.5

            if   lane_mode == "NO LANE":          throttle = THR_BLIND
            elif lane_mode == "L_WHITE_FALLBACK": throttle = THR_BLIND  # very low confidence
            elif is_sharp:                         throttle = THR_SLOW
            elif steer_norm < 0.10:                throttle = THR_FAST
            elif steer_norm < 0.35:                throttle = THR_MID
            else:                                  throttle = THR_SLOW

            if vision_brake_ml:
                throttle = THR_CRAWL
                integral = 0.0   # reset integrator during stops to prevent windup
            if lane_mode == "NO LANE":
                integral = 0.0   # reset integrator when lane lost to prevent windup

            # ==============================================================
            # 5. WRITE TO HARDWARE
            # ==============================================================
            leds    = np.zeros(8, dtype=int)
            leds[4] = int(vision_brake_ml)   # brake LED
            car.write(float(throttle), float(steer), leds)

            # ==============================================================
            # 6. HUD
            # ==============================================================
            # ML banner (top)
            cv2.rectangle(ann, (5, 5), (w_f - 5, 75), (0, 0, 0), -1)
            cv2.putText(ann, f"ML: {ml_status}",
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, ml_color, 2)
            cv2.putText(ann, f"ADAS: {lane_mode}  |  {cdir}  R={crad:.1f}m  Streak:{stop_streak}",
                        (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

            # PID data (bottom)
            cv2.rectangle(ann, (0, h_f - 30), (w_f, h_f), (0, 0, 0), -1)
            cv2.putText(ann,
                        f"Steer:{steer:+.3f}  Thr:{throttle:.4f}  "
                        f"Err:{steer_error:+.3f}  I:{integral:+.3f}",
                        (10, h_f - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 220, 60), 1)

            cv2.imshow("Black Hawks | Master Agent v5", ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Shutdown ──────────────────────────────────────────────────────────
    try:
        car.write(0.0, 0.0, np.zeros(8, dtype=int))
    except Exception:
        pass
    depth_sensor.stop()
    cv2.destroyAllWindows()
    print("Shutdown complete.")


if __name__ == "__main__":
    main()