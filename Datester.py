import time
import cv2
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
VIDEO_SOURCE = r""
LOOP_VIDEO = False

MODEL_PATH = r""
CONF = 0.45
IMGSZ = 640

FRAME_W = 640
FRAME_H = 480

TRACKER_TYPE = "CSRT"          # CSRT, KCF, MOSSE
TRACKER_MAX_MISSES = 8
DETECT_EVERY_N = 2             # detect often while tracking
LOST_HOLD_FRAMES = 30          # how long to try recovering same target after loss

CENTER_BOX_W = 80
CENTER_BOX_H = 60

KP_X, KD_X = 0.030, 0.012
KP_Y, KD_Y = 0.030, 0.012

MAX_CMD_X = 1.0
MAX_CMD_Y = 1.0

REACQUIRE_MAX_DIST = 150       # px
REACQUIRE_MIN_IOU = 0.02
STRONG_MATCH_IOU = 0.10
STRONG_MATCH_DIST = 90         # px
PRINT_EVERY_N = 5

TARGET_CLASS = None            # e.g. "drone" if your model has that class
USE_CARTESIAN_Y = False        # False => top is negative y
SHOW_ALL_DETECTIONS = True


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PD:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = None

    def reset(self):
        self.prev_error = None

    def update(self, error, dt):
        dt = max(dt, 1e-3)
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.kd * derivative


def make_tracker(kind="CSRT"):
    kind = kind.upper()
    legacy = getattr(cv2, "legacy", None)

    if kind == "CSRT":
        ctor = getattr(cv2, "TrackerCSRT_create", None) or getattr(legacy, "TrackerCSRT_create", None)
    elif kind == "KCF":
        ctor = getattr(cv2, "TrackerKCF_create", None) or getattr(legacy, "TrackerKCF_create", None)
    elif kind == "MOSSE":
        ctor = getattr(legacy, "TrackerMOSSE_create", None)
    else:
        raise ValueError(f"Unsupported tracker type: {kind}")

    if ctor is None:
        raise RuntimeError(
            f"Tracker '{kind}' not available in your OpenCV build. "
            f"Install opencv-contrib-python."
        )

    return ctor()


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dist2(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy


def center_of_bbox(b):
    x, y, w, h = b
    return int(x + w / 2), int(y + h / 2)


def detect(model, frame):
    result = model.predict(frame, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    out = []

    if result.boxes is None or len(result.boxes) == 0:
        return out

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, clss):
        label = model.names[int(cid)]
        if TARGET_CLASS and label != TARGET_CLASS:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        out.append({
            "bbox": (x1, y1, w, h),
            "cx": cx,
            "cy": cy,
            "conf": float(conf),
            "label": label,
            "class_id": int(cid)
        })

    return out


def pick_center_target(dets, frame_cx, frame_cy):
    if not dets:
        return None
    return min(dets, key=lambda d: (d["cx"] - frame_cx) ** 2 + (d["cy"] - frame_cy) ** 2)


def pick_best_detection_for_reference(detections, ref_bbox, ref_class_id=None):
    if not detections or ref_bbox is None:
        return None

    rcx, rcy = center_of_bbox(ref_bbox)

    same_class = []
    if ref_class_id is not None:
        same_class = [d for d in detections if d["class_id"] == ref_class_id]

    candidates = same_class if same_class else detections

    best = None
    best_score = -1e9

    for d in candidates:
        this_iou = iou_xywh(d["bbox"], ref_bbox)
        this_d2 = dist2((d["cx"], d["cy"]), (rcx, rcy))
        max_d2 = REACQUIRE_MAX_DIST * REACQUIRE_MAX_DIST

        if this_iou < REACQUIRE_MIN_IOU and this_d2 > max_d2:
            continue

        dist_score = 1.0 / (1.0 + this_d2 / float(max_d2))
        score = 2.8 * this_iou + 1.0 * dist_score + 0.25 * d["conf"]

        if score > best_score:
            best_score = score
            best = d

    return best


def init_tracker_on_detection(frame, det):
    tracker = make_tracker(TRACKER_TYPE)
    bbox = tuple(map(int, det["bbox"]))

    try:
        ok_init = tracker.init(frame, bbox)
    except cv2.error as e:
        print(f"[tracker] init failed: {e}")
        return None, None

    if ok_init is False:
        return None, None

    return tracker, bbox


def in_deadband(cx, cy, fx, fy, bw, bh):
    return abs(cx - fx) <= bw // 2 and abs(cy - fy) <= bh // 2


def quadrant(cx, cy, fx, fy, bw, bh):
    if in_deadband(cx, cy, fx, fy, bw, bh):
        return "CENTER"
    if cx < fx and cy < fy:
        return "TOP-LEFT"
    if cx >= fx and cy < fy:
        return "TOP-RIGHT"
    if cx < fx and cy >= fy:
        return "BOTTOM-LEFT"
    return "BOTTOM-RIGHT"


def compute_errors(cx, cy, fx, fy):
    ex = (cx - fx) / (FRAME_W / 2.0)
    ey = (fy - cy) / (FRAME_H / 2.0) if USE_CARTESIAN_Y else (cy - fy) / (FRAME_H / 2.0)
    return ex, ey


def draw(frame, tracker_bbox, target_label, all_dets, fx, fy, cmd_x, cmd_y, q, locked, trusted_bbox=None):
    h, w = frame.shape[:2]

    if SHOW_ALL_DETECTIONS:
        for d in all_dets:
            x, y, bw, bh = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (120, 120, 120), 1)

    cv2.line(frame, (fx, 0), (fx, h), (70, 70, 70), 1)
    cv2.line(frame, (0, fy), (w, fy), (70, 70, 70), 1)

    dbx1 = fx - CENTER_BOX_W // 2
    dby1 = fy - CENTER_BOX_H // 2
    dbx2 = fx + CENTER_BOX_W // 2
    dby2 = fy + CENTER_BOX_H // 2
    cv2.rectangle(frame, (dbx1, dby1), (dbx2, dby2), (0, 255, 255), 1)
    cv2.drawMarker(frame, (fx, fy), (255, 255, 255), cv2.MARKER_CROSS, 18, 1)

    if trusted_bbox is not None:
        x, y, bw, bh = trusted_bbox
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 1)

    if tracker_bbox is not None:
        x, y, bw, bh = tracker_bbox
        cx, cy = center_of_bbox(tracker_bbox)
        color = (0, 255, 0) if locked else (0, 180, 255)

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
        cv2.line(frame, (fx, fy), (cx, cy), (255, 0, 255), 2)

        cv2.putText(frame, f"{target_label} | {q}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        cv2.putText(frame, f"cmd=({cmd_x:+.3f}, {cmd_y:+.3f})", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "NO TRACKER", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    model = YOLO(MODEL_PATH)

    tracker = None
    tracker_bbox = None
    tracker_label = ""
    tracker_class_id = None

    trusted_bbox = None
    trusted_label = ""
    trusted_class_id = None

    misses = 0
    lost_hold = 0
    frame_idx = 0

    pd_x = PD(KP_X, KD_X)
    pd_y = PD(KP_Y, KD_Y)

    prev_t = time.time()

    while True:
        ok, frame = cap.read()

        if not ok:
            if LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        now = time.time()
        dt = now - prev_t
        prev_t = now
        frame_idx += 1

        fx, fy = FRAME_W // 2, FRAME_H // 2
        detections = []

        # Always detect frequently enough that the detector can pull the tracker back.
        run_detect = (tracker is None) or (frame_idx % DETECT_EVERY_N == 0) or (misses > 0)
        if run_detect:
            detections = detect(model, frame)

        if tracker is None:
            target = None

            if lost_hold > 0 and trusted_bbox is not None and detections:
                target = pick_best_detection_for_reference(detections, trusted_bbox, trusted_class_id)
                lost_hold -= 1

                if target is not None:
                    print(f"[recover] found trusted target again: {target['label']} bbox={target['bbox']}")

            if target is None and detections:
                target = pick_center_target(detections, fx, fy)
                if target is not None:
                    print(f"[detect] acquired new target: {target['label']} bbox={target['bbox']}")

            if target is not None:
                new_tracker, new_bbox = init_tracker_on_detection(frame, target)
                if new_tracker is not None:
                    tracker = new_tracker
                    tracker_bbox = new_bbox
                    tracker_label = target["label"]
                    tracker_class_id = target["class_id"]

                    trusted_bbox = new_bbox
                    trusted_label = tracker_label
                    trusted_class_id = tracker_class_id
                    misses = 0
                else:
                    tracker = None
                    tracker_bbox = None

        else:
            ok_track, bbox = tracker.update(frame)

            if ok_track:
                tracker_bbox = tuple(map(int, bbox))
                misses = 0
            else:
                misses += 1

            # If detector sees the target again, let detector override tracker drift.
            if detections:
                ref_bbox = trusted_bbox if trusted_bbox is not None else tracker_bbox
                ref_class_id = trusted_class_id if trusted_class_id is not None else tracker_class_id

                best = pick_best_detection_for_reference(detections, ref_bbox, ref_class_id)

                if best is not None:
                    best_bbox = tuple(map(int, best["bbox"]))

                    if tracker_bbox is not None:
                        agree_iou = iou_xywh(best_bbox, tracker_bbox)
                        agree_d2 = dist2(center_of_bbox(best_bbox), center_of_bbox(tracker_bbox))
                    else:
                        agree_iou = 0.0
                        agree_d2 = 10**9

                    # detector confirmation updates trusted state
                    trusted_bbox = best_bbox
                    trusted_label = best["label"]
                    trusted_class_id = best["class_id"]

                    # strong disagreement => snap tracker back to detector
                    if (not ok_track) or (agree_iou < STRONG_MATCH_IOU) or (agree_d2 > STRONG_MATCH_DIST * STRONG_MATCH_DIST):
                        new_tracker, new_bbox = init_tracker_on_detection(frame, best)
                        if new_tracker is not None:
                            tracker = new_tracker
                            tracker_bbox = new_bbox
                            tracker_label = best["label"]
                            tracker_class_id = best["class_id"]
                            misses = 0
                            print(f"[snap] tracker -> detector {tracker_label} bbox={tracker_bbox}")
                    else:
                        tracker_label = best["label"]
                        tracker_class_id = best["class_id"]

            if misses >= TRACKER_MAX_MISSES:
                print("[lost] tracker dropped, holding trusted target for recovery")
                tracker = None
                tracker_bbox = None
                tracker_label = trusted_label
                tracker_class_id = trusted_class_id
                lost_hold = LOST_HOLD_FRAMES
                misses = 0
                pd_x.reset()
                pd_y.reset()

        cmd_x, cmd_y = 0.0, 0.0
        q = "NONE"
        locked = False

        if tracker_bbox is not None:
            cx, cy = center_of_bbox(tracker_bbox)
            ex, ey = compute_errors(cx, cy, fx, fy)

            locked = in_deadband(cx, cy, fx, fy, CENTER_BOX_W, CENTER_BOX_H)
            q = quadrant(cx, cy, fx, fy, CENTER_BOX_W, CENTER_BOX_H)

            if locked:
                pd_x.reset()
                pd_y.reset()
            else:
                cmd_x = clamp(pd_x.update(ex, dt), -MAX_CMD_X, MAX_CMD_X)
                cmd_y = clamp(pd_y.update(ey, dt), -MAX_CMD_Y, MAX_CMD_Y)

            if frame_idx % PRINT_EVERY_N == 0:
                print(
                    f"cmd_x={cmd_x:+.3f} cmd_y={cmd_y:+.3f} "
                    f"err_x={ex:+.3f} err_y={ey:+.3f} "
                    f"quadrant={q} locked={locked}"
                )

            # send_servo_commands(cmd_x, cmd_y)

        draw(
            frame,
            tracker_bbox,
            tracker_label,
            detections,
            fx,
            fy,
            cmd_x,
            cmd_y,
            q,
            locked,
            trusted_bbox=trusted_bbox
        )

        cv2.imshow("Single Instance Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()