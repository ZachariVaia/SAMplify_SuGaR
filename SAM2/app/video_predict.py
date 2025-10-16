import os, glob, json, numpy as np, torch, cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor

# ===================== CONFIG via environment variables =====================
DATASET_NAME = os.environ.get("DATASET_NAME", "").strip()

# Default INPUT: /data/in/<DATASET_NAME> if provided, otherwise /data/in
_default_input = f"/data/in/{DATASET_NAME}" if DATASET_NAME else "/data/in"
INPUT = os.environ.get("INPUT", _default_input)
OUT_ROOT = os.environ.get("OUT", "/data/out")
BOX   = os.environ.get("BOX", "")
GUI   = os.environ.get("GUI", "1")
OBJ_ID = int(os.environ.get("OBJ_ID", "1"))
FRAME_IDX = int(os.environ.get("FRAME_IDX", "0"))

# Auto-indexing options
AUTO_INDEX   = os.environ.get("AUTO_INDEX", "1")           # "1" means index frames if needed
INDEX_SUFFIX = os.environ.get("INDEX_SUFFIX", "_indexed")  # suffix for the indexed mirror folder

# Silence mode (default: 1 → fully silent)
QUIET = os.environ.get("QUIET", "1")  # set QUIET=0 to re-enable console output

# Accepted image extensions (add more if needed)
ACCEPT_EXTS = (".jpg", ".jpeg", ".png")


# ===================== SILENCE EVERYTHING =====================
if QUIET == "1":
    import sys, warnings
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore")
    try:
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    except Exception:
        pass  # best effort


# ===================== Helper functions =====================
def _gather_frames(dir_path: str):
    """Return sorted list of absolute paths for accepted image files (case-insensitive)."""
    files = []
    for ext in ACCEPT_EXTS:
        # case-insensitive search by collecting both lower/upper variants
        pattern_lower = os.path.join(dir_path, f"*{ext}")
        pattern_upper = os.path.join(dir_path, f"*{ext.upper()}")
        files.extend(glob.glob(pattern_lower))
        files.extend(glob.glob(pattern_upper))
    return sorted(set(os.path.abspath(p) for p in files))

def list_frames(frames_dir):
    return _gather_frames(frames_dir)

def _is_indexed_name(path: str) -> bool:
    """Check if basename without extension is exactly 6 digits (000000, 000001, ...)."""
    name = os.path.splitext(os.path.basename(path))[0]
    return len(name) == 6 and name.isdigit()

def ensure_indexed(src_dir: str, suffix: str = "_indexed") -> str:
    """
    Create an indexed mirror of frames while preserving original extensions:
      /src_dir/*.{jpg,jpeg,png} → /src_dir_indexed/000000.<ext>, 000001.<ext>, ...
    Tries symlinks; falls back to copy if symlink fails.
    If src_dir already looks indexed (all names 6 digits), just return it.
    """
    import shutil

    if not os.path.isdir(src_dir):
        raise NotADirectoryError(f"ensure_indexed: {src_dir} is not a directory")

    files = list_frames(src_dir)
    if not files:
        raise FileNotFoundError(f"No frames found in {src_dir} (accepted: {', '.join(ACCEPT_EXTS)})")

    # If everything already looks indexed, reuse as-is
    if all(_is_indexed_name(p) for p in files):
        return src_dir

    parent = os.path.dirname(os.path.abspath(src_dir.rstrip("/")))
    base   = os.path.basename(src_dir.rstrip("/"))
    dst_dir = os.path.join(parent, base + suffix)
    os.makedirs(dst_dir, exist_ok=True)

    # Recreate mirror if counts mismatch
    existing = _gather_frames(dst_dir)
    if len(existing) == len(files) and all(
        os.path.exists(os.path.join(dst_dir, f"{i:06d}{os.path.splitext(files[i])[1].lower()}"))
        for i in range(len(files))
    ):
        return dst_dir

    # Build fresh mirror, preserving each file's extension
    for i, f in enumerate(files):
        ext = os.path.splitext(f)[1].lower()
        if ext not in ACCEPT_EXTS:
            # Skip unknown ext just in case
            continue
        new_path = os.path.join(dst_dir, f"{i:06d}{ext}")
        if os.path.exists(new_path):
            continue
        try:
            os.symlink(os.path.abspath(f), new_path)
        except OSError:
            shutil.copy2(f, new_path)
    return dst_dir

def load_frame(frames_dir, frame_idx):
    frames = list_frames(frames_dir)
    if not frames:
        raise FileNotFoundError(f"No frames found in {frames_dir} (accepted: {', '.join(ACCEPT_EXTS)})")
    if frame_idx < 0 or frame_idx >= len(frames):
        raise IndexError(f"frame_idx {frame_idx} out of range [0,{len(frames)-1}]")
    img_bgr = cv2.imread(frames[frame_idx], cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(frames[frame_idx])
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, frames[frame_idx]

def save_prompts_json(path, frame_idx, obj_id, points, labels, image_w, image_h, source_name):
    """Save user prompts (points and labels) in JSON format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "frame_idx": int(frame_idx),
        "obj_id": int(obj_id),
        "points": [[float(x), float(y)] for x, y in points],
        "labels": [int(l) for l in labels],
        "image_w": int(image_w),
        "image_h": int(image_h),
        "source": source_name,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def try_matplotlib_picker(frames_dir, frame_idx, default_box=None):
    """
    Optional interactive GUI for selecting positive/negative points.
    Left-click = positive (green), right-click = negative (red).
    """
    try:
        import matplotlib
        if "agg" in matplotlib.get_backend().lower():
            for bk in ("TkAgg", "Qt5Agg", "GTK3Agg"):
                try:
                    matplotlib.use(bk, force=True)
                    break
                except Exception:
                    pass
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return None, None

    img_rgb, src_path = load_frame(frames_dir, frame_idx)
    h, w = img_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_title("Click points: left=POS, right=NEG | u=undo, Enter=save, q=quit")
    ax.axis("on")

    points, labels = [], []
    pos_scatter = ax.scatter([], [], marker="+", s=120, linewidths=2, c="g")
    neg_scatter = ax.scatter([], [], marker="x", s=120, linewidths=2, c="r")

    if default_box:
        x1, y1, x2, y2 = default_box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        points.append([cx, cy]); labels.append(1)

    def redraw():
        P = np.array(points, dtype=float) if points else np.empty((0, 2))
        L = np.array(labels, dtype=int) if labels else np.empty((0,), dtype=int)
        pos_scatter.set_offsets(P[L == 1] if len(P) and (L == 1).any() else np.empty((0, 2)))
        neg_scatter.set_offsets(P[L == 0] if len(P) and (L == 0).any() else np.empty((0, 2)))
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes != ax or ev.xdata is None or ev.ydata is None:
            return
        x, y = float(ev.xdata), float(ev.ydata)
        if not (0 <= x < w and 0 <= y < h):
            return
        if ev.button == 1:
            points.append([x, y]); labels.append(1)
        elif ev.button == 3:
            points.append([x, y]); labels.append(0)
        redraw()

    state = {"save": False, "quit": False}
    def on_key(ev):
        if ev.key == "u":
            if points: points.pop(); labels.pop(); redraw()
        elif ev.key == "enter":
            state["save"] = True; plt.close(fig)
        elif ev.key in ("q", "escape"):
            state["quit"] = True; plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw(); plt.show()

    if state["quit"]:
        return None, None
    if not points:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.int32)
    return np.asarray(points, np.float32), np.asarray(labels, np.int32)

def to_u8_mask(x):
    """Convert a mask (tensor or array) to 0–255 uint8 image."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().squeeze()
        if x.is_floating_point():
            x = x > 0.5
        x = x.to(torch.uint8).numpy()
    else:
        x = np.asarray(x).squeeze()
        if x.dtype.kind == "f":
            x = x > 0.5
        x = x.astype(np.uint8)
    return x * 255

def save_color_cutout(orig_img_path: str, mask_u8: np.ndarray, out_path: str):
    """Apply binary mask to original image and save the color cutout."""
    img = cv2.imread(orig_img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(orig_img_path)

    # Ensure mask is 0/255 and same size as image
    m = mask_u8
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    # Keep original pixels where mask == 255, otherwise black
    cutout = cv2.bitwise_and(img, img, mask=m)
    cv2.imwrite(out_path, cutout)


# ===================== Main pipeline =====================
# 1) Index frames if needed
if os.path.isdir(INPUT) and AUTO_INDEX == "1":
    INPUT = ensure_indexed(INPUT, INDEX_SUFFIX)

# 2) Define output directories
inp_base = os.path.basename(INPUT.rstrip("/"))
out_name = inp_base if os.path.isdir(INPUT) else os.path.splitext(inp_base)[0]
OUT_DIR = os.path.join(OUT_ROOT, out_name)
os.makedirs(OUT_DIR, exist_ok=True)
PROMPTS_JSON = os.path.join(OUT_DIR, "prompts.json")
OUT_MASKED_DIR = os.path.join(OUT_ROOT, f"{out_name}_masked")
os.makedirs(OUT_MASKED_DIR, exist_ok=True)

# 3) Validate frames and map original names + extensions
if os.path.isdir(INPUT):
    frame_paths = list_frames(INPUT)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {INPUT} (accepted: {', '.join(ACCEPT_EXTS)})")
    idx_to_orig_name = [os.path.basename(os.path.realpath(p)) for p in frame_paths]
    idx_to_orig_ext  = [os.path.splitext(n)[1] for n in idx_to_orig_name]  # includes dot, preserves case
else:
    frame_paths, idx_to_orig_name, idx_to_orig_ext = [], [], []

# 4) Parse optional BOX
box = None
if BOX:
    try:
        x1, y1, x2, y2 = map(int, BOX.split(","))
        box = [x1, y1, x2, y2]
    except Exception:
        box = None

# 5) Load prompts or open GUI picker
points = labels = None
if os.path.isfile(PROMPTS_JSON):
    with open(PROMPTS_JSON, "r") as f:
        J = json.load(f)
    points = np.array(J.get("points", []), dtype=np.float32)
    labels = np.array(J.get("labels", []), dtype=np.int32)
    FRAME_IDX = int(J.get("frame_idx", FRAME_IDX))
    OBJ_ID    = int(J.get("obj_id", OBJ_ID))
elif GUI == "1" and os.path.isdir(INPUT):
    pts, labs = try_matplotlib_picker(INPUT, FRAME_IDX, default_box=box)
    if pts is not None and labs is not None:
        img0, src_path = load_frame(INPUT, FRAME_IDX)
        h, w = img0.shape[:2]
        save_prompts_json(PROMPTS_JSON, FRAME_IDX, OBJ_ID, pts, labs, w, h, os.path.basename(src_path))
        points, labels = pts, labs

# 6) Default fallback points
if points is None or labels is None or len(points) == 0:
    if box is not None:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        points = np.asarray([[cx, cy]], np.float32)
        labels = np.asarray([1], np.int32)
    else:
        if os.path.isdir(INPUT) and frame_paths:
            img0_bgr = cv2.imread(frame_paths[FRAME_IDX], cv2.IMREAD_COLOR)
            h, w = img0_bgr.shape[:2]
        else:
            h, w = 1080, 1920
        points = np.asarray([[w // 2, h // 2]], np.float32)
        labels = np.asarray([1], np.int32)

# 7) Run SAM2 segmentation
pred = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to("cuda")
use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
dtype = torch.bfloat16 if use_bf16 else torch.float16

with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    state = pred.init_state(INPUT)

    # First frame with prompts
    frame_idx, obj_ids, masks = pred.add_new_points_or_box(
        state, FRAME_IDX, OBJ_ID, points=points, labels=labels
    )

    # Save mask & cutout for the prompted frame, preserving original extension
    base_name = idx_to_orig_name[frame_idx] if idx_to_orig_name else f"{frame_idx:06d}"
    stem, orig_ext = os.path.splitext(base_name)
    single_obj = len(obj_ids) == 1
    orig_img_path = os.path.realpath(frame_paths[frame_idx]) if frame_paths else None
    ext_for_save = orig_ext if orig_ext else ".jpg"  # fallback

    for k, oid in enumerate(obj_ids):
        suffix = "" if single_obj else f"_obj{oid}"
        out_name = f"{stem}{suffix}{ext_for_save}"

        # binary mask saved with original extension
        mask_path = os.path.join(OUT_DIR, out_name)
        mask_u8   = to_u8_mask(masks[k])
        cv2.imwrite(mask_path, mask_u8)

        # color cutout saved with original extension
        if orig_img_path:
            cutout_path = os.path.join(OUT_MASKED_DIR, out_name)
            save_color_cutout(orig_img_path, mask_u8, cutout_path)

    # Propagate to remaining frames
    for frame_idx, obj_ids, masks in pred.propagate_in_video(state):
        base_name = idx_to_orig_name[frame_idx] if idx_to_orig_name else f"{frame_idx:06d}"
        stem, orig_ext = os.path.splitext(base_name)
        single_obj = len(obj_ids) == 1
        orig_img_path = os.path.realpath(frame_paths[frame_idx]) if frame_paths else None
        ext_for_save = orig_ext if orig_ext else ".jpg"

        for k, oid in enumerate(obj_ids):
            suffix = "" if single_obj else f"_obj{oid}"
            out_name = f"{stem}{suffix}{ext_for_save}"

            mask_path = os.path.join(OUT_DIR, out_name)
            mask_u8   = to_u8_mask(masks[k])
            cv2.imwrite(mask_path, mask_u8)

            if orig_img_path:
                cutout_path = os.path.join(OUT_MASKED_DIR, out_name)
                save_color_cutout(orig_img_path, mask_u8, cutout_path)
