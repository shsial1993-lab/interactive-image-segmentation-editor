"""BSAI teaching project: promptable segmentation, HCI, and compositing.
Run: python app.py. Model weights are downloaded on first segmentation.
"""
import os
import json
from metrics import load_reference, evaluate, error_map
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageColor, ImageOps

MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-tiny")
MAX_SIDE = 1600
_predictor = None
_lock = threading.Lock()
# The predictor caches an image internally: lock set_image + predict together.
# Images, prompts, and masks belong to gr.State, separately for each browser session.
_exports = tempfile.TemporaryDirectory(prefix="segmentation_editor_")


def fresh(image=None):
    return {
        "image": image,
        "points": [],
        "labels": [],
        "prompt_groups": [],
        "mask": None,
        "raw_mask": None,
        "model_score": None,
        "candidate": None,
        "prediction_seconds": None,
    }


def preview(state):
    if state["image"] is None:
        return None
    canvas = state["image"].copy()
    mask = state["mask"]
    if mask is not None:
        # CG: alpha blending and contour rendering, in RGB channel order.
        canvas[mask] = (0.55 * canvas[mask] + 0.45 * np.array([20, 210, 130])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (255, 230, 30), 2)
    # A circle prompt represents several SAM points but is drawn as one
    # visible circle so the user can understand the interaction footprint.
    prompt_groups = state.get("prompt_groups", [])
    if prompt_groups:
        for group in prompt_groups:
            points = group.get("points", [])
            if not points:
                continue
            label = int(group.get("label", 1))
            color = (20, 230, 80) if label else (255, 65, 65)
            center = tuple(map(int, group.get("center", points[0])))
            radius = int(group.get("radius", 0))
            if radius > 0:
                cv2.circle(canvas, center, radius, color, 2)
                cv2.circle(canvas, center, 4, (255, 255, 255), -1)
                cv2.circle(canvas, center, 3, color, -1)
            else:
                x, y = map(int, points[0])
                cv2.circle(canvas, (x, y), 8, (255, 255, 255), -1)
                cv2.circle(canvas, (x, y), 6, color, -1)
                cv2.putText(canvas, "+" if label else "-", (x - 5, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return canvas

    # Backward-compatible rendering for states created by older app versions.
    for (x, y), label in zip(state["points"], state["labels"]):
        color = (20, 230, 80) if label else (255, 65, 65)
        cv2.circle(canvas, (x, y), 8, (255, 255, 255), -1)
        cv2.circle(canvas, (x, y), 6, color, -1)
        cv2.putText(canvas, "+" if label else "-", (x - 5, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return canvas


def response(state, message):
    # Invalidate previous downloads whenever the image, prompts, or mask changes.
    mask = None if state["mask"] is None else state["mask"].astype(np.uint8) * 255
    # The final five outputs are the measured-evaluation area. They must be
    # cleared whenever a prediction changes, otherwise an old mIoU could be
    # mistaken for the score of the new mask.
    return state, preview(state), mask, None, None, message, None, "", None, None, None


def upload_image(image):
    if image is None:
        return response(fresh(), "Upload an image to begin.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    original_size = image.size
    image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    state = fresh(np.array(image))
    resize_note = f" Resized from {original_size} to {image.size}; exports use this size." if image.size != original_size else ""
    return response(state, "Choose Include and click inside your object." + resize_note)


def _reset_prediction(state):
    state["mask"] = None
    state["raw_mask"] = None
    state["model_score"] = None
    state["candidate"] = None
    state["prediction_seconds"] = None


def _circle_prompt_points(x, y, radius, width, height):
    """Create nine well-spaced same-label prompts inside a visible circle."""
    radius = max(1, int(radius))
    sample_radius = max(1, int(round(radius * 0.68)))
    points = [(int(x), int(y))]
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        px = int(round(x + sample_radius * np.cos(angle)))
        py = int(round(y + sample_radius * np.sin(angle)))
        if 0 <= px < width and 0 <= py < height and (px, py) not in points:
            points.append((px, py))
    return [list(point) for point in points]


def _add_prompt(state, mode, prompt_shape, prompt_radius, evt: gr.SelectData):
    if state["image"] is None:
        raise gr.Error("Upload an image first.")
    x, y = map(int, evt.index)
    h, w = state["image"].shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise gr.Error("Click inside the image.")
    label = 1 if mode == "Include (+)" else 0
    if prompt_shape == "Circle prompt":
        radius = min(max(1, int(prompt_radius)), max(1, min(w, h) // 2))
        prompt_points = _circle_prompt_points(x, y, radius, w, h)
        shape = "Circle"
        description = (
            f"Circle {mode} prompt added at ({x}, {y}), radius {radius}px, "
            f"using {len(prompt_points)} SAM points."
        )
    else:
        radius = 0
        prompt_points = [[x, y]]
        shape = "Single point"
        description = f"{mode} point added at ({x}, {y})."

    state["points"] = state["points"] + prompt_points
    state["labels"] = state["labels"] + [label] * len(prompt_points)
    state.setdefault("prompt_groups", [])
    state["prompt_groups"] = state["prompt_groups"] + [{
        "shape": shape,
        "center": [x, y],
        "radius": radius,
        "points": prompt_points,
        "label": label,
    }]
    _reset_prediction(state)
    return response(state, description + " Click Segment / Refine to update the mask.")


def add_point(state, mode, evt: gr.SelectData):
    """Backward-compatible single-point prompt callback used by tests/examples."""
    return _add_prompt(state, mode, "Single point", 0, evt)


def add_prompt(state, mode, prompt_shape, prompt_radius, evt: gr.SelectData):
    """Canvas callback for either single-point or circle prompts."""
    return _add_prompt(state, mode, prompt_shape, prompt_radius, evt)


def undo_point(state):
    groups = state.get("prompt_groups", [])
    if groups:
        group = groups[-1]
        count = len(group.get("points", []))
        state["points"] = state["points"][:-count] if count else state["points"]
        state["labels"] = state["labels"][:-count] if count else state["labels"]
        state["prompt_groups"] = groups[:-1]
        removed = "last circle" if group.get("radius", 0) else "last point"
    else:
        state["points"] = state["points"][:-1]
        state["labels"] = state["labels"][:-1]
        removed = "last point"
    _reset_prediction(state)
    return response(state, f"Removed {removed}. Run segmentation again.")


def reject(state):
    return response(fresh(state["image"]), "Selection rejected. Original image preserved; select again.")


def predict_mask(image, points, labels, candidate="Best model score"):
    global _predictor
    # Lazy imports keep UI startup independent of the model download.
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    with _lock, torch.inference_mode():
        if _predictor is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=device)
        _predictor.set_image(image)
        masks, scores, _ = _predictor.predict(
            point_coords=np.asarray(points, dtype=np.float32),
            point_labels=np.asarray(labels, dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores)) if candidate == "Best model score" else int(candidate[-1]) - 1
        return masks[best].astype(bool), float(scores[best])


def segment(state, candidate="Best model score"):
    if state["image"] is None or 1 not in state["labels"]:
        raise gr.Error("Upload an image and add at least one Include (+) click.")
    start = time.perf_counter()
    try:
        mask, score = predict_mask(state["image"], state["points"], state["labels"], candidate)
    except Exception as exc:
        raise gr.Error(f"SAM 2 could not run: {exc}. Check installation and first-run internet access.") from exc
    state["mask"] = mask
    state["raw_mask"] = mask.copy()
    elapsed = time.perf_counter() - start
    state["model_score"] = score
    state["candidate"] = candidate
    state["prediction_seconds"] = elapsed
    return response(state, f"Mask ready in {elapsed:.2f}s. Model quality estimate: {score:.3f} "
                    "(not measured accuracy). Add Include/Exclude clicks and refine, or export.")


def compose(image, mask, style, color, background):
    """CG: binary alpha mask, foreground extraction, background compositing."""
    alpha = mask.astype(np.uint8) * 255
    if style == "Transparent":
        return Image.fromarray(np.dstack((image, alpha)))
    if style == "Uploaded background":
        if background is None:
            raise ValueError("Upload a replacement background first.")
        bg = ImageOps.exif_transpose(background).convert("RGB")
        bg = np.array(ImageOps.fit(bg, (image.shape[1], image.shape[0]),
                                   method=Image.Resampling.LANCZOS))
    else:
        rgb = ImageColor.getrgb(color)
        bg = np.empty_like(image)
        bg[:] = rgb[:3]
    # C = alpha * foreground + (1 - alpha) * background.
    a = mask[..., None].astype(np.float32)
    output = a * image.astype(np.float32) + (1 - a) * bg.astype(np.float32)
    return Image.fromarray(output.astype(np.uint8))


def export(state, style, color, background):
    if state["mask"] is None:
        raise gr.Error("Run segmentation before exporting.")
    try:
        output = compose(state["image"], state["mask"], style, color, background)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    folder = Path(tempfile.mkdtemp(dir=_exports.name))
    result_path, mask_path = folder / "result.png", folder / "mask.png"
    output.save(result_path)
    Image.fromarray(state["mask"].astype(np.uint8) * 255).save(mask_path)
    return output, [str(result_path), str(mask_path)], "Export ready: result.png and binary mask.png."



def clean_mask(state, min_area, closing, largest, invert):
    if state.get("raw_mask") is None:
        raise gr.Error("Run segmentation first.")
    mask = state["raw_mask"].astype(np.uint8).copy()
    if invert:
        mask = 1 - mask
    if closing:
        k = int(closing) * 2 + 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    ids = np.flatnonzero(areas >= int(min_area)) + 1
    if largest and len(ids):
        ids = np.array([ids[np.argmax(areas[ids - 1])]])
    state["mask"] = np.isin(components, ids)
    return response(state, "Cleanup applied from the raw SAM mask. Zero controls restore it. Re-evaluate metrics.")


def _percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def _metric_bar(label, value, color):
    shown = _percent(value)
    width = 0.0 if value is None else max(0.0, min(1.0, float(value))) * 100.0
    return (
        "<div style='margin:8px 0'>"
        f"<div style='display:flex;justify-content:space-between;gap:12px;"
        f"font-size:0.9rem'><span>{label}</span><strong>{shown}</strong></div>"
        "<div style='height:8px;background:#e5e7eb;border-radius:99px;overflow:hidden'>"
        f"<div style='height:100%;width:{width:.2f}%;background:{color};border-radius:99px'></div>"
        "</div></div>"
    )


def metric_dashboard(values):
    """Return a compact visual scorecard for the measured binary metrics."""
    miou = values["mIoU (foreground + background)"]
    dice = values["Foreground Dice"]
    precision = values["Foreground precision"]
    recall = values["Foreground recall"]
    accuracy = values["Pixel accuracy"]
    return (
        "<div style='border:1px solid #d1d5db;border-radius:12px;padding:14px;"
        "background:linear-gradient(135deg,#f8fafc,#eefbf5)'>"
        "<div style='display:flex;align-items:baseline;justify-content:space-between;"
        "gap:12px;flex-wrap:wrap'>"
        "<span style='font-size:0.9rem;color:#475569'>Measured binary mIoU</span>"
        f"<strong style='font-size:2rem;color:#047857'>{_percent(miou)}</strong>"
        "</div>"
        "<div style='font-size:0.8rem;color:#64748b;margin:2px 0 12px'>"
        "Mean of foreground and background IoU; zero-union classes are excluded."
        "</div>"
        + _metric_bar("Foreground IoU", values["Foreground IoU"], "#10b981")
        + _metric_bar("Background IoU", values["Background IoU"], "#3b82f6")
        + "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:8px;margin-top:12px'>"
        + "".join(
            f"<div style='background:white;border-radius:8px;padding:8px 10px'>"
            f"<div style='font-size:0.75rem;color:#64748b'>{label}</div>"
            f"<strong>{_percent(value)}</strong></div>"
            for label, value in (
                ("Dice", dice),
                ("Precision", precision),
                ("Recall", recall),
                ("Pixel accuracy", accuracy),
            )
        )
        + "</div></div>"
    )


def evaluate_mask(state, reference):
    if state["mask"] is None or reference is None:
        raise gr.Error("Segment an image and upload its independently annotated reference mask first.")
    try:
        truth = load_reference(reference, state["mask"].shape)
        values = evaluate(state["mask"], truth)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    rows = [[key, "N/A" if value is None else
             (str(value) if key in ("TP", "FP", "FN", "TN") else f"{value:.4f} ({value * 100:.2f}%)")]
            for key, value in values.items()]
    folder = Path(tempfile.mkdtemp(dir=_exports.name))
    report = folder / "metrics.json"
    report.write_text(json.dumps({"metrics": values, "definition":
        "Per-image binary mIoU averages foreground/background IoU; zero-union classes excluded. Undefined ratios are null.",
        "working_shape": list(truth.shape),
        "prompt_count": len(state["points"]),
        "prompt_group_count": len(state.get("prompt_groups", [])),
        "sam_model_score_estimate": state.get("model_score"),
        "selected_candidate": state.get("candidate"),
        "prediction_seconds": state.get("prediction_seconds"),
        "note": "The SAM model score is an estimate; mIoU is measured against the uploaded independent reference mask.",
    }, indent=2), encoding="utf-8")
    miou_percent = None if values["mIoU (foreground + background)"] is None else values["mIoU (foreground + background)"] * 100
    status = f"Evaluation complete. Measured mIoU: {miou_percent:.2f}%."
    return status, miou_percent, metric_dashboard(values), rows, error_map(state["mask"], truth), str(report)


def build_app():
    with gr.Blocks(title="Interactive Image Segmentation Editor", delete_cache=(3600, 3600)) as demo:
        gr.Markdown("# Interactive Image Segmentation Editor\n"
                    "Upload → include/exclude clicks → segment → refine → export. "
                    "Green **+** includes; red **−** excludes. Corrections use point prompts.")
        state = gr.State(fresh())
        with gr.Row():
            source = gr.Image(type="pil", label="1. Upload original image", sources=["upload"])
            canvas = gr.Image(type="numpy", label="2. Click here to select or correct", interactive=False)
        mode = gr.Radio(["Include (+)", "Exclude (-)"], value="Include (+)", label="Click meaning")
        with gr.Row():
            prompt_shape = gr.Radio(
                ["Single point", "Circle prompt"],
                value="Circle prompt",
                label="Prompt tool",
            )
            prompt_radius = gr.Slider(
                minimum=10,
                maximum=300,
                value=60,
                step=5,
                label="Circle radius (working-image pixels)",
            )
        gr.Markdown(
            "Circle prompt adds several same-label SAM points with one click. "
            "Place the circle fully inside the object for Include or fully in the "
            "unwanted area for Exclude; use Single point for precise boundary corrections."
        )
        with gr.Row():
            run = gr.Button("Segment / Refine", variant="primary")
            undo = gr.Button("Undo last click")
            reset = gr.Button("Reject / Clear selection")
        candidate = gr.Dropdown(["Best model score", "Candidate 1", "Candidate 2", "Candidate 3"],
                                value="Best model score", label="SAM candidate (press Segment / Refine after selecting)")
        with gr.Accordion("Advanced mask cleanup", open=False):
            min_area = gr.Slider(0, 5000, value=0, step=10, label="Remove components smaller than this pixel area")
            closing = gr.Slider(0, 5, value=0, step=1, label="Closing radius (0 = off; may fill handle holes)")
            largest = gr.Checkbox(label="Keep only largest remaining component", value=False)
            invert = gr.Checkbox(label="Invert foreground/background", value=False)
            cleanup = gr.Button("Apply cleanup / restore raw mask with zero controls")
        status = gr.Textbox(value="Upload an image to begin.", label="Status", interactive=False)
        with gr.Row():
            mask_view = gr.Image(label="Binary mask: white = selected", interactive=False, format="png")
            result = gr.Image(label="Export preview", interactive=False, format="png", image_mode="RGBA")
        with gr.Row():
            style = gr.Radio(["Transparent", "Solid color", "Uploaded background"],
                             value="Transparent", label="Output background")
            color = gr.ColorPicker(value="#ffffff", label="Solid background color")
            background = gr.Image(type="pil", sources=["upload"], label="Replacement background")
        accept = gr.Button("Accept mask & generate downloads", variant="primary")
        files = gr.File(label="Download PNG files", file_count="multiple")
        with gr.Accordion("Evaluate against ground truth: mIoU and Dice", open=True):
            gr.Markdown("Upload an independently annotated **binary PNG** of this same image: white object, black background. "
                        "Dimensions must match the working image (max side 1600). Do not use the model's own output as ground truth. "
                        "mIoU averages foreground and background IoU for this image; absent zero-union classes are excluded. "
                        "N/A means an undefined denominator. A model score is not measured IoU.")
            reference = gr.Image(type="pil", image_mode="RGBA", sources=["upload"], label="Reference mask")
            measure = gr.Button("Calculate mIoU / Dice")
            miou_score = gr.Number(label="Measured mIoU (%)", value=None, precision=2, interactive=False)
            evaluation_summary = gr.HTML(value="", label="Evaluation scorecard")
            metrics_view = gr.Dataframe(headers=["Metric", "Value"], datatype=["str", "str"], interactive=False)
            errors = gr.Image(label="Errors: green = TP, red = FP, blue = FN, black = TN", interactive=False)
            report = gr.File(label="Download metrics JSON")
        outputs = [state, canvas, mask_view, result, files, status,
                   miou_score, evaluation_summary, metrics_view, errors, report]
        # Serialize operations because callbacks update a session's prompts/mask.
        event_args = {"concurrency_id": "editor", "concurrency_limit": 1}
        source.change(upload_image, source, outputs, **event_args)
        canvas.select(add_prompt, [state, mode, prompt_shape, prompt_radius], outputs, **event_args)
        run.click(segment, [state, candidate], outputs, **event_args)
        undo.click(undo_point, state, outputs, **event_args)
        reset.click(reject, state, outputs, **event_args)
        cleanup.click(clean_mask, [state, min_area, closing, largest, invert], outputs, **event_args)
        measure.click(evaluate_mask, [state, reference],
                      [status, miou_score, evaluation_summary, metrics_view, errors, report], **event_args)
        reference.change(
            lambda: ("Reference changed. Calculate mIoU again.", None, "", [], None, None),
            outputs=[status, miou_score, evaluation_summary, metrics_view, errors, report],
            **event_args,
        )
        source.change(lambda: None, outputs=reference, **event_args)
        accept.click(export, [state, style, color, background], [result, files, status], **event_args)
        # Changing output controls clears exports so old settings cannot look current.
        for control in (style, color, background):
            control.change(lambda: (None, None), outputs=[result, files], **event_args)
    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", share=False)
