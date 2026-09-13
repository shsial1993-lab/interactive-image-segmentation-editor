# Interactive Image Segmentation Editor

A teaching project for fifth-semester BSAI students combining SAM 2.1 point-prompt segmentation, human correction through include/exclude clicks, and computer graphics compositing.

## Setup: Windows PowerShell, Python 3.11, and CPU

These instructions match the setup used to launch the application successfully on your Windows computer. Run the commands in PowerShell or the VS Code PowerShell terminal. Python 3.14 can remain installed; this project uses a separate Python 3.11 environment.

If your `.venv311` environment is already installed and working, skip to **Run the application again**.

### 1. Install Python 3.11 and Git

Check the installed Python versions:

```powershell
python --version
py -0p
```

If Python 3.11 is missing, install it:

```powershell
winget install --exact --id Python.Python.3.11
```

Check Git:

```powershell
git --version
```

If Git is missing, install it:

```powershell
winget install --exact --id Git.Git
```

After installation, close and reopen VS Code or PowerShell, then verify:

```powershell
py -3.11 --version
git --version
```

### 2. Create and activate the project environment

Open the extracted project folder. Change the path below if you stored it elsewhere:

```powershell
cd C:\Users\hp\Desktop\HCI\Projects\segmentation_editor
```

If another virtual environment is active, run `deactivate` first. Then:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python --version
```

The terminal should show `(.venv311)`, and the Python version should be `3.11.x`.

In VS Code, press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and select the interpreter inside `.venv311`.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

This installs CPU versions of PyTorch and TorchVision. Keep these commands in the Python 3.11 environment: the earlier attempt to install this PyTorch version under Python 3.14 failed because no compatible distribution was available.

### 4. Install SAM 2

Use PowerShell syntax to disable the optional CUDA extension during installation:

```powershell
$env:SAM2_BUILD_CUDA = "0"
python -m pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
```

This avoids building the optional CUDA extension. The PyTorch installation above uses the CPU, even if the computer has an NVIDIA GPU. A GPU setup requires a compatible CUDA-enabled PyTorch/TorchVision installation; use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for that separate setup.

The SAM 2 authors recommend WSL2 with Ubuntu for Windows users. The commands in this guide use native Windows PowerShell, matching your successful app launch.

### 5. Verify imports and start the application

```powershell
python -c "import torch, torchvision, gradio; from sam2.sam2_image_predictor import SAM2ImagePredictor; print('Imports successful:', torch.__version__)"
python app.py
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860), or the local URL printed in the terminal.

The first segmentation downloads `facebook/sam2.1-hiera-tiny` from Hugging Face. Internet access is required for this download; model files are then cached. CPU inference can be slow, especially on the first run. No API key is needed.

The application binds to localhost and does not create a public share link. Keep the terminal running while using it. Press **Ctrl+C** to stop it.

### Run the application again

After the initial setup, use only:

```powershell
cd C:\Users\hp\Desktop\HCI\Projects\segmentation_editor
.\.venv311\Scripts\Activate.ps1
python app.py
```

You do not need to recreate the environment or reinstall dependencies each time.

### Record the working environment

After installation succeeds, save the dependency versions:

```powershell
python -m pip freeze > installed.txt
```

The SAM 2 installation command uses the upstream repository without pinning a commit. Retain the resolved SAM 2 revision recorded in the environment output for reproducibility.

## Use

1. Upload an image in the left panel. Start with one clear person or object.
2. Select **Include (+)**. The default **Circle prompt** tool creates several same-label SAM points with one click; adjust the circle radius and click inside the object. A green circle marks the prompt footprint. Use **Single point** when you need a precise correction.
3. Click **Segment / Refine**. The overlay shows the selected region and its boundary. In the binary mask, white means selected and black means background.
4. For corrections, choose **Exclude (-)** and use a circle on unwanted regions, or choose **Include (+)** on missing parts. Place a circle fully inside the intended region and click **Segment / Refine** again after changing prompts. All expanded points are passed to SAM 2 each time.
5. Use **Undo last click** to remove the whole last circle or single-point prompt, then rerun segmentation. Use **Reject / Clear selection** to discard the selection while preserving the original image.
6. Choose **Transparent**, **Solid color**, or **Uploaded background**. For a solid background, select a color; for an uploaded background, supply a replacement image.
7. Click **Accept mask & generate downloads**, then download `result.png` and `mask.png`.

### Output behavior

- Images larger than 1600 pixels on their longest side are resized before editing. Displayed coordinates, masks, and exports use this working resolution.
- Uploaded backgrounds are resized and center-cropped to match the foreground.
- `mask.png` is binary: 255 means selected and 0 means background.
- Transparent output is RGBA PNG. Hard binary edges are intentional; this is segmentation, not hair matting.
- Changing prompts clears the previous mask and exports until segmentation runs again.
- Changing output settings clears previous exports; click **Accept mask & generate downloads** again.

## Teaching map

| Subject | Code | Student learning |
|---|---|---|
| AI | `predict_mask`, `segment` | Pretrained model inference, positive/negative prompts, circle-expanded prompts, candidate-mask selection |
| HCI | `add_prompt`, `undo_point`, `reject`, `build_app` | Visible feedback, circle tool, correction, group undo, rejection, user control |
| Computer Graphics | `preview`, `compose` | Alpha blending, contour rendering, RGB/RGBA, coordinate mapping, compositing |

The highest model score selects a candidate mask. This score is a predicted quality estimate, not measured accuracy. Evaluate Dice/IoU only against real ground-truth masks.

The first reported latency includes model initialization and possibly downloading. Record later requests separately when comparing interaction times in class.

Suggested student extensions:

- Add true brush-based manual mask correction that edits the binary mask directly.
- Compute Dice/IoU against annotated masks.
- Compare clicks, completion time, and errors for five users.
- Investigate thin objects, low contrast, and overlapping objects.

This starter selects one object region per image. It does not assign semantic class names or train a new model.

## Validation and limitations

Run the offline tests from the activated project environment:

```powershell
python -m unittest -v test_editor.py
```

The offline suite contains ten checks. They cover compositing, PNG alpha/mask export, correction/rejection, image resizing, independent session state, circle-prompt expansion and group undo, Gradio interface construction, hand-calculated metric values, invalid references, reversible cleanup, metric export, and stale-result invalidation. They substitute a known mask for the model result, so they do not validate SAM 2 inference or segmentation quality.

You confirmed that the application launches on your Windows setup. Successful launch alone does not confirm model inference. Before class, upload a real image, run segmentation, refine the mask, and download both PNG outputs to check the full workflow.

The shared predictor is locked across image encoding and prediction to prevent cross-session image-cache mixing. Requests are serialized for this small classroom app, and image embeddings are recomputed each time for simplicity. Export files are temporary and removed when the application exits normally.

## Official references

- [SAM 2 installation and models](https://github.com/facebookresearch/sam2)
- [SAM 2 image predictor API](https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py)
- [PyTorch version-specific installation commands](https://pytorch.org/get-started/previous-versions/)
- [Gradio documentation](https://www.gradio.app/docs)

## Advanced edition: evaluation and cleanup

Replace the old project files with this updated package, keeping your existing `.venv311`. No new dependencies are required. The package now includes `metrics.py`; keep it beside `app.py`.

### Measured mIoU

1. Segment your image.
2. Upload an independently annotated reference mask in the evaluation panel. Use an opaque binary PNG with pixel values 0/1 or 0/255. White is the target object. Do not upload an RGB photo, a transparent cutout, or SAM's own prediction as ground truth.
3. Reference dimensions and alignment must match the working image exactly. If the source was resized to a maximum side of 1600, resize the reference to that same size with nearest-neighbor interpolation before uploading. The app rejects mismatched dimensions rather than silently assuming alignment.
4. Click **Calculate mIoU / Dice**. The measured score is shown prominently as **Measured mIoU (%)**, followed by a visual scorecard and the detailed table. Download `metrics.json` if needed.

Foreground IoU = TP / (TP + FP + FN). Background IoU = TN / (TN + FP + FN). Per-image binary mIoU is the mean of these two IoUs, excluding any class whose union is zero. Undefined ratios display N/A (JSON null). Foreground Dice = 2TP / (2TP + FP + FN). Precision, recall, pixel accuracy and confusion counts are also reported. This is not a dataset-level score. Metrics remain unavailable until a reference is supplied.

The scorecard makes the distinction between the SAM model score and measured mIoU explicit. The JSON report also records the selected candidate, prediction time, prompt count, working image shape, and the model score estimate when available.

The error map uses green for correctly selected pixels, red for extra selected pixels, blue for missed object pixels, and black for correctly rejected background. Changing the prediction or reference clears stale metrics. Recalculate after refinement or cleanup.

For the generated mug photograph, no independently annotated ground-truth mask was supplied. Create a careful manual annotation before claiming measured accuracy. Saving a prediction as its own reference would give a misleading perfect score.

### Candidate selection and reversible cleanup

Choose Best model score or Candidate 1/2/3, then press Segment / Refine to generate that candidate. Candidate numbers are SAM's output order, not a quality ranking. Use visual inspection to select a candidate; selecting on test-ground-truth metrics biases evaluation.

Cleanup can remove small components, retain only the largest component, close small gaps, or invert a reversed selection. Press Apply cleanup to commit the controls. Every cleanup starts from the raw prediction; set area and radius to zero and uncheck both boxes to restore it. Segmentation reruns replace the raw prediction and require cleanup to be applied again. Closing can fill legitimate openings such as a mug handle; largest-component filtering can remove disconnected object parts. Defaults leave the prediction unchanged.

A fragmented mask is not proof of a particular installation or model failure. Start with Reject / Clear selection, place an Include point inside the mug body, and refine using a few deliberate Include/Exclude points. Compare candidates before applying cleanup. These controls cannot guarantee a correct segmentation.

Run the expanded offline suite:

```powershell
python -m unittest -v test_editor.py test_metrics.py
```

The new tests cover hand-calculated metric values, empty-class conventions, invalid references, reversible cleanup, metric export, stale-result invalidation, and circle-prompt group undo. Model inference still requires local verification; offline tests do not measure SAM quality.
