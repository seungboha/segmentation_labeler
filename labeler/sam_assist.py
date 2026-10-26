"""Optional, lazy SAM2 image prompting; manual editing needs no ML dependencies."""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path

import numpy as np


class SamUnavailable(RuntimeError):
    pass


class SamAssist:
    def __init__(self, root):
        self.root = root
        self.predictor = None
        self.embedded = None

    def checkpoint(self):
        configured = os.environ.get("SAM2_CHECKPOINT")
        if configured:
            return Path(configured).expanduser()
        candidates = [self.root.parent / "sam2/checkpoints/sam2.1_hiera_tiny.pt"]
        package = importlib.util.find_spec("sam2")
        if package and package.origin:
            candidates.append(Path(package.origin).parent.parent / "checkpoints/sam2.1_hiera_tiny.pt")
        return next((path for path in candidates if path.is_file()), candidates[0])

    def status(self):
        installed = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("sam2") is not None
        checkpoint = self.checkpoint()
        available = installed and checkpoint.is_file()
        message = "Ready; model loads on the first prediction." if available else (
            "Run the server in the SAM2 Python environment." if not installed else
            "Checkpoint missing. Set SAM2_CHECKPOINT to the model file.")
        return {"available": available, "loaded": self.predictor is not None,
                "message": message, "checkpoint": str(checkpoint),
                "device": str(self.predictor.device) if self.predictor else os.environ.get("SAM2_DEVICE", "auto")}

    def get_predictor(self):
        if self.predictor is None:
            state = self.status()
            if not state["available"]:
                raise SamUnavailable(state["message"])
            try:
                import torch
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                config = os.environ.get("SAM2_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
                device = os.environ.get("SAM2_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
                self.predictor = SAM2ImagePredictor(build_sam2(config, str(self.checkpoint()), device=device))
            except ImportError as error:
                raise SamUnavailable("SAM2 dependencies are missing. Run the server in the SAM2 environment.") from error
            except (OSError, RuntimeError, ValueError) as error:
                raise SamUnavailable(f"Could not load SAM2. Check SAM2_CHECKPOINT, SAM2_CONFIG and SAM2_DEVICE. {error}") from error
        return self.predictor

    def predict(self, image, key, points):
        predictor = self.get_predictor()
        import torch
        try:
            cast = torch.autocast("cuda", dtype=torch.bfloat16) if (
                predictor.device.type == "cuda" and torch.cuda.is_bf16_supported()) else nullcontext()
            with torch.inference_mode(), cast:
                if self.embedded != key:
                    self.embedded = None
                    predictor.set_image(np.array(image, copy=True))
                    self.embedded = key
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[point.x, point.y] for point in points], dtype=np.float32),
                    point_labels=np.array([point.label for point in points], dtype=np.int32),
                    multimask_output=True)
            best = int(np.argmax(scores))
            return masks[best].astype(bool), float(scores[best]), str(predictor.device)
        except torch.cuda.OutOfMemoryError as error:
            self.embedded = None
            torch.cuda.empty_cache()
            raise SamUnavailable("GPU memory is full. Close other GPU applications or set SAM2_DEVICE=cpu.") from error
