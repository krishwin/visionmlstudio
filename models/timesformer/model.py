
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
import torch
import kagglehub


class ActionDetectorClass:
    def __init__(self):
        # Download model weights from Kaggle using kagglehub
        model_path = kagglehub.model_download("krishnawin/timesformer_finetuned_workout/pyTorch/2")  # Replace with actual path

        # Load model and processor
        processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        model = TimesformerForVideoClassification.from_pretrained(model_path)
        model.eval()
        self.model = model
        self.processor = processor

    def __call__(self, frames: np.ndarray):
        from PIL import Image
        # Sample frames uniformly
        indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
        frames = [frames[i] for i in indices]
        pil_frames = [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in frames]
        inputs = self.processor(pil_frames, return_tensors="pt")
        print(f"Processing {len(frames)} frames for action detection")
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
        return {"action": f"{self.model.config.id2label[predicted_class]}"}
