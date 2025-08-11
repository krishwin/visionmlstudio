import numpy as np
import torch
import kagglehub
import os
from utils.s3utils import download_file_from_s3 
from .model import RepNet
from torchvision import transforms
from typing import Dict, Tuple, Any


class RepCounterClass:
    def __init__(self):
        model_dir = os.path.join(os.path.dirname(__file__), "") 
        print(model_dir )
        if not os.path.exists(os.path.join(model_dir, "pytorch_weights.pth")):
                print("[RepNet] Weights file not found. Downloading from S3...")
                download_file_from_s3("weights", "RepNet/pytorch_weights.pth", model_dir + "pytorch_weights.pth")
        try:
            self._model = RepNet(num_frames=64, temperature=13.544)
            self._model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_weights.pth"), map_location="cpu"))
            self._model.eval()
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self.STRIDES = [8,4,3,2,1]
            self._input_size = (3, 112, 112)
            self.transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((112, 112)),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=0.5, std=0.5),
                            #transforms.ToTensor()
                        ])
        except OSError as exc:
            raise IOError(f"[RepNet] It seems that files in {model_dir} are corrupted or missing. ")
    
    def __call__(self, frames: np.ndarray):
        processed_frames = []
        for frame in frames:
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
        processed_frames = np.stack(processed_frames)
        print(f"Processed frames shape: {processed_frames.shape}")
        stride, confidence, period_length, period_count, periodicity_score, embeddings = self.predict_raw(processed_frames)
        print(f"Stride: {stride}, Confidence: {confidence:.2f}, Period Length: {period_length:.2f}, Period Count: {period_count.cumsum(dim=0).max().item()}")
        return period_count.cumsum(dim=0).max().item()
    @staticmethod
    def get_counts(raw_period_length: torch.Tensor, raw_periodicity: torch.Tensor, stride: int,
                   periodicity_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the final scores from the period length and periodicity predictions."""
        # Repeat the input to account for the stride
        raw_period_length = raw_period_length.repeat_interleave(stride, dim=0)
        raw_periodicity = raw_periodicity.repeat_interleave(stride, dim=0)
        # Compute the final scores in [0, 1]
        periodicity_score = torch.sigmoid(raw_periodicity).squeeze(-1)
        period_length_confidence, period_length = torch.max(torch.softmax(raw_period_length, dim=-1), dim=-1)
        # Remove the confidence for short periods and convert to the correct stride
        period_length_confidence[period_length < 2] = 0
        period_length = (period_length + 1) * stride
        periodicity_score = torch.sqrt(periodicity_score * period_length_confidence)
        # Generate the final counts and set them to 0 if the periodicity is too low
        period_count = 1 / period_length
        period_count[periodicity_score < periodicity_threshold] = 0
        period_length = 1 / (torch.mean(period_count) + 1e-6)
        #period_count = torch.cumsum(period_count, dim=0)
        confidence = torch.mean(periodicity_score)
        return confidence, period_length, period_count, periodicity_score

    def predict_raw(self, frames: np.ndarray):
         # Test multiple strides and pick the best one
        print('Running inference on multiple stride values...')
        frames = torch.from_numpy(frames)
        print(frames.shape,frames.dtype)
        best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = None, None, None, None, None, None
        for stride in self.STRIDES:
                # Apply stride
                if stride < 3 and best_stride is not None:
                    continue
                stride_frames = frames[::stride]
                stride_frames = stride_frames[:(len(stride_frames) // 64) * 64]
                if len(stride_frames) < 64:
                    continue # Skip this stride if there are not enough frames
                stride_frames = stride_frames.unflatten(0, (-1, 64)).movedim(1, 2) # Convert to N x C x D x H x W
                stride_frames = stride_frames.to("cpu")
                # Run inference
                raw_period_length, raw_periodicity_score, embeddings = [], [], []
                with torch.no_grad(): 
                    for i in range(stride_frames.shape[0]):  # Process each batch separately to avoid OOM
                        batch_period_length, batch_periodicity, batch_embeddings = self._model(stride_frames[i].unsqueeze(0))
                        raw_period_length.append(batch_period_length[0].cpu())
                        raw_periodicity_score.append(batch_periodicity[0].cpu())
                        embeddings.append(batch_embeddings[0].cpu()) 
                # Post-process results
                raw_period_length, raw_periodicity_score, embeddings = torch.cat(raw_period_length), torch.cat(raw_periodicity_score), torch.cat(embeddings)
                confidence, period_length, period_count, periodicity_score = self.get_counts(raw_period_length, raw_periodicity_score, stride)
                if best_confidence is None or confidence > best_confidence:
                    best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings = stride, confidence, period_length, period_count, periodicity_score, embeddings
        #if best_stride is None:
        #        raise RuntimeError('The stride values used are too large and nove 64 video chunk could be sampled. Try different values for --strides.')
        #print(f'Predicted a period length of {best_period_length/fps:.1f} seconds (~{int(best_period_length)} frames) with a confidence of {best_confidence:.2f} using a stride of {best_stride} frames.')
        return best_stride, best_confidence, best_period_length, best_period_count, best_periodicity_score, best_embeddings
