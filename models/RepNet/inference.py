import os
from .model import RepNet
import torch
from torchvision import transforms
from typing import Dict, Tuple, Any
from utils.s3utils import download_file_from_s3 
from utils.decordutils import read_video
import numpy as np
import ray

class RepNetEngine:
    def __init__(self,model_dir=None):
        if model_dir is None:
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
            self.data_path = os.path.join(os.path.dirname(__file__), "data")
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
        except OSError as exc:
            raise IOError(f"[RepNet] It seems that files in {model_dir} are corrupted or missing. ")
    
    def __call__(self, batch: Dict[str, Any], **kwds):
        print(batch.keys())
        print(batch["item"])
        s3_filepath = batch["item"][0]
        file = s3_filepath.split("/")[-1]
        bucket_name, *prefix_parts, filename = s3_filepath.split("/")
        prefix = "/".join(prefix_parts)
        if not os.path.exists(os.path.join(self.data_path, file)):
            download_file_from_s3(bucket_name, prefix+'/'+filename, os.path.join(self.data_path, file))
        frames = read_video(os.path.join(self.data_path, file))
        processed_frames = []
        for frame in frames:
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
        processed_frames =np.stack(processed_frames)
        print(processed_frames.shape)
        clips = batch["clips"][0]
        batch["scenes"] = [[]]
        print(clips)
        for clip in clips:
            start, end = clip
            frames = processed_frames[start:end]
            stride, confidence, period_length, period_count, periodicity_score, embeddings = self.predict_raw(frames)
            if(period_count is None):
                batch["scenes"][0].append([start, end,0])
                continue
            scenes =  self.predictions_to_scenes(period_count)
            print(scenes)
            for scn in scenes:
                batch["scenes"][0].append([scn[0]+start, start+scn[1],scn[2]]) 
        #print(batch["data"].shape[1:], self._input_size)
        #assert len(batch["data"].shape) == 4 and batch["data"].shape[1:] == self._input_size, \
        #    "[RepNet] Input shape must be [frames, height, width, 3]."
        batch['clips'] = [ clip.tolist() for clip in batch['clips']]
        return batch
    
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
    def predictions_to_scenes(self, period_count):
        zero_indices = (period_count == 0).nonzero(as_tuple=True)[0].numpy()
        clips = []
        start = 0
        for i in range(len(zero_indices)-1):
            if zero_indices[i+1] - zero_indices[i] < 100:
                continue
            else:
                end = zero_indices[i]
                clips.append((start, end,period_count[start:end].cumsum(dim=0).max().item()))
                clips.append((end+1, zero_indices[i+1]-1,period_count[end+1:zero_indices[i+1]-1].cumsum(dim=0).max().item()))
                start = zero_indices[i+1]
        if start < len(period_count):
            clips.append((start, len(period_count)-1,period_count[start:].cumsum(dim=0).max().item()))
        return clips
