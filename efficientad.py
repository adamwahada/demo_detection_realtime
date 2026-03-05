import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2

from common import get_autoencoder, get_pdn_small, get_pdn_medium

# ── Constants (must match what was used during training) ──────────────────────
on_gpu       = torch.cuda.is_available()
out_channels = 384
image_size   = 256

# Standard ImageNet normalisation used by EfficientAD
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Model loader ─────────────────────────────────────────────────────────────

def load_anomaly_model(checkpoint_dir: str,
                       model_size: str = 'small',
                       threshold: float = 0.5) -> dict:
    import os

    def _load(fname):
        path = os.path.join(checkpoint_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"EfficientAD checkpoint not found: {path}")
        return torch.load(path, map_location='cpu')

    # Architecture
    if model_size == 'small':
        teacher     = get_pdn_small(out_channels)
        student     = get_pdn_small(2 * out_channels)
    elif model_size == 'medium':
        teacher     = get_pdn_medium(out_channels)
        student     = get_pdn_medium(2 * out_channels)
    else:
        raise ValueError(f"model_size must be 'small' or 'medium', got '{model_size}'")
    autoencoder = get_autoencoder(out_channels)

    # Weights — support both full-model saves (torch.save(model, ...))
    # and state-dict saves (torch.save(model.state_dict(), ...))
    def _load_weights(model, fname):
        raw = _load(fname)
        if isinstance(raw, torch.nn.Module):
            return raw
        model.load_state_dict(raw)
        return model

    teacher     = _load_weights(teacher,     'teacher_final.pth')
    student     = _load_weights(student,     'student_final.pth')
    autoencoder = _load_weights(autoencoder, 'autoencoder_final.pth')

    teacher.eval()
    student.eval()
    autoencoder.eval()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    # Normalisation stats saved alongside models at training time
    norm = _load('normalization.pt')

    def _to_gpu(t):
        return t.cuda() if on_gpu else t

    return {
        'teacher':      teacher,
        'student':      student,
        'autoencoder':  autoencoder,
        'teacher_mean': _to_gpu(norm['teacher_mean']),
        'teacher_std':  _to_gpu(norm['teacher_std']),
        'q_st_start':   _to_gpu(norm['q_st_start']),
        'q_st_end':     _to_gpu(norm['q_st_end']),
        'q_ae_start':   _to_gpu(norm['q_ae_start']),
        'q_ae_end':     _to_gpu(norm['q_ae_end']),
        'threshold':    threshold,
    }


# ── Core inference ────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(image_tensor, teacher, student, autoencoder,
            teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None,
            q_ae_start=None, q_ae_end=None):

    teacher_output     = teacher(image_tensor)
    teacher_output     = (teacher_output - teacher_mean) / teacher_std
    student_output     = student(image_tensor)
    autoencoder_output = autoencoder(image_tensor)

    map_st = torch.mean(
        (teacher_output - student_output[:, :out_channels]) ** 2,
        dim=1, keepdim=True)
    map_ae = torch.mean(
        (autoencoder_output - student_output[:, out_channels:]) ** 2,
        dim=1, keepdim=True)

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def predict_anomaly_score(model: dict, bgr_frame: np.ndarray,
                          orig_size: tuple = None):
    
    orig_h, orig_w = bgr_frame.shape[:2]
    if orig_size is None:
        orig_size = (orig_w, orig_h)

    # BGR → RGB, then apply standard transform
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    tensor = default_transform(rgb).unsqueeze(0)   # (1, 3, 256, 256)
    if on_gpu:
        tensor = tensor.cuda()

    map_combined, _, _ = predict(
        tensor,
        teacher      = model['teacher'],
        student      = model['student'],
        autoencoder  = model['autoencoder'],
        teacher_mean = model['teacher_mean'],
        teacher_std  = model['teacher_std'],
        q_st_start   = model['q_st_start'],
        q_st_end     = model['q_st_end'],
        q_ae_start   = model['q_ae_start'],
        q_ae_end     = model['q_ae_end'],
    )

    # Pad + resize back to original dimensions
    map_combined = F.pad(map_combined, (4, 4, 4, 4))
    map_combined = F.interpolate(map_combined, (orig_size[1], orig_size[0]),
                                 mode='bilinear', align_corners=False)
    heat = map_combined[0, 0].cpu().numpy()   # float

    score = float(np.max(heat))

    # Build colour heatmap (COLORMAP_JET, BGR)
    heat_norm  = np.clip(heat / max(score, 1e-6), 0.0, 1.0)
    heat_uint8 = (heat_norm * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    return score, heatmap