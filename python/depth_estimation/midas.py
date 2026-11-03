"""MiDaS depth estimation from the 2024 report appendix."""

import cv2
import torch


def load_midas_model():
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform if "DPT" in model_type else transforms.small_transform
    return midas, transform, device


def estimate_depth(image_path, model, transform, device):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth_map = prediction.cpu().numpy()
    return depth_map
