import hashlib
import json
import tempfile
import time
from pathlib import Path

from .prepare import _image_path

VGGT_CODE_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
MODEL_ID = "facebook/VGGT-1B"
MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
MIN_CONFIDENCE = 1.5
PREVIEW_STRIDE = 2
OUTPUT_FILES = ("geometry.npz", "preview.ply")


def _read_inputs(manifest_path: Path, image_dir: Path, names: list[str]) -> tuple[str, list[Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_id = manifest["input_id"]
    digests = {image["path"]: image["sha256"] for image in manifest["images"]}

    root = image_dir.resolve()
    paths = []
    for name in names:
        path = _image_path(root, name)
        with path.open("rb") as image:
            digest = hashlib.file_digest(image, "sha256").hexdigest()
        paths.append(path)

    return input_id, paths


def _infer(paths: list[Path]):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    frames = load_and_preprocess_images([str(path) for path in paths], mode="crop")
    model = VGGT.from_pretrained(
        MODEL_ID, 
        revision=MODEL_REVISION, 
        enable_point=False, 
        enable_track=False,
    ).to("cuda").eval()

    inputs = frames.to("cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(inputs)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(prediction["pose_enc"].float(), tuple(inputs.shape[-2:]))

    torch.cuda.synchronize()
    metrics = {
        "device": torch.cuda.get_device_name(),
        "torch_version": str(torch.__version__),
        "inference_seconds": round(time.perf_counter() - started, 3),
        "peak_torch_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    return (
        extrinsics[0].float().cpu().numpy(),
        intrinsics[0].float().cpu().numpy(),
        prediction["depth"][0, ..., 0].float().cpu().numpy(),
        prediction["depth_conf"][0].float().cpu().numpy(),
        (frames.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().byte().numpy(),
        metrics,
    )


def _write_preview(path: Path, depth, confidence, rgb, extrinsics, intrinsics) -> int:
    import numpy as np
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    valid = np.isfinite(depth) & (depth > 0)
    points = unproject_depth_map_to_point_map(
        np.where(valid, depth, 0)[..., None], extrinsics, intrinsics
    )
    mask = valid & np.isfinite(confidence) & (confidence >= MIN_CONFIDENCE)
    mask &= np.isfinite(points).all(axis=-1)
    sample = np.s_[:, ::PREVIEW_STRIDE, ::PREVIEW_STRIDE]
    xyz = points[sample][mask[sample]]
    colors = rgb[sample][mask[sample]]

    count = len(xyz)

    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertices = np.empty(count, dtype=fields)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(vertices.tobytes())
    return count


def _completed(output: Path, request: dict) -> bool:
    try:
        record = json.loads((output / "run.json").read_text(encoding="utf-8"))
        return record["request"] == request and all(
            (output / name).stat().st_size == record["file_sizes"][name] > 0
            for name in OUTPUT_FILES
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        return False


def reconstruct(manifest_path: Path, image_dir: Path, names: list[str]) -> tuple[Path, bool]:
    input_id, paths = _read_inputs(manifest_path, image_dir, names)
    request = {
        "version": 1,
        "input_id": input_id,
        "images": names,
        "model": MODEL_ID,
        "weights_revision": MODEL_REVISION,
        "vggt_revision": VGGT_CODE_REVISION,
        "preprocessing": "crop",
        "dtype": "fp16_autocast",
        "preview_min_confidence": MIN_CONFIDENCE,
        "preview_stride": PREVIEW_STRIDE,
    }
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    run_id = hashlib.sha256(encoded).hexdigest()[:16]
    output = manifest_path.resolve().parent.parent / "reconstruct" / f"vggt-{run_id}"
    if _completed(output, request):
        return output, True

    import numpy as np

    extrinsics, intrinsics, depth, confidence, rgb, metrics = _infer(paths)
    shape = depth.shape

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".write-", dir=output) as folder:
        staged = Path(folder)
        np.savez_compressed(
            staged / "geometry.npz",
            image_names=np.asarray(names),
            intrinsics=intrinsics,
            extrinsics_world_to_camera=extrinsics,
            depth=depth,
            depth_confidence=confidence,
            rgb=rgb,
        )
        count = _write_preview(staged / "preview.ply", depth, confidence, rgb, extrinsics, intrinsics)
        record = {
            "request": request,
            "metrics": metrics,
            "image_size_hw": list(shape[1:]),
            "coordinate_system": "opencv_world_to_camera",
            "scale": "relative",
            "preview_points": count,
            "file_sizes": {name: (staged / name).stat().st_size for name in OUTPUT_FILES},
        }
        (staged / "run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "run.json").unlink(missing_ok=True)
        for name in (*OUTPUT_FILES, "run.json"):
            (staged / name).replace(output / name)
    return output, False
