import hashlib
import json
import re
import tempfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _read_config(config_path: Path) -> tuple[str, list[str]]:
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("설정은 JSON 객체여야 합니다.")

    scene = config.get("scene")
    if not isinstance(scene, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", scene):
        raise ValueError("scene에는 영문, 숫자, 하이픈, 밑줄을 사용할 수 있습니다.")

    names = config.get("images")
    if not isinstance(names, list) or not names:
        raise ValueError("images에 사진 경로를 한 개 이상 넣으세요.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("사진 경로는 비어 있지 않은 문자열이어야 합니다.")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("images에 중복된 사진 경로가 있습니다.")

    return scene, names


def _image_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or relative.as_posix() != name
        or "\\" in name
        or ":" in name
        or ".." in relative.parts
        or relative.suffix.lower() not in IMAGE_SUFFIXES
    ):
        raise ValueError(f"사진 경로는 /로 구분한 JPG·JPEG·PNG 상대 경로여야 합니다: {name}")

    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"사진 경로가 지정한 폴더를 벗어났습니다: {name}")
    if not path.is_file():
        raise FileNotFoundError(f"사진을 찾을 수 없습니다: {path}")
    return path


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            json.dump(manifest, temp, ensure_ascii=False, indent=2)
            temp.write("\n")
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def prepare(config_path: Path, image_dir: Path, runs_dir: Path) -> tuple[Path, int, bool]:
    scene, names = _read_config(config_path)
    root = image_dir.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"사진 폴더를 찾을 수 없습니다: {root}")

    images = []
    for name in names:
        with _image_path(root, name).open("rb") as image:
            digest = hashlib.file_digest(image, "sha256").hexdigest()
        images.append({"path": name, "sha256": digest})

    inputs = {"schema_version": SCHEMA_VERSION, "scene": scene, "images": images}
    canonical = json.dumps(
        inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    input_id = hashlib.sha256(canonical).hexdigest()

    manifest = {**inputs, "input_id": input_id}
    output = runs_dir.resolve() / scene / input_id / "prepare" / "manifest.json"

    try:
        cached = json.loads(output.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        cached = None

    if cached == manifest:
        return output, len(images), True

    _write_manifest(output, manifest)
    return output, len(images), False