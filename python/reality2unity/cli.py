import argparse
from pathlib import Path

from .prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(prog="reality2unity")
    commands = parser.add_subparsers(required=True)
    command = commands.add_parser("prepare", help="입력 사진 목록을 확인하고 저장합니다.")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--image-dir", type=Path, required=True)
    command.add_argument("--runs-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        path, count, reused = prepare(args.config, args.image_dir, args.runs_dir)
    except (OSError, ValueError) as error:
        parser.exit(2, f"prepare: {error}\n")

    status = "reused" if reused else "created"
    print(f"prepare: {status} ({count} images)")
    print(path)


if __name__ == "__main__":
    main()