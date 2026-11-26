import argparse
from pathlib import Path

from .prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(prog="reality2unity")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_command = commands.add_parser("prepare", help="입력 사진 목록을 확인하고 저장합니다.")
    prepare_command.add_argument("--config", type=Path, required=True)
    prepare_command.add_argument("--image-dir", type=Path, required=True)
    prepare_command.add_argument("--runs-dir", type=Path, required=True)

    reconstruct_command = commands.add_parser("reconstruct", help="VGGT로 선택한 사진을 복원합니다.")
    reconstruct_command.add_argument("--manifest", type=Path, required=True)
    reconstruct_command.add_argument("--image-dir", type=Path, required=True)
    reconstruct_command.add_argument("--image", action="append", required=True)
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            path, count, reused = prepare(args.config, args.image_dir, args.runs_dir)
        else:
            from .vggt_reconstruct import reconstruct

            path, reused = reconstruct(args.manifest, args.image_dir, args.image)
            count = len(args.image)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"{args.command}: {error}\n")

    status = "reused" if reused else "created"
    print(f"{args.command}: {status} ({count} images)")
    print(path)


if __name__ == "__main__":
    main()
