"""Preview or restore archived DNN files without overwriting changed files."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent


def restoration_plan(root=ROOT):
    root = Path(root).resolve()
    archive = root / "backupcuoiki"
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    plan = []
    for entry in manifest["files"]:
        if entry["group"] != "dnn":
            continue
        source = (root / entry["archive"]).resolve()
        target = (root / entry["source"]).resolve()
        if not source.is_relative_to(archive.resolve()) or not target.is_relative_to(root) or target.is_relative_to(archive.resolve()):
            raise ValueError("Đường dẫn khôi phục nằm ngoài phạm vi cho phép.")
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Backup không khớp hash: {entry['archive']}")
        if target.exists():
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"File đích đã có nội dung khác; không ghi đè: {entry['source']}")
            continue
        plan.append((source, target))
    return plan


def restore(root=ROOT, apply=False):
    # Validate every path, hash and collision before copying any file.
    plan = restoration_plan(root)
    if apply:
        for source, target in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Khôi phục DNN cuối kỳ; mặc định chỉ xem trước.")
    parser.add_argument("--apply", action="store_true", help="Chép file về vị trí gốc; giữ nguyên backup.")
    arguments = parser.parse_args()
    try:
        files = restore(apply=arguments.apply)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"{'Đã khôi phục' if arguments.apply else 'Có thể khôi phục'} {len(files)} file DNN.")
    if not arguments.apply:
        print("Chưa thay đổi file nào. Thêm --apply khi cần mở lại DNN cho cuối kỳ.")
