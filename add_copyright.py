import os

AMD_COPYRIGHT = "Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved."
OPENMMLAB_COPYRIGHT = "Copyright (c) OpenMMLab. All rights reserved."

SKIP_EXTS = {".md", ".txt", ".in", ".json", ".jsonl", ".cff", ".cfg"}
SKIP_FILES = {".gitkeep", "Jenkinsfile"}


# 区分注释风格
C_STYLE_EXTS = {".cpp", ".c", ".cu", ".cuh", ".hpp", ".h", ".hip"}
PY_STYLE_EXTS = {".py"}


def has_copyright(lines):
    for line in lines[:10]:  # 前几行检查
        if "Copyright" in line:
            return True
    return False


def wrap_copyright(ext, text):
    """根据扩展名选择注释风格，加空行"""
    if ext in C_STYLE_EXTS:
        return f"/* {text} */\n\n"
    elif ext in PY_STYLE_EXTS:
        return f"# {text}\n\n"
    else:
        return f"# {text}\n\n"  # 默认 Python 风格


def process_file(filepath, dry_run=False):
    _, ext = os.path.splitext(filepath)
    filename = os.path.basename(filepath)

    if ext in SKIP_EXTS or filename in SKIP_FILES:
        return

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    content = "".join(lines)
    new_lines = None

    if "hipify" in content:
        if not any("Advanced Micro Devices" in l for l in lines[:5]):
            copyright_line = wrap_copyright(ext, AMD_COPYRIGHT)
            new_lines = [copyright_line] + lines
    else:
        if not has_copyright(lines):
            copyright_line = wrap_copyright(ext, OPENMMLAB_COPYRIGHT)
            if ext in PY_STYLE_EXTS:
                # Python: 确保 shebang 和 encoding 在最前面
                insert_idx = 0
                if lines and lines[0].startswith("#!"):
                    insert_idx = 1
                if len(lines) > insert_idx and "coding" in lines[insert_idx]:
                    insert_idx += 1
                new_lines = lines[:insert_idx] + [copyright_line] + lines[insert_idx:]
            else:
                new_lines = [copyright_line] + lines

    if new_lines:
        if dry_run:
            print(f"[DRY-RUN] Would update: {filepath}")
        else:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            print(f"Updated: {filepath}")


def add_copyright_to_dir(root_dir, dry_run=False):
    for root, _, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(root, file)
            process_file(filepath, dry_run=dry_run)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python add_copyright.py <directory> [--dry-run]")
    else:
        dry_run = "--dry-run" in sys.argv
        add_copyright_to_dir(sys.argv[1], dry_run=dry_run)

