"""Fast recursive directory walk with skip rules and reparse-loop protection."""
from __future__ import annotations

import os
from dataclasses import dataclass

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400


@dataclass
class FileRec:
    path: str
    name: str
    ext: str
    dir: str
    drive: str
    size: int
    mtime: float


def _is_reparse(entry) -> bool:
    try:
        attrs = getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0)
        return bool(attrs & _FILE_ATTRIBUTE_REPARSE_POINT)
    except Exception:
        return False


def walk(paths, cfg):
    """Yield FileRec for every file under each path, honoring skip rules."""
    follow = cfg.get("follow_symlinks", False)
    for root in paths:
        root = os.path.abspath(root)
        drive = os.path.splitdrive(root)[0] or root
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if cfg.skip_dir(entry.name):
                                    continue
                                if not follow and _is_reparse(entry):
                                    continue
                                stack.append(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                st = entry.stat(follow_symlinks=False)
                                name = entry.name
                                yield FileRec(
                                    path=entry.path,
                                    name=name,
                                    ext=os.path.splitext(name)[1].lower(),
                                    dir=os.path.dirname(entry.path),
                                    drive=drive,
                                    size=st.st_size,
                                    mtime=st.st_mtime,
                                )
                        except (PermissionError, OSError):
                            continue
            except (PermissionError, OSError, FileNotFoundError):
                continue
