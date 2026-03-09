"""Class file management utilities.

These helpers isolate classes.txt bootstrapping and read/write behavior so
annotation UI code can stay thin and replaceable.
"""

from pathlib import Path
import shutil


def ensure_classes_file(annot_dir: Path, new_classes_file: Path) -> Path:
    """Ensure `<annot_dir>/classes.txt` exists.

    If missing and `new_classes_file` exists, copy it as initial classes.
    Returns the classes file path.
    """
    annot_dir.mkdir(parents=True, exist_ok=True)
    classes_txt = annot_dir / "classes.txt"

    if not classes_txt.exists() and new_classes_file.exists():
        shutil.copy2(new_classes_file, classes_txt)

    return classes_txt


def load_classes_file(classes_txt: Path) -> list[str]:
    """Load classes as a normalized list from classes.txt."""
    if not classes_txt.exists():
        return []
    with open(classes_txt, "r") as file_obj:
        return [line.strip() for line in file_obj if line.strip()]


def save_classes_file(classes_txt: Path, classes_list: list[str]) -> None:
    """Persist class names to classes.txt, one class per line."""
    classes_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(classes_txt, "w") as file_obj:
        if classes_list:
            file_obj.write("\n".join(classes_list) + "\n")
        else:
            file_obj.write("")
