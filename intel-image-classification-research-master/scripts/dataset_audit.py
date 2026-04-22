from pathlib import Path
from collections import Counter


def main() -> None:
    data_dir = Path("seg_pred")
    if not data_dir.exists():
        print("Папка seg_pred не найдена.")
        return

    image_files = [p for p in data_dir.iterdir() if p.is_file()]
    if not image_files:
        print("В папке seg_pred нет файлов.")
        return

    ext_counter = Counter(p.suffix.lower() for p in image_files)
    print(f"Всего файлов: {len(image_files)}")
    print("Распределение по расширениям:")
    for ext, count in sorted(ext_counter.items()):
        print(f"  {ext or '[без расширения]'}: {count}")


if __name__ == "__main__":
    main()
