import os
import glob
import shutil
from pathlib import Path


if __name__ == "__main__":
    # These .submitit directories can get very big
    base_dirs = [".", os.environ.get("SCRATCH", ".")]
    size_cleared = 0
    for base_dir in base_dirs:
        for f in glob.glob(f"{base_dir}/outputs/*/.submitit"):
            if "n" != input(f"Delete {f}? [y]/n: "):
                megabytes = sum(file.stat().st_size for file in Path(f).rglob('*')) / (1 << 20)
                size_cleared += megabytes
                shutil.rmtree(f)
    print(f"Cleared {size_cleared:.2f}MB")
