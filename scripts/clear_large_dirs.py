import os
import glob
import shutil
from pathlib import Path

SKIP_SMALLER_THAN_MB = 5


if __name__ == "__main__":
    # These .submitit directories can get very big
    base_dirs = [".", os.environ.get("SCRATCH", "."), os.environ.get("OUTPUT_DIR", ".")]
    size_cleared = 0
    for base_dir in base_dirs:
        for f in glob.glob(f"{base_dir}/outputs/*/.submitit"):
            megabytes = sum(file.stat().st_size for file in Path(f).rglob('*')) / (1 << 20)
            if megabytes < SKIP_SMALLER_THAN_MB:
                continue  # not worth it
            if "n" != input(f"Delete {f}? ({megabytes:.2f}MB) [y]/n: "):
                shutil.rmtree(f)
                size_cleared += megabytes

    # The private state folders from datasets with many clients and local adaptors too
    for base_dir in base_dirs:
        for f in glob.glob(f"{base_dir}/outputs/*/*/pvt"):
            # if len(list(glob.glob(f + "/*"))) == 0:
            #     continue
            megabytes = sum(file.stat().st_size for file in Path(f).rglob('*')) / (1 << 20)
            if megabytes < SKIP_SMALLER_THAN_MB:
                continue  # not worth it
            if "n" != input(f"Delete {f}? ({megabytes:.2f}MB) [y]/n: "):
                shutil.rmtree(f)
                size_cleared += megabytes

    print(f"Cleared {size_cleared:.2f}MB")
