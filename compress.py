# %%
import tarfile

def compress_directory_to_tar_gz(tar_gz_path, source_dir):
    with tarfile.open(tar_gz_path, "w:gz") as tar:
        tar.add(source_dir, arcname="")

# Define the directory you want to compress and the destination filename
source_dir = "/path/to/your/directory"
tar_gz_path = "/path/to/your/output/file.tar.gz"

compress_directory_to_tar_gz(tar_gz_path, source_dir)