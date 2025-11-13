import os

os.makedirs("files_to_upload", exist_ok=True)
for i in range(1, 101):
    with open(f"files_to_upload/file_{i}.txt", "w") as f:
        f.write(f"This is file number {i}\n")