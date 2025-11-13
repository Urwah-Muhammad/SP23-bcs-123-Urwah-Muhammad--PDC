import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from supabase import create_client, Client
import time as time

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

buck_name = "pdc_assignment_6"
local_folder = "files_to_upload"
MAX_WORKERS = 3


def upload_file(filename: str):
    """Upload a single file to Supabase storage."""
    file_path = os.path.join(local_folder, filename)
    with open(file_path, "rb") as f:
        file_data = f.read()
    try:
        supabase.storage.from_(buck_name).upload(
            path = filename, 
            file = file_data,
        )
        return f"✅ Uploaded {filename}"
    except Exception as e:
        return f"❌ Failed {filename}: {e}"
    

def main():
    files = [f for f in os.listdir(local_folder) if f.endswith(".txt")]
    print(f"Uploading {len(files)} files in parallel...")

    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(upload_file, f) for f in files]
        for future in as_completed(futures):
            print(future.result())

    print("✅ All uploads completed.")
    
    end_time = time.time()
    print(f"All files uploaded parallely in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()

