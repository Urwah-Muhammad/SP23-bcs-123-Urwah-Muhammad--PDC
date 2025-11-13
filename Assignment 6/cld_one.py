import os
from dotenv import load_dotenv
from supabase import create_client, Client
import time as time

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

buck_name = "pdc_assignment_6"
local_folder = "files_to_upload"

start_time = time.time()

for filename in os.listdir(local_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(local_folder, filename)
        with open(file_path, "rb") as f:
            file_data = f.read()

        # The destination path inside your Supabase bucket
        supabase.storage.from_(buck_name).upload(
            path=filename,  # you can prefix with "foldername/filename" if needed
            file=file_data
        )
        print(f"✅ Uploaded: {filename}")

end_time = time.time()
print(f"All files uploaded sequentially in {end_time - start_time:.2f} seconds.")
