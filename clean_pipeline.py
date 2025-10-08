import os
import shutil
import time

def clean_pipeline():
    """Membersihkan pipeline output dan cache"""
    
    BASE_DIR = r"C:\drug-pipeline"
    DIRECTORIES_TO_CLEAN = [
        os.path.join(BASE_DIR, "output"),
        os.path.join(BASE_DIR, "pipeline_output"),
        os.path.join(BASE_DIR, "serving_models"),
        os.path.join(BASE_DIR, "metadata"),
        os.path.join(BASE_DIR, "beam_temp")
    ]
    
    print("Cleaning pipeline directories...")
    
    for directory in DIRECTORIES_TO_CLEAN:
        if os.path.exists(directory):
            print(f"Removing: {directory}")
            try:
                shutil.rmtree(directory, ignore_errors=True)
                time.sleep(1)
            except Exception as e:
                print(f"Could not remove {directory}: {e}")

if __name__ == "__main__":
    clean_pipeline()
    print("Cleanup completed! Ready for new pipeline run with Tuner.")