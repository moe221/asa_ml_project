import glob
import os
import time
from google.cloud import storage
from sklearn.pipeline import Pipeline
from params import *
import json
import joblib

os.chdir(os.path.dirname(os.path.abspath(__file__)))



def get_sa_json():
    with open("../certs/phiture_automation_sa.json") as sa:
        return json.loads(sa.read())


def load_model(stage="Production") -> Pipeline:
    print("here")
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'
    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print( f"\nLoad latest model from local registry...")

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(f"\nLoad latest model from disk...")

        latest_model = joblib.load(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(f"\nLoad latest model from GCS...")

        client = storage.Client.from_service_account_info(get_sa_json())
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model =joblib.load(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model

if __name__ == "__main__":
    load_model()
