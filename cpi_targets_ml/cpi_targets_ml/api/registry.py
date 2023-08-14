import glob
import os
from colorama import Fore, Style
import joblib


import mlflow
from mlflow.tracking import MlflowClient
from google.cloud import storage

LOCAL_REGISTRY_PATH="./models"
MODEL_TARGET="mlflow"
BUCKET_NAME="phiture-mlflow"

MLFLOW_TRACKING_USERNAME=""
MLFLOW_TRACKING_PASSWORD=""

MLFLOW_TRACKING_URI="https://mlflow-2oymxxy5da-ey.a.run.app/"
MLFLOW_EXPERIMENT="asa-cpi-targets-v2"
MLFLOW_EXPERIMENT_ID=5
MLFLOW_MODEL_NAME="cpi_target"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "../certs/phiture_automation_sa.json"
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD


def load_model(stage="Production"):
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
        local_model_paths = glob.glob(f"{LOCAL_REGISTRY_PATH}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(f"\nLoad latest model from disk...")

        latest_model = joblib.load(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        print(f"\nLoad latest model from GCS...")

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model =joblib.load(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model

    elif MODEL_TARGET == "mlflow":
        print(Fore.BLUE + f"\nLoad [{stage}] model from MLflow..." + Style.RESET_ALL)

        # Load model from MLflow
        model = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        try:
            model_versions = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[stage])
            model_uri = model_versions[0].source
            assert model_uri is not None
        except Exception as e:
            print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}", e)

            return None

        print(model_uri)
        model = mlflow.sklearn.load_model(model_uri=model_uri)

        print("✅ Model loaded from MLflow")
        return model
    else:
        return None


if __name__ == "__main__":
    load_model()
