import glob
import os
import time
import pickle
from colorama import Fore, Style
import joblib


import mlflow
from mlflow.tracking import MlflowClient
from google.cloud import storage


from cpi_targets_ml.params import *


def save_results(params: dict, metrics: dict, model: str) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if MODEL_TARGET == "mlflow":

        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

        # Set mlflow experiment where results will be saved
        mlflow.set_experiment = MLFLOW_EXPERIMENT
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        with mlflow.start_run(run_name=f"{timestamp}-{MLFLOW_MODEL_NAME}"):

            if params is not None:
                mlflow.log_params(params)
            if metrics is not None:
                mlflow.log_metrics(metrics)
            print("✅ Results saved on MLflow")



    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model=None, params=None, metrics=None, input_example: list = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, f"{timestamp}.pkl")
    joblib.dump(model, model_path)

    print("✅ Model saved locally")

    model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"asa_cpi_targets/models/{model_filename}")
    blob.upload_from_filename(model_path)
    print("✅ Model saved to GCS")


    if MODEL_TARGET == "mlflow":


        with mlflow.start_run(run_name=f"{timestamp}-{MLFLOW_MODEL_NAME}"):

            if params is not None:
                mlflow.log_params(params)
            if metrics is not None:
                mlflow.log_metrics(metrics)
            print("✅ Results saved on MLflow")

            mlflow.sklearn.autolog()
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MLFLOW_MODEL_NAME,
                input_example=input_example
            )

            print("✅ Model saved to MLflow")

        return None

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
        except:
            print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {stage}")

            return None

        print(model_uri)
        model = mlflow.sklearn.load_model(model_uri=model_uri)

        print("✅ Model loaded from MLflow")
        return model
    else:
        return None



def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment=MLFLOW_EXPERIMENT

        with mlflow.start_run(experiment_id=MLFLOW_EXPERIMENT_ID):
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper
