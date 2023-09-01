from cpi_targets_ml.ml_logic.data import get_data_with_cache, clean_asa_bq_campaign_data, clean_log_data, aggregate_keyword_level_data, join_log_and_campaign_data, add_new_features, load_data_to_bq
from cpi_targets_ml.ml_logic.preprocessor import preprocess_features
from cpi_targets_ml.ml_logic.registry import save_results, mlflow_run, save_model, load_model
from cpi_targets_ml.ml_logic.model import run_model_benchmarking, train_best_model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from colorama import Fore, Style
from dateutil.parser import parse
from pathlib import Path
from cpi_targets_ml.params import *
from tqdm import tqdm




# load data
def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)


    # Query raw data from BigQuery using `get_data_with_cache`
    start_date = parse(min_date).strftime('%Y-%m-%d')
    end_date = parse(max_date).strftime('%Y-%m-%d')


    # get ASA script logs data
    project_id = 'bigqpr'
    dataset_id = 'dw'
    table_id = 'asa_bid_optimisation_log'
    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{table_id}
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_ASAlog_{start_date}_{end_date}.csv")

    data_query_logs = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # get ASA BQ campaign data

    project_id = 'bigqpr'
    dataset_id = 'adobe'
    table_id = 'apple_asa_ios_app_campaigns'
    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{table_id}
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_ASA_BQ_{start_date}_{end_date}.csv")

    data_query_asa_bq = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # clean data
    data_clean_logs = clean_log_data(data_query_logs)
    data_clean_campaign = clean_asa_bq_campaign_data(data_query_asa_bq)

    # aggregate data
    data_agg_logs = aggregate_keyword_level_data(data_clean_logs)

    # join log + BQ campaign data to get cpi targets
    data_clean = join_log_and_campaign_data(data_agg_logs, data_clean_campaign)

    # add rolling aggregate features
    data_clean = add_new_features(data_clean)


    data_clean = data_clean.drop(columns=['date_dt',
                   'app_adam_id',
                   'campaign_id',
                   'campaign_name'])

    print("✅ Removed campaign, app, and date info from data \n")


    # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # using data.load_data_to_bq()

    # load_data_to_bq(
    #     data_clean,
    #     gcp_project=GCP_PROJECT,
    #     bq_dataset=BQ_DATASET,
    #     table='asa_cpi_targets_model_preprocessed_data',
    #     truncate=True
    # )

    print("✅ preprocess() done \n")

def benchmark(min_date:str = '2009-01-01',
                max_date:str = '2015-01-01',

):
    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset (which should be ordered by date)
    - Store training results and model weights

    Return val_mae as a float
    """

    print(Fore.BLUE + "\nLoading preprocessed training data..." + Style.RESET_ALL)

    # Query data from BigQuery using `get_data_with_cache`
    start_date = parse(min_date).strftime('%Y-%m-%d')
    end_date = parse(max_date).strftime('%Y-%m-%d')


    table_id = "asa_cpi_targets_model_preprocessed_data"

    query = f"""
    SELECT *
    FROM {GCP_PROJECT}.{BQ_DATASET}.{table_id}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None


    # feature scaling and tranformation
    X = data_processed.drop("target_cpi_mean", axis=1)
    y = data_processed["target_cpi_mean"]

    X_processed, preprocessor = preprocess_features(X, X.columns)

    # create benchmarks
    benchmarking_data = run_model_benchmarking(X, y, preprocessor)

    for _, row in tqdm(benchmarking_data.iterrows()):
    ## test save model to ml flow
        val_r2 = row["R2_score"]
        cross_validation_scores = row['cross-validation-scores']
        mean_cross_validation_scores = row['mean-cross-validation-scores']
        std_cross_validation_scores = row['std-cross-validation-scores']
        training_set_size = row["tarining_set_size"]
        model_name = row["model_name"]
        cv_folds = row['cross-validation-folds']


        params = dict(
            context="benchmark",
            training_set_size=training_set_size,
            model_name = model_name,
            cv_folds=cv_folds

        )

        # Save results on the hard drive using taxifare.ml_logic.registry
        save_results(params=params,
                    metrics=dict(r2=val_r2,
                                mean_cross_validation_score=mean_cross_validation_scores,
                                std_cross_validation_score=std_cross_validation_scores),
                    model=model_name)

    print("✅ Model benchmarking done - Results saved in MLflow\n")

def train_model(min_date:str = '2009-01-01',
                max_date:str = '2015-01-01',
                model_name="Random_Forest_Regressor"

):

    print(Fore.BLUE + "\nLoading preprocessed training data..." + Style.RESET_ALL)

    # Query data from BigQuery using `get_data_with_cache`
    start_date = parse(min_date).strftime('%Y-%m-%d')
    end_date = parse(max_date).strftime('%Y-%m-%d')


    table_id = "asa_cpi_targets_model_preprocessed_data"

    query = f"""
    SELECT *
    FROM {GCP_PROJECT}.{BQ_DATASET}.{table_id}
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=True
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None


    # feature scaling and tranformation
    X = data_processed.drop("target_cpi_mean", axis=1)
    y = data_processed["target_cpi_mean"]

    X_reduced = X[['avg_cpa_rolling_mean_7d',
                    'ttr_rolling_mean_7d',
                    'avg_cpt_rolling_mean_7d',
                    'impressions',
                    'conversion_rate_rolling_mean_7d',
                    'avg_cpt',
                    'daily_budget']].copy()


    X_processed, preprocessor = preprocess_features(X_reduced, X_reduced.columns)


    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

    trained_model, mae, mse, r2 = train_best_model(X_train, X_test, y_train, y_test, preprocessor, RandomForestRegressor())

    params = dict(
    context="train",
    training_set_size=X_train.shape[0],
    model_name = model_name,

    )

    input_example = X[['avg_cpa_rolling_mean_7d',
                       'ttr_rolling_mean_7d',
                       'avg_cpt_rolling_mean_7d',
                       'conversion_rate_rolling_mean_7d',
                       'impressions',
                       'avg_cpt',
                       'daily_budget']].iloc[[0]]

    print(input_example)



    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=trained_model,
                params=params,
                metrics=dict(r2=r2,
                            mae=mae,
                            mse=mse),
                input_example=input_example,
                )

if __name__ == '__main__':
    preprocess(min_date='2023-04-01', max_date='2023-08-24')
    benchmark(min_date='2023-04-01', max_date='2023-08-24')
    train_model(min_date='2023-04-01', max_date='2023-08-24', model_name="Random_Forest_Regressor")
    load_model()
    pass
