from cpi_targets_ml.ml_logic.data import get_data_with_cache, clean_asa_bq_campaign_data, clean_log_data, aggregate_keyword_level_data, join_log_and_campaign_data, add_new_features
from colorama import Fore, Style
from dateutil.parser import parse
from pathlib import Path
from cpi_targets_ml.params import *





# load data
def preprocess(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
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
    df = join_log_and_campaign_data(data_agg_logs, data_clean_campaign)

    # add rolling aggregate features
    df = add_new_features(df)


    # feature scaling and tranformation

    pass


    # X = data_clean.drop("fare_amount", axis=1)
    # y = data_clean[["fare_amount"]]

    # X_processed = preprocess_features(X)

    # # Load a DataFrame onto BigQuery containing [pickup_datetime, X_processed, y]
    # # using data.load_data_to_bq()
    # data_processed_with_timestamp = pd.DataFrame(np.concatenate((
    #     data_clean[["pickup_datetime"]],
    #     X_processed,
    #     y,
    # ), axis=1))

    # load_data_to_bq(
    #     data_processed_with_timestamp,
    #     gcp_project=GCP_PROJECT,
    #     bq_dataset=BQ_DATASET,
    #     table=f'processed_{DATA_SIZE}',
    #     truncate=True
    # )

    # print("✅ preprocess() done \n")

# clean data

# preprocess data

# train and eval ML

# feature imprtance

# final ML

# test ML

# save model to GCP bucket

if __name__ == '__main__':
    preprocess(min_date='2023-04-01', max_date='2023-08-11')
