import os
from pathlib import Path


import pandas as pd
import numpy as np
from tqdm import tqdm

from pathlib import Path
from colorama import Fore, Style
from google.cloud import bigquery

from sklearn.impute import SimpleImputer

from cpi_targets_ml.utils import fill_new_bid_amount
from cpi_targets_ml.params import GCP_PROJECT, LOCAL_DATA_PATH, CHUNK_SIZE


def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
        ) -> pd.DataFrame:

    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)

    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line

        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def clean_log_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing irrelevant columns
    """

    print(Fore.BLUE + "\nClean ASA script log data..." + Style.RESET_ALL)

    # Remove duplicates
    df = df.drop_duplicates().copy()

    # add datetime column
    df["date_dt"] = pd.to_datetime(df["date"])


    # drop error, actions, toaster_keyword_id, and overrides
    df = df.drop(columns=["errors",
                    "actions",
                    "toaster_keyword_id",
                    "overrides"])

    # impute missing share of voice with column median
    # TODO: impute on ad group level instead

    imputer = SimpleImputer(strategy="median") # Instanciate a SimpleImputer object with strategy of choice
    imputer.fit(df[['share_of_voice']]) # Call the "fit" method on the object
    df['share_of_voice'] = imputer.transform(df[['share_of_voice']]) # Call the "transform" method on the object


    # clean country code
    df["country_code"] = df["country_code"].map(str.strip)

    # add new features
    df["bid_changed"] = df["new_bid_amount"].apply(lambda x: False if np.isnan(x) else True)
    df["bid_diff"] = df["new_bid_amount"] - df["old_bid_amount"]
    df["bid_diff"].fillna(0, inplace=True)

    df["new_bid_amount"] = df.apply(lambda x: fill_new_bid_amount(x["old_bid_amount"], x["new_bid_amount"]), axis=1)

    print("✅ log data cleaned")

    return df


def clean_asa_bq_campaign_data(df: pd.DataFrame) -> pd.DataFrame:

    print(Fore.BLUE + "\nClean ASA BQ Campaign data..." + Style.RESET_ALL)


    # change column to date type
    df["date_dt"] = pd.to_datetime(df["date"])
    df["country_code"] = df["country_or_region"].map(str.strip)

    # keep only relevant columns
    df = df[
        ['date_dt',
            'app_adam_id',
            'campaign_id',
            'campaign_name',
            'country_code',
            'daily_budget',
            'avg_cpa',
            'avg_cpt',
            'conversion_rate',
            'installs',
            'installs_new_downloads',
            'installs_redownloads',
            'impressions',
            'local_spend',
            'taps',
            'ttr']
        ]

    print("✅ ASA campaign data cleaned")
    return df


def aggregate_keyword_level_data(df: pd.DataFrame) -> pd.DataFrame:

    # CPI targets are unique at the campaign and date level
    print(Fore.BLUE + "\nAggregate keyword data to campaign level..." + Style.RESET_ALL)

    group_columns = [
                    "app_name",
                    "country_code",
                    "campaign_id",
                    "campaign_name",
                    "date_dt"
                    ]

    df = df.groupby(group_columns).aggregate({'target_cpi': ['max', 'min', 'mean'],
                                              'keyword_id': 'nunique',
                                             'impressions': 'sum',
                                             'installs': 'sum',
                                             'spend': 'sum',
                                             'organic_rank': 'mean',
                                             'share_of_voice': 'mean',
                                             'old_bid_amount': 'mean'}).reset_index()


    # rename columns
    df.columns = [col.rstrip("_") for col in list(map('_'.join, df.columns.values))]

    # keep only relevant columns
    df = df[["country_code",
     "app_name",
     "campaign_id",
     "campaign_name",
     "date_dt",
     "target_cpi_mean"]].copy()

    print("✅ Keyword data aggregation done")

    return df


def join_log_and_campaign_data(
        df_targets: pd.DataFrame,
        df_cmp: pd.DataFrame
        ) -> pd.DataFrame:

    print(Fore.BLUE + "\nMerge datasets..." + Style.RESET_ALL)

    df = pd.merge(
        df_cmp,
        df_targets,
        on=["date_dt",
             "campaign_id",
             "country_code",
             "campaign_name"],
        how='inner'
        )

    print("✅ Data merging done \n")


    return df


def add_new_features(df: pd.DataFrame) -> pd.DataFrame:

    print(Fore.BLUE + "\nFeature engineering..." + Style.RESET_ALL)

    # Add overspend flag
    df["spend_vs_budget"] = df["daily_budget"] - df["local_spend"]
    df["is_overspend"] = df.apply(lambda x: False if (x["local_spend"]/x["daily_budget"]) < 0.9 else True, axis=1)

    # rolling aggregates
    # list of key metrics for which we will create rolling aggregate features

    rolling_metrics_sum = ['installs_redownloads',
                    'installs',
                    'taps',
                    'installs_new_downloads',
                    'impressions',
                    'local_spend',
                    'is_overspend',
                    'spend_vs_budget']

    rolling_metrics_mean = ['installs_redownloads',
                    'installs',
                    'taps',
                    'installs_new_downloads',
                    'impressions',
                    'avg_cpa',
                    'avg_cpt',
                    'conversion_rate',
                    'ttr',
                    'local_spend',
                    'spend_vs_budget']


    # create campaign id list
    campaign_ids = df["campaign_id"].unique()

    # loop over campaigns
    df_list = []
    for campaign in tqdm(campaign_ids):
        df_c = df[df["campaign_id"]==campaign].copy()
        # sort df by date
        df_c.sort_values(by="date_dt", ascending=True, inplace=True)

        # Creating rolling aggregate features (sum or mean over the last 7 days) for each specific campaign
        for metric in rolling_metrics_mean:
            df_c[f'{metric}_rolling_mean_7d'] = df_c[metric].rolling(window=7).mean()

        for metric in rolling_metrics_sum:
            df_c[f'{metric}_rolling_sum_7d'] =df_c[metric].rolling(window=7).sum()

        df_list.append(df_c)

    # re merge all datasets
    X_fe = pd.concat(df_list, axis=0)

    # drop all missing rows
    X_fe.dropna(inplace=True)
    X_fe.reset_index(drop=True, inplace=True)

    print("✅ Adding new features done \n")

    return X_fe


def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete
    print(f"✅ Data saved to bigquery, with shape {data.shape}")


if __name__ == "__main__":


    # load log data from bq
    start_date = "2023-04-01"
    end_date = "2023-08-11"

    project_id = 'bigqpr'
    dataset_id = 'dw'
    table_id = 'asa_bid_optimisation_log'
    query = f"""
    SELECT *
    FROM {project_id}.{dataset_id}.{table_id}
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    """

    cache_path = Path(os.path.join(LOCAL_DATA_PATH, "raw", f"query_ASAlog_{start_date}_{end_date}.csv"))


    get_data_with_cache(GCP_PROJECT, query, cache_path)
