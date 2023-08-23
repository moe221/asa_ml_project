from google.oauth2 import service_account
from google.cloud import bigquery
import pandas as pd
import numpy as np
from colorama import Fore, Style
from pathlib import Path
import os




def get_data(keyword_ids: list, start_date: str, end_date: str) -> pd.DataFrame:

    # keyword id are string in BQ
    # campaign and ad groups are INTs

    cache_path = Path(os.path.join("./", "data", f"raw_data_{start_date}_{start_date}.csv"))

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path)

    else:
        print(Fore.BLUE + "\nLoad data from BigQuery server..." + Style.RESET_ALL)

        client = bigquery.Client()

        project_id = 'bigqpr'
        dataset_id = 'adobe'
        keyword_table_id = 'apple_asa_ios_app_all_keywords'
        impression_table_id = 'apple_asa_ios_app_impression_share'
        budget_table_id = 'apple_asa_ios_app_campaigns'

        keywords = ",".join(map(str, keyword_ids))

        query = f"""
                SELECT distinct key.*,
                imp.high_impression_share,
                imp.low_impression_share,
                imp.rank,
                imp.search_popularity,
                bud.daily_budget
                FROM `{project_id}.{dataset_id}.{keyword_table_id}` key
                LEFT JOIN `{project_id}.{dataset_id}.{impression_table_id}` imp
                ON key.app_adam_id = imp.app_adam_id
                AND key.date = imp.date
                AND key.keyword = imp.search_term
                AND key.country_or_region = imp.country_or_region
                LEFT JOIN `{project_id}.{dataset_id}.{budget_table_id}` bud
                ON key.date = bud.date
                AND key.app_adam_id = bud.app_adam_id
                AND key.campaign_id = bud.campaign_id
                AND key.country_or_region = bud.country_or_region
                WHERE key.date between'{start_date}' AND '{end_date}'
                AND key.keyword_id in ('{keywords}')
                """

        query_job = client.query(query)

        df = query_job.to_dataframe()


        if df.shape[0] > 1:
            df.to_csv(cache_path, header=True, index=False)



    print(f"✅ Data loaded, with shape {df.shape}")

    return df


def clean_data(df:pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:

    print(Fore.BLUE + "\nClean data..." + Style.RESET_ALL)

    # turn date to datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Remove duplicates
    df = df.drop_duplicates().copy()

    # drop all paused keywords ~25 rows
    df = df[df["keyword_status"]!="PAUSED"].copy()

    # add SOV share of voice
    df["share_of_voice"] = ((df['low_impression_share'] + df['high_impression_share'])/2 * 100)

    # fill na values of SOV with default [60 - 80]
    df["share_of_voice"] = df["share_of_voice"].fillna(60).copy()
    df["share_of_voice"] = df["share_of_voice"].map(int)

    df["keyword_id"] = df["keyword_id"].map(str)

    # keep only needed columns

    df = df[
            [   'date',
                'campaign_id',
                'campaign_name',
                'ad_group_id',
                'ad_group_name',
                'keyword_id',
                'keyword',
                'country_or_region',
                'match_type',
                'share_of_voice',
                'daily_budget',
                'keyword_status',
                'bid',
                'ttr',
                'taps',
                'local_spend',
                'impressions',
                'installs',
                'avg_cpt',
                'avg_cpa',
            ]
        ].copy()

    #sort by date then campaign id then ad group id, then keyword id
    df = df.sort_values(by=["date", "campaign_id", "ad_group_id", "keyword_id"])



    cache_path = Path(os.path.join("./", "data", f"clean_data_{start_date}_{start_date}.csv"))

    if df.shape[0] > 1:
        df.to_csv(cache_path, header=True, index=False)



    return df


if __name__ == "__main__":
    print("Fetching keyword data")
    keyword_ids = [1260718867]
    start_date = "2023-05-02"
    end_date = "2023-08-22"

    df_keyword = get_data(keyword_ids=keyword_ids, start_date=start_date, end_date=end_date)
    df = clean_data(df_keyword)

    print(df)
