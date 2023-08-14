import pandas as pd

from colorama import Fore, Style

# PIPELINE AND COLUMNTRANSFORMER
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn import set_config; set_config(display="diagram")

# IMPUTERS
from sklearn.impute import SimpleImputer
# SCALERS
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
# ENCODER
from sklearn.preprocessing import OneHotEncoder


def preprocess_features(X: pd.DataFrame, features: list = []):
    """
    Scikit-learn pipeline that transforms a cleaned dataset of shape
    into a preprocessed one of fixed shape

    Stateless operation: "fit_transform()" equals "transform()".
    """

    #select numerical features
    X_num = X.select_dtypes(exclude = ['object', 'datetime', 'bool'])
    X_cat = X.select_dtypes(include=['object', 'bool'])

    print(Fore.MAGENTA + "\nDataset has the following numerical features:" + Style.RESET_ALL)
    print(X_num.columns)

    print(Fore.MAGENTA + "\nDataset has the following categorical features:" + Style.RESET_ALL)
    print(X_cat.columns)

    # define features to scale using each scaling method

    features_robust = [col for col in ['installs_redownloads_rolling_sum_7d',
                   'installs_rolling_sum_7d',
                   'taps_rolling_sum_7d',
                   'installs_new_downloads_rolling_sum_7d',
                   'impressions_rolling_sum_7d',
                   'local_spend_rolling_sum_7d',
                   'installs_redownloads_rolling_mean_7d',
                   'installs_rolling_mean_7d',
                   'taps_rolling_mean_7d',
                   'installs_new_downloads_rolling_mean_7d',
                   'impressions_rolling_mean_7d',
                   'avg_cpa_rolling_mean_7d',
                   'avg_cpt_rolling_mean_7d',
                   'local_spend_rolling_mean_7d'] if col in features]


    features_standard = [col for col in ["conversion_rate",
                        "spend_vs_budget",
                        "spend_vs_budget_rolling_sum_7d",
                        "spend_vs_budget_rolling_mean_7d",
                        "conversion_rate_rolling_mean_7d"] if col in features]


    features_minmax = [col for col in ['daily_budget',
                        'avg_cpa',
                        'avg_cpt',
                        'installs',
                        'installs_new_downloads',
                        'installs_redownloads',
                        'impressions',
                        'local_spend',
                        'taps',
                        'ttr',
                        'ttr_rolling_mean_7d',
                        'is_overspend_rolling_sum_7d'] if col in features]

    # Checking that we didn't forget any numerical feature
    X_num.shape[-1] == len(features_robust + features_standard + features_minmax)


    num_transformer = make_pipeline(
                    ColumnTransformer(
                                [
                                    ("robust_scaler", RobustScaler(), features_robust),
                                    ("standard_scaler", StandardScaler(), features_standard),
                                    ("minmax_scaler", MinMaxScaler(), features_minmax)
                                ])
                )


    cat_transformer = make_pipeline(
                    OneHotEncoder(sparse=False,
                              handle_unknown="ignore",
                              drop='if_binary')
                )


    preprocessor = make_pipeline(
                ColumnTransformer([
                    ("num_transformer", num_transformer, make_column_selector(dtype_include = ["float64","int64"])),
                    ("cat_transformer", cat_transformer, make_column_selector(dtype_include = ["object", 'bool']))
                    ])
                )

    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)
    X_processed = pd.DataFrame(preprocessor.fit_transform(X))

    print("✅ X_processed, with shape", X_processed.shape)

    return X_processed, preprocessor
