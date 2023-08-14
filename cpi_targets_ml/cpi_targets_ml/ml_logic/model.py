
from colorama import Fore, Style


from sklearn.pipeline import Pipeline, make_pipeline
# LINEAR MODELS
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor
# NEIGHBORS
from sklearn.neighbors import KNeighborsRegressor
# TREES AND ENSEMBLE METHODS
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
# SVM
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance


import pandas as pd
import numpy as np

def cpi_regression_models(regression_model, preprocessor):

    piped_regressor = make_pipeline(preprocessor, regression_model)
    return piped_regressor

def run_model_benchmarking(X:pd.DataFrame, y:pd.DataFrame,  preprocessor, cv=5):


    models = [LinearRegression(),
          SGDRegressor(),
          KNeighborsRegressor(),
          SVR(kernel = "linear"),
          SVR(kernel = "poly", degree = 2),
          SVR(kernel = "poly", degree = 3),
          SVR(kernel = "rbf"),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          AdaBoostRegressor(),
          GradientBoostingRegressor()
]

    models_names = ["linear_regression",
                    "sgd_regressor",
                    "kneighbors_regressor",
                    "SVR_linear",
                    "SVR_poly_two",
                    "SVR_poly_three",
                    "SVR_rbf",
                    "decision_tree_regressor",
                    "random_forest_regressor",
                    "ada_boost_regressor",
                    "gradient_boosting_regressor"
    ]

    print(Fore.MAGENTA + f"\nRunning benchmarks with {len(models_names)} models" + Style.RESET_ALL)


    X_train, X_test, y_train, y_test = train_test_split(X, y)

    different_test_scores = []
    all_cv_scores = []
    mean_cv_score = []
    std_cv_score = []

    for model_name, model in zip(models_names, models):

        temp_piped_regressor = cpi_regression_models(model, preprocessor)

        ## add cvs
        print(Fore.MAGENTA + f"Cross validating {model_name} model.."+ Style.RESET_ALL)
        cv_scores = cross_val_score(temp_piped_regressor,
                            X_train,
                            y_train,
                            scoring='r2',
                            cv=cv)

        # Calculate mean and standard deviation of the cross-validation scores

        cv_scores = [round(score, 2) for score in cv_scores]
        cv_mean = round(np.mean(cv_scores),2)
        cv_std = round(np.std(cv_scores),2)

        print("Cross-Validation Scores:", cv_scores)
        print("Mean CV Score:", cv_mean)
        print("CV Score Standard Deviation:", cv_std)

        all_cv_scores.append(cv_scores)
        mean_cv_score.append(cv_mean)
        std_cv_score.append(cv_std)

        temp_piped_regressor.fit(X_train, y_train)
        different_test_scores.append(temp_piped_regressor.score(X_test, y_test))

    comparing_regression_models_cpi = pd.DataFrame(list(zip(models_names,
                                                            different_test_scores,
                                                            all_cv_scores,
                                                            mean_cv_score,
                                                            std_cv_score)),
                                                    columns =['model_name',
                                                              'R2_score',
                                                              'cross-validation-scores',
                                                              'mean-cross-validation-scores',
                                                              'std-cross-validation-scores'])

    comparing_regression_models_cpi["tarining_set_size"] = X_train.shape[0]
    comparing_regression_models_cpi["cross-validation-folds"] = cv


    df_model_bench = round(comparing_regression_models_cpi.sort_values(by = "R2_score", ascending = False),2)

    return df_model_bench


def train_best_model(X_train, X_test, y_train, y_test,  preprocessor, model):

    ml_model = cpi_regression_models(model, preprocessor)
    ml_model.fit(X_train, y_train)

    # Making predictions on the testing set
    y_pred = ml_model.predict(X_test)

    # Calculating evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mae, mse, r2


    print(Fore.MAGENTA + f"Best model traning results.."+ Style.RESET_ALL)

    print("R-squared:", r2)
    print("MAE:", mae)
    print("MSE:", mse)

    return ml_model, mae, mse, r2


def get_important_features(X:pd.DataFrame, y:pd.DataFrame,trained_model):

    print(Fore.MAGENTA + f"Starting feature permutation importance..."+ Style.RESET_ALL)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # feature importance

    permutation_score = permutation_importance(trained_model,
                                            X_train,
                                            y_train,
                                            n_repeats=50,
                                            random_state=42) # Perform Permutation

    importance_df = pd.DataFrame(np.vstack((X_train.columns,
                                            permutation_score.importances_mean)).T) # Unstack results

    importance_df.columns=['feature','score decrease'] # Change column names

    importance_df.sort_values(by="score decrease", ascending = False) # Order by importance

    importance_df = importance_df.sort_values(by="score decrease", ascending = False) # Order by importance

    features = importance_df[importance_df["score decrease"]>=0.02]["feature"].values

    print(Fore.MAGENTA + f"Feature importance:"+ Style.RESET_ALL)
    print(importance_df)

    return features
