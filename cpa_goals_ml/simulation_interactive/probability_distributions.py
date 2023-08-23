import numpy as np
import pandas as pd
from scipy.stats import gamma, zscore


class ProbabilityDistributions:
    def __init__(self) -> None:
        self.df_keyword = pd.read_csv(
            "../raw_data/adobe_asa_keywords_sample.csv"
        )  # keyword_level()
        self.df_daily_budgets = pd.read_csv(
            "../raw_data/df_budget_daily.csv"
        )  # budgets()
        self.df_ad_groups = pd.read_csv(
            "../raw_data/df_ad_group.csv"
        )  # ad_group_level()

        self.hist_budgets = self.get_budgets()
        self.hist_bids = self.get_bid_distribution()
        self.hist_cpa = self.get_cpa_distribution()
        self.hist_ttr = self.get_ttr_distribution()
        self.hist_cvr = self.get_cvr_distribution()

        (
            self.similar_budgets,
            self.lower_budgets,
            self.higher_budgets,
        ) = self.generate_budget_dists()
        self.similar_ttr, self.lower_ttr, self.higher_ttr = self.generate_ttr_dists()
        self.similar_cvr, self.lower_cvr, self.higher_cvr = self.generate_cvr_dists()
        self.similar_cpa, self.lower_cpa, self.higher_cpa = self.generate_cpa_dists()
        (
            self.similar_bids,
            self.lower_bids,
            self.higher_bids,
        ) = self.generate_bids_dists()

        #### lists
        self.budget_distributions = [
            self.hist_budgets,
            self.similar_budgets,
            self.lower_budgets,
            self.higher_budgets,
        ]
        self.ttr_distributions = [
            self.hist_ttr,
            self.similar_ttr,
            self.lower_ttr,
            self.higher_ttr,
        ]
        self.cvr_distributions = [
            self.hist_cvr,
            self.similar_cvr,
            self.lower_cvr,
            self.higher_cvr,
        ]
        self.cpa_distributions = [
            self.hist_cpa,
            self.similar_cpa,
            self.lower_cpa,
            self.higher_cpa,
        ]
        self.bid_distributions = [
            self.hist_bids,
            self.similar_bids,
            self.lower_bids,
            self.higher_bids,
        ]

    def get_cvr_distribution(self):
        # remove all keywords that have 0 taps
        # (We are only interested in the probability of an install given a tap p(tap | install))

        df_keyword_sample_cvr = self.df_keyword[self.df_keyword["taps"] >= 1].copy()
        df_keyword_sample_cvr["cvr"] = df_keyword_sample_cvr.apply(
            lambda x: self._safe_divide(x["installs"], x["taps"]), axis=1
        )

        # drop cvr > 1
        df_keyword_sample_cvr = df_keyword_sample_cvr[df_keyword_sample_cvr["cvr"] <= 1]

        ## conversion rates per keyword in the past 2 months
        cvr_adobe_ad_groups_2_months = df_keyword_sample_cvr["cvr"]
        cvr_adobe_ad_groups_2_months.replace([np.inf, -np.inf], np.nan, inplace=True)

        return cvr_adobe_ad_groups_2_months

    def get_ttr_distribution(self):
        df_keyword_sample_ttr = self.df_keyword[
            self.df_keyword["impressions"] >= 1
        ].copy()
        df_keyword_sample_ttr["ttr"] = df_keyword_sample_ttr.apply(
            lambda x: self._safe_divide(x["taps"], x["impressions"]), axis=1
        )

        # drop ttr > 1
        df_keyword_sample_ttr = df_keyword_sample_ttr[df_keyword_sample_ttr["ttr"] <= 1]

        ## conversion rates per keyword in the past 2 months
        ttr_adobe_ad_groups_2_months = df_keyword_sample_ttr["ttr"]
        ttr_adobe_ad_groups_2_months.replace([np.inf, -np.inf], np.nan, inplace=True)

        return ttr_adobe_ad_groups_2_months

    def get_budgets(self):
        budgets_adobe_daily_campaign = self.df_daily_budgets["daily_budget"]

        return budgets_adobe_daily_campaign

    def get_cpa_distribution(self):
        ## installs per ad group in the past 6 months
        df_ad_group_active = self.df_ad_groups[
            self.df_ad_groups["ad_group_status"] != "PAUSED"
        ]
        df_ad_group_active_grouped = df_ad_group_active.groupby("ad_group_id").sum()

        cpa_adobe_ad_groups_6_months = (
            df_ad_group_active_grouped["installs"]
            / df_ad_group_active_grouped["local_spend"]
        ).sort_values(ascending=False)
        cpa_adobe_ad_groups_6_months.replace([np.inf, -np.inf], np.nan, inplace=True)

        # dropn NA
        cpa_adobe_ad_groups_6_months.dropna(inplace=True)

        return cpa_adobe_ad_groups_6_months

    def get_bid_distribution(self):
        bids_adobe_ad_groups_2_months = self.df_keyword["bid"]
        filtered_bids_series = bids_adobe_ad_groups_2_months[
            (np.abs(zscore(bids_adobe_ad_groups_2_months)) < 3)
        ]

        return filtered_bids_series

    def generate_competitor_budgets(
        self, existing_budgets, num_observations, shape_factor, scale_factor
    ):
        competitor_budgets = np.random.gamma(
            shape_factor, scale=scale_factor, size=num_observations
        )
        competitor_budgets = np.maximum(
            competitor_budgets, 0
        )  # Ensure non-negative budgets
        return competitor_budgets

    def generate_competitor_ttr(
        self, existing_ttr, num_observations, shape_factor, scale_factor
    ):
        competitor_ttr = np.random.gamma(
            shape_factor, scale=scale_factor, size=num_observations
        )
        competitor_ttr = np.maximum(competitor_ttr, 0)
        competitor_ttr = np.clip(competitor_ttr, 0, 1)
        return competitor_ttr

    def generate_competitor_cvr(
        self, existing_cvr, num_observations, mean_factor, std_factor
    ):
        existing_mean = np.mean(existing_cvr)
        existing_std = np.std(existing_cvr)
        competitor_cvr = np.random.normal(
            existing_mean * mean_factor, existing_std * std_factor, num_observations
        )
        competitor_cvr = np.maximum(competitor_cvr, 0)
        competitor_cvr = np.clip(competitor_cvr, 0, 1)  # Ensure non-negative budgets
        return competitor_cvr

    def generate_competitor_cpa(
        self, existing_values, num_observations, shape_factor, scale_factor
    ):
        competitor_values = np.random.gamma(
            shape_factor, scale=scale_factor, size=num_observations
        )
        competitor_values = np.maximum(
            competitor_values, 0
        )  # Ensure non-negative budgets
        return competitor_values

    def generate_competitor_bids(
        self, existing_values, num_observations, shape_factor, scale_factor
    ):
        competitor_values = np.random.gamma(
            shape_factor, scale=scale_factor, size=num_observations
        )
        competitor_values = np.maximum(
            competitor_values, 0
        )  # Ensure non-negative budgets
        return competitor_values

    def generate_budget_dists(self):
        # Parameters
        existing_budgets = self.hist_budgets  # Example existing budget distribution
        num_observations = len(existing_budgets)
        shape_factor = (
            1  # Adjust the shape factor to fit the desired distribution shape
        )
        scale_factor = (
            existing_budgets.median()
        )  # Adjust the scale factor to control the spread of the distribution

        # Generate competitor budget distributions
        similar_budgets = self.generate_competitor_budgets(
            existing_budgets, num_observations, shape_factor, scale_factor
        )
        lower_budgets = self.generate_competitor_budgets(
            existing_budgets, num_observations, shape_factor * 0.6, scale_factor * 0.6
        )
        higher_budgets = self.generate_competitor_budgets(
            existing_budgets, num_observations, shape_factor * 1.3, scale_factor * 1.3
        )

        return similar_budgets, lower_budgets, higher_budgets

    def generate_ttr_dists(self):
        ###### Tap-through-rate

        # Parameters
        existing_ttr = self.hist_ttr  # Example existing ttr distribution
        num_observations = len(self.hist_ttr)
        shape_factor, loc, scale_factor = gamma.fit(existing_ttr)

        # Generate competitor ttr distributions
        similar_ttr = self.generate_competitor_ttr(
            existing_ttr, num_observations, shape_factor, scale_factor
        )
        lower_ttr = self.generate_competitor_ttr(
            existing_ttr, num_observations, shape_factor * 0.6, scale_factor * 0.6
        )
        higher_ttr = self.generate_competitor_ttr(
            existing_ttr, num_observations, shape_factor * 1.3, scale_factor * 1.3
        )

        print(max(existing_ttr), max(similar_ttr), max(lower_ttr), max(higher_ttr))
        return similar_ttr, lower_ttr, higher_ttr

    def generate_cvr_dists(self):
        ###### Conversion rate (install rate)

        # Parameters
        existing_cvr = self.hist_cvr  # Example existing cvr distribution
        num_observations = len(self.hist_cvr)
        mean_factor = 1
        std_factor = 1

        # Generate competitor CVR distributions
        similar_cvr = self.generate_competitor_cvr(
            existing_cvr, num_observations, mean_factor, std_factor
        )
        lower_cvr = self.generate_competitor_cvr(
            existing_cvr, num_observations, mean_factor * 0.6, std_factor * 0.6
        )
        higher_cvr = self.generate_competitor_cvr(
            existing_cvr, num_observations, mean_factor * 1.1, std_factor * 1.1
        )

        return similar_cvr, lower_cvr, higher_cvr

    def generate_cpa_dists(self):
        ###### Average ad-group CPA

        # Parameters
        existing_cpa = self.hist_cpa  # Example existing CAP distribution
        num_observations = len(existing_cpa)
        shape_factor = (
            1.2  # Adjust the shape factor to fit the desired distribution shape
        )
        scale_factor = (
            existing_cpa.median()
        )  # Adjust the scale factor to control the spread of the distribution

        # Generate competitor budget distributions
        similar_cpa = self.generate_competitor_ttr(
            existing_cpa, num_observations, shape_factor, scale_factor
        )
        lower_cpa = self.generate_competitor_ttr(
            existing_cpa, num_observations, shape_factor * 0.6, scale_factor * 0.6
        )
        higher_cpa = self.generate_competitor_ttr(
            existing_cpa, num_observations, shape_factor * 1.3, scale_factor * 1.3
        )

        return similar_cpa, lower_cpa, higher_cpa

    def generate_bids_dists(self):
        ###### Bids
        # Parameters
        existing_bids = self.hist_bids  # Example existing bid distribution
        num_observations = len(existing_bids)
        shape_factor = (
            2  # Adjust the shape factor to fit the desired distribution shape
        )
        scale_factor = (
            existing_bids.median()
        )  # Adjust the scale factor to control the spread of the distribution

        # Generate competitor budget distributions
        similar_bids = self.generate_competitor_bids(
            existing_bids, num_observations, shape_factor, scale_factor
        )
        lower_bids = self.generate_competitor_bids(
            existing_bids, num_observations, shape_factor * 0.6, scale_factor * 0.6
        )
        higher_bids = self.generate_competitor_bids(
            existing_bids, num_observations, shape_factor * 1.5, scale_factor * 1.5
        )

        return similar_bids, lower_bids, higher_bids

    def _safe_divide(self, x, y):
        if y >= 1:
            return x / y
        return 0
