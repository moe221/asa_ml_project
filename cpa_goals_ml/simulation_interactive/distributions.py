import numpy as np
import pandas as pd
from scipy.stats import gamma, zscore
import matplotlib.pyplot as plt


class ProbabilityDistributions:

    def __init__(self, keyword_data, campaign_id, keyword_id) -> None:

        self.df_keyword = keyword_data[keyword_data["keyword_id"]==keyword_id]

        self.campaign_id = campaign_id
        self.keyword_id = keyword_id

        self.hist_budgets = self.get_budgets(campaign_id)
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

        df = self.df_keyword
        # remove all keywords that have 0 taps
        # (We are only interested in the probability of an install given a tap p(tap | install))

        df = df[df["taps"] >= 1].copy()

        df["cvr"] = df.apply(
            lambda x: self._safe_divide(x["installs"], x["taps"]), axis=1
        )

        # drop cvr > 1
        df = df[df["cvr"] <= 1]

        ## conversion rates per keyword in the past 2 months
        cvr_dist = df["cvr"]
        cvr_dist.replace([np.inf, -np.inf], np.nan, inplace=True)


        return cvr_dist

    def get_ttr_distribution(self):
        df = self.df_keyword
        df = df[
            df["impressions"] >= 1
        ].copy()
        df["ttr"] = df.apply(
            lambda x: self._safe_divide(x["taps"], x["impressions"]), axis=1
        )

        # drop ttr > 1
        df = df[df["ttr"] <= 1]

        ## conversion rates per keyword in the past 2 months
        ttr_dist = df["ttr"]
        ttr_dist.replace([np.inf, -np.inf], np.nan, inplace=True)


        return ttr_dist

    def get_budgets(self, campaign_id):

        df =  self.df_keyword[self.df_keyword["campaign_id"] == campaign_id]

        budgets_dist = df["daily_budget"]


        return budgets_dist

    def get_cpa_distribution(self):

        df = self.df_keyword


        df = df[df["installs"] >= 1].copy()

        df["cpa"] = df.apply(lambda x: self._safe_divide(x["local_spend"], x["installs"]), axis=1)

        # drop ttr > 1
        df = df[df["cpa"] <= 1]

        ## conversion rates per keyword in the past 2 months
        cpa_dist = df["cpa"]
        cpa_dist.replace([np.inf, -np.inf], np.nan, inplace=True)


        return cpa_dist

    def get_bid_distribution(self):

        df = self.df_keyword

        bids_dist = df["bid"]

        bids_dist = bids_dist[
            (np.abs(zscore(bids_dist)) < 3)
        ]

        return bids_dist

    def generate_budget_dists(self):

        factor = 1.2
        original_distribution = self.hist_budgets
        # Generate shifted distributions based on the factor
        similar_distribution = self.generate_noisy_distribution(original_distribution)
        similar_distribution = np.maximum(similar_distribution, 0)

        higher_distribution = self.generate_shifted_distribution(original_distribution, factor)
        higher_distribution = np.maximum(higher_distribution, 0)

        lower_distribution = self.generate_shifted_distribution(original_distribution, 1/factor)
        lower_distribution = np.maximum(lower_distribution, 0)

        #self.plot(original_distribution, similar_distribution, higher_distribution, lower_distribution)

        return similar_distribution, lower_distribution, higher_distribution

    def generate_ttr_dists(self):

        factor = 1.2
        original_distribution = self.hist_ttr

        # Generate shifted distributions based on the factor
        similar_distribution = self.generate_noisy_distribution(original_distribution)
        similar_distribution = np.clip(similar_distribution, 0, 1)

        higher_distribution = self.generate_shifted_distribution(original_distribution, factor)
        higher_distribution = np.clip(higher_distribution, 0, 1)

        lower_distribution = self.generate_shifted_distribution(original_distribution, 1/factor)
        lower_distribution = np.clip(lower_distribution, 0, 1)

        #self.plot(original_distribution, similar_distribution, higher_distribution, lower_distribution)

        return similar_distribution, lower_distribution, higher_distribution

    def generate_cvr_dists(self):

        factor = 1.2
        original_distribution = self.hist_cvr
        # Generate shifted distributions based on the factor
        similar_distribution = self.generate_noisy_distribution(original_distribution)
        similar_distribution = np.clip(similar_distribution, 0, 1)

        higher_distribution = self.generate_shifted_distribution(original_distribution, factor)
        higher_distribution = np.clip(higher_distribution, 0, 1)

        lower_distribution = self.generate_shifted_distribution(original_distribution, 1/factor)
        lower_distribution = np.clip(lower_distribution, 0, 1)

        #self.plot(original_distribution, similar_distribution, higher_distribution, lower_distribution)

        return similar_distribution, lower_distribution, higher_distribution

    def generate_cpa_dists(self):

        factor = 1.2
        original_distribution = self.hist_cpa
        # Generate shifted distributions based on the factor
        similar_distribution = self.generate_noisy_distribution(original_distribution)
        similar_distribution = np.clip(similar_distribution, 0, 1)

        higher_distribution = self.generate_shifted_distribution(original_distribution, factor)
        higher_distribution = np.clip(higher_distribution, 0, 1)

        lower_distribution = self.generate_shifted_distribution(original_distribution, 1/factor)
        lower_distribution = np.clip(lower_distribution, 0, 1)

        #self.plot(original_distribution, similar_distribution, higher_distribution, lower_distribution)

        return similar_distribution, lower_distribution, higher_distribution

    def generate_bids_dists(self):

        factor = 1.2
        original_distribution = self.hist_bids
        # Generate shifted distributions based on the factor
        similar_distribution = self.generate_noisy_distribution(original_distribution)
        similar_distribution = np.maximum(similar_distribution, 0)

        higher_distribution = self.generate_shifted_distribution(original_distribution, factor)
        higher_distribution = np.maximum(higher_distribution, 0)

        lower_distribution = self.generate_shifted_distribution(original_distribution, 1/factor)
        lower_distribution = np.maximum(lower_distribution, 0)

        #self.plot(original_distribution, similar_distribution, higher_distribution, lower_distribution)

        return similar_distribution, lower_distribution, higher_distribution

    def _safe_divide(self, x, y):
        if y >= 1:
            return x / y
        return 0

    def generate_noisy_distribution(self, data):
        noise = np.random.normal(0, 0.08, len(data))
        noisy_distribution = data + noise
        return noisy_distribution

    # Function to generate higher or lower distribution based on factor
    def generate_shifted_distribution(self, data, shift_factor):

        noise = np.random.normal(0, 0.1, len(data))
        max_shift = 1.0  # Limit the shift to prevent unrealistic values
        shifted_distribution = data * shift_factor + np.clip(np.random.normal(0, 0.1), -max_shift, max_shift) + noise

        return shifted_distribution

    def plot(self, original_distribution, similar_distribution, higher_distribution, lower_distribution):

        # Plot all distributions
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.hist(original_distribution, bins=30, color='blue')
        plt.title('Original Distribution')

        plt.subplot(2, 2, 2)
        plt.hist(similar_distribution, bins=30, color='green')
        plt.title('Similar Distribution')

        plt.subplot(2, 2, 3)
        plt.hist(higher_distribution, bins=30, color='red')
        plt.title('Higher Distribution')

        plt.subplot(2, 2, 4)
        plt.hist(lower_distribution, bins=30, color='purple')
        plt.title('Lower Distribution')

        plt.tight_layout()
        plt.show()
