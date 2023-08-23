import numpy as np
from distributions import ProbabilityDistributions
from data import get_data, clean_data
import random

class AdGroup:
    def __init__(self,
                 relevancy_level,
                 hist_dataset,
                 keyword_bids = None):
        """
        Initialize an AdGroup object with the given number of keywords,
        the corresponding bids for each keyword, and the relevancy level.

        Parameters:
        - num_keywords: int, the number of keywords in the ad group
        - keyword_bids: float or list, the bids for each keyword in the ad group
        - relevancy_level: int (1, 2, or 3), the relevancy level for the keywords
        """

        self.hist_dataset = hist_dataset

        # Generate random keyword names
        # self.keywords = [f"keyword_{i}" for i in range(self.num_keywords)]
        self.keywords = self.hist_dataset["keyword_id"].unique()
        self.num_keywords = len(self.keywords)

        self.get_historical_data()

        # Check the type of keyword_bids and set the bids accordingly

        if keyword_bids == None:
            self.keyword_bids = {keyword: random.choice(list(self.keyword_bid_dict[keyword][0])) for keyword in self.keywords}

        elif isinstance(keyword_bids, (int, float)):
            self.keyword_bids = {keyword: keyword_bids for keyword in self.keywords}

        elif isinstance(keyword_bids, list) and len(keyword_bids) == self.num_keywords:
            self.keyword_bids = dict(zip(self.keywords, keyword_bids))
        else:
            raise ValueError("Invalid input for keyword_bids")

        # Set the relevancy scores based on the relevancy level
        self.relevancy_level = relevancy_level
        self.keyword_relevancies = self._generate_relevancy_scores()

    def _generate_relevancy_scores(self):
        """
        Generate relevancy scores for the keywords based on the relevancy level.
        """
        if self.relevancy_level == 1:
            return {keyword: np.random.uniform(0.0, 0.4) for keyword in self.keywords}
        elif self.relevancy_level == 2:
            return {keyword: np.random.uniform(0.4, 0.7) for keyword in self.keywords}
        elif self.relevancy_level == 3:
            return {keyword: np.random.uniform(0.7, 1.0) for keyword in self.keywords}
        else:
            raise ValueError(
                "Invalid input for relevancy_level. Choose between 1, 2, or 3."
            )

    def update_keyword_bid(self, keyword, new_bid):
        """
        Update the bid for a specific keyword.

        Parameters:
        - keyword: str, the name of the keyword whose bid we want to update
        - new_bid: float, the new bid amount for the specified keyword
        """
        if keyword in self.keyword_bids:
            self.keyword_bids[keyword] = new_bid
        else:
            print("Keyword not found in this AdGroup")


    def get_historical_data(self):
    # Historical dataset needs to be on a keyword level - daily
    # Use historical dataset to create a distribution of kpis
    # sample these distributions each time a bid is run to determine probabilities of tap and install
    # keep track of actual kpis that are genreated through the simulation
    # each time step, calculate new kpis by sampling and using actual kpis with weights
    # --> predict kpis by giving weight to recent observations

    # get unique keyword ids

        keyword_budget_dict = {}
        keyword_ttr_dict = {}
        keyword_cvr_dict = {}
        keyword_cpa_dict = {}
        keyword_bid_dict = {}

        for keyword in self.keywords:

            campaign = self.hist_dataset[self.hist_dataset["keyword_id"]==keyword]["campaign_id"].unique()[0]
            prob_dist = ProbabilityDistributions(self.hist_dataset, campaign, keyword)

            keyword_budget_dict[keyword] = prob_dist.budget_distributions
            keyword_ttr_dict[keyword] = prob_dist.ttr_distributions
            keyword_cvr_dict[keyword] = prob_dist.cvr_distributions
            keyword_cpa_dict[keyword] = prob_dist.cpa_distributions
            keyword_bid_dict[keyword] = prob_dist.bid_distributions


        self.keyword_budget_dict = keyword_budget_dict
        self.keyword_ttr_dict = keyword_ttr_dict
        self.keyword_cvr_dict = keyword_cvr_dict
        self.keyword_cpa_dict = keyword_cpa_dict
        self.keyword_bid_dict = keyword_bid_dict

    def __str__(self):
        return f"AdGroup with {self.num_keywords} keywords, bids: {self.keyword_bids}, and relevancies: {self.keyword_relevancies}"


if __name__ == "__main__":

    print("Fetching keyword data")
    keyword_ids = [1260718867]
    start_date = "2023-05-02"
    end_date = "2023-08-22"

    df_keyword = get_data(keyword_ids=keyword_ids, start_date=start_date, end_date=end_date)
    df = clean_data(df_keyword, start_date=start_date, end_date=end_date)
    # Example with 5 keywords, a single bid amount for all keywords, and a relevancy level of 1
    ad_group1 = AdGroup(relevancy_level=1, hist_dataset=df)
    print(ad_group1)
