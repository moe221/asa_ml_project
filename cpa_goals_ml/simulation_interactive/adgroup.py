import numpy as np

class AdGroup:

    def __init__(self, num_keywords, keyword_bids, relevancy_level):
        """
        Initialize an AdGroup object with the given number of keywords,
        the corresponding bids for each keyword, and the relevancy level.

        Parameters:
        - num_keywords: int, the number of keywords in the ad group
        - keyword_bids: float or list, the bids for each keyword in the ad group
        - relevancy_level: int (1, 2, or 3), the relevancy level for the keywords
        """
        self.num_keywords = num_keywords

        # Generate random keyword names
        self.keywords = [f'keyword_{i}' for i in range(self.num_keywords)]

        # Check the type of keyword_bids and set the bids accordingly
        if isinstance(keyword_bids, (int, float)):
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
            raise ValueError("Invalid input for relevancy_level. Choose between 1, 2, or 3.")

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

    def __str__(self):
        return f"AdGroup with {self.num_keywords} keywords, bids: {self.keyword_bids}, and relevancies: {self.keyword_relevancies}"


if __name__ == "__main__":
    # Example with 5 keywords, a single bid amount for all keywords, and a relevancy level of 1
    ad_group1 = AdGroup(5, 10.5, 1)
    print(ad_group1)

    # Example with 5 keywords, a list of bid amounts, one for each keyword, and a relevancy level of 3
    ad_group2 = AdGroup(5, [10.5, 9.8, 11.2, 10.0, 10.5], 3)
    print(ad_group2)
