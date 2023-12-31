from adgroup import AdGroup
from advertisers import Advertiser

class Client(Advertiser):

    def __init__(self, name, budget, cpa_goal, max_cpt_bid, avg_cpa, avg_ttr, avg_cvr, ad_group):
        """
        Initialize a Client object with the specified parameters.

        Parameters:
        - name: str, the name of the client
        - budget: float, the daily budget for the client
        - cpa_goal: float, the target Cost Per Acquisition
        - max_cpt_bid: float, the maximum Cost Per Tap bid
        - avg_cpa: float, the historical average Cost Per Acquisition
        - avg_ttr: float, the historical average Tap Through Rate
        - avg_cvr: float, the historical average Conversion Rate
        - ad_group: AdGroup object, the ad group associated with the client
        """
        self.name = name
        self.budget = budget
        self.cpa_goal = cpa_goal
        self.max_cpt_bid = max_cpt_bid
        self.avg_cpa = avg_cpa
        self.avg_ttr = avg_ttr
        self.avg_cvr = avg_cvr
        self.ad_group = ad_group
        self.bids_entered = 0
        self.impressions = 0
        self.taps=0
        self.installations=0
        self.spend=0

    def set_cpa_goal(self, new_cpa_goal):
        """
        Update the CPA goal for the client.

        Parameter:
        - new_cpa_goal: float, the new CPA goal
        """
        self.cpa_goal = new_cpa_goal

    def set_daily_budget(self, new_budget):
        """
        Update the daily budget for the client.

        Parameter:
        - new_budget: float, the new daily budget
        """
        self.budget = new_budget

    def set_max_cpt_bid(self, new_max_cpt_bid):
        """
        Update the maximum Cost Per Tap bid for the client.

        Parameter:
        - new_max_cpt_bid: float, the new maximum CPT bid
        """
        self.max_cpt_bid = new_max_cpt_bid

    def set_ad_group(self, new_ad_group):
        """
        Update the ad group associated with the client.

        Parameter:
        - new_ad_group: AdGroup object, the new ad group to associate with the client
        """
        self.ad_group = new_ad_group


    def reset_client(self):
        self.impressions=0
        self.bids_entered=0
        self.taps=0
        self.installations=0
        self.spend=0

    def __str__(self):
        return f"Client {self.name} with CPA Goal {self.cpa_goal}, Daily Budget {self.budget}, and Max CPT Bid {self.max_cpt_bid}"


if __name__ == "__main__":
    # Example of creating a Client object
    ad_group = AdGroup(5, 10.5, 1)
    client = Client(name="Client_1", budget=1000, cpa_goal=20, max_cpt_bid=2, avg_cpa=25, avg_ttr=0.1, avg_cvr=0.05, ad_group=ad_group)
    print(client)

    # Example of updating the CPA goal, daily budget, max CPT bid, and ad group
    client.set_cpa_goal(18)
    client.set_daily_budget(1200)
    client.set_max_cpt_bid(2.5)

    new_ad_group = AdGroup(7, 11.0, 2)
    client.set_ad_group(new_ad_group)

    print(client)
