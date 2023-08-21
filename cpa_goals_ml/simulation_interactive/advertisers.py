import random
import numpy as np

class Advertiser:
    def __init__(self,
                 name,
                 budget,
                 cpa_goal,
                 max_cpt_bid,
                 avg_cpa,
                 avg_ttr,
                 avg_cvr):

        self.name = name
        self.budget = budget
        self.cpa_goal = cpa_goal
        self.max_cpt_bid = max_cpt_bid
        self.avg_cpa = avg_cpa
        self.avg_ttr = avg_ttr
        self.avg_cvr = avg_cvr
        self.impressions = 0
        self.taps = 0
        self.installations = 0
        self.bids_entered = 0
        self.spend = 0


    def rest_values(self):
        self.impressions = 0
        self.taps = 0
        self.installations = 0
        self.bids_entered = 0
        self.spend = 0


    def get_impression(self):
        self.impressions += 1
        return True

    def gets_tap(self):

        # Random event that generate a direct tap
        random_event = random.random() < 0.1  # 10% chance of a random event
        if random_event: # get tap
                self.taps += 1
                return True


        elif self.avg_ttr == 0:
            # randomly decide if advertiser gets tap
            if random.randint(0, 1) == 1:
                # get tap
                self.taps += 1
                return True

        else:
            # advertiser gets tap based on their ttr
            recieves_tap = random.choices([True, False], weights=[self.avg_ttr, 1 - self.avg_ttr], k=1)[0]
            if recieves_tap:
                # get tap
                self.taps += 1
                return True

        return False


    def gets_install(self):

        # If a suitable CPA goal is selected, chances of an installation are increased
        if self.satisfies_cpa_goal:
            random_bonus = random.choice(np.arange(0.1, 0.2, 0.05))
            random_event = random.random() < min(self.avg_cvr + random_bonus, 0.8)

            if random_event :
                # get install
                self.installations += 1
                return True

        # Random event that generate an install
        random_event = random.random() < 0.1  # 10% chance of a random event
        if random_event:
            # get install
            self.installations += 1

        elif self.avg_cvr == 0:
            # randomly decide if advertiser gets install
            if random.randint(0, 1) == 1:
                # get install
                self.installations += 1

        else:
            # advertiser gets install based on their cvr
            recieves_install = random.choices([True, False], weights=[self.avg_cvr, 1 - self.avg_cvr], k=1)[0]
            if recieves_install:
                # get install
                self.installations += 1


    def bid_winner(self, price):
        if self.get_impression():
            if self.gets_tap():
                # subtract price from bidget
                self.update_budget(price)
                # add to spend
                self.spend += price
                self.gets_install()


    def satisfies_cpa_goal(self):
        return self.avg_cpa <= self.cpa_goal

    def update_budget(self, price):
        self.budget -= price

    def can_afford(self, price):
        return price <= self.budget
