import random
import uuid

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from advertisers import Advertiser
from gym import spaces
from matplotlib.animation import FuncAnimation
from probability_distributions import ProbabilityDistributions


class ASAEnv(gym.Env):
    def __init__(
        self,
        keywords,
        client,
        probabilities,
        auction_competitiveness_level,
        target
    ):
        super(ASAEnv, self).__init__()

        self.client = client
        self.available_keywords = keywords
        self.probabilities = probabilities
        self.auction_competitiveness_level = auction_competitiveness_level
        self.target = target

        self.hist = self.initiate_client_hist()

        self.advertisers = []
        self.is_done = False
        self.keyword_relevancy_level = 2
        self.window_size = 7
        self.current_step = 0

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(4 + self.window_size * 3,)
        )


    def initiate_client_hist(self):
        client_dict = {
            "day": [],
            "nr_searches": [],
            "nr_bids_entered": [],
            "impressions": [],
            "taps": [],
            "installs": [],
            "spend": [],
            "ttr_hist": [],
            "budget_remaining": [],
            "reward": [],
            "avg_cvr": [],
        }

        return pd.DataFrame(client_dict)

    def set_number_of_keywords(self, num_keywords):
        """
        Update the number of keywords in the ad group associated with the client.

        Parameter:
        - num_keywords: int, the new number of keywords in the ad group
        """
        self.client.ad_group.set_number_of_keywords(num_keywords)

    def set_auction_competitiveness(self, level):
        """
        Set the competitiveness level of the auction.

        Parameter:
        - level: int, the competitiveness level (1 for low, 2 for mid, 3 for high)
        """
        self.auction_competitiveness_level = level

    def set_client_cpa_goal(self, cpa_goal):
        """
        Update the CPA goal of the client.

        Parameter:
        - cpa_goal: float, the new CPA goal for the client
        """
        self.client.set_cpa_goal(cpa_goal)

    def set_keyword_relevancy_level(self, level):
        """
        Set the relevancy level of the keywords in the ad group associated with the client.

        Parameter:
        - level: int, the relevancy level (1 for low, 2 for mid, 3 for high)
        """
        self.keyword_relevancy_level = level
        self.client.ad_group.set_keyword_relevancy_level(level)

    def set_client_daily_budget(self, daily_budget):
        """
        Update the daily budget of the client.

        Parameter:
        - daily_budget: float, the new daily budget for the client
        """
        self.client.set_daily_budget(daily_budget)

    def step(self, action):

        # Apply action
        self.client.cpa_goal = action
        # reset client info
        self.client.reset_client_kpis()

        # avg_cpa
        if sum(self.hist["spend"][-self.window_size :]) == 0:
            self.client.avg_cpa = 0

        else:
            self.client.avg_cpa = sum(self.hist["installs"][-self.window_size :]) / sum(
                self.hist["spend"][-self.window_size :]
            )

        iterations = random.randint(300, 700)
        # print(f"Simulating {iterations} Keyword auctions\n")

        # Rub a single simulation
        for _ in range(iterations):
            # select keyword
            self.is_done = False
            self.advertisers = []
            self.keyword_name = self.select_random_keyword()
            self.auction_level = self.auction_competitiveness()
            # reset sample cvr, ttr and budget from historical data
            ttr_level, cvr_level, budget_level = random.sample(range(0, 3), 3)

            ttr_hist = (
                np.mean(self.hist["ttr_hist"][-3 :])
            )

            cvr_hist = (
                np.mean(self.hist["avg_cvr"][-3 :])
            )

            self.client.sample_historical_data(ttr_level=0,
                                               cvr_level=0,
                                               keyword=self.keyword_name,
                                               ttr_hist=ttr_hist,
                                               cvr_hist=cvr_hist)
            self.generate_bidders()
            self.client.set_max_cpt_bid(self.keyword_name)

            # Simulate auction round and update KPIs
            self.perform_auction()

        # print(f"Client won {self.client.impressions} with an average CPA of {self.client.avg_cpa}")

        # update hist
        self.update_client_hist(iterations=iterations)

        # Increase step
        self.current_step += 1

        # Calculate reward
        reward = self.calculate_reward()

        # Prepare observation
        observation = self._get_observation()

        # Check if done
        done = True

        return observation, reward, done, {}
        # same as before

    def calculate_reward(self, impressions_weight=100, cpa_weight=-50):
        impressions_moving_avg = (
            sum(self.hist["impressions"][-self.window_size :]) / self.window_size
        )
        cpa_moving_avg = (
            self.client.avg_cpa
        )  # sum(self.hist["avg_cpa"][-self.window_size:]) / self.window_size

        # Calculate reward
        # reward for selecting a cpa that is not lower than the max keyword bid
        # reward for decreasing the variance in cpa instead of the cpa itself (more spend control)
        # reward for gradual increase in impressions
        # No reward for 0 impressions
        # No reward for 0 cpa
        # USE ROI!! If ROI of the past 3 days is positive, then --> reward else no reward

        # Reward = Conversions * ConversionWeight - Costs * CostWeight

        reward = (
            +impressions_weight * impressions_moving_avg + cpa_weight * cpa_moving_avg
        )

        return reward

    def _get_observation(self):
        # Get data for current and previous steps
        if not self.hist.empty:
            start = max(0, self.current_step - self.window_size + 1)
            window_data = self.hist.iloc[start : self.current_step + 1]

            # Get current values
            current_values = window_data.iloc[-1][
                ["cpa_goal", "avg_cpa", "avg_ttr", "avg_cvr"]
            ].values

            # Get historical values
            cpa_history = window_data["avg_cpa"].values
            ttr_history = window_data["avg_ttr"].values
            cvr_history = window_data["avg_cvr"].values

            # Pad histories with zeros if not long enough
            cpa_history = np.pad(cpa_history, (self.window_size - len(cpa_history), 0))
            ttr_history = np.pad(ttr_history, (self.window_size - len(ttr_history), 0))
            cvr_history = np.pad(cvr_history, (self.window_size - len(cvr_history), 0))

            # Return as a single array
            return np.concatenate(
                [current_values, cpa_history, ttr_history, cvr_history]
            )

        return 0

    def reset(self):
        return self._get_observation()

    def select_random_keyword(self):
        keyword_name = random.choice(list(self.available_keywords))
        return keyword_name

    def auction_competitiveness(self):
        return self.auction_competitiveness_level

    def generate_bidders(self):
        # how many bidders will there be?
        nr_bidders = random.randint(5, 100)
        # print(f"creating {nr_bidders} bidders")

        if self.auction_level == 1:
            # create low level bidders
            index_1 = 1
            index_2 = 1
            index_3 = 1

        elif self.auction_level == 2:
            # create mid level bidders
            index_1 = 1
            index_2 = 1
            index_3 = 1

        elif self.auction_level == 3:
            # create mid high bidders
            index_1 = 3
            index_2 = 3
            index_3 = 1

        for _ in range(nr_bidders):
            name = str(uuid.uuid4())
            max_cpt_bid = self._sample_distributions(
                self.probabilities.bid_distributions, index_1, index_2, index_3
            )
            budget = self._sample_distributions(
                self.probabilities.bid_distributions, index_1, index_2, index_3
            )
            avg_cpa = self._sample_distributions(
                self.probabilities.cpa_distributions, index_1, index_2, index_3
            )
            avg_ttr = self._sample_distributions(
                self.probabilities.ttr_distributions, index_1, index_2, index_3
            )
            avg_cvr = self._sample_distributions(
                self.probabilities.cvr_distributions, index_1, index_2, index_3
            )

            # Each advertiser should have a CPI goal
            cpa_goal = 0  # random.uniform(0, avg_cpa * 1.2)
            keyword_relevance = random.uniform(0.0, 1.0)

        advertiser = Advertiser(
            name=name,
            budget=budget,
            cpa_goal=cpa_goal,
            max_cpt_bid=max_cpt_bid,
            avg_cpa=avg_cpa,
            avg_ttr=avg_ttr,
            avg_cvr=avg_cvr,
        )

        # add keyword relevance
        advertiser.keyword_relevance = keyword_relevance

        self.advertisers.append(advertiser)
        # print("bidder created")

    def get_auction_info(self):
        # get all info on auction participants and their kpis
        max_bid = self.get_highest_bid()

    def _sample_distributions(self, dist, index_1, index_2, index_3):
        return random.choice(
            [
                random.choice(dist[index_1]),
                random.choice(dist[index_2]),
                random.choice(dist[index_3]),
            ]
        )

    def get_highest_bid(self):
        bids = [advertiser.max_cpt_bid for advertiser in self.advertisers]
        return max(bids)

    def calculate_ad_rank(self, advertiser):
        # Simulate ad rank
        return advertiser.max_cpt_bid * advertiser.avg_ttr

    def calculate_quality_score(self, advertiser):
        # Weight the factors and calculate quality score
        relevance_weight = 0.8
        cvr_weight = 0.3
        bid_weight = 0.9
        quality_score = (
            (advertiser.keyword_relevance * relevance_weight)
            + (advertiser.avg_cvr * cvr_weight)
            + (advertiser.max_cpt_bid * bid_weight)
        )
        return quality_score

    def calculate_combination(self, advertiser):
        quality_score = self.calculate_quality_score(advertiser)
        ad_rank = self.calculate_ad_rank(advertiser)
        total_score = quality_score * ad_rank

        return total_score

    def enter_acution(self):
        bid_to_beat = self.get_highest_bid()
        # print("bid to beat", bid_to_beat)
        if self.client.max_cpt_bid >= bid_to_beat:
            return True

        return False

    def enter_high_acution(self):
        bid_to_beat = self.get_highest_bid()
        # print("bid to beat", bid_to_beat)
        if self.client.max_cpt_bid < bid_to_beat:
            return True

    def enter_low_acution(self):
        bid_to_beat = self.get_highest_bid()
        # print("bid to beat", bid_to_beat)
        if self.client.max_cpt_bid > bid_to_beat:
            return True

        return False

    def perform_auction(self):
        highest_score = 0

        # print("auction started")
        # print(self.client.cpa_goal, self.client.avg_cpa)

        enter_auction = True

        # threshold = np.random.uniform(low=0.95, high=1.3)  # Random threshold between 80% and 120%

        # Compute comfort zone limits
        lower_limit = self.client.avg_cpa * 0.7
        upper_limit = self.client.avg_cpa * 1.5

        if self.client.cpa_cap == 0:
            enter_auction = True

        elif lower_limit <= self.client.cpa_cap <= upper_limit:
            # print("client meets cpa criteria")
            # perfrom auction selection
            enter_auction = self.enter_acution()

        elif self.client.cpa_cap > upper_limit:
            # enter advertiser in auctions that are too high
            enter_auction = self.enter_high_acution()

        elif self.client.cpa_cap < upper_limit:
            # enter advertiser in auctions that are too low
            enter_auction = self.enter_low_acution()

        if enter_auction:
            # client enters the acution
            # print("client enters the auction")
            self.client.bids_entered += 1

            self.client.keyword_relevance = self.client.ad_group.keyword_relevancies[
                self.keyword_name
            ]
            # calculate score
            client_score = self.calculate_combination(self.client)

            bid_to_beat = self.get_highest_bid()

            # calculate highest score for all other bidders
            for advertiser in self.advertisers:
                combination = self.calculate_combination(advertiser)
                if combination > highest_score and advertiser.can_afford(bid_to_beat):
                    highest_score = combination

            if client_score >= highest_score and self.client.can_afford(bid_to_beat):
                # print(f"client won the auction with the price: {bid_to_beat}")
                self.client.bid_winner(min(self.client.max_cpt_bid, bid_to_beat))
            else:
                pass
            # print("lost auction")
        else:
            # client exits the auction
            # print("client exited the auction")
            pass

    def render(self, mode="human"):
        df = self.hist.copy()
        processed_df_list = [pd.DataFrame(columns=df.columns)]

        def animate(i):
            # Only process rows up to the current index, i
            if i < len(df):
                new_data = df.iloc[i]  # Get the row at position i
                processed_df_list[0] = pd.concat(
                    [processed_df_list[0], new_data.to_frame().T]
                )  # Use .to_frame().T to transform the Series into a DataFrame

            plt.cla()
            plt.plot(
                processed_df_list[0]["day"],
                processed_df_list[0]["cpa_goal"],
                color="red",
                label="CPA Goal",
            )
            plt.plot(
                processed_df_list[0]["day"],
                processed_df_list[0]["rolling_avg_cpa"],
                color="blue",
                label="Avg. CPA",
            )  # Plot second data series in red

            plt.xlabel("day")
            plt.ylabel("CPA")
            plt.title("ASA 120 Day Simulation - RL Model results")
            plt.legend()
            plt.tight_layout()

        ani = FuncAnimation(plt.gcf(), animate, interval=300)  # update every second
        plt.show()

    def update_client_hist(self, iterations):
        # Define the new row as a dictionary:
        new_row = {
            "day": self.current_step,
            "installs": self.client.installations,
            "taps": self.client.taps,
            "impressions": self.client.impressions,
            "cpa_goal": self.client.cpa_goal,
            "nr_bids_entered": self.client.bids_entered,
            "nr_searches": iterations,
            "spend": self.client.spend,
            "ttr_hist": self.client.avg_ttr,
            "budget_remaining": self.client.budget,
        }

        # Append the new row:
        # self.hist = self.hist.append(new_row, ignore_index=True)
        self.hist = pd.concat([self.hist, pd.DataFrame.from_records([new_row])])

        self.hist["avg_cpa"] = self.hist["spend"] / self.hist["installs"]
        self.hist["avg_ttr"] = self.hist["taps"] / self.hist["impressions"]
        self.hist["avg_cvr"] = self.hist["installs"] / self.hist["taps"]
        self.hist["rolling_avg_cpa"] = (
            self.hist["avg_cpa"].rolling(window=self.window_size).mean()
        )
        self.hist["rolling_avg_cvr"] = (
            self.hist["avg_cpa"].rolling(window=self.window_size).mean()
        )

        self.hist.fillna(0, inplace=True)
