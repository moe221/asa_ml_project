import numpy as np

def fill_new_bid_amount(old_bid, new_bid):

    if np.isnan(new_bid):
        return old_bid

    return new_bid
