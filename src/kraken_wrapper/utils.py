import time
from typing import Callable

from kraken_wrapper.constants import USD_ALIAS


def rate_limiter(refresh_period: int) -> Callable:
    """Limits the refresh period. Not all funcs need 100% accuracy

    :param refresh_period:
        t: int
        Time in seconds

    :return:
        result of rate limited function (cached or un-cached)
    """
    def decorator(func):
        last_call = -1
        last_res = None

        def rate_limited_func(*args, **kwargs):
            nonlocal last_call, last_res
            if time.time() - last_call > refresh_period:
                last_call = time.time()
                last_res = func(*args, **kwargs)
                return last_res
            else:
                print('Old value!')
                return last_res
        return rate_limited_func
    return decorator


def remove_counter(pair):
    """removes ZUSD or USD from the XY pair

    :param pair:
    :return:
    """
    zusd_split = pair.split('ZUSD')
    if len(zusd_split) == 1:
        usd_split = pair.split('USD')
        if len(usd_split) != 2:
            raise ValueError('{} should be of length 2 but was {}'.format(usd_split, len(usd_split)))
        return usd_split[0]
    else:
        if len(zusd_split) != 2:
            raise ValueError('{} should be of length 2 but was {}'.format(zusd_split, len(zusd_split)))
        return zusd_split[0]


def wallet_zero(wallet):
    for _, v in wallet.items():
        if v['volume'] >= 0.001:
            return False
    return True


def get_usd_pair_listing(listing, alt_names=USD_ALIAS):
    return '{}{}'.format(listing, alt_names[listing])

