"""
Acts as a wrapper for Kraken. Used when I want to remote into a jupyter instance and exact trades
since Kraken doesn't have a mobile app

Author: Ian Q.

Notes:
        None
"""
import copy
import pprint
from numbers import Number
import krakenex

from kraken_trader.utils import get_usd_pair_listing, wallet_zero, remove_counter
from kraken_trader.api_key import key as CREDENTIALS


class Interface(object):
    '''
    Serves as a wrapper around the
    '''


    def __init__(self):
        self._exchange = krakenex.API(
            key=CREDENTIALS['API'],
            secret=CREDENTIALS['Private']
        )
        self._conversion_rate = {}

    def _limit_order(self, pair: str, bid: bool, price: Number, quantity: Number):
        """ Interface for orders. Limit orders used for laddering

        :param pair: string: pair to trade
        :param bid: whether to buy or sell
        :param price: price to buy or sell at
        :param quantity: quantity to buy or sell

        :return:

        """
        params = {
            'pair': pair,
            'type': 'buy' if bid else 'sell',
            'ordertype': 'limit',
            'price': str(price),
            'volume': str(quantity),
        }

        res = self._exchange.query_private('AddOrder', params)
        if res['error']:
            print(res['error'])
            return None
        else:
            return res


    ################################################
    #Wallet Propertie
    ################################################

    @property
    def weighted_price(self):
        """Prints the weighted cost for each currency. Only need to consider cost (fee wrapped inside already)

        E.g if you paid $C1 for V1 of X, then paid $C2 for V2 of X,
        weighted is

        ((C1 * V1) + (C2 * V2)) / (V1 + V2)

        :return:
        {
            'ZUSD': {'wallet': 371.3476, 'rate': 1.0},
            'XXRP': {'wallet': 2200.0, 'rate': 0.4571}
        }
        """


        # Setup
        wallet = copy.deepcopy(self.breakdown)
        trade_history = iter(self._exchange.query_private('TradesHistory')['result']['trades'].values())

        accum = {}
        for k in wallet.keys():
            accum[k] = {'volume': 0, 'notional': 0}

        while not wallet_zero(wallet):

            # Error checking - should only hit StopIteration if you don't have
            # too many transactions - thus causing you to exhaust... I think
            try:
                trade = next(trade_history)
            except StopIteration:
                break

            # Else, move on
            # Make sure that we don't accidentally send the wallet to negative

            base = remove_counter(trade['pair'])
            if wallet[base]['volume'] - float(trade['vol']) < 0:
                continue

            wallet[base]['volume'] -= float(trade['vol'])
            accum[base]['volume'] += float(trade['vol'])
            accum[base]['notional'] += float(trade['cost'])

        for k, v in accum.items():
            accum[k]['weighted'] = round(v['notional'] / (v['volume'] + 0.00000000000000000001), 2)
        return accum

    @property
    def breakdown(self):
        """

        :return:
        {
            'ZUSD': {'wallet': 371.3476, 'rate': 1.0},
            'XXRP': {'wallet': 2200.0, 'rate': 0.4571}
        }
        """
        for k, v in self._exchange.query_private('Balance')['result'].items():
            if k == 'ZUSD':
                self._conversion_rate[k] = {
                    'volume': float(v),
                    'rate': float(1)
                }
                continue
            # ELSE: not USD
            pair = get_usd_pair_listing(listing=k)
            ticker_data = self._exchange.query_public('Ticker', {'pair': pair})['result']
            for ticker_key in ticker_data.keys():
                # need to access it this way as pair != pair we access. E.g
                # we queried for ZECUSD, but accessing it we need to use XZECUSD
                rate = float(ticker_data[ticker_key]['c'][0])
                self._conversion_rate[k] = {
                    'volume': float(v),
                    'rate': float(rate)
                }
        return self._conversion_rate

    @property
    def value(self):
        return float(self._exchange.query_private('TradeBalance')['result']['eb'])

    @property
    def unallocated(self):
        usd_total = float(self._exchange.query_private('Balance')['result']['ZUSD'])
        open_orders = self._exchange.query_private('OpenOrders')

        breakdown = {}
        for _id, details in open_orders['result']['open'].items():
            data = details['descr']
            if data['type'] != 'buy':
                continue  # skip sell orders
            cur_amt = breakdown.get(data['pair'], 0)
            order_notional = (float(data['price']) * float(details['vol']))
            breakdown[data['pair']] = cur_amt + order_notional
        return {'ZUSD': usd_total - sum(breakdown.values()), 'allocattions': breakdown}