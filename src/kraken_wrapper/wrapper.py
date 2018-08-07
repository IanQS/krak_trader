"""
Acts as a wrapper for Kraken. Used when I want to remote into a jupyter instance and exact trades
since Kraken doesn't have a mobile app

Author: Ian Q.

Notes:
        None
"""
import copy
from numbers import Number
import krakenex

from kraken_wrapper.utils import get_usd_pair_listing, wallet_zero, remove_counter
from api_key import key as CREDENTIALS


class Interface(object):
    """
    Serves as a wrapper around the Kraken API since it doesn't serve all the features I want
    """

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

    def bulk_limit_orders(self, low, allotment):
        pass

    def cancel_all_orders(self):
        open_orders = self._exchange.query_private('OpenOrders')['result']['open']
        for txid, _ in open_orders.items():
            self._exchange.query_private('CancelOrder', {'txid': txid})

    ################################################
    # Wallet Propertie
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
            accum[k]['weighted'] = round(v['notional'] / (v['volume'] + 0.00000000000000000001), 6)
        return accum

    @property
    def __all_deposits(self, asset: str ='USD') -> float:
        """ Returns all successful FIAT deposits (after fees applied)

        :param asset:
        :return:
        """
        total_deposited = 0
        for method in self._exchange.query_private('DepositMethods')['result']:
            method_ = method['method']
            deposits = self._exchange.query_private('DepositStatus', {'method': method_, 'asset': asset})
            for deposit in deposits['result']:
                if deposit['status'] == "Success":
                    total_deposited += (float(deposit['amount']) - float(deposit['fee']))
        return total_deposited


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
        """ Describes current value, total profit if we liquidate now, and pct-gain

        :return:
        """
        total_deposits = self.__all_deposits
        value = float(self._exchange.query_private('TradeBalance')['result']['eb'])

        profit = value - total_deposits
        pct_gain = profit / total_deposits

        return {'Deposits': total_deposits, 'value': value, 'profit': profit, '%-gain': pct_gain}

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
        return {'ZUSD': usd_total - sum(breakdown.values()), 'allocations': breakdown}
