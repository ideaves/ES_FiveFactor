import FuturesSeries;
from FuturesSeries import ES_FuturesSeries;
from FuturesSeries import EC_FuturesSeries;
from FuturesSeries import US_FuturesSeries;
from FuturesSeries import GC_FuturesSeries;
from FuturesSeries import NQ_FuturesSeries;
from FuturesSeries import BTC_FuturesSeries;

class FuturesHelper: 

    def __init__(self):
        pass

    Global_Futures_Dictionary = {}


    def try_parse_fractional_price(self, price, denominator):
        try:
            parts = price.split('-');
            int_part = float(parts[0])
            frac_part = float(parts[1]) / denominator
        except:
            return None
        finally:
            return int_part + frac_part



    def populate_specific_contract_series(self, root_sym, start_date, num_contracts):
        if root_sym == "ES":
            self.Global_Futures_Dictionary[root_sym] = ES_FuturesSeries(root_sym, [], start_date, num_contracts)

        elif root_sym == "NQ":
            self.Global_Futures_Dictionary[root_sym] = NQ_FuturesSeries(root_sym, [], start_date, num_contracts)

        elif root_sym == "EC":
            self.Global_Futures_Dictionary[root_sym] = EC_FuturesSeries(root_sym, [], start_date, num_contracts)

        elif root_sym == "GC":
            self.Global_Futures_Dictionary[root_sym] = GC_FuturesSeries(root_sym, [], start_date, num_contracts)

        elif root_sym == "US":
            self.Global_Futures_Dictionary[root_sym] = US_FuturesSeries(root_sym, [], start_date, num_contracts)

        elif root_sym == "BTC":
            self.Global_Futures_Dictionary[root_sym] = BTC_FuturesSeries(root_sym, [], start_date, num_contracts)


    def populate_basic_contract_series(self, root_symbol, start_date):
        self.FuturesHelper.populate_specific_contract_series(self, root_symbol, start_date, 3);


    def get_front_contract(self, now, root_symbol):
        if root_symbol not in self.FuturesHelper.Global_Futures_Dictionary.keys():
           self.FuturesHelper.populate_basic_contract_series(self, root_symbol, now) 
        return min(self.FuturesHelper.Global_Futures_Dictionary[root_symbol]._contracts[0])

    def get_second_contract(self, now, root_symbol):
        if root_symbol not in self.FuturesHelper.Global_Futures_Dictionary.keys():
           self.FuturesHelper.populate_basic_contract_series(self, root_symbol, now) 
        return min(self.FuturesHelper.Global_Futures_Dictionary[root_symbol]._contracts[1])

    def get_nth_contract(self, now, seq, root_symbol):
        if root_symbol not in self.Global_Futures_Dictionary.keys() or self.Global_Futures_Dictionary[root_symbol]._FuturesSeries_contracts.length < seq:
           self.populate_specific_contract_series(root_symbol, now, seq) 
        return self.Global_Futures_Dictionary[root_symbol]._contracts[seq-1]



