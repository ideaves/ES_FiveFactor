import datetime;
from datetime import timedelta
from typing import Tuple;
from FuturesPrice import FuturesPrice;
from FuturesContractSpecifications import FuturesContractSpecifications;

class FuturesSeries:
    _symbol_root : str

    _month_number_mapping = {
        0: "F",
        1: "G",
        2: "H",
        3: "J",
        4: "K",
        5: "M",
        6: "N",
        7: "Q",
        8: "U",
        9: "V",
        10: "X",
        11: "Z"
    }

    _month_letter_mapping = {
        "F": 1,
        "G": 2,
        "H": 3,
        "J": 4,
        "K": 5,
        "M": 6,
        "N": 7,
        "Q": 8,
        "U": 9,
        "V": 10,
        "X": 11,
        "Z": 12
    }

    _contract_months = [
        "F",
        "G",
        "H",
        "J",
        "K",
        "M",
        "N",
        "Q",
        "U",
        "V",
        "X",
        "Z"
    ]


    
    _contracts: Tuple[FuturesContractSpecifications,FuturesPrice] = []



    _holidays = []



    def __init__(self, root_sym = "", holidays = {}, num_contracts = 0):
        pass


    def get_nth_weekday_of_month(self, month_start, seq, weekday):
        try_date: datetime = month_start
        try_n = 1
        if weekday is None or seq <= 0 or month_start.day != 1:
            return None
        while try_date.weekday() != weekday:
            try_date = try_date + timedelta(days=1)

        while try_n < seq:
            try_date = try_date + timedelta(days=7)
            try_n = try_n + 1
        while try_date.weekday() == 0 or try_date == 6 or try_date in self._holidays:
            try_date = try_date + timedelta(days=1)
        return try_date

    def get_n_business_days_before_months_end(self, month_start, num):
        year = month_start.year
        month = month_start.month
        days_in_month = 30
        if num < 0 or month_start.Day != 1:
            return None

        if month in { 1, 3, 5, 7, 8, 10, 12}:
            days_in_month = 31
        elif month == 2 and year % 4 == 0 and year % 100 == 0:
            days_in_month = 28
        elif month == 2 and year % 4 == 0:
            days_in_month = 29
        elif month == 2:
            days_in_month = 28

        end_month_date = datetime.datetime(year, month, days_in_month)

        business_days_back = 0
        if num == 0: # Still make sure it's the last good business day of the month
            while end_month_date.weekday() != 0 and end_month_date.weekday() != 6 and (self._holidays == None or self._holidays.Contains(end_month_date)):
                end_month_date = end_month_date + timedelta(days=-1)
            while business_days_back < num: # Normal case for num > 0
                end_month_date = end_month_date + timedelta(days=-1)
            if end_month_date.weekday() != 0 and end_month_date.weekday() != 6 and (self._holidays == None or end_month_date in self._holidays):
                business_days_back = business_days_back + 1

        return end_month_date


    def get_last_good_business_day_of_month(self, month_start, num, weekday):
        year = month_start.year
        month = month_start.month
        if num < 0 or month_start.day != 1:
            return None

        days_in_month = 30
        if month in { 1, 3, 5, 7, 8, 10, 12}:
            days_in_month = 31
        elif month == 2 and year % 4 == 0 and year % 100 == 0:
            days_in_month = 28
        elif month == 2 and year % 4 == 0:
            days_in_month = 29
        elif month == 2:
            days_in_month = 28

        end_month_date = datetime.datetime(year, month, days_in_month)
        while end_month_date.weekday() != weekday:
            end_month_date = end_month_date + timedelta(days=-1)
        while end_month_date.weekday() == 0 or end_month_date.weekday() == 6 or (self._holidays != None and end_month_date in self._holidays):
            end_month_date = end_month_date + timedelta(days=-1);

        return end_month_date


    def get_last_business_day_of_prior_month(self, month_start):
            if month_start.day != 1:
                return None

            end_month_date = month_start + timedelta(days=-1)
            while end_month_date.weekday() == 0 or end_month_date.weekday() == 6 or (self._holidays != None and end_month_date in self._holidays):
                end_month_date = end_month_date + timedelta(days=-1)

            return end_month_date


    def enumerate_contract_series(self, root_symbol, start_date, n_contracts_out):
        yr = start_date.year
        mo = start_date.month
        front_contract_letter = ""
        front_contract_number = -1

        for k_num, v_mo in self._month_number_mapping.items():
            if mo <= k_num + 1 and v_mo in self._FuturesSeries_contract_months:
                front_contract_letter = v_mo
                front_contract_number = k_num + 1
                break

        contract_year = yr
        contract_month = front_contract_number
        contract_letter = front_contract_letter

        series_number = 0
        while series_number < n_contracts_out:
            start_of_month = datetime.datetime(contract_year, contract_month, 1)
            last_trading_date = self.get_last_non_commercial_trading_date(start_of_month)
            if last_trading_date is None:
                return
            if last_trading_date < start_date:
                n_contracts_out = n_contracts_out + 1
                if contract_letter == self._FuturesSeries_contract_months[len(self._FuturesSeries_contract_months)-1]:
                    contract_year = contract_year + 1
                    contract_letter = self._contract_months[0]
                    contract_month = self._month_letter_mapping[contract_letter]
                else:
                    contract_letter = self._FuturesSeries_contract_months[self._FuturesSeries_contract_months.index(contract_letter)+1]
                    contract_month = self._month_letter_mapping[contract_letter]
                continue

            spec = FuturesContractSpecifications(root_symbol, contract_letter, last_trading_date)
            price = FuturesPrice("{}{}{}".format(root_symbol, contract_letter, contract_year % 100))
            spec_price_pair: Tuple[FuturesContractSpecifications,FuturesPrice] = (spec, price)
            self._contracts.append(spec_price_pair)
            series_number = series_number + 1

            if contract_letter == self._FuturesSeries_contract_months[len(self._FuturesSeries_contract_months)-1]:
                contract_year = contract_year + 1
                contract_letter = self._FuturesSeries_contract_months[0]
            else:
                contract_letter = self._FuturesSeries_contract_months[self._FuturesSeries_contract_months.index(contract_letter) + 1]

            contract_month = self._month_letter_mapping[contract_letter]



class ES_FuturesSeries(FuturesSeries):

    _day_of_week = 3
    _seq_in_month = 3

    def __init__(self, root_sym, holidays, start_date, num_contracts):
        super(FuturesSeries, self).__init__()
        self._FuturesSeries_contract_months = ["H", "M", "U", "Z"]
        self._holidays = holidays
        self._symbol_root = root_sym
        self._contracts : Tuple[FuturesContractSpecifications,FuturesPrice] = []
        self.enumerate_contract_series(root_sym, start_date, num_contracts)

    def get_last_non_commercial_trading_date(self, month_start):
        return self.get_nth_weekday_of_month(month_start, 3, 3)


class NQ_FuturesSeries(FuturesSeries):

    _day_of_week = 3
    _seq_in_month = 3

    def __init__(self, root_sym, holidays, start_date, num_contracts):
        super(FuturesSeries, self).__init__()
        self._FuturesSeries_contract_months = ["H", "M", "U", "Z"]
        self._holidays = holidays
        self._symbol_root = root_sym
        self._contracts : Tuple[FuturesContractSpecifications,FuturesPrice] = []
        self.enumerate_contract_series(root_sym, start_date, num_contracts)

    def get_last_non_commercial_trading_date(self, month_start):
        return self.get_nth_weekday_of_month(month_start, 3, 3)


class EC_FuturesSeries(FuturesSeries):

    _day_of_week = 3
    _seq_in_month = 3

    def __init__(self, root_sym, holidays, start_date, num_contracts):
        super(FuturesSeries, self).__init__()
        self._FuturesSeries_contract_months = ["H", "M", "U", "Z"]
        self._holidays = holidays
        self._symbol_root = root_sym
        self._contracts : Tuple[FuturesContractSpecifications,FuturesPrice] = []
        self.enumerate_contract_series(root_sym, start_date, num_contracts)

    def get_last_non_commercial_trading_date(self, month_start):
        return self.get_nth_weekday_of_month(month_start, 3, 3)


class GC_FuturesSeries(FuturesSeries):

    def __init__(self, root_sym, holidays, start_date, num_contracts):
        super(FuturesSeries, self).__init__()
        self._FuturesSeries_contract_months = ["G", "J", "M", "Q", "Z"]
        self._holidays = holidays
        self._symbol_root = root_sym
        self._contracts : Tuple[FuturesContractSpecifications,FuturesPrice] = []
        self.enumerate_contract_series(root_sym, start_date, num_contracts)

    def get_last_non_commercial_trading_date(self, month_start):
        return self.get_last_business_day_of_prior_month(month_start)


class US_FuturesSeries(FuturesSeries):

    _day_of_week = 3
    _seq_in_month = 3

    def __init__(self, root_sym, holidays, start_date, num_contracts):
        super(FuturesSeries, self).__init__()
        self._FuturesSeries_contract_months = ["H", "M", "U", "Z"]
        self._holidays = holidays
        self._symbol_root = root_sym
        self._contracts : Tuple[FuturesContractSpecifications,FuturesPrice] = []
        self.enumerate_contract_series(root_sym, start_date, num_contracts)

    def get_last_non_commercial_trading_date(self, month_start):
        return self.get_nth_weekday_of_month(month_start, 3, 3)



class BTC_FuturesSeries(FuturesSeries):

    _day_of_week = 5

    def __init__(self, root_sym, holidays, start_date, num_contracts):
        super(FuturesSeries, self).__init__()
        self._FuturesSeries_contract_months = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
        self._holidays = holidays
        self._symbol_root = root_sym
        self._contracts : Tuple[FuturesContractSpecifications,FuturesPrice] = []
        self.enumerate_contract_series(root_sym, start_date, num_contracts)

    def get_last_non_commercial_trading_date(self, month_start):
        return self.get_last_good_business_day_of_month(month_start, 1, self._day_of_week)

