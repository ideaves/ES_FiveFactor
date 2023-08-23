import calendar
import time
import dateutil
import datetime

class FuturesContractSpecifications:

    def __init__(self, symbol, month_abbrev, last_traded):
        self._symbol_root = symbol
        self._contract_month = month_abbrev
        self._non_commercial_last_trading_date = last_traded
        self._first_notice_date = None
        self._first_delivery_date = None
        self._last_notice_date = None
        self._expiration_date = None
        self._initial_margin = None
        self._maintenance_margin = None
        self._contract_size = None


