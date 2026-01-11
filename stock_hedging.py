import numpy as np
import pandas as pd

class StockHedging:
    """
    Handles the bookkeeping of a delta-hedged option position along a single stock price path.

    The class takes precomputed stock prices, option prices, and option deltas, and applies
    discrete-time hedging with transaction costs and interest on the cash balance. It does
    not simulate prices or compute Greeks.

    Two output resolutions are supported:
    - `assemble_data()`: results only at hedge rebalancing times
    - `assemble_data_fulltime()`: results at every stock price timestep

    PnL is measured relative to the initial cash balance grown at the risk-free rate.
    """

    def __init__(self, T, options_amount, options_price, options_delta, stock_prices, time_steps=252, cash_balance=0, r=0, transaction_fee_bps=0, integer_stocks=False):
        """
        Initialize a single-path delta-hedging setup.

        :param T: Time to maturity (in years).
        :param options_amount: Number of option contracts held.
        :param options_price: Option prices along the stock price path.
        :param options_delta: Option deltas along the stock price path.
        :param stock_prices: Stock prices along the path.
        :param time_steps: Number of hedge rebalancing times.
        :param cash_balance: Initial cash balance.
        :param r: Risk-free interest rate.
        :param transaction_fee_bps: Transaction cost in basis points of traded notional.
        :param integer_stocks: If True, hedge positions are rounded to integers.
        """
        self.T = T
        self.time_steps = time_steps
        self.options_amount = options_amount
        self.r = r
        self.transaction_fee_bps = transaction_fee_bps
        self.integer_stocks = integer_stocks

        #State
        self.cash_balance = cash_balance
        self.stock_prices = np.asarray(stock_prices)
        self.options_price = np.asarray(options_price)
        self.options_delta = np.asarray(options_delta)

    def assemble_data(self):
        """
        Assemble a dataFrame of the hedging procedure, tracking only the times when hedgning takes place.
        PnL is calculated as difference from the initial cash amount invested at the risk free rate.
        :return: return a pd dataframe containing columns:
                "Option Price",
                "Stock Price"
                "Option Delta",
                "Options held",
                "Shares held",
                "Hedge in this timestep",
                "Transaction costs",
                "Hedging cashflow this timestep",
                "Total Cashflow":
                "Cash Balance",
                "Total Portfolio Value",
                "Total Pnl"
        """
        time_index = np.linspace(0, len(self.stock_prices) - 1, self.time_steps, dtype=int)
        delta_t = self.T / (self.time_steps - 1)

        delta_data = self.options_delta[time_index]
        stock_price_data = self.stock_prices[time_index]
        options_price_data = self.options_price[time_index]

        #Total delta is the delta position of the underlying asset used for hedging
        if self.integer_stocks:
             total_hedge_adjustment = np.rint(-self.options_amount * delta_data).astype(int)
        else:
             total_hedge_adjustment = -self.options_amount * delta_data

        # Calculating the adjustments necessary at each time index
        current_hedge_adjustment = np.empty_like(total_hedge_adjustment)
        current_hedge_adjustment[0] = total_hedge_adjustment[0]

        current_hedge_adjustment[1:] = total_hedge_adjustment[1:] - total_hedge_adjustment[:-1]

        #Calculating transaction costs using bps of notional
        rate = self.transaction_fee_bps / 10000
        transaction_costs = np.abs(current_hedge_adjustment) * stock_price_data * rate

        #Cash used to buy or sell stocks during each timestep
        adjustment_cashflow = - current_hedge_adjustment * stock_price_data - transaction_costs

        #Calculating the total balance and applying interest for each time step
        growth = np.exp(self.r * delta_t)

        cashflow = adjustment_cashflow.astype(float).copy()
        cashflow[0] = cashflow[0] - self.options_amount * options_price_data[0] - np.abs(self.options_amount) * options_price_data[0] * rate

        #Discounting cashflow back to the starting time and dividing by the discount rate (1/g^t)
        #to reach the compounded interest over the period.
        disc = np.ones_like(cashflow,dtype=float)
        disc[1:] = 1.0 / growth
        disc = np.cumprod(disc)

        #total balance in terms of time = 0
        total_cash_balance_0 = self.cash_balance + np.cumsum(cashflow * disc)

        #grow back to current time:
        total_cash_balance_current = total_cash_balance_0 / disc

        #Final balance after liquidating all assets
        total_portfolio_value = (total_cash_balance_current + stock_price_data * total_hedge_adjustment + self.options_amount * options_price_data).astype(float)

        #Excess return compared to initial cash balance invested at risk-free rate
        total_pnl = total_portfolio_value - self.cash_balance / disc

        #Construct DataFrame
        data_collection = {"Option Price": options_price_data,
                           "Stock Price": stock_price_data,
                           "Option Delta": delta_data,
                           "Options held": np.ones_like(options_price_data) * self.options_amount,
                           "Shares held": total_hedge_adjustment,
                           "Hedge in this timestep": current_hedge_adjustment,
                           "Transaction costs": transaction_costs,
                           "Hedging cashflow this timestep": adjustment_cashflow,
                           "Total Cashflow": cashflow,
                           "Cash Balance": total_cash_balance_current,
                           "Total Portfolio Value": total_portfolio_value,
                           "Total Pnl": total_pnl,
                           }

        hedging_overview = pd.DataFrame(data_collection, index=time_index)
        hedging_overview.index.name = "Days"

        return hedging_overview


    def assemble_data_fulltime(self):
        '''
        Assemble a dataFrame of the hedging procedure, tracking only the times when hedgning takes place.
        PnL is calculated as difference from the initial cash amount invested at the risk free rate.
        :return: return a pd dataframe containing columns:
                "Option Price",
                "Stock Price"
                "Option Delta",
                "Options held",
                "Shares held",
                "Hedge in this timestep",
                "Transaction costs",
                "Hedging cashflow this timestep",
                "Total Cashflow":
                "Cash Balance",
                "Total Portfolio Value",
                "Total Pnl"
        '''
        #Calculating times for hedging, total number of days and
        hedge_index = np.linspace(0, len(self.stock_prices) - 1, self.time_steps, dtype=int)
        n = self.stock_prices.size
        delta_t = self.T / (n - 1)

        delta_data = self.options_delta
        stock_price_data = self.stock_prices
        options_price_data = self.options_price

        # Total delta is the delta position of the underlying asset used for hedging
        total_hedge_adjustment_unique = np.empty_like(hedge_index)
        if self.integer_stocks:
            total_hedge_adjustment_unique = np.rint(-self.options_amount * delta_data[hedge_index]).astype(int)
        else:
            total_hedge_adjustment_unique = -self.options_amount * delta_data[hedge_index]

        #Count each interval length in total_hedge_adjustments
        count = np.diff(np.r_[hedge_index, n])

        total_hedge_adjustment = np.repeat(total_hedge_adjustment_unique, count)



        # Calculating the adjustments necessary at each time index
        current_hedge_adjustment = np.empty_like(total_hedge_adjustment)
        current_hedge_adjustment[0] = total_hedge_adjustment[0]

        current_hedge_adjustment[1:] = total_hedge_adjustment[1:] - total_hedge_adjustment[:-1]

        # Calculating transaction costs using bps of notional
        rate = self.transaction_fee_bps / 10000
        transaction_costs = np.abs(current_hedge_adjustment) * stock_price_data * rate

        # Cash used to buy or sell stocks during each timestep
        adjustment_cashflow = - current_hedge_adjustment * stock_price_data - transaction_costs

        # Calculating the total balance and applying interest for each time step
        growth = np.exp(self.r * delta_t)

        cashflow = adjustment_cashflow.astype(float).copy()
        cashflow[0] = cashflow[0] - self.options_amount * options_price_data[0] - np.abs(self.options_amount) * \
                      options_price_data[0] * rate

        # Discounting cashflow back to the starting time and dividing by the discount rate (1/g^t)
        # to reach the compounded interest over the period.
        disc = np.ones_like(cashflow, dtype=float)
        disc[1:] = 1.0 / growth
        disc = np.cumprod(disc)

        # total balance in terms of time = 0
        total_cash_balance_0 = self.cash_balance + np.cumsum(cashflow * disc)

        # grow back to current time:
        total_cash_balance_current = total_cash_balance_0 / disc

        # Final balance after liquidating all assets
        total_portfolio_value = (
                total_cash_balance_current + stock_price_data * total_hedge_adjustment + self.options_amount * options_price_data).astype(
            float)

        # Excess return compared to initial cash balance invested at risk-free rate
        total_pnl = total_portfolio_value - self.cash_balance / disc

        # Construct DataFrame
        data_collection = {"Option Price": options_price_data,
                           "Stock Price": stock_price_data,
                           "Option Delta": delta_data,
                           "Options held": np.ones_like(options_price_data) * self.options_amount,
                           "Shares held": total_hedge_adjustment,
                           "Hedge in this timestep": current_hedge_adjustment,
                           "Transaction costs": transaction_costs,
                           "Hedging cashflow this timestep": adjustment_cashflow,
                           "Total Cashflow": cashflow,
                           "Cash Balance": total_cash_balance_current,
                           "Total Portfolio Value": total_portfolio_value,
                           "Total Pnl": total_pnl,
                           }

        hedging_overview = pd.DataFrame(data_collection)
        hedging_overview.index.name = "Days"

        return hedging_overview


