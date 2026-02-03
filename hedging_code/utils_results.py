import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm

def get_plot(df_plot, start, end):
    if end is None:
        df_plot = df_plot.iloc[start:end]
    else:
        df_plot = df_plot.iloc[start:end]
    #2206, 2245
    prices = []
    x_vals = []

    open_prices = []
    close_prices = []
    open_x = []
    close_x = []

    # Build the curves and markers
    for i, row in df_plot.iterrows():

        # Line: alternating Open/Close to form the zigzag price path
        prices.append(row['Open'])
        prices.append(row['Close'])
        x_vals.append(row.name)
        x_vals.append(row.name)

        # Markers
        open_prices.append(row['Open'])
        close_prices.append(row['Close'])
        open_x.append(row.name)
        close_x.append(row.name)

    # Main price line
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=prices,
        mode='lines',
        name='Price Line',
        line=dict(color='lightblue'),
        hovertemplate='Index: %{x}<br>Price: %{y:.2f}<extra></extra>'
    ))

    # Open price markers (small red dots)
    fig.add_trace(go.Scatter(
        x=open_x,
        y=open_prices,
        mode='markers',
        name='Open',
        marker=dict(color='green', size=6, symbol='circle'),
        hovertemplate='Open: %{y:.2f}<extra></extra>'
    ))

    # Close price markers (small green dots)
    fig.add_trace(go.Scatter(
        x=close_x,
        y=close_prices,
        mode='markers',
        name='Close',
        marker=dict(color='red', size=6, symbol='circle'),
        hovertemplate='Close: %{y:.2f}<extra></extra>'
    ))

    # # Add subtle closing price overlay line (on top, thin, neutral color)
    # fig.add_trace(go.Scatter(
    #     x=df_plot.index,
    #     y=df_plot['Close'],
    #     mode='lines',
    #     name='Close Trace',
    #     line=dict(color='gray', width=1, dash='dot'),  # Thin gray dashed line
    #     hoverinfo='skip',  # Hide hover to avoid clutter
    #     showlegend=False   # Hide from legend for minimalism
    # ))

    # Layout settings
    fig.update_layout(
        title="📊 UPS Price Evolution (Open vs Close)",
        xaxis_title="Time Index",
        yaxis_title="Price",
        template="plotly_dark",
        height=500,
        width=1200,
        font=dict(size=14),
        margin=dict(l=50, r=50, t=60, b=40)
    )

    fig.show()
    return fig



def black_scholes_call_price(
    pricing_date,
    maturity_date,
    spot,
    strike,
    rate,       # risk-free rate (annualized, decimal)
    div_yield,  # dividend yield (annualized, decimal)
    vol         # volatility (annualized, decimal)
):


    pricing_date = np.datetime64(pricing_date, 'D')
    maturity_date = np.datetime64(maturity_date, 'D')
    # Convert date strings to datetime
    if isinstance(pricing_date, str):
        pricing_date = pd.to_datetime(pricing_date).date()
    if isinstance(maturity_date, str):
        maturity_date = pd.to_datetime(maturity_date).date()

    # Compute business days between dates
    bus_days = np.busday_count(pricing_date, maturity_date)
    T = bus_days / 252  # Convert to fraction of a year

    if T <= 0:
        return max(spot - strike, 0.0)  # Option has expired

    # d1 and d2
    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)

    # Black-Scholes call price
    call_price = spot * np.exp(-div_yield * T) * norm.cdf(d1) - strike * np.exp(-rate * T) * norm.cdf(d2)
    return call_price, bus_days

def compute_d1_d2(pricing_date, maturity_date, spot, strike, rate, div_yield, vol):
    pricing_date = np.datetime64(pricing_date, 'D')
    maturity_date = np.datetime64(maturity_date, 'D')
    T = np.busday_count(pricing_date, maturity_date) / 252

    if T <= 0:
        return 0, 0, 0

    d1 = (np.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return d1, d2, T

def call_delta(pricing_date, maturity_date, spot, strike, rate, div_yield, vol):
    d1, _, T = compute_d1_d2(pricing_date, maturity_date, spot, strike, rate, div_yield, vol)
    return np.exp(-div_yield * T) * norm.cdf(d1)

def call_gamma(pricing_date, maturity_date, spot, strike, rate, div_yield, vol):
    d1, _, T = compute_d1_d2(pricing_date, maturity_date, spot, strike, rate, div_yield, vol)
    return (np.exp(-div_yield * T) * norm.pdf(d1)) / (spot * vol * np.sqrt(T))

def call_vega(pricing_date, maturity_date, spot, strike, rate, div_yield, vol):
    d1, _, T = compute_d1_d2(pricing_date, maturity_date, spot, strike, rate, div_yield, vol)
    return spot * np.exp(-div_yield * T) * norm.pdf(d1) * np.sqrt(T)

def get_realized_volatility(data, end_date):
    # Compute log returns
    prices = data.loc[:end_date]['Close']
    log_returns = np.log(prices / prices.shift(1)).dropna()

    # Annualize standard deviation
    realized_vol = log_returns.std() * np.sqrt(252)

    return realized_vol


def get_hedgigng_dataframe(DF_OPTIONS, TICKER_DATA_DATE, notional):

    delta_hedge_daily = [{'date':0, 
                      'delta': 'NaN',
                      'number_of_shares_bought_that_day': 0, 
                      'number_of_shares_sold_that_day': 0, 

                      'price_per_share_that_day': 0, 
                      'mean_buy_value' : 0,
                      'mean_sell_value': 0, 

                      'total_number_bought':0,
                      'total_number_sold': 0,

                      'total_number' : 0,

                      'total_buy_value':0,
                      'total_sell_value': 0,
                      'delta_hedging_cost': 0
                      }]

    res = pd.DataFrame(delta_hedge_daily)

    for i, row in DF_OPTIONS.iterrows():
        dict_day = {}
        index = row.name
        dict_day['date'] = index
        delta = row['Delta']
        dict_day['delta'] = delta
        delta_shares = round(notional * delta, 2)

        old_delta_shares = delta_hedge_daily[-1]['total_number']

        to_buy = delta_shares - old_delta_shares

        price = TICKER_DATA_DATE.loc[index, 'Close']
        if to_buy>0:
            number_of_shares_bought_that_day = to_buy
            number_of_shares_sold_that_day = 0

        else:
            number_of_shares_bought_that_day = 0
            number_of_shares_sold_that_day = - to_buy

        dict_day['number_of_shares_bought_that_day'] = number_of_shares_bought_that_day
        dict_day['number_of_shares_sold_that_day'] = number_of_shares_sold_that_day
        dict_day['price_per_share_that_day'] = price


        old_mean_buy = delta_hedge_daily[-1]['mean_buy_value']
        old_mean_sell = delta_hedge_daily[-1]['mean_sell_value']

        new_mean_buy = (delta_hedge_daily[-1]['total_number_bought'] * old_mean_buy + number_of_shares_bought_that_day * price)/(number_of_shares_bought_that_day + delta_hedge_daily[-1]['total_number_bought'])
        new_mean_sell = (delta_hedge_daily[-1]['total_number_sold'] * old_mean_sell + number_of_shares_sold_that_day * price)/(number_of_shares_sold_that_day + delta_hedge_daily[-1]['total_number_sold'])
        if np.isnan(new_mean_sell):
            new_mean_sell = 0

        dict_day['mean_buy_value'] = new_mean_buy
        dict_day['mean_sell_value'] = new_mean_sell

        total_number_bought = delta_hedge_daily[-1]['total_number_bought'] + number_of_shares_bought_that_day
        total_number_sold = delta_hedge_daily[-1]['total_number_sold'] + number_of_shares_sold_that_day

        dict_day['total_number_bought'] = total_number_bought
        dict_day['total_number_sold'] = total_number_sold


        total_number = total_number_bought - total_number_sold

        dict_day['total_number'] = total_number

        total_buy_value = total_number_bought * new_mean_buy
        total_sell_value = total_number_sold * new_mean_sell

        dict_day['total_buy_value'] = total_buy_value
        dict_day['total_sell_value'] = total_sell_value

        total_value = -(total_sell_value - total_buy_value)
        dict_day['delta_hedging_cost'] = total_value
        
        delta_hedge_daily.append(dict_day)
        
        res = pd.concat([res, pd.DataFrame([dict_day]) ], axis = 0)

    res = res.set_index('date')
    return res.iloc[1:]

def normal_hedging(row, notional, TICKER_DATA_DATE, delta_hedge_daily):
        dict_day = {}
        index = row.name
        dict_day['date'] = index
        dict_day['Schedule'] = 'Close'
        delta = row['Delta']
        dict_day['delta'] = delta
        delta_shares = round(notional * delta, 2)

        
        old_delta_shares = delta_hedge_daily['total_number'].iloc[0]
        to_buy = delta_shares - old_delta_shares
        
        price = TICKER_DATA_DATE.loc[index, 'Close']

        
        if to_buy>0:
            number_of_shares_bought_that_day = to_buy
            number_of_shares_sold_that_day = 0

        else:
            number_of_shares_bought_that_day = 0
            number_of_shares_sold_that_day = - to_buy

        dict_day['number_of_shares_bought_that_day'] = number_of_shares_bought_that_day
        dict_day['number_of_shares_sold_that_day'] = number_of_shares_sold_that_day
        dict_day['price_per_share_that_day'] = price

        
        
        old_mean_buy = delta_hedge_daily['mean_buy_value'].iloc[0]
        old_mean_sell = delta_hedge_daily['mean_sell_value'].iloc[0]

        new_mean_buy = (delta_hedge_daily['total_number_bought'].iloc[0] * old_mean_buy + number_of_shares_bought_that_day * price)/(number_of_shares_bought_that_day + delta_hedge_daily['total_number_bought'].iloc[0])
        new_mean_sell = (delta_hedge_daily['total_number_sold'].iloc[0] * old_mean_sell + number_of_shares_sold_that_day * price)/(number_of_shares_sold_that_day + delta_hedge_daily['total_number_sold'].iloc[0])
        if np.isnan(new_mean_sell):
            new_mean_sell = 0

        dict_day['mean_buy_value'] = new_mean_buy
        dict_day['mean_sell_value'] = new_mean_sell

        total_number_bought = delta_hedge_daily['total_number_bought'] + number_of_shares_bought_that_day
        total_number_sold = delta_hedge_daily['total_number_sold'] + number_of_shares_sold_that_day

        dict_day['total_number_bought'] = total_number_bought
        dict_day['total_number_sold'] = total_number_sold


        total_number = total_number_bought - total_number_sold

        dict_day['total_number'] = total_number

        total_buy_value = total_number_bought * new_mean_buy
        total_sell_value = total_number_sold * new_mean_sell

        dict_day['total_buy_value'] = total_buy_value
        dict_day['total_sell_value'] = total_sell_value

        total_value = -(total_sell_value - total_buy_value)
        dict_day['delta_hedging_cost'] = total_value
        
        
        return pd.DataFrame(dict_day)
