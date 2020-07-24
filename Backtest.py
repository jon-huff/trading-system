'''
Backtest takes in a dataframe with 2 columns: Price series and Signal series.
Only 1 position can be opened at a time.  The signal must be in the form of
sell: -1, flat: 0, buy: 1
It returns a dataframe with additional columns: open_pnl, closed_pnl, total_pnl.
'''

import pandas as pd
import numpy as np
from numba import jit


open_pnl = []
closed_pnl = []
total_pnl = []
position = 0
prev_pos = 0

@jit(nopython=True)
def calc_backtest(prices, signals, payup_tick, tick_val, profit_pnl, stop_pnl, entries, exits):
    price = prices
    signal = signals

    open_pnl = np.zeros(len(price))
    closed_pnl = np.zeros(len(price))
    total_pnl = np.zeros(len(price))
    live_pnl = 0
    position = 0
    prev_position = 0
    entries = np.zeros(len(price))
    exits = np.zeros(len(price))
    early_exit = False


    for i in range(len(price)):
        prev_position=position
        position = signal[i]

        if prev_position == 0:
            if position != 0:
                ## New Position
                if position == 1:
                    curr_price = price[i] + payup_tick
                elif position == -1:
                    curr_price = price[i] - payup_tick
                live_pnl=0
                open_pnl[i] = live_pnl
                closed_pnl[i] = closed_pnl[i-1] if i>0 else 0
                entries[i] = curr_price
                exits[i] = np.nan
                early_exit = False

            else:
                ## No position
                open_pnl[i] = 0
                closed_pnl[i] = closed_pnl[i-1] if i>0 else 0
                live_pnl=0
                entries[i] = np.nan
                exits[i] = np.nan
                early_exit = False

        if prev_position != 0:
            if position != 0:
                if position != prev_position:
                    ## Position flip
                    if prev_position == 1:
                        curr_price = price[i] - payup_tick
                    elif prev_position == -1:
                        curr_price = price[i] + payup_tick

                    if early_exit == False:
                        open_pnl[i] = 0
                        live_pnl += ((curr_price-price[i-1]) * tick_val * prev_position) if i>0 else 0
                        closed_pnl[i] = closed_pnl[i-1] + live_pnl

                        entries[i] = curr_price
                        exits[i] = curr_price

                        live_pnl=0
                        open_pnl[i] = live_pnl

                    else:
                        entries[i] = curr_price
                        exits[i] = np.nan
                        live_pnl=0
                        open_pnl[i] = live_pnl
                        early_exit=False

                elif early_exit==False:
                    ## Running open position
                    curr_price = price[i]
                    live_pnl += ((curr_price-price[i-1]) * tick_val * position) if i>0 else 0

                    ## Close position early
                    if live_pnl >= profit_pnl or live_pnl <= stop_pnl:
                        if position == 1:
                            curr_price = price[i] - payup_tick
                        elif position == -1:
                            curr_price = price[i] + payup_tick

                        open_pnl[i] = 0
                        live_pnl -= (payup_tick * tick_val)
                        closed_pnl[i] = closed_pnl[i-1] + live_pnl
                        live_pnl=0
                        entries[i] = np.nan
                        exits[i] = curr_price
                        early_exit = True

                    else:
                        open_pnl[i] = live_pnl
                        closed_pnl[i] = closed_pnl[i-1] if i>0 else 0
                        entries[i] = np.nan
                        exits[i] = np.nan

            elif early_exit==False:
                ## Close position
                if prev_position == 1:
                    curr_price = price[i] - payup_tick
                elif position == -1:
                    curr_price = price[i] + payup_tick
                open_pnl[i] = 0
                live_pnl += ((curr_price-price[i-1]) * tick_val * prev_position) if i>0 else 0
                closed_pnl[i] = ((closed_pnl[i-1] + live_pnl)) if i>0 else 0
                live_pnl=0
                entries[i] = np.nan
                exits[i] = curr_price
        total_pnl[i] = open_pnl[i]+closed_pnl[i]

    if np.isnan(exits).sum()> np.isnan(entries).sum():
        print('exit action triggered')
        if prev_position == 1:
            curr_price = price[-1] - payup_tick
        elif position == -1:
            curr_price = price[-1] + payup_tick
        live_pnl += (curr_price-price[-2])
        closed_pnl[-1] = (closed_pnl[-2] + live_pnl)
        open_pnl[-1] = 0
        entries[i] = np.nan
        exits[i] = curr_price
        live_pnl = 0
        total_pnl[-1] = open_pnl[-1] + closed_pnl[-1]

    print(len(entries), len(exits))

    return(open_pnl, closed_pnl, total_pnl, entries, exits)



def start(list_of_df, payup_tick = .25, tick_val = 50, profit_pnl=500, stop_pnl=-100):
    assert type(list_of_df) == list, 'Incorrect data input, should be list'
    open_pnl = []
    closed_pnl = []
    total_pnl = []
    entries = []
    exits = []
    prices_list = []
    signals_list = []
    prev_closed_pnl = 0
    prev_total_pnl = 0

    for i in range(len(list_of_df)):
        assert type(list_of_df[i]) == pd.core.frame.DataFrame, f'Incorrect data type in list[{i}]'
        temp = list_of_df[i]
        temp.fillna(0, inplace=True)
        prices = np.array(temp.iloc[:, 0])
        signals = np.array(temp.iloc[:, 1])
        assert len(prices)==len(signals), 'Prices and Signals have different lengths'

        print('Running backtest...')
        r1, r2, r3, r4, r5 = calc_backtest(prices, signals, payup_tick, tick_val, profit_pnl, stop_pnl, entries=np.array([]), exits=np.array([]))
        open_pnl.extend(list(r1))
        closed_pnl.extend(list(r2 + prev_closed_pnl))
        total_pnl.extend(list(r3 + prev_total_pnl))
        entries.extend(list(r4))
        exits.extend(list(r5))
        prices_list.extend(list(prices))
        signals_list.extend(list(signals))

        prev_closed_pnl = closed_pnl[-1]
        prev_total_pnl = total_pnl[-1]

        print('Done')


    backtest_df = pd.DataFrame({'price':prices_list,
                                'signal':signals_list,
                                'open_pnl':open_pnl,
                                'closed_pnl':closed_pnl,
                                'total_pnl':total_pnl})

    return(backtest_df, entries, exits)