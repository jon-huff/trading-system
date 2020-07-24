import pandas as pd
import numpy as np
import time
from numba import jit
from multiprocessing import Pool
from scipy.stats import skew, kurtosis
import gc

indicators = list(['bbands',
                 'consec_ticks',
                 'eff_ratio',
                 'ema',
                 'high_low',
                 'intensity',
                 'kurtosis',
                 'range',
                 'rolling_volume_delta',
                 'session_volume_delta',
                 'skewness',
                 'sma',
                 'volatility',
                 'volume_delta',
                 'volume_intensity',
                 'vwap',
                 'window_nc'])


def list_indicators():
    return(indicators)


#def start(df, intervals, tick_window, n_periods, indicator_list = [], min_price_increment = .25, calc_inputs=True, calc_outputs=True):
#    """
#    Parameters
#    ----------
#    df : Formatted DataFrame \n
#    intervals : List of intervals specifying session breaks. \n
#    tick_list_input : List of rolling window widths. \n
#    indicator_list : Specify indicators to calculate.  Default is [], calculates all \n
#    calc_inputs : Calculate X DataFrame.  The default is True. \n
#    calc_outputs : Calculate y DataFrame.  The default is True. \n
#
#    Returns
#    -------
#    X, y DataFrames.
#
#    """
#    t1 = time.perf_counter()
#
#    assert type(df) == pd.core.frame.DataFrame, 'Incorrect df type'
#    assert type(indicator_list) == list, 'Incorrect indicator_list type'
#
#    if len(indicator_list)==0:
#        indicator_list = indicators


def map_functions(df, intervals, tick_window, n_periods, indicator_list, min_price_increment, calc_inputs, calc_outputs):


    if calc_inputs:
        # Session Features
        df = session_features(df, intervals, tick_window, indicator_list, min_price_increment)

        # Rolling Features
        features = rolling_features(df, intervals, tick_window, indicator_list)

        # Displaced Features
        displaced_features = displaced_predictors(features, intervals, tick_window, n_periods)
        X = pd.concat([df.loc[features.index,:], features, displaced_features], axis=1)
    else:
        features = df
    if calc_outputs:
        # Displaced Targets
        y = displaced_targets(features, intervals, tick_window, n_periods)


    if calc_inputs:
        if calc_outputs:
            return(intervals[0], X, y)
        else: return(intervals[0], X)
    else: return(intervals[0], y)

# =============================================================================
# # =============================================================================
# # Calculate Session Features
# # =============================================================================
# =============================================================================
def session_features(df, intervals, tick_window, indicator_list, min_price_increment):

    # =============================================================================
    # Volume Session
    # =============================================================================
    for i in range(len(intervals)-1):
        start = intervals[i]
        stop = intervals[i+1]-1
        df.loc[start:stop, 'volume'] = df.loc[start:stop, 'Qty'].cumsum()

    # =============================================================================
    # VWAP Session
    # =============================================================================
    for i in range(len(intervals)-1):
        start = intervals[i]
        stop = intervals[i+1]-1
        vwap = ((df.loc[start:stop, 'Qty']*df.loc[start:stop,'net_change']).cumsum())/df.loc[start:stop,'volume']
        df.loc[start:stop,'vwap_session'] = vwap

    # =============================================================================
    # Consecutive Ticks Session
    # =============================================================================
    if 'consec_ticks' in indicator_list:
        global local_high, local_low, num_upticks, num_downticks, min_tick
        min_tick = min_price_increment
        local_high = 0
        local_low = 0
        num_upticks = 0
        num_downticks = 0


        def check_ticks(p):
            global local_high, local_low, num_upticks, num_downticks, min_tick
            if p > local_high:
                num_upticks += 1
                num_downticks = 0
                local_high = p
                local_low = p - min_tick

            elif p < local_low:
                num_downticks -= 1
                num_upticks = 0
                local_low = p
                local_high = p + min_tick

            if num_upticks > 0 and num_downticks < 0:
                return(float('nan'))
            elif num_upticks > 0:
                return(num_upticks)
            else: return(num_downticks)

        consec_ticks = []
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1

            #initialize high and low
            local_high = df.loc[start, 'Price']
            local_low = df.loc[start, 'Price']

            num_upticks = 0
            num_downticks = 0
            consec_ticks.extend(df.loc[start:stop, 'Price'].apply(check_ticks).values)

        df['consec_ticks'] = consec_ticks

    # =============================================================================
    # Volume Delta Session
    # =============================================================================
    if 'volume_delta' in indicator_list:
        @jit(nopython=True)
        def session_volume_delta(side, qty):
            running_delta = 0
            delta_array = np.repeat(np.nan, len(side))
            for s in range(len(side)):
                running_delta += (qty[s] * side[s])
                delta_array[s] = running_delta
            return(delta_array)

        for i in range(len(intervals)-1):
            start = intervals[0]
            stop = intervals[i+1]-1
            side = df.Side.loc[start:stop].apply(lambda x: 1 if x=='B' else -1).values
            qty = df.Qty.loc[start:stop].values
            df.loc[start:stop, 'session_volume_delta'] = session_volume_delta(side, qty)


    # =============================================================================
    # High/Low Session
    # =============================================================================
        if 'high_low' in indicator_list:
            @jit(nopython=True)
            def get_high_low(prices):
                highs = np.zeros(len(prices))
                lows = np.zeros(len(prices))
                highs[0] = prices[0]
                lows[0] = prices[0]
                high = prices[0]
                low = prices[0]
                for i in range(1, len(prices)):
                    if prices[i] > high:
                        high = prices[i]
                    if prices[i] < low:
                        low = prices[i]
                    highs[i] = high
                    lows[i] = low
                return(highs, lows)

            # Distance from High/Low
            for i in range(len(intervals)-1):
                start = intervals[i]
                stop = intervals[i+1]-1

                highs, lows = get_high_low(np.array(df.loc[start:stop,'net_change']))

                df.loc[start:stop, 'ticks_from_high'] = df.loc[start:stop, 'net_change'] - pd.Series(highs, index = pd.RangeIndex(start, stop+1))
                df.loc[start:stop, 'ticks_from_low'] = df.loc[start:stop, 'net_change'] - pd.Series(lows, index = pd.RangeIndex(start, stop+1))

                session_range = pd.Series(highs, index = pd.RangeIndex(start, stop+1)) - pd.Series(lows, index = pd.RangeIndex(start, stop+1))
                df.loc[start:stop, 'session_range'] = session_range

    gc.collect()
    return(df)


# =============================================================================
# # =============================================================================
# # Calculate Rolling Features
# # =============================================================================
# =============================================================================

def rolling_features(df, intervals, tick_window, indicator_names):
    inputs = pd.DataFrame(data=None, index = df.index)

    # =============================================================================
    # OHLC
    # =============================================================================
    if 'ohlc' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            prices = df.loc[start:stop, 'net_change']
            inputs.loc[start:stop, f'open_{tick_window}'] = prices.rolling(tick_window).apply(lambda x: x[0])
            inputs.loc[start:stop, f'high_{tick_window}'] = prices.rolling(tick_window).apply(lambda x: x.max())
            inputs.loc[start:stop, f'low_{tick_window}'] = prices.rolling(tick_window).apply(lambda x: x.min())
            inputs.loc[start:stop, f'close_{tick_window}'] = prices.rolling(tick_window).apply(lambda x: x[-1])
        


    # =============================================================================
    # VWAP Rolling
    # =============================================================================
    if 'vwap' in indicator_names:
        @jit(nopython=True)
        def vwap_calc(p, q, ticks, array):
            start=ticks
            while start<=p.size:
                array[start-1] = (np.sum(p[start-ticks:start] * q[start-ticks:start]) / np.sum(q[start-ticks:start]))
                start+=1
            return(array)

        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            prices = np.array(df.loc[start:stop, 'net_change'])
            qtys = np.array(df.loc[start:stop,'Qty'])
            array = np.repeat(np.nan, prices.size)
            inputs.loc[start:stop, f'vwap_{tick_window}'] = vwap_calc(prices, qtys, tick_window, array)


    # =============================================================================
    # Consecutive Ticks
    # =============================================================================
    if 'consec_ticks' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'consec_ticks_{tick_window}'] = df.loc[start:stop,'consec_ticks'].rolling(tick_window).mean()


    # =============================================================================
    # Volume Delta
    # =============================================================================
    if 'volume_delta' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            side = df.Side.loc[start:stop].apply(lambda x: 1 if x=='B' else -1).values
            qty = df.Qty.loc[start:stop].values
            inputs.loc[start:stop, f'volume_delta_{tick_window}'] = (pd.Series(side*qty).rolling(tick_window).sum()).values


    # =============================================================================
    # Rolling Volatility
    # =============================================================================
    if 'volatility' in indicator_names:
        def rolling_vol(arr, ticks):
            x = np.array(arr)
            res = np.std(x)*np.sqrt(ticks)
            return(res)

        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'vol_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).apply(lambda x: rolling_vol(x, tick_window), raw=True)


    # =============================================================================
    # Range
    # =============================================================================
    if 'range' in indicator_names:
        def range_calc(arr):
            x = np.array(arr)
            res = np.max(x)-np.min(x)
            return(res)

        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'range_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).apply(range_calc, raw=True)


    # =============================================================================
    # Rolling High/Low
    # =============================================================================
    if 'high_low' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'rolling_high_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).apply(lambda x: np.max(x)-x[0], raw=True)
            inputs.loc[start:stop, f'rolling_low_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).apply(lambda x: np.min(x)-x[0], raw=True)


    # =============================================================================
    # Rolling Window Net Change
    # =============================================================================
    if 'window_nc' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'window_nc_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).apply(lambda x: x[-1]-x[0], raw=True)


    # =============================================================================
    # Efficiency Ratio
    # =============================================================================
    if 'eff_ratio' in indicator_names:
        def eff_ratio(arr):
            x = np.array(arr)
            arr_max = np.max(x)
            arr_min = np.min(x)
            if arr_max - arr_min == 0:
                return(0)
            else:
                res = (x[-1]-x[0])/(arr_max - arr_min)
            return(res)

        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'eff_ratio_{tick_window}'] = df.loc[start:stop, 'Price'].rolling(tick_window).apply(eff_ratio, raw=True)


    # =============================================================================
    # Tick Intensity
    # =============================================================================
    if 'intensity' in indicator_names:
        def intensity(arr, ticks):
            x = np.array(arr)
            delta = (np.max(x)-np.min(x))
            res = ticks/max(1,delta)
            return(res)

        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'intensity_{tick_window}'] = df.loc[start:stop, 'Timestamp'].apply(lambda x: time.mktime(x.timetuple()))\
                .rolling(tick_window).apply(lambda x: intensity(x, tick_window), raw=True)


    # =============================================================================
    # Volume Intensity
    # =============================================================================
    if 'volume_intensity' in indicator_names:
        @jit(nopython=True)
        def vol_intensity_calc(t, q, ticks, array):
            start=ticks
            while start<=t.size:
                maxx = np.max(t[start-ticks:start])
                minn = np.min(t[start-ticks:start])
                if maxx-minn != 0:
                    array[start-1] = np.sum(q[start-ticks:start])/(maxx-minn)
                else:
                    array[start-1] = 0
                start+=1
            return(array)

        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            t = np.array(df.loc[start:stop, 'Timestamp'].apply(lambda x: float(x.strftime('%S.%f'))))
            q = np.array(df.loc[start:stop, 'Qty'])
            array = np.repeat(np.nan, t.size)
            inputs.loc[start:stop, f'volume_intensity_{tick_window}'] = vol_intensity_calc(t, q, tick_window, array)


    # =============================================================================
    # Simple Moving Average
    # =============================================================================
    if 'sma' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'sma_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).mean()


    # =============================================================================
    # Exponential Moving Average
    # =============================================================================
    if 'ema' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'ema_{tick_window}'] = df.loc[start:stop, 'net_change'].ewm(span=tick_window, adjust=False, min_periods=tick_window).mean()


    # =============================================================================
    # Bollinger Bands
    # =============================================================================
    if 'bbands' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            stdev = df.loc[start:stop, 'net_change'].rolling(tick_window).std()
            sma = df.loc[start:stop, 'net_change'].rolling(tick_window).mean()
            inputs.loc[start:stop, f'bbands_low_{tick_window}'] = (sma - 2*stdev)
            inputs.loc[start:stop, f'bbands_high_{tick_window}'] = (sma + 2*stdev)
            inputs.loc[start:stop, f'bbands_width_{tick_window}'] = \
                inputs.loc[start:stop, f'bbands_high_{tick_window}'] - inputs.loc[start:stop, f'bbands_low_{tick_window}']
            
    
    # =============================================================================
    # Skewness
    # =============================================================================
    if 'skewness' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'skewness_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).apply(skew, raw=True)


    # =============================================================================
    # Kurtosis
    # =============================================================================
    if 'kurtosis' in indicator_names:
        for i in range(len(intervals)-1):
            start = intervals[i]
            stop = intervals[i+1]-1
            inputs.loc[start:stop, f'kurtosis_{tick_window}'] = df.loc[start:stop, 'net_change'].rolling(tick_window).apply(kurtosis, raw=True)


    # =============================================================================
    # Create Null Values for Last Row in Sessions
    # =============================================================================
    for i in intervals[1:]:
        inputs.loc[i-1,:] = float('nan')

    gc.collect()
    return(inputs)


# =============================================================================
# # =============================================================================
# #  Displaced Historical Predictors
# # =============================================================================
# =============================================================================

def displaced_predictors(df, intervals, tick_window, n_periods):
#    displacement = 100
#    temp = df.shift(displacement)
#    temp_cols = [f"{col[:col.rfind('_')]}_100_1" for col in df.columns]
#    temp.columns = temp_cols
#    output = pd.concat([output, temp], axis=1)
    start = intervals[0]
    stop = intervals[1]-1
    output = pd.DataFrame(data=None)
    for i in range(1, n_periods+1):
        displacement = i * tick_window
        temp = df.shift(displacement)
        temp_cols = [f"{col}_{i}" for col in df.columns]
        temp.columns = temp_cols
        temp = temp.loc[start:stop,:]
        output = pd.concat([output, temp], axis=1)

    gc.collect()
    return(output)



# =============================================================================
# # =============================================================================
# # Displaced Future Targets
# # =============================================================================
# =============================================================================

def displaced_targets(df, intervals, tick_window, n_periods):
#    displacement = 100
#    temp = df.shift(displacement)
#    temp_cols = [f"{col[:col.rfind('_')]}_100_1" for col in df.columns]
#    temp.columns = temp_cols
#    output = pd.concat([output, temp], axis=1)
    start = intervals[0]
    stop = intervals[1]-1
    output = pd.DataFrame(data=None)
    for i in range(1, n_periods+1):
        displacement = (-i) * tick_window
        temp = df.shift(displacement)
        temp_cols = [f"{col}_{i}" for col in df.columns]
        temp.columns = temp_cols
        output = pd.concat([output, temp], axis=1)

    gc.collect()
    return(output)


# =============================================================================
# # =============================================================================
# #  Instantiate Process Pool and Return Data
# # =============================================================================
# =============================================================================

def start(df, intervals, tick_window, n_periods, indicator_list = [], min_price_increment = .25, calc_inputs=True, calc_outputs=True):
    t1 = time.perf_counter()

    assert type(df) == pd.core.frame.DataFrame, 'Incorrect df type'
    assert type(indicator_list) == list, 'Incorrect indicator_list type'

    if len(indicator_list)==0:
        indicator_list = indicators

    args = [[df.loc[intervals[i]:intervals[i+1]-1, :], [intervals[i], intervals[i+1]], tick_window, n_periods, indicator_list, min_price_increment, calc_inputs, calc_outputs] for i in range(len(intervals)-1)]
    print('Starting process pool')
    pool = Pool()
    result = pool.starmap(map_functions, args)
    pool.close()
    pool.join()

    gc.collect()

    result = sorted(result, key=lambda x: x[0])
    X = pd.DataFrame(data=None)
    y = pd.DataFrame(data=None)
    for i in result:
        X = pd.concat([X, i[1]])
        y = pd.concat([y, i[2]])
        gc.collect()
    y = pd.concat([X.net_change, y], axis=1)


    t2 = time.perf_counter()
    print('Complete','\n',f'Total time: {int((t2-t1)//60)}m {int((t2-t1)%60)}s')
    return(X, y)

#result = map_functions(df, intervals, tick_window, n_periods, indicator_list, min_price_increment, calc_inputs, calc_outputs)


# result = sorted(result, key=lambda x: x[0])
