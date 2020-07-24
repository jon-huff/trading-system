import pandas as pd
import numpy as np



def start(df):
    df = df.copy()
    print('Beginning format')
    df.drop(['BidQty','BidPrc','AskPrc','AskQty'], axis=1, inplace=True)
    df['hour'] = df.Timestamp.apply(lambda x: pd.to_datetime(x).hour)
    df = df.loc[df.hour<16, :]
    dates = df.Timestamp.apply(lambda x: pd.to_datetime(x).date()).unique()
    df['Date'] = df.Timestamp.apply(lambda x: x.date())
    df.reset_index(drop=True, inplace=True)
    print('Done')

    print('Calculating previous settles')
    settles_idx = []
    for i in dates:
        settles_idx.append((df.loc[df.Date == i,'hour'] == 15).idxmax())
    print('Done')

    print('Calculating session intervals')
    intervals = [0]
    for i in dates:
        intervals.append((df.loc[intervals[-1]:,'Date'] == i).idxmin())
    intervals[-1] = (len(df))
    print('Done')

    print('Appending previous settles')
    df.loc[:, 'prev_settle'] = float('nan')
    for i in range(1,len(intervals)-1):
        start = intervals[i]
        stop = intervals[i+1]-1
        df.loc[start:stop, 'prev_settle'] = df.loc[settles_idx[i-1], 'Price']
    print('Done')

    print('Restting session intervals')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    data = dates[1:]
    intervals = [0]
    for i in dates:
        intervals.append((df.loc[intervals[-1]:,'Date'] == i).idxmin())
    intervals[-1] = (len(df))
    intervals = intervals[1:]
    df['net_change'] = df['Price'] - df['prev_settle']
    df.drop(['Date'], axis=1,inplace=True)
    print('Done')

    return(df, intervals)




