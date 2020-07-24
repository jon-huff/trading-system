# Query data from MySQL to Pandas dataframe

import os
import pymysql
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import time
from numba import jit

def sql_connect(schema='ES'):
    db = pymysql.connect('localhost', 'root', 'rootroot', schema)
    return(db)

db = sql_connect('ES')

# SQL close connection
def sql_close():
    db.close()

# SQL query
def qry(query, schema = 'ES'):
    df = pd.read_sql(query, db)
    return df

# SQL Update
def update(query, db=db):
    db = db
    db_cursor = db.cursor()
    db_cursor.execute(query)
    db.commit()


# =============================================================================
# Query database
# =============================================================================


update("use `ES`;")
table_list = qry('show tables')
table_list
table_list = sorted(list(table_list['Tables_in_es']))
table_list = list(pd.DataFrame(zip(list(map(lambda x: datetime.datetime.strptime(x[(x.index('_')+1):], '%Y-%m-%d'), table_list)), table_list)).sort_values(by=0)[1])
table_list
columns = list(qry(f"select * from `{table_list[0]}` limit 1;").columns)
columns.extend(['Side', 'Action'])
columns
df_all = pd.DataFrame(data=None, columns = columns)

def ret_action(x):
    x = np.array(x)
    if x[1]>x[0]:
        return(1)
    elif x[1]<x[0]:
        return (-1)
    else: return(0)

for i in table_list[:-1]:
    temp = qry(f"select * from `{i}`;").reset_index(drop=True)
    ffill = temp[['BidPrc','AskPrc']].fillna(method='ffill')
    temp.loc[:,'BidPrc'] = ffill.BidPrc
    temp.loc[:, 'AskPrc'] = ffill.AskPrc
    temp = temp.loc[temp.ReqId =='trade', :]
    side = temp.loc[:,['BidPrc','AskPrc','Price']].apply(lambda x: 'S' if x[2]==x[0] else 'B', axis=1)
    temp['Side'] = side
    action = temp.loc[:, 'BidPrc'].rolling(2).apply(lambda x: ret_action(x), raw=True)
    temp['Action'] = action
    print('Appending', i)
    df_all = df_all.append(temp, ignore_index=True).reset_index(drop=True)

df_all.drop(['Index_key'], axis=1, inplace=True)

df_all.to_pickle('df_all.pkl')