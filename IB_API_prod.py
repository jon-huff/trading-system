#!/usr/bin/env python

# =============================================================================
# Import libraries
# =============================================================================
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.ticktype import TickTypeEnum
import pandas as pd
import datetime
import math
import time
import datetime
import queue
import pymysql
import sqlalchemy
from sqlalchemy import create_engine
import threading
import queue
from queue import Queue


flag = threading.Lock()

# Export queue takes retrieves ticks from the API and immediately hands
# them off to be stored and eventually written to DB
export_q = Queue()

# tick_list gets ticks from the export_q and appends them until it reaches
# a len threshold to export to DB
tick_list = []

# df_queue takes the tick_list to be exported so the tick_list can be 
#$ immediately reset to accept new ticks
df_queue = Queue()

# reqId codes specified in the GetTicks function, converted to strings
reqId_codes = {1:'price', 2:'trade'}



# =============================================================================
# MySQL Functions
# =============================================================================

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

# SQL update
def update(query, db=db):
    db = db
    db_cursor = db.cursor()
    db_cursor.execute(query)
    db.commit()

# Allows writing in the DB
update("set sql_safe_updates=0")

# engine allows batch writing into DB
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root", pw="rootroot", db='ES'))

# Create DB table name for new session
table_name = str('ES_')+str(datetime.datetime.now().year)+'-'+ \
str(datetime.datetime.now().month)+'-' + \
str(datetime.datetime.now().day+1)

update("use `ES`;")

update(f"create table if not exists `ES`.`{table_name}` (\
`Index_key` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, \
`Timestamp` DATETIME, \
`ReqId` CHARACTER(16), \
`BidQty` SMALLINT UNSIGNED, \
`BidPrc` FLOAT8 UNSIGNED, \
`AskPrc` FLOAT8 UNSIGNED, \
`AskQty` SMALLINT UNSIGNED, \
`Price` DECIMAL(6,2) UNSIGNED, \
`Qty` SMALLINT UNSIGNED, \
PRIMARY KEY (`Index_key`)) \
ENGINE = InnoDB \
AUTO_INCREMENT = 0 \
DEFAULT CHARACTER SET = utf8mb4;")


# =============================================================================
# Data Handling
# =============================================================================

# Pulls data from export_q into tick_list
def q_to_list():
    global tick_list
    tick_list.append(export_q.get())
    flag.acquire()
    if len(tick_list)>=1000:
        df_queue.put(tick_list)
        tick_list = []
        sql_export()
    flag.release()

# Pulls data from the df_queue, converts to a DF and pushes to DB; called by q_to_list function
def sql_export():
    tick_df = pd.DataFrame(df_queue.get())
    tick_df['timestamp'] = tick_df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S.000'))
    tick_df.to_sql(table_name, con=engine, if_exists='append', chunksize=1000, index=False)
    print(datetime.datetime.now(), ": ", 'Pushed to DB', sep = '')


# =============================================================================
# IB Data Feeds
# =============================================================================


class TickFeed(EWrapper, EClient):

    def __init__(self, export_func):
        EClient.__init__(self,self)
        self.export_func = export_func
        global export_q


    def error(self, reqId, errorCode, errorString):
        print(datetime.datetime.now(),"Msg Code: ", reqId, " ",errorCode, " ", errorString)


    def tickByTickBidAsk(self, reqId: int, time: int, bidPrice: float, askPrice: float,
                         bidSize: int, askSize: int, tickAttribBidAsk: int):

        super().tickByTickBidAsk(reqId, time, bidPrice, askPrice, bidSize,
             askSize, tickAttribBidAsk)
        new_price = {'timestamp':time, 'reqId':reqId_codes[reqId], 'bidQty':bidSize, 'bidPrc':bidPrice, 'askPrc':askPrice, 'askQty':askSize}
        #print(new_price)
        #flag.acquire()
        export_q.put(new_price)
        #flag.release()
        #get_queue_thread.run()
        self.export_func()

    def tickByTickAllLast(self, reqId: int, tickType: int, time: int, price: float,
                           size: int, tickAtrribLast: int, exchange: str,
                           specialConditions: str):

        super().tickByTickAllLast(reqId, tickType, time, price, size, tickAtrribLast,
             exchange, specialConditions)
        new_trade = {'timestamp':time, 'reqId':reqId_codes[reqId], 'price':price, 'qty':size}
        #print(new_trade)
        #flag.acquire()
        export_q.put(new_trade)
        #flag.release()
        #get_queue_thread.run()
        self.export_func()


def GetTicks():
    app = TickFeed(export_func = q_to_list)

    app.connect("127.0.0.1", 4002, 0)

    contract = Contract()
    contract.secType = "FUT"
    contract.exchange = "GLOBEX"
    contract.currency = "USD"
    contract.localSymbol = "ESH0"

    app.reqMarketDataType(1)
    app.reqTickByTickData(1, contract, "BidAsk", 0, False)
    app.reqTickByTickData(2, contract, "AllLast", 0, False)

    app.run()

if __name__ == '__main__':
    GetTicks()