Several .py scripts are included in this respository.  

IB_API_prod.py
Connects to an Interactive Brokers gateway to collect tick data
in real-time and pushes batches to a MySQL database.

Query.py
Queries data from the database to a pandas dataframe

Format_DF.py
Pre-processes the queried data and formats it for further analysis.

IndicatorsDisplaced_MP.py
Performs feature engineering on the formatted data.  The script is modular 
such that new indicators and statistical measures can be easily added.
This time series data is cross-sectional so historical displaced data is
included in each observation in time.  The script utilizes a process pool
to run calculations across all available CPU cores.

Backtest.py
The backtest script is still very basic in terms of functionality (e.g. only
one position can be open at a time).  However, this provides a glimpse into
the performance of the strategy on out-of-sample data.  
