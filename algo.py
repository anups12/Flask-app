import requests
import yaml
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
import time
from NorenApi import NorenApi  # Ensure that NorenApi is properly imported
#import shoonya_login as login
import logging
import telebot
from helper import placeOrder, rename_previous_log_file
import os
import pandas_ta as ta

current_directory = os.path.dirname(os.path.abspath(__file__))

# Need to create the file by name "total_points.txt" initially before the start of algo
total_points_file_name = "total_points.txt"
total_points_file_path = os.path.join(current_directory,total_points_file_name)

def write_total_points_to_file(number):
    global total_points_file_path
    with open(total_points_file_path, 'w') as file:
        file.write(str(number))

# Function to read a number from a file
def read_total_points_from_file():
    global total_points_file_path
    with open(total_points_file_path, 'r') as file:
        return float(file.read().strip())  # Read and convert to float


def load_config(config_file=os.path.join(current_directory,"config.yaml")):
    """
    Loads the config file date.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
config = load_config()

last_entry_time_flag = False

telegram_bot_flag = False

#Starts telegram bot
try:
    telegram_key = config['telegram_key']
    telegram_chat_id = config['chat_id']
    telegram_bot = telebot.TeleBot(str(telegram_key))
    telegram_bot_flag = True
    telegram_error_message = ""
except Exception as e:
    print(e)
    telegram_bot = None
    telegram_error_message = str(e)
    telegram_bot_flag = False

#Text placeholder for live or paper trading, to send in telegram chat
trade_type_text = ""
if config.get('paper_trading', 1) == 0:
    trade_type_text = "LIVE TRADE"
else:
    trade_type_text = "PAPER TRADE"


#Formats UTC time to IST time , in log file
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Convert the record's creation time from UTC to IST
        utc_time = datetime.fromtimestamp(record.created, timezone.utc)
        ist_time = utc_time.replace(tzinfo=pytz.utc).astimezone(pytz.timezone('Asia/Kolkata'))
        if datefmt:
            return ist_time.strftime(datefmt)
        else:
            return ist_time.strftime('%Y-%m-%d %H:%M:%S')  # Default time format

# Set up the logger
logger = logging.getLogger()

# Set the logging level
logger.setLevel(logging.INFO)
log_file_name = "app.log"
log_file_path = os.path.join(current_directory,log_file_name)
rename_previous_log_file(log_file_path)
print(log_file_path)
# Create a file handler
handler = logging.FileHandler(log_file_path)
handler.setLevel(logging.INFO)

# Create and set the formatter (using the custom IST formatter)
# Note: Use `%(asctime)s` for time, not the `strftime` format directly
formatter = ISTFormatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)



# Initialize the Shoonya API
def init_login():
    try:
        print("Initiating shoonya login")
        api = NorenApi()
        api.token_setter()  # Make sure this sets the necessary tokens

        # Login function (assuming it logs in and sets tokens)
        #def login():
         #   login.Shoonya_login()
        logger.info("Broker logged in successfully")
        #time.sleep(0.2)
        return api
    except Exception as e:
        logger.info(f"Broker login error -- {e}")
        time.sleep(0.2)

api = init_login()

#Get current price of index, or contract, first through websocket, if that fails, then through historical api.
def get_current_price(index_name, flask_base_url,type,token=None,flask_timeout=2):
    #time.sleep(0.6) #DELAY ADDED FOR RATELIMIT IN HISTORICAL API
    index_map = {
        'Nifty 50': 'NSE:Nifty 50',
        'Nifty Bank': 'NSE:Nifty Bank',
        'Nifty Fin Service': 'NSE:Nifty Fin Service'
    }
    token_map = {
        'Nifty 50': '26000',
        'Nifty Bank': '26009',
        'Nifty Fin Service': '26037'
    }
    logger.info(f"delete me {index_name}")
    if type == "index":
        instrument = index_map.get(index_name)
    else :
        instrument = index_name
    logger.info(f"reached here index {instrument}")

    url = f"{flask_base_url}/ltp?instrument={instrument}"
    hist_flag = 0
    logger.info(f"reached here url {url}")
    try:
        # Adding a timeout of 2 seconds
        # response = requests.get(url, timeout=(flask_timeout, flask_timeout))
        import random
        response = {
            "status_code":200, 
            "text": random.randint(10, 100)
        }
        logger.info(f"request  {response}")
        hist_flag = 1
    except requests.exceptions.Timeout:
        pass
        print("The request timed out")
    except requests.exceptions.RequestException as e:
        pass
        print(f"An error occurred: {e}")
    else:
        print("response ", response)
        if response.status_code == 200:
            price = float(response.text)
            if isinstance(price,float) and price is not None:
                hist_flag = 1
                print(f"Current price of {index_name}: {price}")
                try:
                    if float(price) > 0.0001:
                        return price
                except Exception as e :
                    logger.info(f"Error in websocket, Data: {price}, error: {str(e)}")
                    get_current_price(index_name, flask_base_url,type,token=token,flask_timeout=2)
        else:
            print("201 response")
            hist_flag = 0 
    if hist_flag == 0:
       if type == "index":
           token = token_map.get(index_name)
           exchange = "NSE"
       else:
           exchange = "NFO"
           token = str(token)
       print(f"Failed to get current price for {instrument}, Fetching Historical API")
       time.sleep(0.5)

       #RETRIES BY CALLING THE FUNCTION RECURSIVELY, UNTIL A VALID RESPONSE FOR PRICE IS GOTTEN FROM EITHER WEBSOCKET OR HISTORICAL API.
       try:
           ret = api.get_quotes(exchange=exchange, token=token)
           
           print(f"Current price of {instrument}: {ret['lp']}")
           if ret['lp'] is not None and  float(ret['lp']) > 0.0001:
               return float(ret['lp'])
           else:
               logger.info(f"Error in historical, Data: {ret}, error: {str(e)}")
               get_current_price(index_name, flask_base_url,type,token=token,flask_timeout=2) 
       except Exception as e:
           logger.info(f"Error in historical, Data: {ret}, error: {str(e)}")
           get_current_price(index_name, flask_base_url,type,token=token,flask_timeout=2)

#Gets the list of contracts to check, for entry, based on index price, band, and number of strikes given
def get_contracts(index_price, symbol, n, band, atm_band):
    # Round the index price to the nearest band
    index_price = round(index_price / atm_band) * atm_band
    
    # Adjust symbol names
    if symbol == 'NIFTY50':
        symbol = 'NIFTY'
    elif symbol == "NIFTYBANK":
        symbol = "BANKNIFTY"
    elif symbol == 'NIFTYFINSERVICE':
        symbol = "FINNIFTY"
    # Load and clean data
    df = pd.read_csv(os.path.join(current_directory,"filtered_data.csv"))
    df['StrikePrice'] = pd.to_numeric(df['StrikePrice'], errors='coerce')
    df = df.dropna(subset=['StrikePrice'])
    
    # Filter for the specific symbol and ensure StrikePrice aligns with band
    df = df[df['Symbol'] == symbol]
    df = df[(df['StrikePrice'] % band).abs() < 1e-8]
    
    # Initialize an empty list to store the results
    results = []

    # Loop for both option types CE and PE
    for option_type in ['CE', 'PE']:
        # Filter the DataFrame for the current Option Type
        option_df = df[df['OptionType'] == option_type]

        # Find the contract exactly at the index_price (ATM)
        atm_df = option_df[option_df['StrikePrice'] == index_price]

        # If it's CE, get ITM strikes below the index price
        if option_type == 'CE':
            itm_ce_df = option_df[option_df['StrikePrice'] < index_price]
            itm_ce_df = itm_ce_df.sort_values('StrikePrice', ascending=False).head(n)
            results.append(atm_df)  # Append the ATM contract for CE
            results.append(itm_ce_df)  # Append n ITM contracts below the index price for CE

        # If it's PE, get ITM strikes above the index price
        elif option_type == 'PE':
            itm_pe_df = option_df[option_df['StrikePrice'] > index_price]
            itm_pe_df = itm_pe_df.sort_values('StrikePrice').head(n)
            results.append(atm_df)  # Append the ATM contract for PE
            results.append(itm_pe_df)  # Append n ITM contracts above the index price for PE

    # Concatenate all results into a single DataFrame, remove duplicates, and reset index
    final_df = pd.concat(results).drop_duplicates()
    final_df.reset_index(drop=True, inplace=True)

    return final_df


#Fetches 1 MINUTE candle data, for contracts, first through websocket, if that fails, then through historical API
def check_and_fetch_missing_candles(contracts_df, time_frame, flask_base_url, market_start_time,algo_start_time,flask_timeout=2):
    hist_flag = 0
    aggregated_candles = {}
    # Get the current time in IST
    num_required_candles = int(time_frame)*2
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)

    # Calculate the time for the last time_frame*2 candles based on the current time
    last_expected_candle_time = current_time.replace(second=0, microsecond=0) - timedelta(minutes=1)
    for index, row in contracts_df.iterrows():
        instrument = row['TradingSymbol']  # Get TradingSymbol
        token = row['Token']
        #instrument = "NFO:BAKNIFTY09OCT24C48100"
        #token = "4407" 
        instrument_key = f"NFO:{instrument}"
        # Fetch the last time_frame*2 one-minute candles from the Flask endpoint
        url = f"{flask_base_url}/candles?instrument={instrument_key}"
        try:
            response = requests.get(url,timeout=(flask_timeout, flask_timeout))
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            pass
        else:
          if response.status_code == 200:
            hist_flag = 1
            candles = response.json()
            if candles: # and instrument_key != "NFO:BANKNIFTY09OCT24P52900":
                # Convert the list of candles to a DataFrame
                candles_df = pd.DataFrame(candles)
                candles_df['time'] = pd.to_datetime(candles_df['time'])
                # Ensure other columns are numeric
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    candles_df[col] = pd.to_numeric(candles_df[col], errors='coerce')
                # Ensure we have the last num_required_candles (e.g., 9)
                if len(candles_df) >= num_required_candles:
                    # Get the last num_required_candles (e.g., 9) candles
                    last_candles_df = candles_df.tail(num_required_candles)

                    # Calculate the expected candle times for the last num_required_candles (e.g., 9) minutes
                    expected_times = [last_expected_candle_time - timedelta(minutes=i) for i in range(num_required_candles)][::-1]
                    expected_times_str = [dt.strftime('%Y-%m-%d %H:%M') for dt in expected_times]
                    #print(expected_times_str)
                    # Compare the actual times of the last candles with the expected times
                    actual_times = last_candles_df['time'].tolist()
                    actual_times_str = [dt.strftime('%Y-%m-%d %H:%M') for dt in actual_times]
                    #print(actual_times_str)
                    #actual_times_as_datetime = [ts.to_pydatetime() for ts in actual_times]
                    #print(actual_times_as_datetime)
                    #print(candles_df)
                    #print(candles_df.dtypes)
                    if actual_times == expected_times :
                        aggregated_candles[instrument_key] = aggregate_candles(candles_df,time_frame,market_start_time,algo_start_time, instrument_key)
                        print(f"All required candles are present for {instrument_key}.")
                        continue  # Continue to the next instrument if candles are correct
                    else:
                        logger.info(f"Incorrect candle times for {instrument_key}. Fetching historical data.")
                        logger.info(f"Actual Times: {actual_times}, Ecpected times: {expected_times}")
                        print(f"Incorrect candle times for {instrument_key}. Fetching historical data.")
                else:
                    logger.info(f"Not enough candles for {instrument_key}. Expected {num_required_candles}, but got {len(candles_df)}.")
                    print(f"Not enough candles for {instrument_key}. Expected {num_required_candles}, but got {len(candles_df)}.")
            else:
                logger.info(f"No candle data for {instrument_key}.")
                try:
                    logger.info(f"Error candle data: {candles}.")
                except Exception as e:
                    logger.info(f"Error: {e}")
                print(f"No candle data for {instrument_key}.")
          else:
            logger.info(f"Failed to fetch candle data for {instrument_key} from Flask endpoint. Status code: {response.status_code}")
            print(f"Failed to fetch candle data for {instrument_key} from Flask endpoint. Status code: {response.status_code}")

        # If candles are missing or incorrect, fetch historical data
        historical_df = fetch_historical_data(token)
        if historical_df is not None:
            # Update the aggregated candles
            print(1)
            aggregated_candles[instrument_key] = aggregate_candles(historical_df,time_frame,market_start_time,algo_start_time, instrument_key)
            print(aggregated_candles[instrument_key])
            print(2)
        else:
            print("No data for today")
    return aggregated_candles

#Aggregates the 1 MINUTE candle data, to the specified time frame candles
def aggregate_candles(df,time_frame,market_start_time,algo_start_time, instument_key):
    #print(df)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    # Align resampling with the given start_time (e.g., 9:15)
    start_datetime = df.index.min()  # First timestamp in the data
    aligned_start_time = start_datetime.replace(hour=market_start_time.hour, minute=market_start_time.minute, second=0, microsecond=0)
    backdate_days = config.get("HISTORICAL_DATA_BACK_DAYS")
    aligned_start_time = aligned_start_time - timedelta(days=backdate_days)

    # Ensure aligned_start_time is not earlier than the first data point
    while aligned_start_time < start_datetime:
        aligned_start_time += timedelta(minutes=time_frame)

    # Resample the data to the desired time frame, starting from aligned start time
    resampled_df = df.resample(f'{time_frame}min', origin=aligned_start_time).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
    }).dropna()

    # Remove the first candle if websocket starts before we have full data for the fist candle
    first_resampled_time = resampled_df.index.min()
    if first_resampled_time < df.index.min():
        resampled_df = resampled_df.iloc[1:]

    # Remove the last candle if it's incomplete
    last_time = resampled_df.index.max()
    last_candle_end_time = last_time + timedelta(minutes=time_frame)
    if last_candle_end_time > df.index.max() + timedelta(minutes=1):
        resampled_df = resampled_df[:-1]

    # Reset index to make 'time' a column again
    resampled_df.reset_index(inplace=True)
    resampled_df['EMA_20'] = ta.ema(resampled_df['close'], length=20)
    last_2_candles = resampled_df.tail(2)


    print(f"Calculate ema data {instument_key} /n", last_2_candles)
    logger.info(f"Calculate ema data {instument_key} /n {last_2_candles}" )
    #print(last_2_candles)
    algo_start_datetime = last_2_candles['time'].iloc[0].replace(
        hour=algo_start_time.hour, 
        minute=algo_start_time.minute, 
        second=0, 
        microsecond=0
    )
    #print(algo_start_datetime)
    # Check if the first candle of the last 2 is greater than the algo_start_time
    if last_2_candles['time'].iloc[0] >= algo_start_datetime:
        return last_2_candles
    else:
       logger.info("Not enough candles after algo start time.")
       print("No enough candles after algo start time.")
       return pd.DataFrame()

#Fetch historical candle data.
def fetch_historical_data(instrument):
    time.sleep(0.2)
    global api
    return_df = []
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz)
    
    # Set market open time to 9:15 AM
    market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
    backdate_days = config.get("HISTORICAL_DATA_BACK_DAYS")

    market_open -= timedelta(days=backdate_days)

    market_open = market_open.timestamp()

    
    print(f"Fetching historical data for token {instrument}")
    logger.info(f"Fetching historical data for token {instrument}")
    #print(api)
    #candles = api.get_time_price_series(exchange='NFO', token="47262",
     #             starttime=market_open, interval=int(time_frame))
    
    candles = api.get_time_price_series(exchange='NFO', token=str(instrument),
                  starttime=float(market_open), interval=1)
    #print(type(instrument),instrument,market_open,type(market_open))
    try:
        dataframe = pd.DataFrame(candles)
        print(dataframe)
        if candles:
            candles.reverse()
            for response in candles:
                data = {
                   'time': response['time'],
                   'open': float(response['into']),
                   'high': float(response['inth']),
                   'low': float(response['intl']),
                   'close': float(response['intc']),
                   'volume': int(response['intv']),
                }
                #print(data)
                return_df.append(data)
            df = pd.DataFrame(return_df)
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = df['time'].dt.tz_localize('Asia/Kolkata')
            return df
        else:
            print(f"1Retrying candle data from historical api, received data : {candles} ")
            #If login expires, login again in loop for token. Once the token candle data is received, then next token candle data is fetched
            #api = init_login()
            fetch_historical_data(instrument)
    except Exception as e:
        print(f"2Retrying candle data from historical api, received data : {candles}, error : {e}")
        logger.info(f"Error fetching historical candle data from API, Error: {str(e)}, Trying again")
        #If login expires, login again in loop for token. Once the token candle data is received, then next token candle data is fetched
        #api = init_login()
        fetch_historical_data(instrument)

def check_entry_conditions(candles):
    """
    Function to check entry based on last 2 candles.

    Parameters:
    candles (pd.DataFrame): A DataFrame containing the last 2 candles with columns: 'open', 'high', 'low', 'close', 'volume'.

    """
    # Extract the relevant data from the 3 candles
    T1, O1, H1, L1, C1, V1, EMA1 = candles.iloc[0][['time','open', 'high', 'low', 'close', 'volume', 'EMA_20']]
    T2, O2, H2, L2, C2, V2, EMA2 = candles.iloc[1][['time','open', 'high', 'low', 'close', 'volume', 'EMA_20']]
    logger.info(f"TIME: {T1}")
    logger.info(f"OPEN: {O1}")
    logger.info(f"HIGH: {H1}")
    logger.info(f"LOW: {L1}")
    logger.info(f"CLOSE: {C1}")
    logger.info(f"VOLUME: {V1}")
    logger.info(f"EMA_20: {EMA1}")
    logger.info(" ")
    logger.info(f"TIME: {T2}")
    logger.info(f"OPEN: {O2}")
    logger.info(f"HIGH: {H2}")
    logger.info(f"LOW: {L2}")
    logger.info(f"CLOSE: {C2}")
    logger.info(f"VOLUME: {V2}")
    logger.info(f"EMA_20: {EMA2}")

    # Compute x (second candle volume - first candle volume)
    x = V2 - V1
    logger.info(" ")
    logger.info(f"Volume Difference X: {x}")
    entry_upper_limit = ((C2+O2)/2)+25
    entry_lower_limit = (C2+O2)/2
    # Compute derived values for entry checks
    compute1 = O2 - 5
    compute2 = H2 - 4
    compute3 = L2 - 2
    compute4 = C2 - 5
    # Entry Conditions
    if C1 > O1 and C2 > O2:
        if 100 < x < 100000000:
            if compute2 > 0 and compute3 > 0:
                # Entry condition satisfied
                logger.info("Conditions matched ---->")
                return {'action':"buy",'lower_limit':entry_lower_limit,'upper_limit':entry_upper_limit,'compute1':compute1}

    return None
    
#Get the order status of a given order_id
def get_order_status(order_id):
    while True:
        order_status = api.single_order_status(order_id)
        order_status = order_status[0]
        if order_status['stat'] == "Ok":
            if order_status['status'] == "COMPLETE":
                ltp = float(order_status['avgprc'])
                return [True,ltp]
            elif order_status['status'] == "REJECTED":
                logger.info(f"Exit order Rejected, Reason: {order_status['rejreason']}")
                return [False,0]
        else:
            logger.info(f"Exit Order Not placed. Reason Below")
            logger.info(f"{order_status}")
            return [False,0]
        time.sleep(0.2)

#Executes the trade.
def execute_trade(data,paper_trading):
    global api
    print(data)
    if data['status'] == "pending":
        t_type = "BUY"
    elif data['status'] == "open":
        logger.info(f"Contract: {data['contract']}, Entry: {data['entry_price']}, 'sl': {data['sl']}, 'target': {data['target']}, 'Total previous trades': {data.get('trades_count',0)} ")
        t_type = "SELL"
    else:
        return 1
    quantity = data['quantity']
    price = 0
    variety = "regular"
    order_type = "MARKET"
    inst = data['contract']
    order_data = placeOrder(inst ,t_type,quantity,order_type,price,variety, api,paper_trading)
    if order_data[0]:
        if order_data[1] == 0:
            logger.info(f"PAPER TRADE placed {t_type} {order_type} order for contract: {inst}, quantity: {quantity}, order number: {order_data[1]}")
            #if telegram_bot_flag:
            #    telegram_bot.send_message(telegram_chat_id,f"{t_type} {order_type} order for contract: {inst}, quantity: {quantity}, order number: {order_data[1]}, {trade_type_text}")

        else:
            logger.info(f"Successfully placed {t_type} {order_type} order for contract: {inst}, quantity: {quantity}, order number: {order_data[1]}")
            #if telegram_bot_flag:
             #   telegram_bot.send_message(telegram_chat_id,f"{t_type} {order_type} order for contract: {inst}, quantity: {quantity}, order number: {order_data[1]}, {trade_type_text}")
    else:
        logger.info(f"FAILED placing {t_type} {order_type} order for contract: {inst}, quantity: {quantity}, Error: {order_data[1]}")
        if telegram_bot_flag:
            telegram_bot.send_message(telegram_chat_id,f"FAILED placing {t_type} {order_type} order for contract: {inst}, quantity: {quantity}, Error: {order_data[1]}, {trade_type_text}")

    return order_data

#dd = {'contract': 'NFO:BANKNIFTY16OCT24C51100', 'token': 43698, 'entry_price': 480.05, 'quantity': 15, 'status': 'pending', 'sl': 470.449}
#print(22222)
#sos = execute_trade(dd,0)
"""
while 1 == 2:
    order_status = api.single_order_status(sos[1])
    print(1111111,order_status)
    if order_status['stat'] == "Ok":
        print(22222,order_status)
        if order_status['status'] == "COMPLETE":
           ltp = float(order_status['avgprc'])
           logger.info(f"Entered at: {ltp}")
           open_position_flag = True
           break
        elif order_status['status'] == "REJECTED":
           print(3333333,order_status)
           logger.info(f"Entry order Rejected, Reason: {order_status['rejreason']}")
           if telegram_bot_flag:
                telegram_bot.send_message(telegram_chat_id,f"Entry order Rejected, Reason: {order_status['rejreason']}")
                close_bot = True
                break
"""

#time.sleep(1111)

def main():
 
 
 try:
    global config,telegram_bot_flag, telegram_error_message, telegram_bot, telegram_chat_id, trade_type_text, last_entry_time_flag

    if telegram_bot_flag:
        logger.info("Telegram Bot started")
    else:
        logger.info("Error starting telegram bot: {telegram_error_message}")

    # Define the base URL for the Flask server
    flask_base_url = 'http://127.0.0.1:5001'
    trades_count = 0
    close_bot = False       
    flask_timeout = config.get('websocket_timeout', 2)
    trade_nifty = config['trade_nifty']
    trade_bnf = config['trade_bnf']
    trade_fin = config['trade_fin']
    if trade_nifty == 1:
       index_name = "Nifty 50"
    elif trade_bnf == 1:
       index_name = "Nifty Bank"
    elif trade_fin == 1:
       index_name = "Nifty Fin Service"
    else:
       logger.info("No Index selected. Stopping BOT")
       if telegram_bot_flag:
           telegram_bot.send_message(telegram_chat_id,"No Index selected. Stopping BOT")
       return
    #total_points = config['total_points']
    total_points = read_total_points_from_file()
    # write_total_points_to_file(total_points)
    time_frame = int(config['time_frame'])
    market_start_time_str = config['market_start_time']
    algo_start_time_str = config['algo_start_time']
    algo_end_time_str =  config['algo_end_time']
    last_entry_time_str = config['last_entry']
    number_of_strike_prices = int(config['number_of_trade_strike_prices'])
    band = int(config['band'])
    max_trades = int(config['max_trades'])
    paper_trading = int(config['paper_trading'])
    target_percentage = float(config['target_percentage'])
    
    #STOPLOSS PERCENTAGE IS HARD CODED HERE
    sl_percentage = 0.80

    #UNCOMMENT BELOW LINE IF YOU WANT TO USE THE SL_PERCENTAGE FROM CONFIG.YAML FILE
    #sl_percentage = float(config['sl_percentage'])
    
    lots = int(config['lots'])
    open_position = {}
    open_position_flag = False
    logger.info("Retrieved data from the config file. Starting bot with the following parameters:")
    logger.info("------------------------------------------------------------------------------------")
    logger.info(f"START TIME: {algo_start_time_str}, INDEX: {index_name}, TIMEFRAME: {time_frame}, LOTS: {lots} ")
    logger.info(f"NUMBER OF STRIKES: {number_of_strike_prices}, BAND: {band}, MAX TRADES: {max_trades}")
    logger.info("----------------------------------------------------------------------------------------")
    #if telegram_bot_flag:
     #   telegram_bot.send_message(telegram_chat_id,f"Bot starting at {algo_start_time_str}, Index: {index_name}")
    print(index_name,time_frame)

    # Convert start_time to datetime.time object
    market_start_time = datetime.strptime(market_start_time_str, '%H:%M').time()
    algo_start_time =  datetime.strptime(algo_start_time_str, '%H:%M').time()
    algo_end_time =  datetime.strptime(algo_end_time_str, '%H:%M').time()
    last_entry_time = datetime.strptime(last_entry_time_str, '%H:%M:%S').time()
    india_tz = pytz.timezone('Asia/Kolkata')

    # Wait for the start time
    while True:
        current_time = datetime.now(india_tz).time()
        print(current_time)
        if current_time >= algo_start_time:
            logger.info(f"Bot started at {current_time}")
            if telegram_bot_flag:
                telegram_bot.send_message(telegram_chat_id,f"Bot starting at {current_time}, Index: {index_name}")
            break
        datetime1 = datetime.combine(datetime.today(), current_time)
        datetime2 = datetime.combine(datetime.today(), algo_start_time)
        time_difference = int((datetime2 - datetime1).total_seconds())
        print(f"Waiting {time_difference} seconds to start BOT...")
        time.sleep(time_difference/5) #DELAY TO WAIT FOR ALGO START TIME
    previous_minute = current_time.strftime("%M")
    condition_check = {}
    index_price = get_current_price(index_name, flask_base_url,"index",flask_timeout=flask_timeout)
    global_contracts_df = get_contracts(index_price, index_name.upper().replace(' ', ''), number_of_strike_prices, band, band)
    
    # MAIN LOOP RUNS THE ENTIRE LOGIC (CANDLE FORMATION, CHECKING ENTRY CONDITIONS, ENTRY PRICE, EXIT PRICE)
    while True:
        time.sleep(0.1) # DELAY BEFORE EVERY LOOP , IT CAN BE REMOVED
        current_date = datetime.now(india_tz)
        current_time = current_date.time()
        current_minute = current_time.strftime("%M")
        market_start_datetime = datetime.combine(current_date.date(), market_start_time)
        market_start_datetime = india_tz.localize(market_start_datetime)
        time_diff = (current_date - market_start_datetime).total_seconds() // 60
        print(time_diff)
        
        new_candle = time_diff % time_frame == 0
        if time_diff < time_frame*2:
            new_candle = False
        if current_time >= algo_end_time:
           close_bot = True
           logger.info(f"Square-off and Bot shutdown at {current_time}")
           if open_position_flag :
               open_entry_price = open_position['entry_price']
               open_token = open_position['token']
               open_contract = open_position['contract'] 
               open_ltp = get_current_price(open_contract,flask_base_url,"contract",token=open_token,flask_timeout=flask_timeout)
               order_result = execute_trade(open_position,paper_trading)
               if order_result[0]:
                    open_position_flag = False
                    if order_result[1] == 0:
                        ltp = open_ltp
                    else:
                        order_status = get_order_status(order_result[1])
                        if order_status[0]:
                            ltp = float(order_status[1])
                        else:
                            logger.info(f"Error placing SELL order")
                            if telegram_bot_flag:
                                telegram_bot.send_message(telegram_chat_id,"Error placing sell order, {order_status[1]}")
                            close_bot = False
                            open_position_flag = True
                    if open_position_flag == False:
                        if ltp >= open_entry_price:
                            total_points = total_points + (float(ltp) - float(open_entry_price))
                            write_total_points_to_file(total_points)
                        else:
                            total_points = total_points - (float(open_entry_price) - float(ltp))
                            write_total_points_to_file(total_points)
                        logger.info(f"Trade exited at price : {ltp}")
                        logger.info(f"Number of Trades: {trades_count}")
                        logger.info(f"Total Points: {total_points}")
                        logger.info("Bot Shutdown")
                        if telegram_bot_flag:
                            telegram_bot.send_message(telegram_chat_id,f"SELL MARKET, Square off time, Entry: {open_entry_price}, Exit: {ltp}, Order Id: {order_result[1]}, Total Trades: {trades_count},  Total points: {total_points}, {trade_type_text}")
                            #telegram_bot.send_message(telegram_chat_id,f"Bot Shutdown, Trades: {trades_count}, Total points: {total_points}")
                        break
           else:
                break 
        else:
            if open_position_flag:
               open_token = open_position['token']
               open_contract = open_position['contract']
               open_entry_price = open_position['entry_price']
               open_sl = open_position['sl']
               open_target = open_position['target']
               open_quantity = open_position['quantity']
               open_ltp = get_current_price(open_contract,flask_base_url,"contract",token=open_token,flask_timeout=flask_timeout)
               logger.info(f"Contract: {open_contract}, Entry: {open_entry_price}, LTP: {open_ltp}, Target: {open_target}, SL: {open_sl}")
               if open_ltp is not None and open_ltp >= open_target:
                  close_bot = True
                  order_result = execute_trade(open_position,paper_trading)
                  if order_result[0]:
                      open_position_flag = False
                      if order_result[1] == 0:
                          ltp = open_ltp
                      else:
                          order_status = get_order_status(order_result[1])
                          if order_status[0]:
                              ltp = float(order_status[1])
                          else:
                              close_bot = False
                              open_position_flag = True
                  if open_position_flag == False:
                      profit = ltp - open_entry_price
                      total_points = total_points + profit
                      write_total_points_to_file(total_points)
                      logger.info("Target Reached.")
                      logger.info(f"Entry: {open_entry_price}, Exit: {open_ltp}")
                      logger.info(f"Number of Trades: {trades_count}")
                      logger.info(f"Total Points: {total_points}")
                      if telegram_bot_flag:
                          telegram_bot.send_message(telegram_chat_id,f"SELL MARKET order, Target Reached, Entry: {open_entry_price}, Exit: {ltp}, Order Id: {order_result[1]},  Trades: {trades_count}, Total points: {total_points}, {trade_type_text}")
               elif open_ltp is not None and open_ltp <= open_sl:
                  order_result = execute_trade(open_position,paper_trading)
                  if order_result[0]:
                       open_position_flag = False
                       if order_result[1] == 0:
                           ltp = open_ltp
                       else:
                           order_status = get_order_status(order_result[1])
                           if order_status[0]:
                               ltp = float(order_status[1])
                           else:
                               open_position_flag = True
                       if open_position_flag == False:
                           open_position = {}
                           total_points = (float(ltp) - float(open_entry_price)) + float(total_points)
                           write_total_points_to_file(total_points)
                           logger.info(f"Stop-loss hit for trade number {trades_count}, Entry: {open_entry_price}, Exit: {ltp}, Total Points: {total_points}")
                           if telegram_bot_flag:
                               telegram_bot.send_message(telegram_chat_id,f"SELL MARKET order, Stop-loss hit. Entry: {open_entry_price}, Exit: {ltp}, Order Id: {order_result[1]} ,Trades: {trades_count}, Total points: {total_points}, {trade_type_text}")
                           if trades_count >= max_trades:
                               logger.info("Closing Bot. Max trade limit reached")
                               logger.info(f"Number of Trades: {trades_count}")
                               logger.info(f"Total Points: {total_points}")
                               if telegram_bot_flag:
                                   telegram_bot.send_message(telegram_chat_id,f"Trade limit reached, Trades: {trades_count}, Total points: {total_points}")
                               close_bot = True
            if close_bot:
                 logger.info("Bot Shutdown")
                 if telegram_bot_flag:
                     telegram_bot.send_message(telegram_chat_id,"Bot Shutdown")
                 break
            #elif new_candle and last_entry_time > current_time:
            elif  previous_minute != current_minute and last_entry_time > current_time:
                print(current_time)
                previous_minute = current_minute
                if new_candle:
                    print(".")
                    # Step 1: Get current price of the index
                    index_price = get_current_price(index_name, flask_base_url,"index",flask_timeout=flask_timeout)
                    if band > 100:
                       atm_band  = 100
                    else:
                        atm_band = band
                    index_price_atm = round(index_price / atm_band) * atm_band
                    print(index_name.upper().replace(' ', ''))
                    contracts_df = get_contracts(index_price, index_name.upper().replace(' ', ''), number_of_strike_prices, band, atm_band)
                    logger.info(f"Index LTP:  {index_price}, ATM: {index_price_atm}")
                    #logger.info(f"Contracts: {contracts_df}")
                    logger.info(f" ")
                    time.sleep(1) #DELAY TO WAIT FOR CANDLE FORMATION

                    # Step 2 continued: Get required strike prices and contract names
                    print(contracts_df)
                    global_contracts_df = contracts_df

                    all_candles = check_and_fetch_missing_candles(contracts_df, time_frame,flask_base_url , market_start_time,algo_start_time,flask_timeout=flask_timeout)
                    condition_check ={}
                    # Step 6: Save each candle to a variable or data structure for further processing
                    # For demonstration, print the aggregated candles
                    for instrument, df in all_candles.items():
                        print(f"Checking for {instrument}")
                        logger.info(f"Last two candles of {instrument}")
                        try:
                            result = check_entry_conditions(df)
                        except Exception as e:
                            print(e)
                            logger.info(f"Error: {e}")
                            result = None
                        if result is not None:
                            condition_check[instrument] = result
                    #df.to_csv(instrument+".csv")
                    print(condition_check)
                    logger.info("-------------------------------------------------------------------------")
            elif previous_minute == current_minute and last_entry_time > current_time:
                ##if last_entry_time > current_time:
                    print("..")
                    #condition_check["NFO:BANKNIFTY09OCT24C53000"] = {'action':'buy','lower_limit':250,'upper_limit':420}
                    for instrument, data in condition_check.items():
                        print(instrument,data, open_position_flag)
                        if data['action'] == "buy" and not open_position_flag:
                            token_result = global_contracts_df[global_contracts_df['TradingSymbol'] == instrument[4:]]['Token']
                            lot_size_result = global_contracts_df[global_contracts_df['TradingSymbol'] == instrument[4:]]['LotSize']
                            try:
                                token = token_result.iloc[0]
                                lot_size = lot_size_result.iloc[0]
                                quantity = int(lot_size)*int(lots)
                                ltp = get_current_price(instrument,flask_base_url,"contract",token=token,flask_timeout=flask_timeout)
                                print("-----", ltp)
                                if  data['lower_limit'] < ltp < data['upper_limit']:
                                    sl_tmp = sl_percentage * float(data['compute1'])
                                    if open_position.get('entry_price', 10000000) > ltp:
                                        open_position = {'contract':instrument,'token':token,'entry_price':ltp,'quantity':quantity,'status':'pending','sl_tmp':sl_tmp}
                            except Exception as e:
                               print("Error while checking ", e)
                               print("Token Result: ", token_result)
                               print(lot_size_result)
                               break
                    if open_position.get('status',"open") == "pending":
                        #Enter Trade
                        order_result = execute_trade(open_position,paper_trading)
                        if order_result[0]:
                            if order_result[1] == 0:
                                ltp = open_position['entry_price']
                                open_position_flag = True
                            else:
                                open_position['order_id'] = order_result[1]
                                while True:
                                    order_status = api.single_order_status(order_result[1])
                                    order_status = order_status[0]
                                    if order_status['stat'] == "Ok":
                                        if order_status['status'] == "COMPLETE":
                                           ltp = float(order_status['avgprc'])
                                           logger.info(f"Entered at: {ltp}")
                                           open_position_flag = True
                                           break
                                        elif order_status['status'] == "REJECTED":
                                            logger.info(f"Entry order Rejected, Reason: {order_status['rejreason']}")
                                            if telegram_bot_flag:
                                                telegram_bot.send_message(telegram_chat_id,f"Entry order Rejected, Reason: {order_status['rejreason']}")
                                            close_bot = True
                                            break
                                    else:
                                       logger.info(f"Order Not placed. Reason Below")
                                       logger.info(f"{order_status}")
                                    time.sleep(0.2)
                            if open_position_flag:
                                logger.info(f"{sl_percentage} SL_Percentage Logger" )
                                sl = float(open_position['sl_tmp'])
                                logger.info(f"{sl}SL Logger ")
                                if trades_count == 0:
                                    target = (target_percentage*ltp/100)+ltp
                                else:
                                    target = abs(total_points) +  ((target_percentage*ltp/100)+ltp)
                                open_position['entry_price'] = ltp
                                open_position['sl']= sl
                                open_position['target'] = target
                                trades_count += 1
                                open_position['status'] = "open"
                                open_position['order_id'] = order_result
                                open_position['open_position_flag'] = open_position_flag
                                open_position['trades_count'] = trades_count
                                logger.info(f"Entered Trade No: {trades_count}")
                                logger.info(f"Contract: {open_position['contract']}, Entry: {open_position['entry_price']}, 'sl': {open_position['sl']}, 'target': {open_position['target']}, Order Id: {order_result[1]}, Total trades: {open_position.get('trades_count',0)} ")
                                if telegram_bot_flag:
                                    telegram_bot.send_message(telegram_chat_id,f"BUY Entry: Contract: {open_position['contract']}, Entry: {open_position['entry_price']}, 'sl': {open_position['sl']}, 'target': {open_position['target']}, Order Id: {order_result[1]}, Total trades: {open_position.get('trades_count',0)}, {trade_type_text} ")
                                    
                        else:
                            print(str(order_result[1]))
                            logger.info(f"Error entering trade: {order_result[1]}")
                            if telegram_bot_flag:
                                telegram_bot.send_message(telegram_chat_id,f"BUY trade Error entering: {order_result[1]}")
                            close_bot = True
            elif last_entry_time < current_time:
               if not last_entry_time_flag:
                   logger.info(f"Last entry time {last_entry_time} crossed. No new entries will be taken")
                   last_entry_time_flag = True
                   if telegram_bot_flag:
                       telegram_bot.send_message(telegram_chat_id,f"Last entry time {last_entry_time} crossed. No new entries will be taken")
 except Exception as e:
   logger.info(f"{e}")

if __name__ == "__main__":
    main()

