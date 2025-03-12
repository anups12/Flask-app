#WEBSOCKET AND FLASK WEB SERVER FILE, RUN AT 9:11 AM THOUGH CRONJON (crontab -e)
import os
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
import pandas as pd
from flask import Flask, request, jsonify
import pytz
import yaml
from NorenApi import NorenApi  # Ensure that NorenApi is properly imported
#import shoonya_login as login    # Ensure shoonya_login is available


current_directory = os.path.dirname(os.path.abspath(__file__))

#GET THE LIST OF CONTRACTS, BASED ON INDEX PRICE, BAND, SYMBOL AND THE NUMBER OF STRIKES, BASED ON filtered_data.csv file
def get_contracts(last_price, symbol, n, band):
    try:
        # Read the data from the filtered CSV file
        df = pd.read_csv(os.path.join(current_directory,"filtered_data.csv"))

        # Ensure 'StrikePrice' is numeric
        df['StrikePrice'] = pd.to_numeric(df['StrikePrice'], errors='coerce')
        df = df.dropna(subset=['StrikePrice'])

        # Filter the DataFrame for the given symbol
        symbol_df = df[df['Symbol'] == symbol]

        # Initialize an empty list to store the results
        results = []

        # Loop over both Option Types: 'CE' and 'PE'
        for option_type in ['CE', 'PE']:
            # Filter for the current Option Type
            option_df = symbol_df[symbol_df['OptionType'] == option_type]

            # Filter for strike prices that are multiples of 'band'
            option_df = option_df[(option_df['StrikePrice'] % band).abs() < 1e-8]

            # Get contracts with StrikePrice above or equal to last_price
            above_df = option_df[option_df['StrikePrice'] >= last_price]
            above_df = above_df.sort_values('StrikePrice').head(n)

            # Get contracts with StrikePrice below last_price
            below_df = option_df[option_df['StrikePrice'] <= last_price]
            below_df = below_df.sort_values('StrikePrice', ascending=False).head(n)

            # Append the above and below DataFrames to the results list
            results.extend([above_df, below_df])

        # Concatenate all results into a single DataFrame
        final_df = pd.concat(results).drop_duplicates()
        final_df.reset_index(drop=True, inplace=True)

        # Create the dictionary {"NFO:TradingSymbol": "NFO|Token"}
        contract_dict = {
            f"NFO:{row['TradingSymbol']}": f"NFO|{row['Token']}"
            for _, row in final_df.iterrows()
        }

        return contract_dict

    except FileNotFoundError:
        print("Error: 'filtered_data.csv' file not found.")
        return {}
    except KeyError as e:
        print(f"Error: Missing column {e} in the data.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}
        
class MarketDataStreamer:
    def __init__(self, config_file=os.path.join(current_directory,"config.yaml")):
        # Load configuration
        self.config = self.load_config(config_file)
        self.t1 = datetime.now()
        # Initialize variables
        self.api = NorenApi()
        self.LTPDICT = {}
        self.CANDLE_DATA = {}
        self.CANDLE_DATA_COMPLETE = {}
        self.symbolDict = {}
        self.instrumentList = []
        self.discon = []
        self.tickList = []
        self.tickDict = {}
        for stl in self.tickList:
            self.tickDict[stl] = []
        print(self.tickDict)
        # Initialize the Shoonya API
        self.login()

        # Generate symbols based on configuration
        self.generate_symbols()

        # Initialize Flask app
        self.app = Flask(__name__)
        self.setup_routes()

    def load_config(self, config_file):
        # Read the config file
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def login(self):
        # Login to the Shoonya API
        #login.Shoonya_login()
        #time.sleep(0.2)
        self.api.token_setter()  # Make sure this sets the necessary tokens

    def generate_symbols(self):
        # Access configuration parameters
        Flag_Nifty = self.config['Flag_Nifty']
        Flag_Banknifty = self.config['Flag_Banknifty']
        Flag_Finnifty = self.config['Flag_Finnifty']

        Band_Nifty = self.config['Band_Nifty']
        Band_Banknifty = self.config['Band_Banknifty']
        Band_Finnifty = self.config['Band_Finnifty']

        Num_of_ITM_strikes = self.config['Num_of_websocket_strikes']

        # Generate symbols for Nifty, Bank Nifty, and Fin Nifty
        if Flag_Nifty == 1:
            ltp_nifty = float(self.api.get_quotes(exchange="NSE", token="26000")['lp'])
            print("NIFTY LTP:", ltp_nifty)
            self.symbolDict.update(get_contracts(ltp_nifty, "NIFTY", Num_of_ITM_strikes, Band_Nifty))

        if Flag_Banknifty == 1:
            ltp_banknifty = float(self.api.get_quotes(exchange="NSE", token="26009")['lp'])
            print("BANKNIFTY LTP:", ltp_banknifty)
            self.symbolDict.update(get_contracts(ltp_banknifty, "BANKNIFTY", Num_of_ITM_strikes, Band_Banknifty))

        if Flag_Finnifty == 1:
            ltp_finnifty = float(self.api.get_quotes(exchange="NSE", token="26037")['lp'])
            print("FINNIFTY LTP:", ltp_finnifty)
            self.symbolDict.update(get_contracts(ltp_finnifty, "FINNIFTY", Num_of_ITM_strikes, Band_Finnifty))

        # Add indices to symbolDict
        self.symbolDict['NSE:Nifty 50'] = "NSE|26000"
        self.symbolDict['NSE:Nifty Bank'] = "NSE|26009"
        self.symbolDict['NSE:Nifty Fin Service'] = "NSE|26037"

        self.instrumentList = list(self.symbolDict.values())
        #print("Symbol to Token Map:")
        #for k in self.symbolDict.keys():
         #   print(k)
        #print('------------------------')
        #print("Instrument List:")
        #print(self.instrumentList)

    # Application callbacks
    def event_handler_order_update(self, message):
        print("ORDER :", time.strftime('%d-%m-%Y %H:%M:%S'), message)

    def event_handler_quote_update(self, tick):
        # Handle quote updates
        key = tick['e'] + '|' + tick['tk']
        ltp = tick['lp']
        self.LTPDICT[key] = ltp

        # Extract tick timestamp
        tick_timestamp = self.get_tick_timestamp(tick)
        if tick_timestamp is None:
            # If timestamp is not available, skip processing this tick
            return
        if key in self.tickList:
            self.tickDict[key].append({"time":tick_timestamp,'ltp':ltp})
        # Extract volume from tick
        volume = self.get_tick_volume(tick)
        if volume is None:
            # If volume is not available, set volume to zero
            volume = 0

        # Update candle data
        self.update_candle_data(key, ltp, tick_timestamp, volume)

    def get_tick_timestamp(self, tick):
        # Extract the timestamp from the tick data
        try:
            timestamp = int(tick['ft'])

            # Convert UNIX timestamp to UTC datetime (Recommended way)
            utc_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

            # Convert UTC datetime to Indian Standard Time (IST)
            india_tz = pytz.timezone('Asia/Kolkata')
            tick_timestamp = utc_dt.astimezone(india_tz)

            # Round down to the nearest minute
            tick_timestamp = tick_timestamp.replace(second=0, microsecond=0)

            return tick_timestamp
        except KeyError:
            print("Timestamp not found in tick data.")
            return None
        except ValueError:
            print("Timestamp format error.")
            return None
        except Exception as e:
            print(e)
            return None

    def get_tick_volume(self, tick):
        # Extract volume from the tick data
        try:
            volume = float(tick.get('v', 0))
            return volume
        except (KeyError, ValueError):
            print("Volume not found or invalid in tick data.")
            return 0

    def update_candle_data(self, symbol, ltp, tick_timestamp, volume):
        candle_time = tick_timestamp.replace(second=0, microsecond=0)
        if symbol not in self.CANDLE_DATA:
            # Initialize candle data for the symbol
            start_volume = volume if volume != 0 else None  # Set start_volume only if volume is not zero
            self.CANDLE_DATA[symbol] = {
                'timestamp': candle_time,
                'open': ltp,
                'high': ltp,
                'low': ltp,
                'close': ltp,
                'start_volume': start_volume,
                'end_volume': start_volume
            }
            self.CANDLE_DATA_COMPLETE[symbol] = []
        else:
            candle = self.CANDLE_DATA[symbol]
            if candle['timestamp'] == candle_time:
                # Update existing candle
                candle['high'] = max(candle['high'], ltp)
                candle['low'] = min(candle['low'], ltp)
                candle['close'] = ltp

                if volume != 0:
                    if candle['start_volume'] is None:
                        candle['start_volume'] = volume  # Set start_volume if it's the first valid volume
                    candle['end_volume'] = volume  # Update end_volume with the latest valid volume
            else:
                # Candle for the previous minute is complete
                if candle.get('start_volume') is not None and candle.get('end_volume') is not None:
                    candle_volume = candle['end_volume'] - candle['start_volume']
                    # Ensure volume is not negative
                    candle_volume = max(candle_volume, 0)
                else:
                    candle_volume = 0  # Volume is zero if we didn't receive valid volume data

                # Append completed candle to CANDLE_DATA_COMPLETE
                self.CANDLE_DATA_COMPLETE[symbol].append({
                    "time": candle['timestamp'],
                    "open": candle['open'],
                    "high": candle['high'],
                    "low": candle['low'],
                    "close": candle['close'],
                    "volume": candle_volume
                })

                # Initialize new candle
                start_volume = candle.get('end_volume', None)
                end_volume = volume if volume != 0 else None
                self.CANDLE_DATA[symbol] = {
                    'timestamp': candle_time,
                    'open': ltp,
                    'high': ltp,
                    'low': ltp,
                    'close': ltp,
                    'start_volume': start_volume,
                    'end_volume': end_volume
                }

    def open_callback(self):
        print('WebSocket connection opened.')
        utc_now =datetime.now(pytz.utc)
        ist_timezone = pytz.timezone('Asia/Kolkata')
        ist_now = utc_now.astimezone(ist_timezone)
        formatted_time = ist_now.strftime('%H:%M:%S.%f')[:-3]
        self.discon.append({'Connected':formatted_time})
        # Subscribe to instruments
        self.api.subscribe(self.instrumentList)

    def close_callback(self):
        print('WebSocket connection closed.')
        utc_now = datetime.now(pytz.utc)
        ist_timezone = pytz.timezone('Asia/Kolkata')
        ist_now = utc_now.astimezone(ist_timezone)
        formatted_time = ist_now.strftime('%H:%M:%S.%f')[:-3]
        self.discon.append({"disconnected":formatted_time})
        os._exit(1)

    # Flask server setup
    def setup_routes(self):
        @self.app.before_request
        def validate_passcode():
            # Allow requests from localhost without passcode check
            if request.remote_addr == '127.0.0.1':
                return  # Proceed with the request

            # Check for 'passcode' in query parameters for non-localhost requests
            passcode = request.args.get('passcode')
            if passcode != self.config['Flask_passcode']:
                # Return error if passcode is missing or incorrect
                return jsonify({"error": "Invalid request"}), 403
                
        @self.app.route('/')
        def hello_world():
            return 'Hello World'

        @self.app.route('/ltp')
        def getLtp():
            try:
                #os._exit(1)
                instrument = request.args.get('instrument')
                ltp = self.LTPDICT[self.symbolDict[instrument]]
                return str(ltp)
            except Exception as e:
                print("Exception occurred while getting LTPDICT()")
                print(e)
                return jsonify([]), 201

        @self.app.route('/candles')
        def getCandle():
            try:
                #return jsonify([]), 201
                instrument = request.args.get('instrument')
                candles = self.CANDLE_DATA_COMPLETE[self.symbolDict[instrument]]
                for candle in candles:
                    if isinstance(candle['time'], datetime):
                        candle['time'] = candle['time'].isoformat()

                        #TODO find this check how much data is available here


                # Return the last 3 candles
                if len(candles) > 1:
                    print("DELETE ME CANDLES", candles)
                    return jsonify(candles[1:]), 200
                else:
                    return jsonify([]), 201
            except Exception as e:
                print("Exception occurred while getting candle data")
                print(e)
                return jsonify([]), 201

        @self.app.route('/ticks')
        def getTicks():
            try:
                instrument = request.args.get('instrument')
                candles = self.tickDict[self.symbolDict[instrument]]
                for candle in candles:
                    if isinstance(candle['time'], datetime):
                        candle['time'] = candle['time'].isoformat()
                # Return the last 3 candles
                return jsonify(candles)
            except Exception as e:
                print("Exception occurred while getting candle data")
                print(e)
                return jsonify([]),201

        @self.app.route('/discon')
        def getDiscon():
            return jsonify(self.discon), 200

        @self.app.route('/symbols')
        def getSymbols():
            return jsonify(self.symbolDict), 200

    def start_server(self):
        print("Starting Flask Server...")
        # Start the Flask app in a separate thread
        threading.Thread(target=self.app.run, kwargs={'host': '0.0.0.0', 'port': 5001}, daemon=True).start()

    def main(self):
        # Start the Flask server
        self.start_server()
        time.sleep(1)

        # Start the WebSocket
        self.api.start_websocket(
            order_update_callback=self.event_handler_order_update,
            subscribe_callback=self.event_handler_quote_update,
            socket_open_callback=self.open_callback,
            socket_close_callback=self.close_callback
        )

        # Keep the main thread alive
        while True:
            time.sleep(1)

if __name__ == "__main__":
    streamer = MarketDataStreamer()
    streamer.main()




