# USED TO RENAME THE LOG FILE DAILY AND PLACE THE ORDERS IN TEH MARKET
import os
import datetime

def rename_previous_log_file(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Extract directory, filename, and extension
        directory, file_name = os.path.split(file_path)
        file_base, file_extension = os.path.splitext(file_name)

        # Get the current date and time in a suitable format
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct the new file name
        new_file_name = f"{file_base}_{current_time}{file_extension}"

        # Create a backup directory if it doesn't exist
        backup_dir = os.path.join(directory, "backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # Create the new file path in the backup directory
        new_file_path = os.path.join(backup_dir, new_file_name)

        # Rename (and effectively move) the file to the backup directory
        os.rename(file_path, new_file_path)
        print(f"File renamed and moved to: {new_file_path}")
    else:
        print(f"The file '{file_path}' does not exist.")


def placeOrder(inst ,t_type,qty,order_type,price,variety, api,papertrading=0):
    exch = inst[:3]
    symb = inst[4:]
    #paperTrading = 0 #if this is 1, then real trades will be placed
    if( t_type=="BUY"):
        t_type="B"
    else:
        t_type="S"

    if(order_type=="MARKET"):
        order_type="MKT"
        price = 0
    elif(order_type=="LIMIT"):
        order_type="LMT"

    try:
        if(papertrading == 0):
            print(t_type)
            print(exch)
            print(symb)
            print(qty)
            print(order_type)
            print(price)
            order_id = api.place_order(buy_or_sell=t_type,  #B, S
                                       product_type="I", #C CNC, M NRML, I MIS
                                       exchange=exch,
                                       tradingsymbol=symb,
                                       quantity = qty,
                                       discloseqty=qty,
                                       price_type= order_type, #LMT, MKT, SL-LMT, SL-MKT
                                       price = price,
                                       trigger_price=price,
                                       amo="NO",#YES, NO
                                       retention="DAY"
                                       )
            print(" => ", symb , order_id['norenordno'] )
            print(order_id)
            return [True,order_id['norenordno']]

        else:
            order_id=0
            return [True,order_id]

    except Exception as e:
        print(" => ", symb , "Failed : {} ".format(e))
        return [False,str(e)]
