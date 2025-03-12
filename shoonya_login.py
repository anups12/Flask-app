from NorenApi import NorenApi
import pyotp
from datetime import datetime, timedelta, timezone
import pytz
import time
import pandas as pd
import logging

t1=datetime.now()
api = NorenApi()
api.token_setter()  # Make sure this sets the necessary tokens
def Shoonya_login():
    #credentials
    api = NorenApi()
    user    = 'FA17716'
    pwd     = 'Chart@123'
    vc      = 'FA17716_U'
    otp_token = "C3EXZ75245I5Z7K2675XHF3U33622762"
    factor2=pyotp.TOTP(otp_token).now()
    app_key = '530b1f3fefda775b2aa1a05a722ecc47'
    imei    = 'abc1234'
    accesstoken = ''

    #make the api call
    ret = api.login(userid=user, password=pwd, twoFA=factor2, vendor_code=vc, api_secret=app_key, imei=imei)
    #print(ret)
    print("User  Token      :", ret['susertoken'])
    print("Login API Status:", ret['stat'])
    #print(api)
    if ret['stat'] != None:
        print("Login Successful")

Shoonya_login()
t2=datetime.now()
print(t2-t1)