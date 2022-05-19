from typing import Optional

from fastapi import FastAPI
import pickle
import sklearn
import pandas as pd
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
app = FastAPI(
        title="Air Quality Prediction",
        description="Mine Air Quality Prediction",
        openapi_tags=[
            {"name": "AQI Prediction ", "description": "AQI prediction",},
        ],
    )
I_h1=dict()
I_l0=dict()
I_h1['v good']=50
I_h1['good']=100
I_h1['moderate']=200
I_h1['poor']=300
I_h1['v poor']=500
I_l0['v good']=0
I_l0['good']=50
I_l0['moderate']=100
I_l0['poor']=200
I_l0['v poor']=300


@app.get("/calculate_aqi",tags=['AQI Prediction Using Formula and ML Techniques'])
def calculate_AQI(Cp_So2:float,Cp_No2:float,Cp_PM_10:float,Cp_PM_25:float,temp_in_c:float,humidity:float):
    category=""
    if(Cp_So2<=2.5):
        Bp_H1_So2=2.5
        Bp_L0_So2=0
        category="v good"
    elif(Cp_So2<=4):
        Bp_H1_So2=4
        Bp_L0_So2=2.5
        category="good"
    elif(Cp_So2<=6):
        Bp_H1_So2=6
        Bp_L0_So2=4
        category="moderate"
    elif(Cp_So2<=8):
        Bp_H1_So2=8
        Bp_L0_So2=6
        category="poor"
    else:
        Bp_H1_So2=Cp_So2
        Bp_L0_So2=8
        category="v poor"
    Ip_So2=(((I_h1[category]-I_l0[category])*(Cp_So2-Bp_L0_So2))/(Bp_H1_So2-Bp_L0_So2))+I_l0[category]
    print(f"IpSo2= {Ip_So2}")
#Delta for No2 
    category=""
    if(Cp_No2<=1):
        Bp_H1_No2=1
        Bp_L0_No2=0
        category="v good"
    elif(Cp_No2<=2):
        Bp_H1_No2=2
        Bp_L0_No2=1
        category="good"
    elif(Cp_No2<=3):
        Bp_H1_No2=3
        Bp_L0_No2=2
        category="moderate"
    elif(Cp_No2<=4):
        Bp_H1_No2=4
        Bp_L0_No2=3
        category="poor"
    else:
        Bp_H1_No2=Cp_No2
        Bp_L0_No2=4
        category="v poor"
    Ip_No2=(((I_h1[category]-I_l0[category])*(Cp_No2-Bp_L0_No2))/(Bp_H1_No2-Bp_L0_No2))+I_l0[category]
    print(f"IpNo2= {Ip_No2}")

#Delta for PM_10 
    if(Cp_PM_10<=54):
        Bp_H1_PM_10=54
        Bp_L0_PM_10=0
        category="v good"
    elif(Cp_PM_10<=154):
        Bp_H1_PM_10=154
        Bp_L0_PM_10=54
        category="good"
    elif(Cp_PM_10<=254):
        Bp_H1_PM_10=254
        Bp_L0_PM_10=154
        category="moderate"
    elif(Cp_PM_10<=354):
        Bp_H1_PM_10=354
        Bp_L0_PM_10=254
        category="poor"
    else:
        Bp_H1_PM_10=Cp_PM_10
        Bp_L0_PM_10=354
        category="v poor"
    Ip_PM_10=(((I_h1[category]-I_l0[category])*(Cp_PM_10-Bp_L0_PM_10))/(Bp_H1_PM_10-Bp_L0_PM_10))+I_l0[category]
    print(f"Ip_PM10= {Ip_PM_10}")



#Delta for PM_25 
    if(Cp_PM_25<=15.4):
        Bp_H1_PM_25=15.4
        Bp_L0_PM_25=0
        category="v good"
    elif(Cp_PM_25<=40.4):
        Bp_H1_PM_25=40.4
        Bp_L0_PM_25=15.4
        category="good"
    elif(Cp_PM_25<=65.4):
        Bp_H1_PM_25=65.4
        Bp_L0_PM_25=40.4
        category="moderate"
    elif(Cp_PM_25<=150.4):
        Bp_H1_PM_25=150.4
        Bp_L0_PM_25=65.4
        category="poor"
    else:
        Bp_H1_PM_25=Cp_PM_25
        Bp_L0_PM_25=150.4
        category="v poor"
    Ip_PM_25=(((I_h1[category]-I_l0[category])*(Cp_PM_25-Bp_L0_PM_25))/(Bp_H1_PM_25-Bp_L0_PM_25))+I_l0[category]
    print(f"Ip_PM25= {Ip_PM_25}")

    temp_in_f=((temp_in_c*9)/5)+32
    a=2.0491523*temp_in_f
    b=10.14333127*humidity
    c=0.22475541*temp_in_f*humidity
    d=(6.83783/1000)*temp_in_f*temp_in_f
    e=(5.481717/100)*humidity*humidity
    f=(1.22874/1000)*temp_in_f*temp_in_f*humidity
    g=(8.5282/10000)*temp_in_f*humidity*humidity
    h=(1.99/1000000)*temp_in_f*temp_in_f*humidity*humidity
    TCI=-42.379+a+b-c-d+-e+f+g-h
    print(f"TCI= {TCI}")

    MAQI=(Ip_No2+Ip_PM_10+Ip_PM_25+Ip_So2)/4
    print(f"MAQI :{MAQI}")
    MeI=(0.7*MAQI)+(0.3*TCI)
    print(f"MEI:{MeI}")

    lr = pickle.load(open("lr.sav", 'rb'))
    mlp = pickle.load(open("mlp.sav", 'rb'))
    xgb_model = pickle.load(open("xgb.sav", 'rb'))
    knn_model = pickle.load(open("knn.sav", 'rb'))
    input_list=[Cp_PM_10,Cp_PM_25,Cp_So2,Cp_No2,humidity,temp_in_c]
    lr_output=lr.predict([input_list])
    lr_output=lr_output.tolist()[0]
    arr=np.array([input_list])
    xgb_output=xgb_model.predict(arr)
    xgb_output=xgb_output.tolist()[0]
    mlp_output=mlp.predict([input_list])
    mlp_output=mlp_output.tolist()[0]
    knn_output=knn_model.predict([input_list])
    knn_output=knn_output.tolist()[0]
    return({"MeI Using Formula":MeI,"MeI Using Linear Regression":lr_output[0],"MeI Using MLP Regressor":mlp_output,"MeI Using KNN Regressor": knn_output[0],
    "MeI Using XgBoost Regressor":xgb_output
    })



