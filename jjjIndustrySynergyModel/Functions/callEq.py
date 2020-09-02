#!/usr/bin/env python
# -*- coding: utf-8 -*-

#引包：引入所需python包
import xlrd
import os
import re
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

from sklearn.metrics import roc_curve,auc
import matplotlib as mpl
from matplotlib import pyplot as plt
from numpy import nan
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import seaborn as sns
import time
from collections import Counter   #引入Counter
from linearmodels import IV3SLS
from scipy.stats import kstest
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

def callEq(Eqid,exogs = exogs ,df_ev = df_ev, preds = preds):
    global exogs
    global df_ev
    global preds
    global Eqs
    #print(len(exogs))
    #print(df_ev.shape)   
    #print(preds.shape)
    
    if Eqid == "Eq1":
        #-----1.判断内生变量还未生成，并且自变量数据完整；
        y = "gdp1r_bj"
        x_list = ["cul_area_bj","d_gdp1r_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
            #-----2 :建立OLS模型
            test1 = df_ev[["gdp1r_bj","cul_area_bj","d_gdp1r_bj"]]
            test1["gdplr_bj_s1"] = test1["gdp1r_bj"].shift(1)  ##直接写进表达式，结果有差异
            test1 = test1[(test1.index>1996) & (test1.index <2018)]
            results = sm.ols(formula = "gdp1r_bj ~ cul_area_bj  + gdplr_bj_s1 + d_gdp1r_bj",data = test1).fit()
            # 衍生变量
            
            #------3 : 写结果
            test1[y] = results.predict()
               # 预测结果
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)
               # 更新数据源相应外生变量，方便后面作为变量进行其他模型拟合
            df_ev[y] = preds[y]
               # 将该内生变量从内生变量清单中删除 ，已证明内生变量只会计算一次 ：
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )         

        
    if Eqid == "Eq2": #建立定义模型
            y = "gdpn_bj"
            x_list = ["gdp1n_bj","gdp2n_bj","gdp3n_bj"]
            if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
                
           # try :
                #-----2 :
                test1 = df_ev[["gdpn_bj","gdp1n_bj","gdp2n_bj","gdp3n_bj"]]
                #test1 = test1[(test1.index>1996) & (test1.index <2018)]
                #------3 : 写结果
                test1[y] = test1["gdp1n_bj"] + test1["gdp2n_bj"] + test1["gdp3n_bj"]
                # 预测结果
                preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)
                # 更新数据源相应外生变量，方便后面作为变量进行其他模型拟合
                df_ev[y] = preds[y]
                # 将该内生变量从内生变量清单中删除 ，已证明内生变量只会计算一次 ：
                exogs.remove(y)
                Eqs.remove(Eqid)
                print(Eqid ,y,"----------------   求解成功 。")
            else:
                print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 

    if Eqid == "Eq3": #建立定义模型
            y = "gdpr_bj"
            x_list =["gdp1r_bj","gdp2r_bj","gdp3r_bj"]
            if (y in exogs) &( len([x for x in x_list if x in exogs])==0): 
           # try :                
                test1 = df_ev[["gdpr_bj","gdp1r_bj","gdp2r_bj","gdp3r_bj"]]
                #test1 = test1[(test1.index>1996) & (test1.index <2018)]                
                test1[y] = test1["gdp1r_bj"] + test1["gdp2r_bj"] + test1["gdp3r_bj"]             
                preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
                df_ev[y] = preds[y]                 
                exogs.remove(y)
                Eqs.remove(Eqid)
                print(Eqid ,y," ----------------   求解成功 。")
            else:
                print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
    if Eqid == "Eq4": #建立定义模型
            y = "gdp1n_bj"
            x_list =["gdp1r_bj","gdp1d_bj"]
            if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
           # try :                
                test1 = df_ev[["gdp1n_bj","gdp1r_bj","gdp1d_bj"]]
                #test1 = test1[(test1.index>1996) & (test1.index <2018)]
                test1["y"] = test1["gdp1r_bj"]*test1["gdp1d_bj"]/100            
                preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
                df_ev[y] = preds[y]                 
                exogs.remove(y)
                Eqs.remove(Eqid)
                print(Eqid ,y," ----------------   求解成功 。")
            else:
                print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )    
    if Eqid == "Eq5": #建立定义模型
        y = "gdp2_indan_bj"
        x_list =["gdp2_indar_bj","gdp2_indad_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp2_indan_bj","gdp2_indar_bj","gdp2_indad_bj"]]
            #test1 = test1[(test1.index>1996) & (test1.index <2018)]
            test1[y] = test1["gdp2_indar_bj"]*test1["gdp2_indad_bj"]/100           
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 

    if Eqid == "Eq6": #建立定义模型
        y = "gdp2_bindan_bj"
        x_list =["gdp2_bindar_bj","gdp2_bindad_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp2_bindan_bj","gdp2_bindar_bj","gdp2_bindad_bj"]]
            test1 = test1[(test1.index>1996) & (test1.index <2018)]
            test1[y] = test1["gdp2_bindar_bj"]*test1["gdp2_bindad_bj"]/100          
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
            
    if Eqid == "Eq7": #建立定义模型
        y = "gdp2n_bj"
        x_list =["gdp2r_bj","gdp2d_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp2n_bj","gdp2r_bj","gdp2d_bj"]]
            #test1 = test1[(test1.index>1996) & (test1.index <2018)]
            test1[y] = test1["gdp2r_bj"]*test1["gdp2d_bj"]/100        
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq8": #建立定义模型
        y = "gdp2r_bj2"
        x_list =["gdp2_indar_bj","gdp2_bindar_bj","gdp2r_erro_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp2r_bj2","gdp2_indar_bj","gdp2_bindar_bj","gdp2r_erro_bj"]]
            #test1 = test1[(test1.index>1996) & (test1.index <2018)]
            test1[y] = test1["gdp2_indar_bj"] + test1["gdp2_bindar_bj"] + test1["gdp2r_erro_bj"]       
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 

    if Eqid == "Eq9":       
        y = "gdp2r_bj1"
        x_list = ["gdp_tecn_bj","gdp2d_bj","d_gdp2r_bj_1"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp2r_bj1","gdp_tecn_bj","gdp2d_bj","d_gdp2r_bj_1"]]
            test1["gdp2r_bj1_s1"] = test1["gdp2r_bj1"].shift(1)
            test1["gdp_tecn_gdp2d"] = test1["gdp_tecn_bj"]/test1["gdp2d_bj"]
            test1 = test1[(test1.index>2003) & (test1.index <2019)]
            results = sm.ols(formula = "gdp2r_bj1 ~ gdp_tecn_gdp2d  + gdp2r_bj1_s1 + d_gdp2r_bj_1",data = test1 ).fit()            
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq10": #建立定义模型
        y = "gdp2r_bj"
        x_list =["gdp2r_bj1","gdp2r_bj2"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp2r_bj","gdp2r_bj1","gdp2r_bj2"]]
            #test1 = test1[(test1.index>1996) & (test1.index <2018)]
            test1[y] = (test1["gdp2r_bj1"] + test1["gdp2r_bj2"])/2      
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )

    if Eqid == "Eq11": #建立定义模型
        y = "gdp3n_bj"
        x_list =["gdp3r_bj","gdp3d_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3n_bj","gdp3r_bj","gdp3d_bj"]]
            #test1 = test1[(test1.index>1996) & (test1.index <2018)]
            test1[y] = (test1["gdp3r_bj"] * test1["gdp3d_bj"])/100     
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 

    if Eqid == "Eq12":       
        y = "gdp2_indar_bj"
        x_list = ["consr_bj","investr_bj","exusd_bj","exrate_cn","d_gdp2_indar_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp2_indar_bj","consr_bj","investr_bj","exusd_bj","exrate_cn","d_gdp2_indar_bj"]]
            test1 = test1[(test1.index>1995) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(gdp2_indar_bj) ~ np.log(consr_bj)  + np.log(investr_bj) + np.log(exusd_bj * exrate_cn) + d_gdp2_indar_bj",data = test1 ).fit()           
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )   
            
    if Eqid == "Eq13": 
        y = "gdp2_bindar_bj"
        x_list = ["investr_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp2_bindar_bj","investr_bj"]]
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            ar = (0,1) 
            ma = 0
            mod = SARIMAX(endog = np.log(test1["gdp2_bindar_bj"])  , exog = np.log(test1["investr_bj"]) ,order =(ar,0,ma),enforce_invertibility= False , trend = 'c', enforce_stationarity = False)
            results = mod.fit()
            test1[y] = np.exp(results.predict())[2:]  # 和AR系数有关             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )   
            
    if Eqid == "Eq14":       
        y = "gdp3r_bj"
        x_list = ["gdp2r_bj","exusd_bj","exrate_cn"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3r_bj","gdp2r_bj","exusd_bj","exrate_cn"]]
            test1["exusexrte"] = test1["exusd_bj"]*test1["exrate_cn"]
            test1["gdp3r_bj_s1"] = test1["gdp3r_bj"].shift(1)           
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(gdp3r_bj) ~ np.log(gdp2r_bj)  + np.log(exusexrte) +  np.log(gdp3r_bj_s1)",data = test1 ).fit()
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
           
    if Eqid == "Eq15": #建立定义模型
        y = "gdp3r_bj"
        x_list =["con_pr_bj" , "con_gr_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["con_fr_bj" ,"con_pr_bj" , "con_gr_bj"]]
            test1[y] = (test1["con_pr_bj"] + test1["con_gr_bj"])   
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )   
    if Eqid == "Eq16": #建立定义模型
        y = "con_gr_bj"
        x_list =["con_gn_bj" , "pcon_gd_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["con_gr_bj" ,"con_gn_bj" , "pcon_gd_bj"]]
            test1[y] = (test1["con_gn_bj"] /test1["pcon_gd_bj"])*100  
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq17": #建立定义模型
        y = "con_pr_bj"
        x_list =["con_pur_bj" , "con_prr_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["con_pr_bj" ,"con_pur_bj" , "con_prr_bj"]]
            test1[y] = (test1["con_pur_bj"] + test1["con_prr_bj"]) 
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
    if Eqid == "Eq18": #建立定义模型
        y = "con_fn_bj"
        x_list =["con_pn_bj" , "con_gn_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["con_fn_bj" ,"con_pn_bj" , "con_gn_bj"]]
            test1["预测"] = (test1["con_pn_bj"] + test1["con_gn_bj"])
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
    if Eqid == "Eq19": #建立定义模型
        y = "con_pn_bj"
        x_list =["con_pun_bj" , "con_prn_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["con_pn_bj" ,"con_pun_bj" , "con_prn_bj"]]
            test1[y] = (test1["con_pun_bj"] + test1["con_prn_bj"])
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
            
    if Eqid == "Eq20": #建立定义模型
        y = "con_pun_bj"
        x_list =["con_pur_bj","pcon_pud_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["con_pun_bj" ,"con_pur_bj","pcon_pud_bj"]]
            test1[y] = (test1["con_pun_bj"] + test1["con_prn_bj"])
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq21": #建立定义模型
        y = "con_prn_bj"
        x_list =["con_prr_bj" , "pcon_prd_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["con_prn_bj" ,"con_prr_bj" , "pcon_prd_bj"]]
            test1[y] = (test1["con_prr_bj"] * test1["pcon_prd_bj"])/100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )      
            
    if Eqid == "Eq22":       
        y = "con_gn_bj"
        x_list = ["fen_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["con_gn_bj","fen_bj"]]
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "con_gn_bj ~ fen_bj",data = test1).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq23": #建立定义模型
        y = "investn_bj"
        x_list =["investr_bj" , "faipi_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["investn_bj" ,"investr_bj" , "faipi_bj"]]
            test1[y] = (test1["investr_bj"] * test1["faipi_bj"]) /100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
            
    if Eqid == "Eq24": #建立定义模型
        y = "inv_fon_bj"
        x_list =["inv_for_bj" , "pinv_fod_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["inv_fon_bj" ,"inv_for_bj" , "pinv_fod_bj"]]
            test1[y] = (test1["inv_for_bj"] * test1["pinv_fod_bj"])/100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
                  
    if Eqid == "Eq25": #建立定义模型
        y = "inv_chn_bj"
        x_list =["inv_chr_bj" , "pinv_chd_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["inv_chn_bj" ,"inv_chr_bj" , "pinv_chd_bj"]]
            test1[y] = (test1["inv_chr_bj"] * test1["pinv_chd_bj"])/100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])
                  
    if Eqid == "Eq26": #建立定义模型
        y = "consr_bj"
        x_list =["consn_bj" , "rpi_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["consr_bj" ,"consn_bj" , "rpi_bj"]]
            test1[y] = (test1["consn_bj"] / test1["rpi_bj"])*100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])
            
    if Eqid == "Eq27": #建立定义模型
        y = "inv_for_bj"
        x_list =["inv_for_bj" ,"inv_fir_bj" , "inv_chr_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["inv_for_bj" ,"inv_fir_bj" , "inv_chr_bj"]]
            test1[y] = test1["inv_fir_bj"] + test1["inv_chr_bj"]
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])           
 
    if Eqid == "Eq28": #建立定义模型
        y = "inv_fin_bj"
        x_list =["inv_fir_bj" , "pinv_fid_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["inv_fin_bj" ,"inv_fir_bj" , "pinv_fid_bj"]]
            test1[y] = (test1["inv_fir_bj"] * test1["pinv_fid_bj"])/100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])  
            
    if Eqid == "Eq29": #建立定义模型
        y = "gdper_bj"
        x_list =["con_fr_bj","inv_for_bj","nex_gsr_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdper_bj" ,"con_fr_bj","inv_for_bj","nex_gsr_bj"]]
            test1[y] = test1["con_fr_bj"] + test1["inv_for_bj"] + test1["nex_gsr_bj"]
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])    
            
    if Eqid == "Eq30": #建立定义模型
        y = "gdpen_bj"
        x_list =["con_fn_bj" , "inv_fon_bj","nex_gsn_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdpen_bj" ,"con_fn_bj" , "inv_fon_bj","nex_gsn_bj"]]
            test1[y] = test1["con_fn_bj"] + test1["inv_fon_bj"] + test1["nex_gsn_bj"]
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs]) 
            
    if Eqid == "Eq31": #建立定义模型
        y = "nex_gsn_bj"
        x_list =["nex_gsr_bj" , "pnex_gsd_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["nex_gsn_bj" ,"nex_gsr_bj" , "pnex_gsd_bj"]]
            test1[y] = (test1["nex_gsr_bj"] * test1["pnex_gsd_bj"])/100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])             
            
    if Eqid == "Eq32":       
        y = "con_pur_bj"
        x_list = ["yht_un_bj" , "pcon_pud_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["con_pur_bj" ,"yht_un_bj" , "pcon_pud_bj"]]
            test1["yht_pcon_100"] = test1["yht_un_bj"]/test1["pcon_pud_bj"]*100
            test1 = test1[(test1.index>2005) & (test1.index <2019)]
            results = sm.ols(formula = "con_pur_bj ~ yht_pcon_100",data = test1 ).fit()           
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq33":       
        y = "con_prr_bj"
        x_list = ["yht_rn_bj" , "pcon_prd_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["con_prr_bj" ,"yht_rn_bj" , "pcon_prd_bj"]]
            test1["yht_pcon_100"] = test1["yht_rn_bj"]/test1["pcon_prd_bj"]*100
            test1["diff_con_prr"] = test1["con_prr_bj"].shift(1)-test1["con_prr_bj"].shift(2)
            test1 = test1[(test1.index>2007) & (test1.index <2019)]
            results = sm.ols(formula = "con_prr_bj ~ yht_pcon_100 + diff_con_prr",data = test1 ).fit()          
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 


    if Eqid == "Eq34":       
        y = "investr_bj"
        x_list = ["loann_bj" , "faipi_bj","d_investr_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["investr_bj" ,"loann_bj" , "faipi_bj","d_investr_bj"]]
            test1["loan_fai_100"] = test1["loann_bj"]/test1["faipi_bj"]*100
            test1 = test1[(test1.index>2005) & (test1.index <2019)]
            results = sm.ols(formula = "investr_bj ~ loan_fai_100 + d_investr_bj",data = test1 ).fit()          
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq35":       
        y = "inv_fir_bj"
        x_list = ["investr_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["inv_fir_bj" ,"investr_bj" ]]
            test1["inv_fir_bj_s1"] = test1["inv_fir_bj"].shift(1)
            test1 = test1[(test1.index>2006) & (test1.index <2019)]
            results = sm.ols(formula = "inv_fir_bj ~ investr_bj + inv_fir_bj_s1",data = test1 ).fit()           
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq36":       
        y = "nex_gsr_bj"
        x_list = ["gdpr_bj" ,"d_nex_gsr_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["nex_gsr_bj","gdpr_bj" ,"d_nex_gsr_bj" ]]
            test1["nex_gsr_bj_s1"] = test1["nex_gsr_bj"].shift(1)
            test1 = test1[(test1.index>2006) & (test1.index <2019)]
            results = sm.ols(formula = "nex_gsr_bj ~ gdpr_bj + nex_gsr_bj_s1 + d_nex_gsr_bj",data = test1 ).fit()          
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
    if Eqid == "Eq37":       
        y = "consn_bj"
        x_list = ["yht_n_bj" , "d_consn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["consn_bj" ,"yht_n_bj" , "d_consn_bj"]]
            test1["yht_n_bj_s1"] = test1["yht_n_bj"].shift(1)
            test1["consn_bj_s1"] = test1["consn_bj"].shift(1)
            test1["n_s1_diff"] = test1["yht_n_bj"] - test1["yht_n_bj_s1"]
            results = sm.ols(formula = "consn_bj ~ n_s1_diff +consn_bj_s1 + d_consn_bj",data = test1 ).fit()           
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq38":       
        y = "exusd_bj"
        x_list = ["exusd_ch" , "d_exusd_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["exusd_bj" ,"exusd_ch" , "d_exusd_bj"]]
            test1["exusd_bj_s1"] = test1["exusd_bj"].shift(1) 
            test1["exusd_bj_diff"] = test1["exusd_bj"].shift(1) - test1["exusd_bj"].shift(2)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]  # 本来没这行 。但不加的话，可能预测值对不齐
            results = sm.ols(formula = "exusd_bj ~ exusd_ch +exusd_bj_s1 + exusd_bj_diff + d_exusd_bj",data = test1 ).fit()         
            test1[y] = results.predict() 
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 

    if Eqid == "Eq39":       
        y = "imusd_bj"
        x_list = ["exusd_bj" , "d_imusd_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["imusd_bj" ,"exusd_bj" , "d_imusd_bj"]]
            test1["imusd_bj_s1"] = test1["imusd_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]  # 本来没这行 。但不加的话，可能预测值对不齐
            results = sm.ols(formula = "np.log(imusd_bj) ~ np.log(imusd_bj_s1) + np.log(exusd_bj) + d_imusd_bj",data = test1 ).fit()                      
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq40": #建立定义模型
        y = "yht_n_bj"
        x_list =["yht_rn_bj","yht_un_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["yht_n_bj" ,"yht_rn_bj","yht_un_bj"]]
            test1[y] = test1["yht_rn_bj"] + test1["yht_un_bj"]
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])            
    if Eqid == "Eq41": #建立定义模型
        y = "yh_un_bj"
        x_list =["yht_un_bj","popu_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["yh_un_bj" ,"yht_un_bj","popu_bj"]]
            test1[y] = (test1["yht_un_bj"] / test1["popu_bj"])
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])
            
    if Eqid == "Eq42": #建立定义模型
        y = "yh_rn_bj"
        x_list =["yht_rn_bj","popr_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["yh_rn_bj" ,"yht_rn_bj","popr_bj"]]
            test1[y] = (test1["yht_rn_bj"] / test1["popr_bj"])
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])

    if Eqid == "Eq43": #建立定义模型
        y = "yhn_bj"
        x_list =["yht_n_bj","totpop_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["yhn_bj" ,"yht_n_bj","totpop_bj"]]
            test1[y] = (test1["yht_n_bj"] / test1["totpop_bj"])
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])            
            

    if Eqid == "Eq44": #建立定义模型
        y = "conet_un_bj"
        x_list =["cone_un_bj","popu_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["conet_un_bj" ,"cone_un_bj","popu_bj"]]
            test1[y] = test1["cone_un_bj"] * test1["popu_bj"]
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])            
    if Eqid == "Eq45": #建立定义模型
        y = "conet_rn_bj"
        x_list =["cone_rn_bj","popr_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["conet_rn_bj" ,"cone_rn_bj","popr_bj"]]
            test1[y] = test1["cone_rn_bj"] * test1["popr_bj"]
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])
            
            

    if Eqid == "Eq46":       
        y = "yht_un_bj"
        x_list = ["gdp2n_bj","gdp3n_bj","taxn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["yht_un_bj","gdp2n_bj","gdp3n_bj","taxn_bj"]]
            test1["yht_un_bj_s1"] = test1["yht_un_bj"].shift(1)
            test1["gdp2gdp3tax"] = test1["gdp2n_bj"] + test1["gdp3n_bj"] - test1["taxn_bj"]
            results = sm.ols(formula = "yht_un_bj ~ gdp2gdp3tax + yht_un_bj_s1",data = test1 ).fit()          
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 

    if Eqid == "Eq47":       
        y = "yht_rn_bj"
        x_list = ["gdp1n_bj","gdp3n_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["yht_rn_bj","gdp1n_bj","gdp3n_bj"]]
            test1["yht_rn_bj_s1"] = test1["yht_rn_bj"].shift(1)
            results = sm.ols(formula = "yht_rn_bj ~ gdp1n_bj + gdp3n_bj + yht_rn_bj_s1",data = test1 ).fit()          
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq48":       
        y = "cone_un_bj"
        x_list = ["yht_un_bj" , "yh_un_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["cone_un_bj" , "yh_un_bj" ]]
            test1["un_diff"] = test1["yh_un_bj"] - test1["yh_un_bj"].shift(1)
            test1["cone_un_bj_s1"] = test1["cone_un_bj"].shift(1)
            results = sm.ols(formula = "cone_un_bj ~ un_diff + cone_un_bj_s1 ",data = test1 ).fit()           
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             
            
 
    if Eqid == "Eq49":       
        y = "cone_rn_bj"
        x_list = ["yh_rn_bj" ,"d_cone_rn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["cone_rn_bj" , "yh_rn_bj" ,"d_cone_rn_bj" ]]
            results = sm.ols(formula = "cone_rn_bj ~ yh_rn_bj+ d_cone_rn_bj ",data = test1 ).fit()          
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq50":       
        y = "frn_bj"
        x_list = ["taxn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["frn_bj","taxn_bj"]]
            test1["frn_bj_s1"] = test1["frn_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            results = sm.ols(formula = "frn_bj ~ taxn_bj + frn_bj_s1 ",data = test1 ).fit()           
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )   
    if Eqid == "Eq51": 
        y = "taxn_bj1"
        x_list = ["tax_vadn_bj","tax_cinn_bj","tax_pinn_bj","d_taxn_bj_1"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["taxn_bj1","tax_vadn_bj","tax_cinn_bj","tax_pinn_bj","d_taxn_bj_1"]]
            test1["log_tax_sum"] =np.log( test1["tax_vadn_bj"] + test1["tax_cinn_bj"] + test1["tax_pinn_bj"])
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            ar = 1
            ma = 0
            mod = SARIMAX(endog = np.log(test1["taxn_bj1"])  , exog = test1[["log_tax_sum","d_taxn_bj_1"] ],order =(ar,0,ma),enforce_invertibility= False , trend = 'c', enforce_stationarity = False)
            results = mod.fit()
            test1[y] = np.exp(results.predict())[1:]  # 和AR系数有关             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
            
    if Eqid == "Eq52":       
        y = "taxn_bj2"
        x_list = ["gdpn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["taxn_bj2","gdpn_bj"]]
            test1["taxn_bj2_s1"] = test1["taxn_bj2"].shift(1)
            results = sm.ols(formula = "taxn_bj2 ~ gdpn_bj+ taxn_bj2_s1 ",data = test1 ).fit()         
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq53": #建立定义模型
        y = "taxn_bj"
        x_list =["taxn_bj1","taxn_bj2"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["taxn_bj","taxn_bj1","taxn_bj2"]]
            test1[y] = (test1["taxn_bj1"] + test1["taxn_bj2"])/2
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])  
            
    if Eqid == "Eq54":       
        y = "tax_vadn_bj"
        x_list = ["gdpn_bj","d_tax_vadn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["tax_vadn_bj","gdpn_bj","d_tax_vadn_bj"]]
            test1["tax_vadn_bj_s1"] = test1["tax_vadn_bj"].shift(1)
            test1["log_tax_vadn"] = np.log(test1["tax_vadn_bj"])
            results = sm.ols(formula = "np.log(tax_vadn_bj) ~ np.log(gdpn_bj) + np.log(tax_vadn_bj_s1) + d_tax_vadn_bj ",data = test1 ).fit()     
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
            
    if Eqid == "Eq55":       
        y = "tax_cinn_bj"
        x_list = ["gdpn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["tax_cinn_bj","gdpn_bj"]]
            test1["tax_cinn_bj_s1"] = test1["tax_cinn_bj"].shift(1)
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "tax_cinn_bj ~ gdpn_bj+ tax_cinn_bj_s1 ",data = test1 ).fit()        
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
            
    if Eqid == "Eq56":       
        y = "tax_pinn_bj"
        x_list = ["yht_n_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["tax_pinn_bj","yht_n_bj"]]
            test1["tax_pinn_bj_s1"] = test1["tax_pinn_bj"].shift(1)
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "tax_pinn_bj ~ yht_n_bj+ tax_pinn_bj_s1 ",data = test1 ).fit()        
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
    if Eqid == "Eq57": 
        y = "fen_bj1"
        x_list = ["frn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["fen_bj1","frn_bj"]]
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            ar = 1 
            ma = 0
            mod = SARIMAX(endog = test1["fen_bj1"] , exog = test1["frn_bj"] ,order =(ar,0,ma),enforce_invertibility= False , trend = 'c', enforce_stationarity = False)
            results = mod.fit()
            test1[y] = results.predict()[1:]  # = AR系数有关             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )
            
            
    if Eqid == "Eq58":       
        y = "fen_bj2"
        x_list = ["fe_edun_bj","fe_tecn_bj","fe_culn_bj","fe_ssen_bj","fe_medn_bj","fe_eepn_bj","d_fen_bj_2"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["fen_bj2","fe_edun_bj","fe_tecn_bj","fe_culn_bj","fe_ssen_bj","fe_medn_bj","fe_eepn_bj","d_fen_bj_2"]]
            test1["fen_sum"] = test1[["fe_edun_bj","fe_tecn_bj","fe_culn_bj","fe_ssen_bj","fe_medn_bj","fe_eepn_bj"]].sum(axis =1)
            test1 = test1[(test1.index>2005) & (test1.index <2019)]
            results = sm.ols(formula = "fen_bj2 ~ fen_sum + d_fen_bj_2 ",data = test1 ).fit()         
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )            
    if Eqid == "Eq59": #建立定义模型
        y = "fen_bj"
        x_list =["fen_bj1","fen_bj2"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["fen_bj","fen_bj1","fen_bj2"]]
            test1[y] = (test1["fen_bj1"]+test1["fen_bj2"])/2
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])             
 
            
    if Eqid == "Eq60":       
        y = "depn_bj"
        x_list = ["dep_san_bj","d_depn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["depn_bj","dep_san_bj","d_depn_bj"]]
            test1["depn_bj_s1"] = test1["depn_bj"].shift(1)
            test1 = test1[(test1.index>2004) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(depn_bj) ~ np.log(dep_san_bj) + np.log(depn_bj_s1) + d_depn_bj ",data = test1 ).fit()       
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
    if Eqid == "Eq61":       
        y = "dep_san_bj"
        x_list = ["yht_n_bj","d_dep_san_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["dep_san_bj","yht_n_bj","d_dep_san_bj"]]
            test1["dep_san_bj_s1"] = test1["dep_san_bj"].shift(1)
            test1 = test1[(test1.index>2001) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(dep_san_bj) ~ np.log(yht_n_bj) + np.log(dep_san_bj_s1) + d_dep_san_bj ",data = test1 ).fit()

            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )            
    if Eqid == "Eq62":       
        y = "loan_ln_bj"
        x_list = ["depn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           

            test1 = df_ev[["loan_ln_bj","depn_bj"]]
            test1["loan_ln_bj_s1"] = test1["loan_ln_bj"].shift(1)
            test1 = test1[(test1.index>2003) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(loan_ln_bj) ~ np.log(depn_bj) + np.log(loan_ln_bj_s1) ",data = test1 ).fit()
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             

    if Eqid == "Eq63":       
        y = "loann_bj"
        x_list = ["depn_bj","intratel_cn","d_loann_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["loann_bj","depn_bj","intratel_cn","d_loann_bj"]]
            test1["loann_bj_s1"] = test1["loann_bj"].shift(1)
            test1["depn_bj_sum"] = test1["depn_bj"] + test1["depn_bj"].shift(1)
            test1 = test1[(test1.index>2006) & (test1.index <2019)]
            results = sm.ols(formula = "loann_bj ~ depn_bj_sum + loann_bj_s1  + intratel_cn + d_loann_bj ",data = test1 ).fit()
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
            
    if Eqid == "Eq64": #建立定义模型
        y = "emp1_bj"
        x_list =["totemp_bj" , "emp2_bj" , "emp3_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["emp1_bj" , "totemp_bj" , "emp2_bj" , "emp3_bj"]]
            test1[y] = test1["totemp_bj"] - test1["emp2_bj"] - test1["emp3_bj"]
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])      
    if Eqid == "Eq65": #建立定义模型
        y = "popr_bj"
        x_list =["totpop_bj" , "popu_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["popr_bj" , "totpop_bj" , "popu_bj" ]]
            test1[y] = test1["totpop_bj"] - test1["popu_bj"] 
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs])            
            
    if Eqid == "Eq66":       
        y = "totemp_bj"
        x_list = ["gdpr_bj","d_totemp_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["totemp_bj","gdpr_bj","d_totemp_bj"]]
            test1["gdpr_bj_diff"] = test1["gdpr_bj"] - test1["gdpr_bj"].shift(1)
            test1["totemp_bj_s1"] = test1["totemp_bj"].shift(1)
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "totemp_bj ~ gdpr_bj_diff + totemp_bj_s1  + d_totemp_bj ",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq67":       
        y = "emp2_bj"
        x_list = ["gdpr_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["emp2_bj","gdpr_bj"]]
            test1["gdpr_bj_pch"] =  (test1["gdpr_bj"] - test1["gdpr_bj"].shift(1) )/  test1["gdpr_bj"].shift(1)  #增长率
            test1["emp2_bj_pch"] = (test1["emp2_bj"] - test1["emp2_bj"].shift(1) )/  test1["emp2_bj"].shift(1)
            test1["emp2_bj_s1"] = test1["emp2_bj"].shift(1)
            test1 = test1[(test1.index>1998) & (test1.index <2019)]
            results = sm.ols(formula = "emp2_bj ~ gdpr_bj + gdpr_bj_pch  + emp2_bj_pch + emp2_bj_s1 ",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
            
    if Eqid == "Eq68":       
        y = "emp3_bj"
        x_list = ["gdp3r_bj" ,"d_emp3_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["emp3_bj","gdp3r_bj" ,"d_emp3_bj"]]
            test1["gdp3r_bj_pch"] =  (test1["gdp3r_bj"] - test1["gdp3r_bj"].shift(1) )/  test1["gdp3r_bj"].shift(1)  #增长率
            test1["emp3_bj_s1"] = test1["emp3_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            results = sm.ols(formula = "emp3_bj ~ gdp3r_bj_pch + emp3_bj_s1  + d_emp3_bj",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq69":       
        y = "totpop_bj"
        x_list = ["gdpr_bj","d_totpop_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["totpop_bj","gdpr_bj","d_totpop_bj"]]
            test1["gdpr_bj_diff"] = test1["gdpr_bj"] - test1["gdpr_bj"].shift(1)
            test1["totpop_bj_s1"] = test1["totpop_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            results = sm.ols(formula = "totpop_bj ~ gdpr_bj_diff + totpop_bj_s1  + d_totpop_bj",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )              
    if Eqid == "Eq70": 
        y = "popu_bj"
        x_list = ["d_popu_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["popu_bj" , "d_popu_bj" ]]
            test1["popu_bj_s1"] = test1["popu_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            ar = 1
            ma = 0
            mod = SARIMAX(endog = test1["popu_bj"]  , exog = test1[["popu_bj_s1","d_popu_bj"]] ,order =(ar,0,ma),enforce_invertibility= False , trend = 'c', enforce_stationarity = False)
            results = mod.fit()
            test1[y] = results.predict()[1:]  # 和AR系数有关             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 

    if Eqid == "Eq71":       
        y = "rpi_bj"
        x_list = ["cpi_bj","d_rpi_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["rpi_bj","cpi_bj","d_rpi_bj"]]
            test1["cpi_bj_diff"] = test1["cpi_bj"] - test1["cpi_bj"].shift(1)
            test1["rpi_bj_s1"] = test1["rpi_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            results = sm.ols(formula = "rpi_bj ~ cpi_bj_diff + rpi_bj_s1  + d_rpi_bj",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )              
    if Eqid == "Eq72": 
        y = "cpi_bj"
        x_list = ["gdpd_bj" ,"d_cpi_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["cpi_bj","gdpd_bj" ,"d_cpi_bj"]]
            test1["cpi_bj_s1"] = test1["cpi_bj"].shift(1)
            test1["gdpd_bj_diff"] = test1["gdpd_bj"] - test1["gdpd_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            ar = 1
            ma = 0
            mod = SARIMAX(endog = test1["cpi_bj"]  , exog = test1[["cpi_bj_s1","gdpd_bj_diff","d_cpi_bj"]] ,order =(ar,0,ma),enforce_invertibility= False , trend = 'c', enforce_stationarity = False)
            results = mod.fit()
            test1[y] = np.exp(results.predict())[1:]  # 和AR系数有关             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq73":       
        y = "gdp1d_bj"
        x_list = ["rpi_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp1d_bj","rpi_bj"]]
            test1["rpi_bj_diff"] = test1["rpi_bj"] - test1["rpi_bj"].shift(1)
            test1["gdp1d_bj_s1"] = test1["gdp1d_bj"].shift(1)
            test1 = test1[(test1.index>1997) & (test1.index <2019)]
            results = sm.ols(formula = "gdp1d_bj ~ rpi_bj_diff + gdp1d_bj_s1",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )   
    if Eqid == "Eq74":       
        y = "gdp2d_bj"
        x_list = ["ppi_bj","faipi_bj","d_gdp2d_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp2d_bj","ppi_bj","faipi_bj","d_gdp2d_bj"]]
            test1 = test1[(test1.index>1995) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(gdp2d_bj) ~ np.log(ppi_bj) + np.log(faipi_bj) + d_gdp2d_bj",data = test1 ).fit()
            test1[y] = np.exp(results.predict())           
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )              
    if Eqid == "Eq75":       
        y = "gdp3d_bj"
        x_list = ["gdp2d_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3d_bj","gdp2d_bj"]]
            test1["gdp3d_bj_s1"] = test1["gdp3d_bj"].shift(1)
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "gdp3d_bj ~ gdp2d_bj + gdp3d_bj_s1",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )              
    if Eqid == "Eq76":       
        y = "gdpd_bj"
        x_list = ["gdp1d_bj","gdp2d_bj","gdp3d_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdpd_bj","gdp1d_bj","gdp2d_bj","gdp3d_bj"]]
            test1 = test1[(test1.index>1995) & (test1.index <2019)]
            results = sm.ols(formula = "gdpd_bj ~ gdp1d_bj + gdp2d_bj + gdp3d_bj",data = test1 ).fit()
            test1[y] = results.predict()              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq77":       
        y = "ppi_bj"
        x_list = ["rpi_bj","d_ppi_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["ppi_bj","rpi_bj","d_ppi_bj"]]
            test1["rpi_bj_diff"] = np.log(test1["rpi_bj"])- np.log(test1["rpi_bj"].shift(1))
            test1["ppi_bj_s1"] = test1["ppi_bj"].shift(1)
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(ppi_bj) ~ rpi_bj_diff + np.log(ppi_bj_s1)  + d_ppi_bj ",data = test1 ).fit()
            test1[y] = np.exp(results.predict() )             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq78":       
        y = "faipi_bj"
        x_list = ["ppi_bj","d_faipi_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["faipi_bj","ppi_bj","d_faipi_bj"]]
            test1["ppi_bj_diff"] = test1["ppi_bj"] - test1["ppi_bj"].shift(1)
            test1["faipi_bj_s1"] = test1["faipi_bj"].shift(1)
            test1 = test1[(test1.index>1996) & (test1.index <2019)]
            results = sm.ols(formula = "np.log(faipi_bj) ~ ppi_bj_diff + np.log(faipi_bj_s1)  + d_faipi_bj ",data = test1 ).fit()
            test1[y] = np.exp(results.predict())              
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
            
    if Eqid == "Eq79": #建立定义模型
        y = "gdp3_tspar_bj"
        x_list =["gdp3_tspan_bj","pgdp3_tspad_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3_tspar_bj","gdp3_tspan_bj","pgdp3_tspad_bj"]]
            test1[y] = test1["gdp3_tspan_bj"] / test1["pgdp3_tspad_bj"] *100    
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq80":       
        y = "gdp3_tspan_bj"
        x_list = ["gdp3_tratp_bj","gdp3_tragt_bj","gdp3_trapt_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3_tspan_bj","gdp3_tratp_bj","gdp3_tragt_bj","gdp3_trapt_bj"]]
            test1 = test1[(test1.index>1999) & (test1.index <2019)]
            results = sm.ols(formula = " gdp3_tspan_bj~ gdp3_tratp_bj + gdp3_tragt_bj  + gdp3_trapt_bj ",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             
            
            
    if Eqid == "Eq81": #建立定义模型
        y = "gdp3_wrar_bj"
        x_list =["gdp3_wran_bj","pgdp3_wrad_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3_wrar_bj","gdp3_wran_bj","pgdp3_wrad_bj"]]
            test1[y] = test1["gdp3_wran_bj"] / test1["pgdp3_wrad_bj"] *100    
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             
    if Eqid == "Eq82":       
        y = "gdp3_wran_bj"
        x_list = ["consn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3_wran_bj","consn_bj" ]]
            test1["consn_bj_diff"] = test1["consn_bj"] - test1["consn_bj"].shift(1)
            test1["gdp3_wran_bj_s1"] = test1["gdp3_wran_bj"].shift(1)
            test1 = test1[(test1.index>2000) & (test1.index <2019)]
            results = sm.ols(formula = " gdp3_wran_bj~ consn_bj_diff + gdp3_wran_bj_s1 ",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq83": #建立定义模型
        y = "gdp3_fiar_bj"
        x_list =["gdp3_fian_bj","pgdp3_fiad_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3_fiar_bj","gdp3_fian_bj","pgdp3_fiad_bj"]]
            test1[y] = test1["gdp3_fian_bj"] / test1["pgdp3_fiad_bj"] *100    
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
            
    if Eqid == "Eq84":       
        y = "gdp3_fian_bj"
        x_list = ["depn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3_fian_bj","depn_bj"]]
            test1["gdp3_fian_bj_s1"] = test1["gdp3_fian_bj"].shift(1)
            test1 = test1[(test1.index>2003) & (test1.index <2019)]
            results = sm.ols(formula = " gdp3_fian_bj~ depn_bj + gdp3_fian_bj_s1 ",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq85": 
        y = "gdp3_rear_bj"
        x_list =["gdp3_rean_bj","pgdp3_read_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3_rear_bj","gdp3_rean_bj","pgdp3_read_bj"]]
            test1[y] = test1["gdp3_rean_bj"] / test1["pgdp3_read_bj"] *100    
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq86":       
        y = "gdp3_rean_bj"
        x_list = ["inv_rean_bj","d_gdp3_rean_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3_rean_bj","inv_rean_bj","d_gdp3_rean_bj"]]
            test1["gdp3_rean_bj_s1"] = test1["gdp3_rean_bj"].shift(1)
            test1 = test1[(test1.index>2002) & (test1.index <2019)]
            results = sm.ols(formula = " gdp3_rean_bj~ inv_rean_bj + gdp3_rean_bj_s1 + d_gdp3_rean_bj ",data = test1 ).fit()            
            test1[y][1:] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             
    if Eqid == "Eq87": 
        y = "gdp3_infr_bj"
        x_list =["gdp3_infn_bj","pgdp3_infd_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3_infr_bj","gdp3_infn_bj","pgdp3_infd_bj"]]
            test1[y] = test1["gdp3_infn_bj"] / test1["pgdp3_infd_bj"] *100  
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid ,y, "还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
    if Eqid == "Eq88":       
        y = "gdp3_infn_bj"
        x_list = ["gdp_tran_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3_infn_bj","gdp_tran_bj"]]
            test1 = test1[(test1.index>2014) & (test1.index <2019)]
            results = sm.ols(formula = " gdp3_infn_bj~ gdp_tran_bj ",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq89": 
        y = "gdp3_renr_bj"
        x_list =["gdp3_renn_bj","pgdp3_rend_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3_renr_bj","gdp3_renn_bj","pgdp3_rend_bj"]]
            test1[y] = test1["gdp3_renn_bj"] / test1["pgdp3_rend_bj"] *100
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")  
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )    
            
    if Eqid == "Eq90":       
        y = "gdp3_renn_bj"
        x_list = ["gdp_momn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3_renn_bj","gdp_momn_bj"]]
            test1 = test1[(test1.index>2005) & (test1.index <2019)]
            results = sm.ols(formula = " gdp3_renn_bj~ gdp_momn_bj ",data = test1 ).fit()
            print("Eq90: " ,test1.shape , len(results.predict()))
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
    if Eqid == "Eq91": 
        y = "gdp3_sicr_bj"
        x_list =["gdp3_sicn_bj","pgdp3_sicd_bj"]
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0): 
       # try :                
            test1 = df_ev[["gdp3_sicr_bj","gdp3_sicn_bj","pgdp3_sicd_bj"]]
            test1[y] = test1["gdp3_sicn_bj"] / test1["pgdp3_sicd_bj"] *100 
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)              
            df_ev[y] = preds[y]                 
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")            
    if Eqid == "Eq92":       
        y = "gdp3_sicn_bj"
        x_list = ["gdp3_sicn_bj","gdp_momn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp3_sicn_bj","gdp_momn_bj"]]
            test1["gdp3_sicn_bj_s1"] = test1["gdp3_sicn_bj"].shift(1)
            test1 = test1[(test1.index>2005) & (test1.index <2019)]
            results = sm.ols(formula = " gdp3_sicn_bj~ gdp_momn_bj + gdp3_sicn_bj_s1",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )   
            
    if Eqid == "Eq93":       
        y = "gdp_tran_bj"
        x_list = ["gdp_tecn_bj","d_gdp_tran_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp_tran_bj","gdp_tecn_bj","d_gdp_tran_bj"]]
            test1 = test1[(test1.index>2014) & (test1.index <2019)]
            results = sm.ols(formula = " gdp_tran_bj~ gdp_tecn_bj + d_gdp_tran_bj",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )   
    if Eqid == "Eq94":       
        y = "gdp_tecn_bj"
        x_list = ["lse_bj","tta_bj","d_gdp_tecn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp_tecn_bj","lse_bj","tta_bj","d_gdp_tecn_bj"]]
            test1["tta_bj_diff"] = test1["tta_bj"] - test1["tta_bj"].shift(1)
            test1["gdp_tecn_bj_s1"] = test1["gdp_tecn_bj"].shift(1)
            test1 = test1[(test1.index>2004) & (test1.index <2019)]
            results = sm.ols(formula = " gdp_tecn_bj ~ lse_bj + tta_bj_diff + gdp_tecn_bj_s1 + d_gdp_tecn_bj",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )  
            
    if Eqid == "Eq95":       
        y = "tta_f_bj"
        x_list = ["tta_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["tta_f_bj","tta_bj"]]
            test1 = test1[(test1.index>2002) & (test1.index <2019)]
            results = sm.ols(formula = " tta_f_bj~ tta_bj ",data = test1 ).fit()
            print(test1.shape ,results.predict() )
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             
    if Eqid == "Eq96":       
        y = "gdp_mosn_bj"
        x_list = ["gdp3_infn_bj","gdp3_renn_bj","gdp3_sicn_bj","d_gdp_mosn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp_mosn_bj","gdp3_infn_bj","gdp3_renn_bj","gdp3_sicn_bj","d_gdp_mosn_bj"]]
            test1 = test1[(test1.index>2003) & (test1.index <2019)]
            results = sm.ols(formula = " gdp_mosn_bj ~ gdp3_infn_bj + gdp3_renn_bj + gdp3_sicn_bj + d_gdp_mosn_bj",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             
    if Eqid == "Eq97":       
        y = "gdp_momn_bj"
        x_list = ["gdp_tecn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp_momn_bj","gdp_tecn_bj"]]
            test1["gdp_tecn_bj_diff"] = test1["gdp_tecn_bj"] - test1["gdp_tecn_bj"].shift(1)
            test1["gdp_momn_bj_s1"] = test1["gdp_momn_bj"].shift(1)
            test1 = test1[(test1.index>2005) & (test1.index <2019)]
            results = sm.ols(formula = " gdp_momn_bj ~ gdp_tecn_bj_diff + gdp_momn_bj_s1",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
    if Eqid == "Eq98":       
        y = "gdp_infn_bj"
        x_list = ["gdp_tecn_bj","d_gdp_infn_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["gdp_infn_bj","gdp_tecn_bj","d_gdp_infn_bj"]]
            test1["gdp_infn_bj_s1"] = test1["gdp_infn_bj"].shift(1)
            test1 = test1[(test1.index>2004) & (test1.index <2019)]
            results = sm.ols(formula = " gdp_infn_bj ~ gdp_tecn_bj + gdp_infn_bj_s1 + d_gdp_infn_bj",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] ) 
            
    if Eqid == "Eq99":       
        y = "tta_bj"
        x_list = ["lse_bj","d_tta_bj"]        
        if (y in exogs) & (len([x for x in x_list if x in exogs])==0):           
            test1 = df_ev[["tta_bj","lse_bj","d_tta_bj"]]
            test1["tta_bj_diff"] = test1["tta_bj"] - test1["tta_bj"].shift(1)
            test1 = test1[(test1.index>2002) & (test1.index <2019)]
            results = sm.ols(formula = " tta_bj ~ lse_bj + tta_bj_diff + d_tta_bj",data = test1 ).fit()
            test1[y] = results.predict()             
            preds = pd.merge(preds,test1[[y]],how = "left",left_index =True ,right_index = True)            
            df_ev[y] = preds[y]
            exogs.remove(y)
            Eqs.remove(Eqid)
            print(Eqid ,y," ----------------   求解成功 。")
        else:
            print(Eqid , y,"还不能求解 ，因还未获得外生变量:  ",[x for x in x_list if x in exogs] )             
           
    return  exogs , df_ev ,preds,Eqs