#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:53:32 2019

This class is used to calculate the return for the portfolio, formed by 
the asset weights and their returns in the market.
The class can return arithmatic and geometric return, IR ratio.

@author: yuxing
"""


import numpy as np
import pandas as pd


class Portfolio:
    
    def __init__(self, asset_weights, asset_returns, stock_symbol, freq = 12):
        
        """
        input asset weights and asset returns
        
        """
        
        self.wgt = asset_weights
        self.rtn = asset_returns
        self.freq = freq   # whether it's monthly (12) or daily model (252).
        self.stock = stock_symbol
        
        self.cumulative_return()
        
    def cumulative_return(self, compounding = "arithmetic"):
        
        self.cumu_return = pd.DataFrame((self.wgt * self.rtn).cumsum(axis = 0),
                                        columns = [self.stock])
       
    def IR(self, bench_weights, bench_returns):
        
#        bench_weights = bench_wgt
        
#        bench_returns = stock_return
        
        
        
        bench_rtn = bench_weights * bench_returns
        bench_cumu = bench_rtn.cumsum(axis = 0)
        
        cumu_return = pd.DataFrame(columns = ['my_return', "bench_return"])
        
        cumu_return['my_return'] = self.cumu_return[self.stock]
        cumu_return['bench_return'] = bench_cumu[self.stock]
        
        ax = cumu_return.plot(figsize = (8,6), fontsize = 14, lw = 3)
        
        ax.set_xlabel("Months into the future", fontsize = 16)
        ax.set_ylabel("Cumulative return", fontsize = 16)
        ax.set_title("Comparison between my strategy and the market return", fontsize = 16)
        
        
        # calculation of information retio
    
        excess_return = self.wgt * self.rtn - bench_rtn
        
        excess_return_annual = excess_return.mean(axis = 0) * self.freq
        
        # anuualized volatility of excess return
        excess_return_annual_vol = excess_return.std(axis = 0) * np.sqrt(self.freq)
        
        IR = excess_return_annual / excess_return_annual_vol
        print("The IR for the strategy is: {}".format(round(IR[0],2)))
        
        return IR
        