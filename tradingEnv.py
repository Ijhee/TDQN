# coding=utf-8

"""
Goal: Implement a trading environment compatible with OpenAI Gym.
Authors: Thibaut Théate and Damien Ernst
Institution: University of Liège
"""

###############################################################################
################################### Imports ###################################
###############################################################################

import os
import gym
import math
import numpy as np

import pandas as pd
pd.options.mode.chained_assignment = None

from matplotlib import pyplot as plt

from dataDownloader import AlphaVantage
from dataDownloader import YahooFinance
from dataDownloader import CSVHandler
from fictiveStockGenerator import StockGenerator



###############################################################################
################################ Global variables #############################
###############################################################################

# Boolean handling the saving of the stock market data downloaded
saving = True

# Variable related to the fictive stocks supported
fictiveStocks = ('LINEARUP', 'LINEARDOWN', 'SINUSOIDAL', 'TRIANGLE')



###############################################################################
############################## Class TradingEnv ###############################
###############################################################################

class TradingEnv(gym.Env):
    """
    GOAL: Implement a custom trading environment compatible with OpenAI Gym.
    
    VARIABLES:  - data: Dataframe monitoring the trading activity.
                - state: RL state to be returned to the RL agent.
                - reward: RL reward to be returned to the RL agent.
                - done: RL episode termination signal.
                - t: Current trading time step.
                - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - stateLength: Number of trading time steps included in the state.
                - numberOfShares: Number of shares currently owned by the agent.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                                
    METHODS:    - __init__: Object constructor initializing the trading environment.
                - reset: Perform a soft reset of the trading environment.
                - step: Transition to the next trading time step.
                - render: Illustrate graphically the trading environment.
    """

    def __init__(self, marketSymbol, startingDate, endingDate, money, stateLength=30,
                 transactionCosts=0, startingPoint=0):
        """
        GOAL: Object constructor initializing the trading environment by setting up
              the trading activity dataframe as well as other important variables.
        
        INPUTS: - marketSymbol: Stock market symbol.
                - startingDate: Beginning of the trading horizon.
                - endingDate: Ending of the trading horizon.
                - money: Initial amount of money at the disposal of the agent.
                - stateLength: Number of trading time steps included in the RL state.
                - transactionCosts: Transaction costs associated with the trading
                                    activity (e.g. 0.01 is 1% of loss).
                - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """
        # CASE 1: Fictive stock generation
        if(marketSymbol in fictiveStocks):
            stockGeneration = StockGenerator()
            if(marketSymbol == 'LINEARUP'):
                self.data = stockGeneration.linearUp(startingDate, endingDate)
            elif(marketSymbol == 'LINEARDOWN'):
                self.data = stockGeneration.linearDown(startingDate, endingDate)
            elif(marketSymbol == 'SINUSOIDAL'):
                self.data = stockGeneration.sinusoidal(startingDate, endingDate)
            else:
                self.data = stockGeneration.triangle(startingDate, endingDate)
 
        # CASE 2: Real stock loading
        else:
            # Check if the stock market data is already present in the database
            csvConverter = CSVHandler()
            csvName = "".join(['Data/', marketSymbol, '_', startingDate, '_', endingDate])
            exists = os.path.isfile(csvName + '.csv')
            
            # If affirmative, load the stock market data from the database
            if(exists):
                self.data = csvConverter.CSVToDataframe(csvName)
            # Otherwise, download the stock market data from Yahoo Finance and save it in the database
            else:  
                downloader1 = YahooFinance()
                downloader2 = AlphaVantage()
                try:
                    self.data = downloader1.getDailyData(marketSymbol, startingDate, endingDate)
                except:
                    self.data = downloader2.getDailyData(marketSymbol, startingDate, endingDate)

                if saving == True:
                    csvConverter.dataframeToCSV(csvName, self.data)

        # Interpolate in case of missing data
        self.data.replace(0.0, np.nan, inplace=True)
        self.data.interpolate(method='linear', limit=5, limit_area='inside', inplace=True)
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(method='bfill', inplace=True)
        self.data.fillna(0, inplace=True)
        
        # Set the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = float(money)
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:stateLength].tolist(),
                      self.data['Low'][0:stateLength].tolist(),
                      self.data['High'][0:stateLength].tolist(),
                      self.data['Volume'][0:stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Set additional variables related to the trading activity
        self.marketSymbol = marketSymbol
        self.startingDate = startingDate
        self.endingDate = endingDate
        self.stateLength = stateLength
        self.t = stateLength
        self.numberOfShares = 0
        self.transactionCosts = transactionCosts
        self.epsilon = 0.1

        # If required, set a custom starting point for the trading activity
        if startingPoint:
            self.setStartingPoint(startingPoint)


    def reset(self):
        """
        GOAL: Perform a soft reset of the trading environment. 
        
        INPUTS: /    
        
        OUTPUTS: - state: RL state returned to the trading strategy.
        """

        # Reset the trading activity dataframe
        self.data['Position'] = 0
        self.data['Action'] = 0
        self.data['Holdings'] = 0.
        self.data['Cash'] = self.data['Cash'][0]
        self.data['Money'] = self.data['Holdings'] + self.data['Cash']
        self.data['Returns'] = 0.

        # Reset the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][0:self.stateLength].tolist(),
                      self.data['Low'][0:self.stateLength].tolist(),
                      self.data['High'][0:self.stateLength].tolist(),
                      self.data['Volume'][0:self.stateLength].tolist(),
                      [0]]
        self.reward = 0.
        self.done = 0

        # Reset additional variables related to the trading activity
        self.t = self.stateLength
        self.numberOfShares = 0

        return self.state

    
    def computeLowerBound(self, cash, numberOfShares, price):
        """
        GOAL: Compute the lower bound of the complete RL action space, 
              i.e. the minimum number of share to trade.
        
        INPUTS: - cash: Value of the cash owned by the agent.
                - numberOfShares: Number of shares owned by the agent.
                - price: Last price observed.
        
        OUTPUTS: - lowerBound: Lower bound of the RL action space.
        """

        # Computation of the RL action lower bound
        deltaValues = - cash - numberOfShares * price * (1 + self.epsilon) * (1 + self.transactionCosts)
        if deltaValues < 0:
            lowerBound = deltaValues / (price * (2 * self.transactionCosts + (self.epsilon * (1 + self.transactionCosts))))
        else:
            lowerBound = deltaValues / (price * self.epsilon * (1 + self.transactionCosts))
        return lowerBound
    

    def step(self, action):
        """
        GOAL: Transition to the next trading time step based on the
              trading position decision made (either long or short).
        
        INPUTS: - action: Trading decision (1 = long, 0 = short).    
        
        OUTPUTS: - state: RL state to be returned to the RL agent.
                 - reward: RL reward to be returned to the RL agent.
                 - done: RL episode termination signal (boolean).
                 - info: Additional information returned to the RL agent.
        """

        # Stting of some local variables
        t = self.t
        numberOfShares = self.numberOfShares
        customReward = False


        # CASE 1: 100% 
        if action == 2:
            
            self.data['Position'][t] = 2
            
            # Case no position => 100%
            if self.data["Position"][t - 1] == 0:
                sal_jusik = math.floor(self.data["Cash"][t - 1]/ (self.data["Close"][t] * (1 + self.transactionCosts)))

                # 현재 보유한 현금에서자본을 투자 (주식을 사는 것임)
                self.data["Cash"][t] = self.data["Cash"][t - 1] - (sal_jusik) * self.data["Close"][t] * (1 + self.transactionCosts)
                
                # 기존 주식 보유량에 산 주식을 더함
                self.numberOfShares = sal_jusik

                # 현재 자산
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]

                # action 상태 변경
                self.data['Action'][t] = 2


            # Case a: 100% => 100%
            elif self.data["Position"][t - 1] == 2:
                # 따라서, 돈도 이전 시점의 돈과 같음
                self.data["Cash"][t] = self.data["Cash"][t - 1]
                # 현재 홀딩하고 있는 자본은 아래와 같음
                self.data["Holdings"][t] = self.numberOfShares * self.data["Close"][t]
                
                #self.data["Action"][t] = 2

            # Case b: 50% -> 100%
            # 무조건  long. 
            elif self.data["Position"][t - 1] == 1: # 50%-> 100%
                # 근데 cash가 한 주 살 돈 도 없으면 
                if (self.data["Cash"]  <= (self.data["Close"][t])).all():                
                    # 따라서, 돈도 이전 시점의 돈과 같음
                    self.data["Cash"][t] = self.data["Cash"][t - 1]
                    
                    # 현재 홀딩하고 있는 자본은 아래와 같음
                    self.data["Holdings"][t] = self.numberOfShares * self.data["Close"][t]
                    
                    # action이 바뀌었으므로
                    self.data['Action'][t] = 2

                else:
                    
                    # 현재 갖고 있는 cash로 살 수 있는 물량은 아래와 같이 계산될 수 있음                                        
                    # 살 수 있는 물량: 보유하고 있는 현금/주식 종가 * (1+수수료)
                    sal_jusik = math.floor(self.data["Cash"][t - 1]/ (self.data["Close"][t] * (1 + self.transactionCosts)))

                    # 현재 보유한 현금에서자본을 투자 (주식을 사는 것임)
                    self.data["Cash"][t] = self.data["Cash"][t - 1] - (sal_jusik) * self.data["Close"][t] * (1 + self.transactionCosts)
                    
                    # 기존 주식 보유량에 산 주식을 더함
                    self.numberOfShares = sal_jusik + self.numberOfShares 

                    # 샀으니 Holding 금액은 아래와 같음
                    self.data["Holdings"][t] = (self.numberOfShares) * self.data["Close"][t]

                    # action이 바뀌었으므로
                    self.data['Action'][t] = 2
            
            # Case c: -100% -> 100%
            else:

                # 숏 포지션 청산(빌린 주식 수를 현재 종가에 맞춰 상환) -> 이전에 주식을 빌리고 현금을 받은 상황이므로, 기존 cash에서 빌린만큼 제함
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)

                # 돈을 갚고 난 다음 남은 현금으로 살 수 있는 주식 수 계산
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))

                # 현금 잔액 ( 한주 살 돈 모자랄수 있음 )
                self.data['Cash'][t] = self.data['Cash'][t] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)

                # 주식 산만큼 보유 주식수 업데이트
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]

                # 상태 업데이트
                self.data['Action'][t] = 2
        
        # CASE 2: 50%
        elif action == 1:
            # 포지션 정의
            self.data['Position'][t] = 1

            # 100% => 50% 
            if self.data["Position"][t - 1] == 2:
                
                # 현재 보유 주식이 줄게 된다
                pan_jusik = math.floor(self.numberOfShares/2) # 50% 매도

                # 주식보유량 업데이트
                self.numberOfShares = self.numberOfShares - pan_jusik
                
                # 기존금액에서 판 만큼 현금 업데이트
                self.data['Cash'][t] = self.data['Cash'][t - 1] + pan_jusik * self.data['Close'][t] * (1 - self.transactionCosts)
                
                # 주식 보유 금액 업데이트
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]
                
                #액션이 변경되었으므로
                self.data['Action'][t] = 1

            # 50% => 50%             
            elif self.data["Position"][t - 1] == 1: # No position일 때
                # 주식이 올랐을 경우 => 자산에서 주식비율이 커짐 => 주식을 팔아야함
                # 주식이 내렸을 경우 => 자산에서 주식비율이 작아짐 => 주식을 사야함
                # 현재 holding 금액 업데이트
                self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]

                # 주식이 오른 경우
                if (self.data['Holdings'][t] > self.data['Holdings'][t - 1]):
                    
                    # 주식을 팔아야함                    

                    # 팔 주식의 양 계산
                    # 팔 주식  = ((보유 주식 금액 - 현금) * 0.5)/종가
                    pal_jusik = math.floor( ((self.data['Holdings'][t] - self.data['Cash'][t-1])*0.5)/self.data['Close'][t]  )
                    
                    # 현재 주식 보유량 업데이트
                    self.numberOfShares = self.numberOfShares - pal_jusik

                    # 현재 현금 보유량 업데이트
                    self.data['Cash'][t] = self.data['Cash'][t-1] + pal_jusik*self.data['Close'][t]* (1 - self.transactionCosts)

                    # 주식 보유 금액 업데이트
                    self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]

                    # action은 업데이트 ㄴㄴ

                # 주식이 내린 경우
                elif (self.data['Holdings'][t] < self.data['Holdings'][t - 1]):
                    # 주식을 사야함                    

                    # 살 주식의 양 계산
                    # 살 주식  = ((보유 주식 금액 - 현금) * 0.5)/종가
                    sal_jusik = math.floor( (( self.data['Cash'][t-1] - self.data['Holdings'][t])*0.5)/self.data['Close'][t]  )
                    
                    # 현재 주식 보유량 업데이트
                    self.numberOfShares = self.numberOfShares + sal_jusik

                    # 현재 현금 보유량 업데이트
                    self.data['Cash'][t] = self.data['Cash'][t-1] - sal_jusik*self.data['Close'][t]* (1 + self.transactionCosts)

                    # 주식 보유 금액 업데이트
                    self.data['Holdings'][t] = self.numberOfShares * self.data['Close'][t]

                    # action은 업데이트 ㄴㄴ
                
                else:
                    self.data['Cash'][t] = self.data['Cash'][t-1]                   

            # -100% => 50% 
            elif self.data["Position"][t - 1] == -1:
                
                # 빌린 주식 청산
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)

                # 숏 포지션 청산(빌린 주식 수를 현재 종가에 맞춰 상환) -> 이전에 주식을 빌리고 현금을 받은 상황이므로, 기존 cash에서 빌린만큼 제함
                self.data['Cash'][t] = self.data['Cash'][t - 1] - self.numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)

                # 앞으로 살 주식 수 계산
                # 살 주식  = ((보유 주식 금액 - 현금) * 0.5)/종가
                self.numberOfShares = math.floor( (( self.data['Cash'][t])*0.5) / self.data['Close'][t]  )

                # holidng 업데이트
                self.data['Holdings'][t] = self.numberOfShares*self.data['Close'][t]*(1 - self.transactionCosts)

                # cash 업데이트
                self.data['Cash'][t] =  self.data['Cash'][t] - self.data['Holdings'][t]
                
                # action 업데이트
                self.data['Action'][t] = 1
            else:
                self.numberOfShares = math.floor( (( self.data['Cash'][t-1])*0.5) / self.data['Close'][t]  )

                # holidng 업데이트
                self.data['Holdings'][t] = self.numberOfShares*self.data['Close'][t]*(1 - self.transactionCosts)

                # cash 업데이트
                self.data['Cash'][t] =  self.data['Cash'][t-1] - self.data['Holdings'][t]
                
                # action 업데이트
                self.data['Action'][t] = 1



        # CASE 3: -100
        elif(action == 0):
            self.data['Position'][t] = -1
            
            # Case a: Short -> Short
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    self.data['Cash'][t] = self.data['Cash'][t - 1]
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), self.numberOfShares)
                    self.numberOfShares -= numberOfSharesToBuy
                    self.data['Cash'][t] = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    self.data['Holdings'][t] =  - self.numberOfShares * self.data['Close'][t]
                    customReward = True

            # Case b: No position -> Short(-100%)
            elif(self.data['Position'][t - 1] == 0):
                self.numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1
                
            # Case c: Long(50%)& Long(100%) -> Short 
            else:
                # 현재 돈 = 기존 돈 + 현재 들고 있는 주식 수만큼 팔았을 때의 돈 -> 보유 주식 전량 매도
                self.data['Cash'][t] = self.data['Cash'][t - 1] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                # 현재 돈으로 빌릴 수 있는 주식 수 계산
                self.numberOfShares = math.floor(self.data['Cash'][t]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                self.data['Cash'][t] = self.data['Cash'][t] + self.numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                self.data['Holdings'][t] = - self.numberOfShares * self.data['Close'][t]
                self.data['Action'][t] = -1

        # CASE 3: PROHIBITED ACTION
        else:
            raise SystemExit("Prohibited action! Action should be either 1 (long) or 0 (short).")

        # Update the total amount of money owned by the agent, as well as the return generated
        self.data['Money'][t] = self.data['Holdings'][t] + self.data['Cash'][t]
        self.data['Returns'][t] = (self.data['Money'][t] - self.data['Money'][t-1])/self.data['Money'][t-1]

        # Set the RL reward returned to the trading agent
        if not customReward:
            self.reward = self.data['Returns'][t]
        else:
            self.reward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]

        # Transition to the next trading time step
        self.t = self.t + 1
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1  

        # Same reasoning with the other action (exploration trick)
        otherAction = int(not bool(action))
        customReward = False
        if(otherAction == 1):
            otherPosition = 1
            if(self.data['Position'][t - 1] == 1):
                otherCash = self.data['Cash'][t - 1]
                otherHoldings = numberOfShares * self.data['Close'][t]
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash - numberOfShares * self.data['Close'][t] * (1 + self.transactionCosts)
                otherHoldings = numberOfShares * self.data['Close'][t]
        else:
            otherPosition = -1
            if(self.data['Position'][t - 1] == -1):
                lowerBound = self.computeLowerBound(self.data['Cash'][t - 1], -numberOfShares, self.data['Close'][t-1])
                if lowerBound <= 0:
                    otherCash = self.data['Cash'][t - 1]
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                else:
                    numberOfSharesToBuy = min(math.floor(lowerBound), numberOfShares)
                    numberOfShares -= numberOfSharesToBuy
                    otherCash = self.data['Cash'][t - 1] - numberOfSharesToBuy * self.data['Close'][t] * (1 + self.transactionCosts)
                    otherHoldings =  - numberOfShares * self.data['Close'][t]
                    customReward = True
            elif(self.data['Position'][t - 1] == 0):
                numberOfShares = math.floor(self.data['Cash'][t - 1]/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - numberOfShares * self.data['Close'][t]
            else:
                otherCash = self.data['Cash'][t - 1] + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                numberOfShares = math.floor(otherCash/(self.data['Close'][t] * (1 + self.transactionCosts)))
                otherCash = otherCash + numberOfShares * self.data['Close'][t] * (1 - self.transactionCosts)
                otherHoldings = - self.numberOfShares * self.data['Close'][t]
        otherMoney = otherHoldings + otherCash
        if not customReward:
            otherReward = (otherMoney - self.data['Money'][t-1])/self.data['Money'][t-1]
        else:
            otherReward = (self.data['Close'][t-1] - self.data['Close'][t])/self.data['Close'][t-1]
        otherState = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [otherPosition]]
        self.info = {'State' : otherState, 'Reward' : otherReward, 'Done' : self.done}

        # Return the trading environment feedback to the RL trading agent
        return self.state, self.reward, self.done, self.info


    def render(self):
        """
        GOAL: Illustrate graphically the trading activity, by plotting
              both the evolution of the stock market price and the 
              evolution of the trading capital. All the trading decisions
              (long and short positions) are displayed as well.
        
        INPUTS: /   
        
        OUTPUTS: /
        """

        # Set the Matplotlib figure and subplots
        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211, ylabel='Price', xlabel='Time')
        ax2 = fig.add_subplot(212, ylabel='Capital', xlabel='Time', sharex=ax1)

        # Plot the first graph -> Evolution of the stock market price
        self.data['Close'].plot(ax=ax1, color='blue', lw=2)
        ax1.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Close'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='orange', label='Long 50')
        ax1.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Close'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='blue', label='Short')
        ax1.plot(self.data.loc[self.data['Action'] == 2.0].index, 
                 self.data['Close'][self.data['Action'] == 2.0],
                 '^', markersize=5, color='green', label='Long 100')
        
        # Plot the second graph -> Evolution of the trading capital
        self.data['Money'].plot(ax=ax2, color='blue', lw=2)
        ax2.plot(self.data.loc[self.data['Action'] == 1.0].index, 
                 self.data['Money'][self.data['Action'] == 1.0],
                 '^', markersize=5, color='orange', label='Long 50')
        ax2.plot(self.data.loc[self.data['Action'] == -1.0].index, 
                 self.data['Money'][self.data['Action'] == -1.0],
                 'v', markersize=5, color='blue', label='Short')
        ax2.plot(self.data.loc[self.data['Action'] == 2.0].index, 
                 self.data['Money'][self.data['Action'] == 2.0],
                 '^', markersize=5, color='green', label='Long 100')
        
        # Generation of the two legends and plotting
        ax1.legend(["Price", "Long 50",  "Short", "Long 100"])
        ax2.legend(["Capital", "Long 50", "Short", "Long 100"])
        plt.savefig(''.join(['Figures/', str(self.marketSymbol), '_Rendering', '.png']))
        self.data.to_csv('test_result.csv')
        #plt.show()




    def setStartingPoint(self, startingPoint):
        """
        GOAL: Setting an arbitrary starting point regarding the trading activity.
              This technique is used for better generalization of the RL agent.
        
        INPUTS: - startingPoint: Optional starting point (iteration) of the trading activity.
        
        OUTPUTS: /
        """

        # Setting a custom starting point
        self.t = np.clip(startingPoint, self.stateLength, len(self.data.index))

        # Set the RL variables common to every OpenAI gym environments
        self.state = [self.data['Close'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Low'][self.t - self.stateLength : self.t].tolist(),
                      self.data['High'][self.t - self.stateLength : self.t].tolist(),
                      self.data['Volume'][self.t - self.stateLength : self.t].tolist(),
                      [self.data['Position'][self.t - 1]]]
        if(self.t == self.data.shape[0]):
            self.done = 1
    