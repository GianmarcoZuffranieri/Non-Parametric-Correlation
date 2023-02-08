"""
NP Assignment.py

Purpose:
    Non-Parametric approach for stock correlation

Version:
    1  

Date:
    2022/10/07

Author:
    Gianmarco Zuffranieri
"""
###########################################################
### Imports
from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###########################################################
### df= ReadCAPM(asStocks, sIndex, sTB, sPer)
def ReadCAPM(asStocks,sPer):
    """
    Purpose:
        Read CAPM data

    Inputs:
        asStocks    list, with list of stocks,
        sIndex      string with sp500,
        sTB         string with TB indicator
        sPer        string, indicator for period

    Return value: 
    """
    df= pd.DataFrame()
    # Get SP500 data, only Adj Close, by Date
    #df[sIndex]= pd.read_csv('Data/%s_%s.csv' % (sIndex, sPer), index_col='Date')['Adj Close']

    # Add in TB, here the date is called DATE
    #df[sTB]= pd.read_csv('Data/%s_%s.csv' % (sTB, sPer), index_col='DATE',na_values='.')

    # Add in stocks, one by one
    for sStock in asStocks:
        df[sStock]= pd.read_csv('Data/%s_%s.csv' % (sStock, sPer), index_col='Date')['Adj Close']

    df.index.name= 'Date'   # Fix the name of the index column
    df.index= pd.to_datetime(df.index)
    #df.index = df.index.strftime('%d/%m/%Y')    # Ensure it is a true datetime

    # For simplicity, drop full rows with nans;
    # df= df.dropna(axis=0, how='any')
    # For less simplicity, drop rows with only nans;
    df= df.dropna(axis=0, how='all')

    return df

###########################################################
### AddExcessRet(df, 252, 'DTB3')
def AddExcessRet(df, asStocks):
    """
    Purpose:
        Add excess returns

    Inputs:
        df      dataframe with TB, and stock prices
        iDoP    integer, number of days to use for rescaling TB

    Outputs:
        df      dataframe with TB, and stock prices, and returns of stock prices, and returns ex market risk

    Return value:
        None
    """
    # Get daily risk-free rate
    #srRF= df[tb]/iDoP
    #asS= np.setdiff1d(df.columns, tb)
    for sStock in asStocks:
        df[f'r{sStock}']= 100*np.log(df[sStock]).diff()
        #df[f'r{sStock}Ex']= df[f'r{sStock}'] - srRF

    print ('Number of missings:\n', df.isna().sum(axis=0))
    df.dropna(axis=0, how='any', inplace= True)



###########################################################
### Uniform Kernel
def Ku(vZ):
    """
    Purpose:
        Provide the uniform kernel

    Inputs:
        vZ      iN vector (or iN x iM matrix, whatever...)

    Return value:
        vK      iN vector (or whatever)
    """
    vK= np.zeros_like(vZ)
    vK[np.fabs(vZ) <= 1]= 0.5

    return vK

###########################################################
### Triangular kernel
def Kt(vZ):
    """
    Purpose:
        Provide the triangular/bartlett kernel

    Inputs:
        vZ      iN vector (or iN x iM matrix, whatever...)

    Return value:
        vK      iN vector (or whatever)
    """
    vK= np.zeros_like(vZ)
    vI= np.fabs(vZ) <= 1
    vK[vI]= (1-np.fabs(vZ[vI]))

    return vK

###########################################################
### KernRegr(vY, mX, vXk, mXe, vXe, dH, fnK)
def KernRegr(vY, mX, vXk, mXe, vXe, dH, fnK, bSilent=False):
    """
    Purpose:
        Perform kernel regression

    Inputs:
        vY      iN vector of data
        mX      iN x iK matrix of explanatory variables
        vXk     iN vector of locations, used for kernel weights
        mXe     iE x iK matrix of explanatory variables, at evaluation points
        vXe     iE vector of locations, used for kernel weights, at evaluation
                points
        dH      double, bandwidth
        fnK     (optional, default= Ku) function, kernel
        bSilent (optional, default= False) boolean, if True information on
                screen is given

    Return values:
        vYe     iE vector, fitted heights
        mB      iK x iE matrix, estimated beta's
    """
   
    mX= mX.reshape(len(mX), 1)
    (iN, iK)= mX.shape
    iE=  len(vXe)

    vYe= np.zeros(iE)
    mB= np.zeros((iK, iE))

    if (not bSilent):
        print ('\nRunning %i KernRegressions' % iE)
    for i in range(iE):
        if (not bSilent):
            if ( i % 70 == 0):
                print ('\ni=%5i ' % i, end='')
            print ('.', end='')
        vK= fnK((vXk - vXe[i])/dH)
        mK= np.diag(vK)

        mXtKX= mX.T@mK@mX
        mXtKy= mX.T@mK@vY
        vB= np.linalg.lstsq(mXtKX, mXtKy, rcond=None)[0]
        mB[:,i]= vB
        vYe[i]= mXe[i,:]@vB
    if (not bSilent):
        print ('\n')

    return (vYe, mB)


###########################################################
### LocRegr(vY, mX, sBase)
def LocRegr(vY, mX, dH, fnK, sBase):
    """
    Purpose:
        Perform local regression
    """
    iN= len(vY)
    vXk= np.arange(iN)
    mXe= mX             # Note: a copy, no use to take separate variables
    vXe= vXk

    try:
        # Try to use earlier results, if available
        cRes= np.load('Output/est'+sBase+'_kernel.npz')
        (vYe, mB)= (cRes['vYe'], cRes['mB'])
    except:
        # Else re-compute, and store
        (vYe, mB)= KernRegr(vY, mX, vXk, mXe, vXe, dH, fnK)
        np.savez_compressed('Output/est'+sBase+'_kernel.npz', vYe=vYe, mB=mB)

    return (vYe, mB)
###############################################
### Local regression for each stock

def stock_LocReg(df,vX,dH,stock,ret,kern_shape):
    if ret == "Ret":
        sBase = f"Mean_{stock}_{dH}_days_{kern_shape.__name__}"
        vY = df[stock].values
        (vYe,  mB)= LocRegr(vY, vX, dH, kern_shape, sBase)
        return mB
      
    else:
        sBase = f"Mean2_{stock}_{dH}_days_{kern_shape.__name__}"
        vY = (df[stock].values)**2
        (vYe, mB2)= LocRegr(vY, vX, dH, Ku, sBase)
        return mB2



###############################################
### Daily correlation

def daily_corr(df,vX,dH,kern_shape):
    """
    Purpose:
        Compute correlation with NW estimator
    """
    corr = pd.DataFrame(stock_LocReg(df,vX,dH,"rAMZN","Ret",kern_shape).T, index = df.index, columns=["Mean_AMZN"])
    corr["Mean2_AMZN"] = stock_LocReg(df,vX,dH,"rAMZN","Ret2",kern_shape).T
    corr["Mean_UNH"] = stock_LocReg(df,vX,dH,"rUNH","Ret",kern_shape).T
    corr["Mean2_UNH"] = stock_LocReg(df,vX,dH,"rUNH","Ret2",kern_shape).T
    corr["VAR_AMZN"]  =  corr["Mean2_AMZN"] - corr["Mean_AMZN"]**2
    corr["VAR_UNH"]  =  corr["Mean2_UNH"] - corr["Mean_UNH"]**2
    corr["AMZN"] = df["rAMZN"].values
    corr["UNH"] = df["rUNH"].values
    corr["Covariance"] = (corr["AMZN"] - corr["Mean_AMZN"])*(corr["UNH"] - corr["Mean_UNH"])
    corr["Daily_Covariance"] = stock_LocReg(corr,vX,dH,"Covariance","Ret",kern_shape).T
    corr["Correlation"] = corr["Daily_Covariance"]/(np.sqrt(corr["VAR_AMZN"]*corr["VAR_UNH"]))
    return corr
  

###############################################
### Get same month as intraday data (March 2022)

def get_specific_month(df):
    """
    Purpose:
        Specify the single month of interest
    """
    df.index = pd.to_datetime(df.index)
    specific_month = df.query("20220228 < index < 20220401")
    return specific_month["Correlation"]


###############################################
### Plot

def plot(x,y):

    plt.figure(figsize=(7, 5))      # Choose alternate size
    #plt.subplot(2, 1, 1)            # Work with 2x1 grid, first plot
    plt.plot(x["VAR_AMZN"],label= "Nadaraya-Watson")
    plt.plot(y["VAR_AMZN"], color ="tab:red")  
    plt.ylabel("Correlation")
    #plt.xlabel("Time")               # Simply plot the white noise
    plt.legend()
    plt.title(f"Daily Correlation for March 2022")  
      

    #plt.subplot(2, 1, 2)            # Start with second plot
    #plt.plot(y,label= "Freq = 15 Min", color ="tab:red") 
    plt.ylabel("Correlation")
    plt.xlabel("Monthly Days")
    plt.legend()
    #plt.title(f"Daily Correlation with {kern_shape} kernel") 
   
    plt.savefig(f'Graphs/1022_NW_ID_15Min.pdf', bbox_inches='tight') 
    """
    plt.subplot(4, 1, 3) 
    plt.plot(z)  # Plot here some cross-plots
    plt.ylabel("Correlation")
    plt.xlabel("Time")
    plt.legend("H=1y/2")
    plt.title("Daily Correlation with triangular kernel") 

    plt.subplot(4, 1, 4) 
    plt.plot(w)  # Plot here some cross-plots
    plt.ylabel("Correlation")
    plt.xlabel("Time")
    plt.legend("H=2y/2")
    plt.title("Daily Correlation with triangular kernel") 
    """
    plt.tight_layout(pad=5.0) 

    plt.show()     

###############################################
### Read intraday data
"""
def get_intraday_data(sData,iN,sFreq):

    df= pd.read_csv(sData, nrows= iN, parse_dates=[['DATE', 'TIME_M']])
    df.set_index(['DATE_TIME_M'], inplace=True)
    vI= [type(sSuf) is float for sSuf in df['SYM_SUFFIX']]
    print ('Missing suffix: %i, non-missing suffix (hence to be dropped): %i' % (np.sum(vI), len(df) - np.sum(vI)))
    df= df[vI]
    # Select single exchange
    srC= df['EX'].value_counts()
    print ("Finding the following counts of exchanges:\n", srC)
    print ("Most common: Exchange %s at %i/%i=%g" % (srC.index[0], srC[0], srC.sum(), srC[0]/srC.sum()))
    vI= df['EX'] == srC.index[0]
    df= df[vI]
    # Select timeslot 9.30 - 16.00
    asTime= ['9:30', '16:00']
    vT= pd.to_datetime(asTime, format='%H:%M').time     # Use only the time part of the datetime item
    # Compare that time to the time of the index
    vI= (df.index.time >= vT[0]) & (df.index.time <= vT[1])
    df= df[vI]          # Select only trades inside the window.
    # Move to specified frequency
    dfFr= df['PRICE'].resample(sFreq).ohlc().dropna()
    dfFr.to_csv(f"Output/UNH_intraday_{sFreq}", index=True)
"""
   

###############################################
### Intraday data return

def intraday_return(lData5,lData1,lData15):
    """
    Purpose:
        Read intra-day data
    """

    df5 = pd.DataFrame()
    for sStocks in lData5:

        df5[sStocks] = pd.read_csv(f"Output/{sStocks}",index_col='DATE_TIME_M')["close"]
        df5[f'r_{sStocks}']= np.log(df5[sStocks]).diff()
        df5[f'r2_{sStocks}']= (np.log(df5[sStocks]).diff())**2
    
    df5.index = pd.to_datetime(df5.index)
    df5.dropna(axis=0, how='any', inplace= True)
 
    df1 = pd.DataFrame()
    for sStocks in lData1:
    
        df1[sStocks] = pd.read_csv(f"Output/{sStocks}",index_col='DATE_TIME_M')["close"]
        df1[f'r_{sStocks}']= np.log(df1[sStocks]).diff()
        df1[f'r2_{sStocks}']= (np.log(df1[sStocks]).diff())**2
    df1.index = pd.to_datetime(df1.index)
    df1.dropna(axis=0, how='any', inplace= True)

        
    df15 = pd.DataFrame()
    for sStocks in lData15:

        df15[sStocks] = pd.read_csv(f"Output/{sStocks}",index_col='DATE_TIME_M')["close"]
        df15[f'r_{sStocks}']= np.log(df15[sStocks]).diff()
        df15[f'r2_{sStocks}']= (np.log(df15[sStocks]).diff())**2
    df15.index = pd.to_datetime(df15.index)
    df15.dropna(axis=0, how='any', inplace= True)

        
      
    dfunique = pd.read_csv("Output/AMZN_intraday_5Min", index_col= "DATE_TIME_M")
    dfunique.index = pd.to_datetime(dfunique.index)
    unique = dfunique.index.map(lambda t: t.date()).unique()
    return df5,df1,df15,unique


def intraday_variance(df,unique,stock,sFreq):
    """
    Purpose:
        Compute daily variance from intra-day data
    """

    lInt_var = []
    for ddI in unique:
        vI= df.index.date == ddI
        lInt_var.append(df[f"r2_{stock}_intraday_{sFreq}"][vI].sum())
    return lInt_var

def intraday_covariance(df,unique,stock1,stock2,sFreq):
    """
    Purpose:
        Compute daily covariance from intra-day data
    """

    lInt_cov = []
    for ddI in unique:
        vI= df.index.date == ddI
        lInt_cov.append((df[f"r_{stock1}_intraday_{sFreq}"]*df[f"r_{stock2}_intraday_{sFreq}"][vI]).sum())
    return lInt_cov

def intraday_correlation(unique,cov,var1,var2):
    """
    Purpose:
        Compute daily correlation from intra-day data
    """

    pd.to_datetime(unique)
    intraday_correlation_df = pd.DataFrame(cov, columns=["Daily_Covariance"], index = unique)
    intraday_correlation_df["Daily_Variance_AMZN"] = var1
    intraday_correlation_df["Daily_Variance_UNH"] = var2
    intraday_correlation_df["Daily_Correlation"] = intraday_correlation_df["Daily_Covariance"]/(np.sqrt(intraday_correlation_df["Daily_Variance_AMZN"] * intraday_correlation_df["Daily_Variance_UNH"]))
   
    return intraday_correlation_df
    

def main():
    # Magic numbers
    asStocks= ["AMZN","UNH"]
    sPer= '1022'
    iDoP= 252          # Days per period 
    iYear= 2015         # Year to highlight
    sBase= "Mean"
    
    sData= 'Data/UNH_intraday.csv'
    lData_5Min= ['AMZN_intraday_5Min', 'UNH_intraday_5Min']  
    lData_1Min = ['AMZN_intraday_1Min',"UNH_intraday_1Min"]
    lData_15Min = ['AMZN_intraday_15Min','UNH_intraday_15Min']
    #asTime= ['9:30', '16:00']
    #iN= None                        
    #sFreq= ["5Min","1Min","15Min"]
    
    # Initialisation

    # NW estimator
    ret = "Ret"        #Ret for normal Return, Ret2 for squared Return 
    dH= iDoP/2         # N-width bandwidth (iDoP/2,iDoP/4,iDoP/6)
    kern_shape = Kt   # Choose kernel shape (Ku,Kt)
    df= ReadCAPM(asStocks,sPer)
    AddExcessRet(df,asStocks)

    iN =  len(df)
    vX = np.ones((iN,1))
 
    # NP estimation
    (df_5Min,df_1Min,df_15Min,unique) = intraday_return(lData_5Min,lData_1Min,lData_15Min)

    lVariance1 = intraday_variance(df_5Min,unique,"AMZN","5Min")     # Insert frequence
    lVariance2 = intraday_variance(df_5Min,unique,"UNH","5Min")      # Insert frequence
    lCovariance = intraday_covariance(df_5Min,unique,"AMZN","UNH","5Min")   # Insert 2 stocks and frequence
    df_correlation_intraday = intraday_correlation(unique,lCovariance,lVariance1,lVariance2)   # Insert covariance list
    #plot(daily_corr(df,vX,dH=iDoP/2 ,kern_shape=Ku),daily_corr(df,vX,dH=iDoP/6,kern_shape=Ku),"Uniform")   #Insert x,y and kernel shape: Uniform (Ku) or Triangular (Kt)
    #plot(get_specific_month(daily_corr(df,vX,dH=iDoP/2 ,kern_shape=Ku)),df_correlation_intraday)


    return plot(daily_corr(df,vX,dH=iDoP/2,kern_shape=Ku),daily_corr(df,vX,dH=iDoP/4,kern_shape=Ku))    #Force dH and ker_shape to output the specific dataframe

###########################################################
### start main
if __name__ == '__main__':
    main()  

