import pandas as pd
import scipy.stats
import numpy as np

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawadowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
        
    })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    Load and format the Hedge fund index returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                     header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_files(filetype):
    """
    Load and format Ken French 30 Industry Portfolios file
    """
    known_types = ["returns", "nfirms", "size"]
    if filetype not in known_types:
        sep = ','
        raise ValueError(f'filetype must be one of :{sep.join(known_types)}')
    if filetype is "returns":
        name = "vw_rets"
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms":
        divisor = 1
    ind = pd.read_csv(f"data/ind30/_m_{name}.csv", header=0, index_col=0)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """
    return get_ind_file("returns")

def get_ind_nfirms():
    """
    Load and format Ken French 30 Industry Portfolios Average number of firms
    """
    return get_ind_file("nfirms")

def get_ind_size():
    """
    Load and format Ken French 30 industry Portfolios Average size (market cap)
    """
    return get_ind_file("size")

def get_total_market_reutrns():
    """
    Load the 20 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_firms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_marketcap = ind_size * ind_firms
    total_marketcap = ind_marketcap.sum(axis=1)
    ind_capweight = ind_marketcap.divide(total_marketcap, axis="rows")
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a series or a dataframe
    """
    
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied series or dataframe
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the skewness of the supplied series or dataframe
    Returns a float or a series
    """
    demeaned_r = r - r.mean()
    # use population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a series is normal or not
    Test is applied at the %1 level by defualt
    Returns True if the hypothesis of normality is accepted, false or otherwise
    """
    statistic, p_values = scipy.stats.jarque_bera(r)
    return p_values > level

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns Parametric Gaussian VaR of a Series or DataFrame
    If "modified is True, then the modified VaR is returned "
    """
    
    # Compute the z-score assuming it was gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score baed on observed skewness and Kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1)*s/6 + 
             (z**3 - 3*z)*(k-3)/24 - 
             (2*z**3 - 5*z)*(s**2)/36
            )
        
    return -(r.mean() + z*r.std(ddof=0))

def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a series or dataframe")
    

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a series or DataFrame")
        