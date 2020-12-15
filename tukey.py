# -*= coding:utf-8 -*-
import numpy as np
import scipy.stats
import pandas as pd
import math
import scipy.optimize
from statsmodels.stats.libqsturng import psturng, qsturng


def tukey_hsd(args, ddof = 1):
    """
    Usage
    ------
    tukey(args, ddof=1.0)
    
    Parameters
    --------
    args : pandas dataframe datasets
            pd.DataFrame({'A':[x11,x12,x13,,,,x1i],
                   'B':[x21,x22,x23,,,,x2j],
                   'K':[..................]
                   'N':[xn1,xn2,xn3,,,,xnk]])
    ddof : delta degrees of freedom, default is 1.
    
    Returns
    -------
    dict of 'summary', 'p', 'v','var_e'
    summary : contains mean and var within group, and size of each groups.
    p       : contains dataframe of p-values between each groups.
    t       : contains dataframe of t-values between each groups.
    var_e   : contains df(degrees of freedom of error-variance), and error-variance between groups.
    Comments
    -------
    Multiple comparison with Tukey-Kramer's range test.
    This program is a ported python-script from R-script called tukey, originally 
    coded by Dr Shigeyuki Aoki.
    psturng and qsturng functions were imported from statsmodels.
    
    This program was wirtten by Kyoichiro Higashi, (khigashi@my-pharm.ac.jp)
    """
    
    ddof = ddof
    data = args
    alldata = args.unstack().values
    groups = len(args.columns)		# number of groups
    n_wig = args.notna().sum(0)      # cases within groups
    phi_e = (n_wig - ddof).sum()                    # sum of degrees of freedom
    mean_wig = data.mean()       # means of each groups
    var_wig = data.var()       # variance of each groups
    idx_group = args.columns
    summary = pd.DataFrame({'size':n_wig, 'mean':mean_wig, 'var':var_wig},
    index=idx_group)           # statics of each groups

    var_e = sum((n_wig - ddof) * var_wig) / phi_e                # Error variance (Variance within group)
    t = np.array( [np.array( [ abs(mean_wig[i] - mean_wig[j]) / np.sqrt(var_e * (1. / n_wig[i] + 1. / n_wig[j])) for i in range(groups)] ) for j in range(groups)] )  # t statistics of paired comparison         
    prob = psturng(t*np.sqrt(2.0), groups, phi_e)

    t = pd.DataFrame(data = t,index = idx_group, columns = idx_group)
    prob = pd.DataFrame(data = prob,index = idx_group, columns = idx_group)
    variance = pd.DataFrame(data = [phi_e, var_e], index = ['df','var'], columns = ['error'])

    return {'summary':summary, 't':t, 'p':prob, 'var_e':variance}
