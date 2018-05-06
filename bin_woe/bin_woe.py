import pandas as pd
import sys
sys.path.append('/Users/sarawang/Documents/工作文件/scorecard_wxy/bin_method')
from ChiMerge_con import ChiMerge_con
from ChiMerge_class import ChiMerge_class


def var_bin(df,y,classified_var,continuous_var):
    '''
    ----------所有变量分箱---------
    分别进行分类变量和连续变量的分箱
    return:
    '''
    bin_classified=pd.DataFrame()
    if len(classified_var)!=0:
        for col in classified_var:
            smb=ChiMerge_class(df,col,y,confidenceVal=3.841, maxbin=10,minBinPcnt=0.05)
            bin_classified=pd.concat([bin_classified,smb],axis=0)
        bin_classified['type'] = 'classified'

    #连续变量分箱
    bin_continuous=pd.DataFrame()
    if len(continuous_var)!=0:
        for col in continuous_var:
            smb=ChiMerge_con(df, col, y, confidenceVal=3.841, maxbin=10,minBinPcnt=0.05)
            bin_continuous=pd.concat([bin_continuous,smb],axis=0)
        bin_continuous['type'] = 'continuous'

    var_bin = pd.concat([bin_classified,bin_continuous],axis=0)
    return var_bin

def is_BadRateMonotone(bin_filter):
    var = bin_filter['var'].drop_duplicates().tolist()
    BadRateMonotone = []
    for col in var:
        badrate = bin_filter[bin_filter['var']==col]['target1_rate']
        badRateMonotone = [badrate[i]<badrate[i+1] and badrate[i] > badrate[i-1] or badrate[i]>badrate[i+1] and badrate[i] < badrate[i-1]
                               for i in range(1,len(badrate)-1)]
        is 'False' in badRateMonotone:





def tran_classified(df,classified,bin_iv):
    '''
    ----------分类变量：替换woe-----------
    '''
    df1 = df[classified].fillna('NaN')
    for col in classified:
        cutpoint = list(bin_iv[bin_iv['var']==col]['cutpoint'])
        woe = list(bin_iv[bin_iv['var']==col]['woe'])
        woe_dict = dict(zip(cutpoint,woe))
        for point in woe_dict.keys():
            original_value = str(point).split(',')
            #print(original_value)
            df1[col] = df1[col].replace(original_value,woe_dict[point])
        print('分类变量_'+col+'woe转换完成!')
    return df1[classified]


def tran_continuous(df,continuous,bin_iv):
    '''
    ------------连续变量：替换woe--------------
    '''
    df1 = df[continuous].fillna(-999)
    for col in continuous:
        cutpoint = list(bin_iv[bin_iv['var']==col]['cutpoint'])
        woe = list(bin_iv[bin_iv['var']==col]['woe'])
        woe_dict = dict(zip(cutpoint,woe))

        df1_col_na = df1[df1[col]==-999]
        df1_col_notnull = df1[df1[col]!=-999]

        if len(df1_col_na)!=0:
            df1_col_na[col] = df1_col_na[col].replace(-999,woe_dict['NaN'])
            del woe_dict['NaN']

        woe_df = pd.DataFrame(list(woe_dict.items()),columns=['cutpoint','woe'])
        new_cut = list(woe_df['cutpoint'])
        new_cut.sort()
        del new_cut[-1]
        new_cut.insert(0,-10e10)
        new_cut.append(10e10)
        woe_df['minband'] = new_cut[:-1]
        woe_df['maxband'] = new_cut[1:]

        bin_df = pd.Series()
        for i in range(woe_df.__len__()):
            df_temp = df1_col_notnull[(df1_col_notnull[col]>woe_df['minband'][i]) & (df1_col_notnull[col]<=woe_df['maxband'][i])]
            df_temp[col] = woe_df['woe'][i]
            bin_df=pd.concat([bin_df,df_temp[col]])
        df1[col] = pd.concat([bin_df,df1_col_na[col]])
        print('连续变量_'+col+'woe转换完成!')
    return df1[continuous]



def tran_woe(df,bin_iv):
    '''
    -----------把df中的变量转为woe---------------
    '''
    varset = bin_iv[['var','type']].drop_duplicates()
    classified = list(varset[varset.type=='classified']['var'])
    continuous = list(varset[varset.type=='continuous']['var'])
    df_woe_classified = tran_classified(df,classified,bin_iv)
    df_woe_continuous = tran_continuous(df,continuous,bin_iv)

    df_woe = pd.concat([df_woe_classified,df_woe_continuous],axis=1)
    return df_woe

# def tran_bin(df,bin_iv):
