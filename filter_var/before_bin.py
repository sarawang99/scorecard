import pandas as pd
from sklearn.model_selection import train_test_split

def missing_rate(df,rate):
    '''
    @rate:缺失率容忍值，一般为0.8
    return:返回缺失率低于rate的数据
    '''
    checknull=df.isnull().sum(axis=0).sort_values(ascending=False)/len(df)
    keeplist=checknull[checknull<rate].index.tolist()
    return df[keeplist]

def single_rate(df,rate):
    '''
    @rate:单一值率容忍值，一般为0.8
    return:返回单一值率低于rate的变量及其单一值比率
    '''
    single_result=df.apply(lambda col:col.value_counts(dropna=False).reset_index(drop=True)[0]/len(df)).reset_index()
    single_result.columns=['var','uniquerate']
    return_var = single_result[single_result.uniquerate < rate]
    return return_var

def sample_cover(df,keyvar,exclude_column=[]):
    df1 = df.drop(exclude_column,axis=1)
    checknull=df.isnull().sum(axis=1).sort_values(ascending=False)/len(df.columns)
    df_key = df[keyvar]
    checknull_keyvar = df_key.isnull().sum(axis=1).sort_values(ascending=False)/len(df_key.columns)
    return checknull,checknull_keyvar


#计算变量的缺失率和单一值比率
def single_miss(df):
    miss_result = df.apply(lambda col: 1 - col.count() / len(df)).reset_index()
    miss_result.columns=['var','missrate']
    single_result=df.apply(lambda col:col.value_counts(dropna=False).reset_index(drop=True)[0]/len(df)).reset_index()
    single_result.columns=['var','uniquerate']
    single_miss_result=pd.merge(miss_result,single_result,on='var')
    return single_miss_result


def single_ks(df,flag,exclude_column=[]):
    '''
    ---------单变量ks计算---------
    '''
    df1 = df.drop(exclude_column,axis=1)
    var = df1.drop(flag,axis=1).columns
    var_ks = {}
    var_ksvalue = {}
    for col in var:
        total = df1.groupby([col])[flag].count()
        bad = df1.groupby([col])[flag].sum()
        all = pd.DataFrame({'total':total, 'bad':bad})
        all['good'] = all['total'] - all['bad']
        all = all.sort_index(ascending=False).reset_index()
        all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
        all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
        all['KS'] = all.apply(lambda x: abs(x.badCumRate - x.goodCumRate), axis=1)
        max_index = all['KS'].idxmax()
        ks = max(all['KS'])
        max_ks_Value = all[col][max_index]
        var_ks[col] = ks
        var_ksvalue[col] = max_ks_Value
    ksframe1 = pd.DataFrame(list(var_ks.items()),columns=['var','ks'])
    ksframe2 = pd.DataFrame(list(var_ksvalue.items()),columns=['var','ksvalue'])
    ksframe = pd.merge(ksframe1,ksframe2,on='var',how='left')
    ksframe = ksframe.sort_values(by='ks',ascending=False)
    return ksframe


'''
判断是否为分类变量
1、通过变量type判断
2、通过变量类别个数判断
'''
def class_var(df,exclude_column=[]):
    df1 = df.drop(exclude_column,axis=1)
    var_type=df1.dtypes.reset_index()
    var_type.columns=['var','type']
    classified=var_type[var_type.type=='object']['var'].tolist()
    continuous=var_type[var_type.type!='object']['var'].tolist()
    return classified,continuous

def class_var2(df,classnum,exclude_column=[]):
    df1 = df.drop(exclude_column,axis=1)
    categoricalFeatures = []
    numericalFeatures = []
    for var in df1.columns:
        if (len(df[var].drop_duplicates()) > classnum):
            numericalFeatures.append(var)
        else:
            categoricalFeatures.append(var)
    return categoricalFeatures,numericalFeatures

def train_test(df,flag,testrate):
    '''
    :param df:待抽样数据
    :param flag:数据因变量
    :param testrate:测试集样本比例
    :return:训练集和测试集
    '''
    #from sklearn.model_selection import train_test_split
    y=df[flag]
    x = df.drop(flag,axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testrate, random_state=23)
    return x_train,y_train,x_test,y_test


def tran_type(df,classified_var,continuous_var):
    if len(classified_var)!=0:
        df[classified_var] = df[classified_var].astype(str)
    if len(continuous_var)!=0:
        df[continuous_var] = df[continuous_var].astype(float)
    return df

def tran_na(df,not_in_list=["None", "NaN", "NA", "nan", None, "-999", "-999.0", -999, "-1111", "-1111.0", -1111]):
    import numpy as np
    if len(not_in_list)!=0:
        df1 = df.replace(not_in_list,np.nan)
        return df1
    else:
        return df
