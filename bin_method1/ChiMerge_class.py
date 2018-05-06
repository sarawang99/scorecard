def ChiMerge_class(df, var, flag, confidenceVal=3.841,maxbin=10,minBinPcnt=0):
    '''
    分类变量分箱
    Parameters
    -----------
    df:dataframe,传入一个数据框仅包含一个需要卡方分箱的变量与正负样本标识(响应样本为1，非响应为0)
    var:变量名。var需在df中
    flag:y_train
    confidenceVal:P(K^2≥k)=0.05 表示组间相近的概率为5%，k越大，p越小
    maxbin:最多箱子数量
    minBinPcnt:组内样本最低占比

    Returns
    ---------
    result_data:dataframe
          包含该变量分箱点,woe,iv等信息
    '''
    import numpy as np
    import math
    import pandas as pd

    df1 = pd.concat([df,flag],axis=1)
    total_num = df1.groupby([var])['label'].count()
    target1_class = df1.groupby([var])['label'].sum()
    regroup = pd.DataFrame({'total_num':total_num,'target1_class': target1_class}).reset_index()
    regroup['target0_class'] = regroup['total_num'] - regroup['target1_class']
    regroup = regroup.drop('total_num', axis=1)
    np_regroup = np.array(regroup)  # 把数据框转化为numpy（提高运行效率）
    print('已完成数据读入,正在进行数据初处理')

    # 处理连续没有正样本或负样本的区间，并进行区间的合并（以免卡方值计算报错）
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] == 0 and np_regroup[i + 1, 1] == 0) or (np_regroup[i, 2] == 0 and np_regroup[i + 1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 响应样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 非响应样本
            np_regroup[i, 0] = np_regroup[i, 0]+','+ np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    #计算分组样本占比，把低于5%的分组合并
    i = 0
    while (i <= np_regroup.shape[0] - 2):
        if ((np_regroup[i, 1] + np_regroup[i,2])/sum(np_regroup[:,1]+np_regroup[:,2])<minBinPcnt):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i + 1, 1]  # 响应样本
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i + 1, 2]  # 非响应样本
            np_regroup[i, 0] = np_regroup[i, 0]+','+ np_regroup[i + 1, 0]
            np_regroup = np.delete(np_regroup, i + 1, 0)
            i = i - 1
        i = i + 1

    # 对相邻两个区间进行卡方值计算
    chi_table = np.array([])  # 创建一个数组保存相邻两个区间的卡方值  n(ad-bc)^2/(a+b)(c+d)(a+c)(b+d)
    for i in np.arange(np_regroup.shape[0] - 1):
        a = np_regroup[i, 1]
        b = np_regroup[i, 2]
        d = np_regroup[i + 1, 2]
        c = np_regroup[i + 1, 1]
        chi = (a + b + c + d) * (a*d - b*c) ** 2 /((a + b) * (c + d) * (a + c) * (b + d))
        chi_table = np.append(chi_table, chi)
    print('已完成数据初处理，正在进行卡方分箱核心操作')

    #把卡方值最小的两个区间进行合并（卡方分箱核心）
    while (1):
        if (len(chi_table)!=0):
            if (len(chi_table) <= (maxbin - 1) and min(chi_table) >= confidenceVal):
                break
            #找出卡方值最小所在的索引、合并相邻的分组
            chi_min_index = np.argwhere(chi_table == min(chi_table))[0]  # 找出卡方值最小的位置索引
            np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index + 1, 1]
            np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index + 1, 2]
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index, 0] +','+np_regroup[chi_min_index + 1, 0]
            np_regroup = np.delete(np_regroup, chi_min_index + 1, 0)

            '''
            计算合并后分组相关联的卡方值并替换
            1、当chi_min_index位于chi_table最后一个时：计算合并后分组与前一个分组的卡方值
            2、
            3、否则：计算合并后分组与前一个分组、后一个分组的卡方值
            '''
            if (chi_min_index == np_regroup.shape[0] - 1):
                a = np_regroup[chi_min_index - 1, 1]
                b = np_regroup[chi_min_index - 1, 2]
                c = np_regroup[chi_min_index, 1]
                d = np_regroup[chi_min_index, 2]
                chi_table[chi_min_index - 1] = (a + b + c + d) * (a*d - b*c) ** 2 /((a + b) * (c + d) * (a + c) * (b + d)) #计算与前一组的卡方值
                chi_table = np.delete(chi_table, chi_min_index, axis=0) # 删除替换前的卡方值
            elif chi_min_index ==0:
                a1 = np_regroup[chi_min_index, 1]
                b1 = np_regroup[chi_min_index, 2]
                c1 = np_regroup[chi_min_index + 1, 1]
                d1 = np_regroup[chi_min_index + 1, 2]
                chi_table[chi_min_index] = (a1 + b1 + c1 + d1) * (a1*d1 - b1*c1) ** 2 /((a1 + b1) * (c1 + d1) * (a1 + c1) * (b1 + d1)) #计算与后一组的卡方值
                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)  #删除被替换组的卡方值
            else:
                a = np_regroup[chi_min_index - 1, 1]
                b = np_regroup[chi_min_index - 1, 2]
                c = np_regroup[chi_min_index, 1]
                d = np_regroup[chi_min_index, 2]
                chi_table[chi_min_index - 1] = (a + b + c + d) * (a*d - b*c) ** 2 /((a + b) * (c + d) * (a + c) * (b + d)) #计算与前一组的卡方值

                a1 = np_regroup[chi_min_index, 1]
                b1 = np_regroup[chi_min_index, 2]
                c1 = np_regroup[chi_min_index + 1, 1]
                d1 = np_regroup[chi_min_index + 1, 2]
                chi_table[chi_min_index] = (a1 + b1 + c1 + d1) * (a1*d1 - b1*c1) ** 2 /((a1 + b1) * (c1 + d1) * (a1 + c1) * (b1 + d1)) #计算与后一组的卡方值

                chi_table = np.delete(chi_table, chi_min_index + 1, axis=0)  #删除被替换组的卡方值
        print('已完成卡方分箱核心操作，正在保存结果')
        break

    # 把分箱结果保存成一个数据框
    #增加缺失组
    dfnull = df1[df1[var].isnull().values==True]
    if len(dfnull)!=0:
        cutpoint = np_regroup[:, 0].tolist()+['NaN']
        target_0 = np_regroup[:, 2].tolist()+[dfnull.label.value_counts()[0]]
        target_1 = np_regroup[:, 1].tolist()+[dfnull.label.value_counts()[1]]
    else:
        cutpoint = np_regroup[:, 0]
        target_0 = np_regroup[:, 2]
        target_1 = np_regroup[:, 1]

    result_data = pd.DataFrame()  # 创建一个保存结果的数据框
    result_data['var'] = [var] * np_regroup.shape[0]  # 结果表第一列：变量名
    result_data['cutpoint'] = cutpoint  # 切割点
    result_data['interval'] = cutpoint
    result_data['target_0'] =  target_0 # 结果表第三列：0样本数目
    result_data['target_1'] =  target_1# 结果表第四列：1样本数目
    result_data['bin_count'] = list(map(lambda x,y:x+y,result_data['target_1'], result_data['target_0']))
    result_data['bin_rate'] = result_data['bin_count']/result_data['bin_count'].sum()

    tot_target1 = df1['label'].sum()
    tot_target0 = len(df1) - tot_target1

    result_data['target1_rate'] = result_data['target_1']/result_data['bin_count']
    result_data['dist_target1'] = result_data['target_1'] / tot_target1
    result_data['dist_target0'] = result_data['target_0'] / tot_target0

    result_data['woe'] = list(
        map(lambda x, y: math.log(x/(y+0.00000000001)) if y==0 else math.log((x+0.0000000001)/y) if x==0 else math.log(x / y), result_data['dist_target1'],
            result_data['dist_target0']))
    result_data['iv_group'] = list(
        map(lambda x, y, z: (x - y)*z if x!=0 else 0,result_data['dist_target1'],
            result_data['dist_target0'], result_data['woe']))
    result_data['iv'] = sum(result_data['iv_group'])
    return result_data
