def iv_choose(woe_result,threold):
    iv_result=woe_result[['var','iv']].drop_duplicates().reset_index(drop=True)
    iv_result=iv_result.sort_values(by='iv',ascending=False)
    return iv_result[iv_result.iv>=threold]


#指标的相关性筛选
def corr_func(data, coef, iv, exclude_column=None, corr_method='pearson'):
    '''

    data:包含全部变量的数据框
    coef:相关系数限制条件
    iv:只包含var和iv值的数据框
    corr_method：默认是pearson，可以自定义为其他类型
    '''
    import numpy as np
    import pandas as pd
    import copy
    if exclude_column is None:
        exclude_column = [self.ID, self.target]
    data_new = data
    iterate = 1
    while True:
        temp_data = data_new.drop(exclude_column,axis=1)
        corr_result = temp_data.corr(corr_method)
        corr_result_array = np.array(corr_result)
        x, y = np.nonzero(np.triu(corr_result_array, 1) > coef)
        temp_dict = {}
        temp_dict['var1'] = list(np.array(corr_result.index)[x])
        temp_dict['var2'] = list(np.array(corr_result.columns)[y])
        corr_delete_var = pd.DataFrame(temp_dict)
        corr_delete_var = corr_delete_var[['var1', 'var2']]
        if len(corr_delete_var) > 0:
            iv_1 = copy.deepcopy(iv)
            iv_1.columns = ['var1', 'var1_iv']
            corr_delete_var = corr_delete_var.merge(iv_1, on='var1', how='left')
            iv_2 = copy.deepcopy(iv)
            iv_2.columns = ['var2', 'var2_iv']
            corr_delete_var = corr_delete_var.merge(iv_2, on='var2', how='left')
            corr_delete_var['delete_var'] = list(map(lambda x, y, z, w: x if z <= w else y, corr_delete_var['var1'],corr_delete_var['var2'], corr_delete_var['var1_iv'],corr_delete_var['var2_iv']))
            corr_delete_unique = list(set(corr_delete_var['delete_var'].tolist()))
            data_new = data_new.drop(corr_delete_unique,axis=1)
            # iterate += 1
        else:
            break
    return data_new
