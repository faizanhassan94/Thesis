import pandas as pd
from django.shortcuts import render
from .forms import MyForm
import pm4py
import os
import numpy as np

def Get_Event_Log(dataframe):
    eventlog = dataframe.copy()
    eventlog = pm4py.format_dataframe(eventlog, case_id='Case ID', activity_key='Activity', timestamp_key='Complete Timestamp')
    eventlog = pm4py.convert_to_event_log(eventlog)
    return eventlog

def Preprocessing(dataframe):
    dataframe = dataframe.drop_duplicates()
    columns = ['case:concept:name','org:resource','concept:name','time:timestamp','case:enddate_planned','case:enddate','Due Date', 'Invoice_Paid', 'SAP_DOC_NO - Inv. Doc. No.']
    common_columns = set(columns).intersection(dataframe.columns)
    dataframe = dataframe[common_columns]
    column_mapping = {'case:concept:name': 'Case ID',
                  'concept:name': 'Activity',
                   'org:resource': 'Resource',
                   'time:timestamp': 'Complete Timestamp',
                  'SAP_DOC_NO - Inv. Doc. No.': 'Invoice No',
                   'case:enddate_planned': 'Due Date',
                  'Due Date': 'Due Date',
                  'Invoice_Paid' : 'End Date',
                    'case:enddate': 'End Date',
                  
                 }

    dataframe = dataframe.rename(columns=column_mapping)
    desired_order = ['Case ID', 'Activity', 'Resource', 'Complete Timestamp','Invoice No', 'Due Date', 'End Date']
    dataframe = dataframe.reindex(columns=desired_order)
    date_columns = ['Complete Timestamp', 'Due Date', 'End Date']
    common_time_zone = 'UTC'

    for column in date_columns:
        dataframe[column] = pd.to_datetime(dataframe[column], utc=True).dt.tz_convert(common_time_zone)


    dataframe = dataframe.reset_index()
    dataframe.drop("index", axis=1, inplace=True)

    return dataframe

def Get_All_Resources(log):
    import pm4py
    df = pd.DataFrame()
    resources = pm4py.get_event_attribute_values(log, "org:resource")
    df['Resource'] = resources.keys()
    return df

def Due_Date_Compliance(dataframe):
    dataframe['Payment'] = ' '
    
    for i in range(len(dataframe)):
        if pd.isna(dataframe.loc[i, 'Due Date']):
            dataframe['Payment'][i] = 'Not Applicable'   
        elif dataframe['Due Date'][i] >= dataframe['End Date'][i]:
            dataframe['Payment'][i] = 'Paid on time'
        else:
            dataframe['Payment'][i] = 'late payment'
            
    df1 = dataframe[dataframe['Payment']== 'Paid on time']
    df3 = dataframe[dataframe['Payment']== 'late payment']
    
    df1 = df1['Case ID'].unique()
    df3 = df3['Case ID'].unique()
    
    df2 = dataframe[dataframe['Case ID'].isin(df1)]
    df4 = dataframe[dataframe['Case ID'].isin(df3)]
    
    dataframe_good = df2[df2['Payment'] != 'late payment']
    dataframe_bad = df4[df4['Payment'] != 'Paid on time']
    
    dataframe_good = dataframe_good.reset_index()
    dataframe_good.drop("index", axis=1, inplace=True)
    
    dataframe_bad = dataframe_bad.reset_index()
    dataframe_bad.drop("index", axis=1, inplace=True)
    return dataframe_good,dataframe_bad

def Lead_Time(dataframe):
    dataframe = dataframe.copy()
    dataframe['Complete Timestamp'] = pd.to_datetime(dataframe['Complete Timestamp'])
    dataframe['Due Date'] = pd.to_datetime(dataframe['Due Date'])
    dataframe['End Date'] = pd.to_datetime(dataframe['End Date'])
    # Group the data by the Case ID
    grouped_df = dataframe.groupby('Case ID')

    # Calculate the purchase order cycle time for each group
    purchase_order_cycle_time = grouped_df.apply(lambda x: (x['Complete Timestamp'].max() - x['Complete Timestamp'].min()).days)
    purchase_order_cycle_time = purchase_order_cycle_time.reset_index(name='Total_Days')

    # Merge the original dataframe with the purchase_order_cycle_time dataframe on the Case ID column
    dataframe = pd.merge(dataframe, purchase_order_cycle_time, on='Case ID')
    mean_total_days = int(dataframe['Total_Days'].mean())
    dataframe_good = dataframe[dataframe['Total_Days'] <= mean_total_days]
    dataframe_bad = dataframe[dataframe['Total_Days'] > mean_total_days]
    dataframe_good = dataframe_good.reset_index()
    dataframe_good.drop("index", axis=1, inplace=True)
    dataframe_bad = dataframe_bad.reset_index()
    dataframe_bad.drop("index", axis=1, inplace=True)
    return dataframe_good,dataframe_bad

def Rework(dataframe):
    df = dataframe.copy()
    cols_to_check = df.columns[df.columns != 'Complete Timestamp']
    df['Rework'] = df.duplicated(subset=cols_to_check).groupby(df['Case ID']).transform(lambda x: 'Yes' if x.any() else 'No')

    df1 = df[df['Rework']== 'No']
    df3 = df[df['Rework']== 'Yes']

    df1 = df1['Case ID'].unique()
    df3 = df3['Case ID'].unique()

    df2 = df[df['Case ID'].isin(df1)]
    df4 = df[df['Case ID'].isin(df3)]

    dataframe_good = df2[df2['Rework'] != 'Yes']
    dataframe_bad = df4[df4['Rework'] != 'No']

    dataframe_good = dataframe_good.reset_index()
    dataframe_good.drop("index", axis=1, inplace=True)

    dataframe_bad = dataframe_bad.reset_index()
    dataframe_bad.drop("index", axis=1, inplace=True)
    
    return dataframe_good,dataframe_bad

def Good_vs_Bad_log(dataframe,kpi):
    if kpi == 'Due Date Compliance':
        dataframe_good,dataframe_bad = Due_Date_Compliance(dataframe)
    if kpi == 'Lead Time':
        dataframe_good,dataframe_bad = Lead_Time(dataframe)
    if kpi == 'Rework':
        dataframe_good,dataframe_bad = Rework(dataframe)
 
    return dataframe_good,dataframe_bad

def Get_Activities_of_a_resource(dataframe,resource,start_activity):
    df_filtered = dataframe.copy()
    df2 = df_filtered[(df_filtered['Resource']==resource) & (df_filtered['Activity'] != start_activity)]
    df2 = df2.reset_index()
    df2. drop("index", axis=1, inplace=True)

#extracting the previous activity of each case
    df3 = df_filtered[(df_filtered['Resource']==resource) & (df_filtered['Activity'] != start_activity)].index-1
    df4 = df_filtered.loc[df_filtered.index[df3]]

#getting two datasets sorted in ascending order of timestamp
    df2 = df2.sort_values(by=['Complete Timestamp']) #contains the activities performed by resource
    df4 = df4.sort_values(by=['Complete Timestamp']) #contains information on how the cases were assigned
    df2['Execution'] = 'Completed'
    df4['Execution'] = 'Assigned'
    df5 = df2.append(df4, ignore_index=True)
    df5 = df5.sort_values(by=['Complete Timestamp'])
    df5= df5.reset_index()
    df5. drop("index", axis=1, inplace=True)
    return df5

def Get_FIFO_Stats(df5):
    from collections import Counter
    l1 = []
    l2 = []
    fifo = 0
    lifo = 0
    random = 0
    count = 0
    count1 = 0
    task_completed = 0
    fifo_cases = []
    lifo_cases = []
    random_cases = []


    while(count != len(df5)):
        if df5['Execution'][count] == 'Assigned':
            l1.append(df5['Case ID'][count])
            count = count + 1
        elif df5['Execution'][count] == 'Completed':
            count1 = count
            l2.append(df5['Case ID'][count1])
            if count1 == len(df5)-1:
                count1 = count1
            else:
                while(df5['Execution'][count1+1]!= 'Assigned'):
                    count1 = count1+1
                    l2.append(df5['Case ID'][count1])
                    if count1 == len(df5)-1:
                        break
            temp = Counter(l1)
            l1 = [*temp]
            temp = Counter(l2)
            l2 = [*temp]
        
            while len(l2) != 0:
                if len(l1) == 0:
                    l2.clear()
                    break
                elif l1[0] == l2[0]:
                    fifo +=1
                    fifo_cases.append(l2[0])
                    task_completed += 1
                    try:
                        l1.remove(l2[0])
                    except ValueError:
                        pass
                    del l2[0]
                elif l1[-1] == l2[0]:
                    lifo += 1
                    lifo_cases.append(l2[0])
                    task_completed += 1
                    try:
                        l1.remove(l2[0])
                    except ValueError:
                        pass
                    del l2[0]
                else:
                    random +=1
                    random_cases.append(l2[0])
                    task_completed += 1
                    try:
                        l1.remove(l2[0])
                    except ValueError:
                        pass
                    del l2[0]
            count = count1 + 1
    return fifo,lifo,random,task_completed,fifo_cases,lifo_cases,random_cases

def Resource_Behaviour(resources,processed_dataframe,dataframe_good,dataframe_bad,start_activity):
    from collections import Counter
    df = resources.copy()
    dataframe = processed_dataframe.copy()
    df['Task_Completed'] = 0
    df['FIFO'] = 0
    df['LIFO'] = 0
    df['Random'] = 0
    df['Good_FIFO'] = 0
    df['Good_LIFO'] = 0
    df['Good_Random'] = 0
    df['Bad_FIFO'] = 0
    df['Bad_LIFO'] = 0
    df['Bad_Random'] = 0

    for i in range(len(df)):   #len(df)
        df5 = Get_Activities_of_a_resource(dataframe,df['Resource'][i],start_activity)
        a,b,c,d,fc,lc,rc = Get_FIFO_Stats(df5)
        df['FIFO'][i] = a
        df['LIFO'][i] = b
        df['Random'][i] = c
        df['Task_Completed'][i] = d

        if fc != []:
            e = Counter(fc)
            df6 = pd. DataFrame. from_dict(e.keys())
            df6['Value'] = e.values()
            df6.columns = ['Case ID', 'Value']
            df6['Case ID'] = df6['Case ID'].isin(dataframe_good['Case ID'])
            x = df6.loc[df6['Case ID'] == True, 'Value'].sum()
            y= df6.loc[df6['Case ID'] == False, 'Value'].sum()
            df['Good_FIFO'][i] = x
            df['Bad_FIFO'][i] = y


            #df7 = pd. DataFrame. from_dict(e.keys())
            #df7['Value'] = e.values()
            #df7.columns = ['Case ID', 'Value']
            #df7['Case ID'] = df7['Case ID'].isin(dataframe_bad['Case ID'])
            #x = df7.loc[df7['Case ID'] == True, 'Value'].sum()
            #df['Bad_FIFO'][i] = x

        if lc != []:
            f = Counter(lc)
            df8 = pd. DataFrame. from_dict(f.keys())
            df8['Value'] = f.values()
            df8.columns = ['Case ID', 'Value']
            df8['Case ID'] = df8['Case ID'].isin(dataframe_good['Case ID'])
            x = df8.loc[df8['Case ID'] == True, 'Value'].sum()
            y= df8.loc[df8['Case ID'] == False, 'Value'].sum()
            df['Good_LIFO'][i] = x
            df['Bad_LIFO'][i] = y

            #df9 = pd. DataFrame. from_dict(f.keys())
            #df9['Value'] = f.values()
            #df9.columns = ['Case ID', 'Value']
            #df9['Case ID'] = df9['Case ID'].isin(dataframe_bad['Case ID'])
            #x = df9.loc[df9['Case ID'] == True, 'Value'].sum()
            #df['Bad_LIFO'][i] = x


        if rc != []:
            g = Counter(rc)
            df10 = pd. DataFrame. from_dict(g.keys())
            df10['Value'] = g.values()
            df10.columns = ['Case ID', 'Value']
            df10['Case ID'] = df10['Case ID'].isin(dataframe_good['Case ID'])
            x = df10.loc[df10['Case ID'] == True, 'Value'].sum()
            y= df10.loc[df10['Case ID'] == False, 'Value'].sum()
            df['Good_Random'][i] = x
            df['Bad_Random'][i] = y

            #df11 = pd. DataFrame. from_dict(g.keys())
            #df11['Value'] = g.values()
            #df11.columns = ['Case ID', 'Value']
            #df11['Case ID'] = df11['Case ID'].isin(dataframe_bad['Case ID'])
            #x = df11.loc[df11['Case ID'] == True, 'Value'].sum()
            #df['Bad_Random'][i] = x
            
    df = df[df['Task_Completed'] >20]
    df= df.reset_index()
    df. drop("index", axis=1, inplace=True)
    new_order = ['Resource','Task_Completed', 'FIFO', 'Good_FIFO','Bad_FIFO','LIFO','Good_LIFO','Bad_LIFO','Random','Good_Random','Bad_Random']
    df = df.reindex(columns=new_order)
    working_behaviour = df[['Resource','Task_Completed', 'FIFO','LIFO','Random']]
    working_behaviour_evaluated = df[['Resource','Task_Completed', 'FIFO', 'Good_FIFO','Bad_FIFO','LIFO','Good_LIFO','Bad_LIFO','Random','Good_Random','Bad_Random']]
    
    working_behaviour_top20= working_behaviour.nlargest(20, 'Task_Completed')
    working_behaviour_top20 = working_behaviour_top20.reset_index(drop=True)
    
    working_behaviour_evaluated_top20 = working_behaviour_evaluated.nlargest(20, 'Task_Completed')
    working_behaviour_evaluated_top20 = working_behaviour_evaluated_top20.reset_index(drop=True)
    
    return working_behaviour,working_behaviour_top20,working_behaviour_evaluated,working_behaviour_evaluated_top20

def Get_Weighted_Success_Rate(working_behaviour_evaluated):
    import math
    df = working_behaviour_evaluated.copy()
    df['Weighted_Success_Rate_FIFO']= 0.0
    df['Weighted_Success_Rate_LIFO']= 0.0
    df['Weighted_Success_Rate_Random']= 0.0
    for i in range(len(df)):
        df['Weighted_Success_Rate_FIFO'][i] = ((df['Good_FIFO'][i]) / df['FIFO'][i])
        df['Weighted_Success_Rate_LIFO'][i] = ((df['Good_LIFO'][i] ) / df['LIFO'][i])
        df['Weighted_Success_Rate_Random'][i] = ((df['Good_Random'][i]) / df['Random'][i])
            
    df = df[['Resource','Task_Completed','Weighted_Success_Rate_FIFO','Weighted_Success_Rate_LIFO','Weighted_Success_Rate_Random']]
    df.columns = ['Resource','Task_Completed', 'Success_Rate_FIFO','Success_Rate_LIFO','Success_Rate_Random']
    
    weighted_success_rate_top20= df.nlargest(20, 'Task_Completed')
    weighted_success_rate_top20 = weighted_success_rate_top20.reset_index(drop=True)
    
    return df,weighted_success_rate_top20


def Activity_based_batch(dataframe, resources, dataframe_good):
    df1 = resources.copy()
    resource = df1['Resource'].unique()
    df1['Total_Executions'] = 0
    df1['Total_Batch_Execution'] = 0
    df1['Good_Batching'] = 0
    df1['Bad_Batching'] = 0
    df1['Success_Rate'] = 0.0

    for j in range(len(resource)):
        df = dataframe[dataframe['Resource'] == resource[j]]
        df.sort_values(['Resource', 'Complete Timestamp'], inplace=True)
        df = df.reset_index(drop=True)

        df['Batch'] = 'Single'
        for i in range(len(df)-1):
            if (df.loc[i, 'Activity'] == df.loc[i+1, 'Activity']) and (df.loc[i, 'Case ID'] != df.loc[i+1, 'Case ID']):
                df.loc[i, 'Batch'] = 'Batch'
                df.loc[i+1, 'Batch'] = 'Batch'
        batch_count = df['Batch'].value_counts()
        column_name = 'Batch'
        column_name1 = 'Single'

        batch_count = batch_count.reindex([column_name, column_name1], fill_value=0)

        df1['Total_Executions'][j] = batch_count['Single'] + batch_count['Batch']
        df1['Total_Batch_Execution'][j] = batch_count['Batch']

        sublog = df.loc[df['Batch'] != 'Single']
    
        sublog['Case ID'] = sublog['Case ID'].isin(dataframe_good['Case ID'])
        sublog = sublog[['Case ID']]
        sublog = sublog.reset_index(drop=True)


        df1['Good_Batching'][j] = (sublog['Case ID'] == True).sum()
        df1['Bad_Batching'][j] = (sublog['Case ID'] == False).sum()
        #df1['Total_Batch_Execution'][j] = df1['Good_Batching'][j] + df1['Bad_Batching'][j]
        df1['Success_Rate'][j] = df1['Good_Batching'][j] / (df1['Total_Batch_Execution'][j])
    
    df1['Success_Rate'] = np.nan_to_num(df1['Success_Rate'], nan=0.0)
    Batching_AR_AB = df1[['Resource', 'Total_Executions', 'Total_Batch_Execution']]
    Batching_Evaluated_AR_AB = df1[['Resource', 'Total_Executions', 'Total_Batch_Execution','Good_Batching','Bad_Batching']]
    Batching_SR_AR_AB = df1
    
    Batching_top20_AB = Batching_AR_AB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_top20_AB = Batching_top20_AB.reset_index(drop=True)


    Batching_Evaluated_top20_AB = Batching_Evaluated_AR_AB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_Evaluated_top20_AB = Batching_Evaluated_top20_AB.reset_index(drop=True)


    Batching_SR_top20_AB = Batching_SR_AR_AB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_SR_top20_AB = Batching_SR_top20_AB.reset_index(drop=True)


    return Batching_AR_AB,Batching_top20_AB,Batching_Evaluated_AR_AB,Batching_Evaluated_top20_AB,Batching_SR_AR_AB,Batching_SR_top20_AB

import pandas as pd
def Time_based_batch(dataframe, resources, dataframe_good,threshold=pd.Timedelta(minutes=5)):
    df2 = resources.copy()
    resource = df2['Resource'].unique()
    df2['Total_Executions'] = 0
    df2['Total_Batch_Execution'] = 0
    df2['Good_Batching'] = 0
    df2['Bad_Batching'] = 0
    df2['Success_Rate'] = 0.0

    for j in range(len(resource)):
        df = dataframe[dataframe['Resource'] == resource[j]]
        df.sort_values(['Complete Timestamp'], inplace=True)
        df = df.reset_index(drop=True)
        df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
        df['Batch'] = 'Single'

        for i in range(len(df) - 1):
            time_diff = df.loc[i + 1, 'Complete Timestamp'] - df.loc[i, 'Complete Timestamp']
            if time_diff <= threshold:
                df.loc[i, 'Batch'] = 'Batch'
                df.loc[i+1, 'Batch'] = 'Batch'
        batch_count = df['Batch'].value_counts()
        column_name = 'Batch'
        column_name1 = 'Single'

        batch_count = batch_count.reindex([column_name, column_name1], fill_value=0)

        df2['Total_Executions'][j] = batch_count['Single'] + batch_count['Batch']
        df2['Total_Batch_Execution'][j] = batch_count['Batch']

        sublog = df.loc[df['Batch'] != 'Single']

        sublog['Case ID'] = sublog['Case ID'].isin(dataframe_good['Case ID'])
        sublog = sublog[['Case ID']]
        sublog = sublog.reset_index(drop=True)


        df2['Good_Batching'][j] = (sublog['Case ID'] == True).sum()
        df2['Bad_Batching'][j] = (sublog['Case ID'] == False).sum()
        #df2['Total_Batch_Execution'][j] = df2['Good_Batching'][j] + df2['Bad_Batching'][j]
        df2['Success_Rate'][j] = df2['Good_Batching'][j] / (df2['Total_Batch_Execution'][j])

    df2['Success_Rate'] = np.nan_to_num(df2['Success_Rate'], nan=0.0)   
    Batching_AR_TB = df2[['Resource', 'Total_Executions', 'Total_Batch_Execution']]
    Batching_Evaluated_AR_TB = df2[['Resource', 'Total_Executions', 'Total_Batch_Execution','Good_Batching','Bad_Batching']]
    Batching_SR_AR_TB = df2
    
    Batching_top20_TB = Batching_AR_TB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_top20_TB = Batching_top20_TB.reset_index(drop=True)


    Batching_Evaluated_top20_TB = Batching_Evaluated_AR_TB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_Evaluated_top20_TB = Batching_Evaluated_top20_TB.reset_index(drop=True)


    Batching_SR_top20_TB = Batching_SR_AR_TB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_SR_top20_TB = Batching_SR_top20_TB.reset_index(drop=True)
    
    return Batching_AR_TB,Batching_top20_TB,Batching_Evaluated_AR_TB,Batching_Evaluated_top20_TB,Batching_SR_AR_TB,Batching_SR_top20_TB


import pandas as pd
def Size_based_batch(dataframe, resources, dataframe_good, batch_size=10, threshold=pd.Timedelta(minutes=5)):
    df3 = resources.copy()
    resource = df3['Resource'].unique()
    df3['Total_Executions'] = 0
    df3['Total_Batch_Execution'] = 0
    df3['Good_Batching'] = 0
    df3['Bad_Batching'] = 0
    df3['Success_Rate'] = 0.0

    for j in range(len(resource)):
        df = dataframe[dataframe['Resource'] == resource[j]]
        df.sort_values(['Complete Timestamp'], inplace=True)
        df = df.reset_index(drop=True)
        df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
        df['Batch'] = 'Single'

        counter = 0

        for i in range(len(df) - 1):
            time_diff = df.loc[i + 1, 'Complete Timestamp'] - df.loc[i, 'Complete Timestamp']
            if time_diff <= threshold:
                counter += 1
            else:
                counter = 0

            if counter >= batch_size:
                df.loc[i - counter + 1:i+1, 'Batch'] = 'Batch'
                counter = 0

        batch_count = df['Batch'].value_counts()
        column_name = 'Batch'
        column_name1 = 'Single'

        batch_count = batch_count.reindex([column_name, column_name1], fill_value=0)

        df3['Total_Executions'][j] = batch_count['Single'] + batch_count['Batch']
        df3['Total_Batch_Execution'][j] = batch_count['Batch']

        sublog = df.loc[df['Batch'] != 'Single']

        sublog['Case ID'] = sublog['Case ID'].isin(dataframe_good['Case ID'])
        sublog = sublog[['Case ID']]
        sublog = sublog.reset_index(drop=True)


        df3['Good_Batching'][j] = (sublog['Case ID'] == True).sum()
        df3['Bad_Batching'][j] = (sublog['Case ID'] == False).sum()
        #df3['Total_Batch_Execution'][j] = df3['Good_Batching'][j] + df3['Bad_Batching'][j]
        df3['Success_Rate'][j] = df3['Good_Batching'][j] / df3['Total_Batch_Execution'][j]
        
    df3['Success_Rate'] = np.nan_to_num(df3['Success_Rate'], nan=0.0)    
    Batching_AR_SB = df3[['Resource', 'Total_Executions', 'Total_Batch_Execution']]
    Batching_Evaluated_AR_SB = df3[['Resource', 'Total_Executions', 'Total_Batch_Execution','Good_Batching','Bad_Batching']]
    Batching_SR_AR_SB = df3
    
    
    Batching_top20_SB = Batching_AR_SB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_top20_SB = Batching_top20_SB.reset_index(drop=True)


    Batching_Evaluated_top20_SB = Batching_Evaluated_AR_SB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_Evaluated_top20_SB = Batching_Evaluated_top20_SB.reset_index(drop=True)


    Batching_SR_top20_SB = Batching_SR_AR_SB.sort_values(by='Total_Batch_Execution', ascending=False).head(20)
    Batching_SR_top20_SB = Batching_SR_top20_SB.reset_index(drop=True)

    return Batching_AR_SB,Batching_top20_SB,Batching_Evaluated_AR_SB,Batching_Evaluated_top20_SB,Batching_SR_AR_SB,Batching_SR_top20_SB


def Evaluate_Batching(Batching_AR_AB,Batching_AR_TB,Batching_AR_SB):
    Batching_AR_AB_R = Batching_AR_AB.copy()
    Batching_AR_TB_R = Batching_AR_TB.copy()
    Batching_AR_SB_R = Batching_AR_SB.copy()
    
    Batching_AR_AB_R['Rank_AB'] = Batching_AR_AB_R['Total_Batch_Execution'].rank(ascending=False, method='min')
    Batching_AR_TB_R['Rank_TB'] = Batching_AR_TB_R['Total_Batch_Execution'].rank(ascending=False, method='min')
    Batching_AR_SB_R['Rank_SB'] = Batching_AR_SB_R['Total_Batch_Execution'].rank(ascending=False, method='min')

    # Sort the dataframe based on the 'Total_Batch_Execution' column in descending order
    Batching_AR_AB_R = Batching_AR_AB_R.sort_values(by='Rank_AB', ascending=True)
    Batching_AR_TB_R = Batching_AR_TB_R.sort_values(by='Rank_TB', ascending=True)
    Batching_AR_SB_R = Batching_AR_SB_R.sort_values(by='Rank_SB', ascending=True)

    Batching_AR_AB_R = Batching_AR_AB_R.reset_index(drop=True)
    Batching_AR_TB_R = Batching_AR_TB_R.reset_index(drop=True)
    Batching_AR_SB_R = Batching_AR_SB_R.reset_index(drop=True)
    
    return Batching_AR_AB_R,Batching_AR_TB_R,Batching_AR_SB_R
    

def Evaluate_Batching_SuccessRate(Batching_SR_AR_AB,Batching_SR_AR_TB,Batching_SR_AR_SB):
    Batching_SR_AR_AB_R = Batching_SR_AR_AB.copy()
    Batching_SR_AR_TB_R = Batching_SR_AR_TB.copy()
    Batching_SR_AR_SB_R = Batching_SR_AR_SB.copy() 
    
    Batching_SR_AR_AB_R['Rank_AB'] = Batching_SR_AR_AB_R['Success_Rate'].rank(ascending=False, method='min')
    Batching_SR_AR_TB_R['Rank_TB'] = Batching_SR_AR_TB_R['Success_Rate'].rank(ascending=False, method='min')
    Batching_SR_AR_SB_R['Rank_SB'] = Batching_SR_AR_SB_R['Success_Rate'].rank(ascending=False, method='min')

    # Sort the dataframe based on the 'Total_Batch_Execution' column in descending order
    Batching_SR_AR_AB_R = Batching_SR_AR_AB_R.sort_values(by='Rank_AB', ascending=True)
    Batching_SR_AR_TB_R = Batching_SR_AR_TB_R.sort_values(by='Rank_TB', ascending=True)
    Batching_SR_AR_SB_R = Batching_SR_AR_SB_R.sort_values(by='Rank_SB', ascending=True)

    Batching_SR_AR_AB_R = Batching_SR_AR_AB_R.reset_index(drop=True)
    Batching_SR_AR_TB_R = Batching_SR_AR_TB_R.reset_index(drop=True)
    Batching_SR_AR_SB_R = Batching_SR_AR_SB_R.reset_index(drop=True)
    
    return Batching_SR_AR_AB_R,Batching_SR_AR_TB_R,Batching_SR_AR_SB_R

def Evaluate_Batching_Ranking(Batching_AR_AB_R,Batching_AR_TB_R,Batching_AR_SB_R):
    merged_df = pd.merge(Batching_AR_TB_R[['Resource', 'Total_Batch_Execution', 'Rank_TB']],
                     Batching_AR_SB_R[['Resource', 'Total_Batch_Execution', 'Rank_SB']],
                     on='Resource',
                     how='outer')

    merged_df = pd.merge(merged_df,
                             Batching_AR_AB_R[['Resource', 'Total_Batch_Execution', 'Rank_AB']],
                             on='Resource',
                             how='outer')

    merged_df = merged_df.rename(columns={'Total_Batch_Execution_x': 'Total_Batch_Execution_TB',
                                              'Total_Batch_Execution_y': 'Total_Batch_Execution_SB',
                                              'Total_Batch_Execution': 'Total_Batch_Execution_AB'})

    merged_df['Geometric Mean'] = np.power(merged_df['Rank_TB'] * merged_df['Rank_SB'] * merged_df['Rank_AB'], 1/3)

    column_order = ['Resource', 'Total_Batch_Execution_AB', 'Total_Batch_Execution_TB', 'Total_Batch_Execution_SB',
                        'Rank_AB', 'Rank_TB', 'Rank_SB', 'Geometric Mean']

    merged_df = merged_df[column_order]

        # Sort the dataframe based on the desired order (e.g., rank)
    merged_df = merged_df.sort_values(by='Geometric Mean', ascending=True)

        # Add a column for the rank based on geometric mean
    merged_df['Aggregated Rank'] = range(1, len(merged_df) + 1)

        # Reset the index if needed
    merged_df = merged_df.reset_index(drop=True)


    return merged_df

def Evaluate_Batching_SuccessRate_Ranking(Batching_SR_AR_AB_R,Batching_SR_AR_TB_R,Batching_SR_AR_SB_R):
    merged_df = pd.merge(Batching_SR_AR_TB_R[['Resource', 'Success_Rate', 'Rank_TB']],
                     Batching_SR_AR_SB_R[['Resource', 'Success_Rate', 'Rank_SB']],
                     on='Resource',
                     how='outer')

    merged_df = pd.merge(merged_df,
                         Batching_SR_AR_AB_R[['Resource', 'Success_Rate', 'Rank_AB']],
                         on='Resource',
                         how='outer')

    merged_df = merged_df.rename(columns={'Success_Rate_x': 'Success_Rate_TB',
                                          'Success_Rate_y': 'Success_Rate_SB',
                                          'Success_Rate': 'Success_Rate_AB'})

    merged_df['Geometric Mean'] = np.power(merged_df['Rank_TB'] * merged_df['Rank_SB'] * merged_df['Rank_AB'], 1/3)

    column_order = ['Resource', 'Success_Rate_AB', 'Success_Rate_TB', 'Success_Rate_SB',
                    'Rank_AB', 'Rank_TB', 'Rank_SB', 'Geometric Mean']

    merged_df = merged_df[column_order]

    # Sort the dataframe based on the desired order (e.g., rank)
    merged_df = merged_df.sort_values(by='Geometric Mean', ascending=True)

    # Add a column for the rank based on geometric mean
    merged_df['Aggregated Rank SR'] = range(1, len(merged_df) + 1)

    # Reset the index if needed
    merged_df = merged_df.reset_index(drop=True)


    return merged_df

def Workload_Resource(dataframe,resources,dataframe_good):
    resource = resources['Resource'].unique()

    min_timestamp = pd.to_datetime(dataframe['Complete Timestamp'].min())
    min_timestamp = min_timestamp.to_period('M').to_timestamp()
    min_timestamp = min_timestamp.tz_localize('UTC')

    max_timestamp = pd.to_datetime(dataframe['Complete Timestamp'].max())
    max_timestamp = max_timestamp + pd.offsets.MonthEnd(0)

    months = pd.date_range(start=min_timestamp, end=max_timestamp, freq='MS')

    df = pd.DataFrame(index=resource,columns=months.strftime('%B %Y'))
    workload_AR = pd.DataFrame(index=resource,columns=months.strftime('%B %Y'))


    for j in range(len(resource)):
        df1 = dataframe[dataframe['Resource'] == resource[j]]
        df1.sort_values(['Complete Timestamp'], inplace=True)
        df1 = df1.reset_index(drop=True)
        for i in range(len(months)-1):
            start_date = months[i]
            end_date = months[i+1]
            time_period_df = df1[(df1['Complete Timestamp'] >= start_date) & (df1['Complete Timestamp'] < end_date)]
            unique_case_ids = time_period_df['Case ID'].unique()
            workload = len(unique_case_ids)
            case_ids_series = pd.Series(unique_case_ids)
            count = case_ids_series.isin(dataframe_good['Case ID']).sum()

            df.loc[resource[j],start_date.strftime('%B %Y')] = workload
            workload_AR.loc[resource[j],start_date.strftime('%B %Y')] = workload
            
        
            df.loc[resource[j],start_date.strftime('%B %Y') + 'success'] = count

    month_years = [col[:-7] for col in df.columns if 'success' in col]
    success_cols = [col for col in df.columns if 'success' in col]

        # Create new column names by pairing each month with its corresponding success column
    new_columns = []
    for month_year in month_years:
        success_col = f"{month_year}success"
        month_col = month_year
        new_columns.extend([month_col, success_col])

        # Rearrange the columns in the DataFrame
    df = df[new_columns]
    workload_evaluated_AR = df.copy()


        # Get the list of month columns and success columns
    month_columns = [col for col in df.columns if 'success' not in col]
    success_columns = [col for col in df.columns if 'success' in col]

        # Create new column names for success ratio
    ratio_columns = [col + " success ratio" for col in month_columns]

        # Calculate the success ratio for each month and store the values in the ratio columns
    for month_col, success_col, ratio_col in zip(month_columns, success_columns, ratio_columns):
        df[ratio_col] = df[success_col] / df[month_col].replace(0, float('nan'))

        # Rearrange the columns by grouping month, success, and ratio columns together
    new_columns = []
    for month_col, success_col, ratio_col in zip(month_columns, success_columns, ratio_columns):
        new_columns.extend([month_col, success_col, ratio_col])


    df = df[new_columns]
    
    workload_sr_AR = df.copy()
    
    month_year_cols = df.filter(regex=r'(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}$')

    # Sum the values for each row under the month and year columns
    df['Total'] = month_year_cols.apply(lambda row: row.sum() if row.notnull().all() else np.nan, axis=1)

    top_20_resources = df.nlargest(20, 'Total')
    top_20_resources = top_20_resources.drop('Total', axis=1)
    top_20_resources = top_20_resources.reset_index().rename(columns={'index': 'Resource'})
    
    workload_ranges = [
    (1, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
    (71, 80),
    (81, 90),
    (91, 100),
    (101, float('inf'))
    ]
    data = []


    for resource in top_20_resources['Resource']:
        resource_data = {'Resource': resource}
        matched_row = top_20_resources.loc[top_20_resources['Resource'] == resource]
        month_year_cols = matched_row.filter(regex=r'(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}$')
        success_ratio_cols = matched_row.filter(regex=r'\w+\s\d{4}\s(success\s)?ratio$')

        workload_values = month_year_cols.values[0]
        success_ratio_values = success_ratio_cols.values[0]

        range_success_ratio = {}

    # Iterate over the workload ranges
        for i, (start, end) in enumerate(workload_ranges):
        # Filter the workload values within the current range
            filtered_values = [value for value in workload_values if start <= value < end]

        # Filter the corresponding success ratio values
            filtered_success_ratio = [ratio for j, ratio in enumerate(success_ratio_values) if start <= workload_values[j] < end]

        # Calculate the mean success ratio for multiple workloads in the same range
            if len(filtered_success_ratio) > 0:
                mean_success_ratio = sum(filtered_success_ratio) / len(filtered_success_ratio)
            else:
                mean_success_ratio = float('nan')

        # Store the mean success ratio in the dictionary with the range label as the key
            range_label = f'{start}-{end}'
            range_success_ratio[range_label] = mean_success_ratio
            resource_data[range_label] = mean_success_ratio
        data.append(resource_data)
        SR_Workload = pd.DataFrame(data)

# Set 'Resource' as the index
        SR_Workload.set_index('Resource', inplace=True)
    
    
    workload_top20 = workload_AR.copy()
    workload_top20['TotalTasks'] = workload_top20.sum(axis=1)
    workload_top20 = workload_top20.sort_values('TotalTasks', ascending=False)
    workload_top20 = workload_top20.head(20)
    workload_top20 = workload_top20.drop('TotalTasks', axis=1)
    
    workload_evaluated_top20 = workload_evaluated_AR.copy()
    workload_evaluated_top20['TotalTasks'] = workload_evaluated_top20[[col for col in workload_evaluated_top20.columns if 'success' not in col]].sum(axis=1)
    workload_evaluated_top20 = workload_evaluated_top20.sort_values('TotalTasks', ascending=False)
    workload_evaluated_top20 = workload_evaluated_top20.head(20)
    workload_evaluated_top20 = workload_evaluated_top20.drop('TotalTasks', axis=1)

    workload_AR = workload_AR.reset_index().rename(columns={'index': 'Resource'})
    workload_top20 = workload_top20.reset_index().rename(columns={'index': 'Resource'})
    workload_evaluated_AR = workload_evaluated_AR.reset_index().rename(columns={'index': 'Resource'})
    workload_evaluated_top20 = workload_evaluated_top20.reset_index().rename(columns={'index': 'Resource'})
    workload_sr_AR = workload_sr_AR.reset_index().rename(columns={'index': 'Resource'})
    SR_Workload = SR_Workload.reset_index().rename(columns={'index': 'Resource'})
    
    
    return workload_AR,workload_top20,workload_evaluated_AR,workload_evaluated_top20,workload_sr_AR,top_20_resources,SR_Workload


def Workload_Resource_Year(dataframe, resources, dataframe_good):
    resource = resources['Resource'].unique()

    min_timestamp = pd.to_datetime(dataframe['Complete Timestamp'].min())
    min_timestamp = min_timestamp.to_period('Y').to_timestamp()
    min_timestamp = min_timestamp.tz_localize('UTC')

    max_timestamp = pd.to_datetime(dataframe['Complete Timestamp'].max())
    max_timestamp = max_timestamp + pd.offsets.YearEnd(0)

    years = pd.date_range(start=min_timestamp, end=max_timestamp, freq='YS')

    df = pd.DataFrame(index=resource, columns=years.strftime('%Y'))
    workload_AR = pd.DataFrame(index=resource, columns=years.strftime('%Y'))

    for j in range(len(resource)):
        df1 = dataframe[dataframe['Resource'] == resource[j]]
        df1.sort_values(['Complete Timestamp'], inplace=True)
        df1 = df1.reset_index(drop=True)
        for i in range(len(years)-1):
            start_date = years[i]
            end_date = years[i+1]
            time_period_df = df1[(df1['Complete Timestamp'] >= start_date) & (df1['Complete Timestamp'] < end_date)]
            unique_case_ids = time_period_df['Case ID'].unique()
            workload = len(unique_case_ids)
            case_ids_series = pd.Series(unique_case_ids)
            count = case_ids_series.isin(dataframe_good['Case ID']).sum()

            df.loc[resource[j], start_date.strftime('%Y')] = workload
            workload_AR.loc[resource[j], start_date.strftime('%Y')] = workload

            df.loc[resource[j], start_date.strftime('%Y') + ' success'] = count

    year_cols = [col[:-8] for col in df.columns if 'success' in col]
    success_cols = [col for col in df.columns if 'success' in col]

    # Create new column names by pairing each year with its corresponding success column
    new_columns = []
    for year_col in year_cols:
        success_col = f"{year_col} success"
        new_columns.extend([year_col, success_col])

    # Rearrange the columns in the DataFrame
    df = df[new_columns]
    workload_evaluated_AR = df.copy()

    # Get the list of year columns and success columns
    year_columns = [col for col in df.columns if 'success' not in col]
    success_columns = [col for col in df.columns if 'success' in col]

    # Create new column names for success ratio
    ratio_columns = [col + " success ratio" for col in year_columns]

    # Calculate the success ratio for each year and store the values in the ratio columns
    for year_col, success_col, ratio_col in zip(year_columns, success_columns, ratio_columns):
        df[ratio_col] = df[success_col] / df[year_col].replace(0, float('nan'))

    # Rearrange the columns by grouping year, success, and ratio columns together
    new_columns = []
    for year_col, success_col, ratio_col in zip(year_columns, success_columns, ratio_columns):
        new_columns.extend([year_col, success_col, ratio_col])

    df = df[new_columns]
    workload_sr_AR = df.copy()

    year_cols = df.filter(regex=r'^\d{4}$')

    # Sum the values for each row under the year columns
    df['Total'] = year_cols.apply(lambda row: row.sum() if row.notnull().all() else np.nan, axis=1)

    top_20_resources = df.nlargest(20, 'Total')
    top_20_resources = top_20_resources.drop('Total', axis=1)
    top_20_resources = top_20_resources.reset_index().rename(columns={'index': 'Resource'})

    workload_ranges = [
    (1, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
    (71, 80),
    (81, 90),
    (91, 100),
    (101, float('inf'))
    ]

    data = []

    for resource in top_20_resources['Resource']:
        resource_data = {'Resource': resource}
        matched_row = top_20_resources.loc[top_20_resources['Resource'] == resource]
        year_cols = matched_row.filter(regex=r'^\d{4}$')
        success_ratio_cols = matched_row.filter(regex=r'\d{4}\s(success\s)?ratio$')

        workload_values = year_cols.values[0]
        success_ratio_values = success_ratio_cols.values[0]

        range_success_ratio = {}

        # Iterate over the workload ranges
        for i, (start, end) in enumerate(workload_ranges):
            # Filter the workload values within the current range
            filtered_values = [value for value in workload_values if start <= value < end]

            # Filter the corresponding success ratio values
            filtered_success_ratio = [ratio for j, ratio in enumerate(success_ratio_values) if
                                      start <= workload_values[j] < end]

            # Calculate the mean success ratio for multiple workloads in the same range
            if len(filtered_success_ratio) > 0:
                mean_success_ratio = sum(filtered_success_ratio) / len(filtered_success_ratio)
            else:
                mean_success_ratio = float('nan')

            # Store the mean success ratio in the dictionary with the range label as the key
            range_label = f'{start}-{end}'
            range_success_ratio[range_label] = mean_success_ratio
            resource_data[range_label] = mean_success_ratio
        data.append(resource_data)
    SR_Workload = pd.DataFrame(data)

    # Set 'Resource' as the index
    SR_Workload.set_index('Resource', inplace=True)

    workload_top20 = workload_AR.copy()
    workload_top20['TotalTasks'] = workload_top20.sum(axis=1)
    workload_top20 = workload_top20.sort_values('TotalTasks', ascending=False)
    workload_top20 = workload_top20.head(20)
    workload_top20 = workload_top20.drop('TotalTasks', axis=1)
    
    workload_evaluated_top20 = workload_evaluated_AR.copy()
    workload_evaluated_top20['TotalTasks'] = workload_evaluated_top20[[col for col in workload_evaluated_top20.columns if 'success' not in col]].sum(axis=1)
    workload_evaluated_top20 = workload_evaluated_top20.sort_values('TotalTasks', ascending=False)
    workload_evaluated_top20 = workload_evaluated_top20.head(20)
    workload_evaluated_top20 = workload_evaluated_top20.drop('TotalTasks', axis=1)
    
    workload_AR = workload_AR.reset_index().rename(columns={'index': 'Resource'})
    workload_top20 = workload_top20.reset_index().rename(columns={'index': 'Resource'})
    workload_evaluated_AR = workload_evaluated_AR.reset_index().rename(columns={'index': 'Resource'})
    workload_evaluated_top20 = workload_evaluated_top20.reset_index().rename(columns={'index': 'Resource'})
    workload_sr_AR = workload_sr_AR.reset_index().rename(columns={'index': 'Resource'})
    SR_Workload = SR_Workload.reset_index().rename(columns={'index': 'Resource'})

    return workload_AR,workload_top20,workload_evaluated_AR,workload_evaluated_top20,workload_sr_AR,top_20_resources,SR_Workload


def Workload_Resource_Quarter(dataframe, resources, dataframe_good):
    resource = resources['Resource'].unique()

    min_timestamp = pd.to_datetime(dataframe['Complete Timestamp'].min())
    min_timestamp = min_timestamp.to_period('Q').to_timestamp()
    min_timestamp = min_timestamp.tz_localize('UTC')

    max_timestamp = pd.to_datetime(dataframe['Complete Timestamp'].max())
    max_timestamp = max_timestamp + pd.offsets.QuarterEnd(0)

    quarters = pd.date_range(start=min_timestamp, end=max_timestamp, freq='QS')

    df = pd.DataFrame(index=resource, columns=quarters.strftime('%Y Q%m'))
    workload_AR = pd.DataFrame(index=resource, columns=quarters.strftime('%Y Q%m'))

    for j in range(len(resource)):
        df1 = dataframe[dataframe['Resource'] == resource[j]]
        df1.sort_values(['Complete Timestamp'], inplace=True)
        df1 = df1.reset_index(drop=True)
        for i in range(len(quarters)-1):
            start_date = quarters[i]
            end_date = quarters[i+1]
            time_period_df = df1[(df1['Complete Timestamp'] >= start_date) & (df1['Complete Timestamp'] < end_date)]
            unique_case_ids = time_period_df['Case ID'].unique()
            workload = len(unique_case_ids)
            case_ids_series = pd.Series(unique_case_ids)
            count = case_ids_series.isin(dataframe_good['Case ID']).sum()

            df.loc[resource[j], start_date.strftime('%Y Q%m')] = workload
            workload_AR.loc[resource[j], start_date.strftime('%Y Q%m')] = workload

            df.loc[resource[j], start_date.strftime('%Y Q%m') + ' success'] = count

    year_cols = [col[:-8] for col in df.columns if 'success' in col]
    success_cols = [col for col in df.columns if 'success' in col]

    # Create new column names by pairing each year with its corresponding success column
    new_columns = []
    for year_col in year_cols:
        success_col = f"{year_col} success"
        new_columns.extend([year_col, success_col])

    # Rearrange the columns in the DataFrame
    df = df[new_columns]
    workload_evaluated_AR = df.copy()

    # Get the list of year columns and success columns
    year_columns = [col for col in df.columns if 'success' not in col]
    success_columns = [col for col in df.columns if 'success' in col]

    # Create new column names for success ratio
    ratio_columns = [col + " success ratio" for col in year_columns]

    # Calculate the success ratio for each year and store the values in the ratio columns
    for year_col, success_col, ratio_col in zip(year_columns, success_columns, ratio_columns):
        df[ratio_col] = df[success_col] / df[year_col].replace(0, float('nan'))

    # Rearrange the columns by grouping year, success, and ratio columns together
    new_columns = []
    for year_col, success_col, ratio_col in zip(year_columns, success_columns, ratio_columns):
        new_columns.extend([year_col, success_col, ratio_col])

    df = df[new_columns]
    workload_sr_AR = df.copy()

    year_cols = df.filter(regex=r'^\d{4}$')

    # Sum the values for each row under the year columns
    df['Total'] = year_cols.apply(lambda row: row.sum() if row.notnull().all() else np.nan, axis=1)

    top_20_resources = df.nlargest(20, 'Total')
    top_20_resources = top_20_resources.drop('Total', axis=1)
    top_20_resources = top_20_resources.reset_index().rename(columns={'index': 'Resource'})

    workload_ranges = [
    (1, 10),
    (11, 20),
    (21, 30),
    (31, 40),
    (41, 50),
    (51, 60),
    (61, 70),
    (71, 80),
    (81, 90),
    (91, 100),
    (101, float('inf'))
    ]

    data = []

    for resource in top_20_resources['Resource']:
        resource_data = {'Resource': resource}
        matched_row = top_20_resources.loc[top_20_resources['Resource'] == resource]
        year_cols = matched_row.filter(regex=r'^\d{4}\sQ\d{2}$')
        success_ratio_cols = matched_row.filter(regex=r'\d{4}\sQ\d{2}\ssuccess\sratio$')

        workload_values = year_cols.values[0]
        success_ratio_values = success_ratio_cols.values[0]

        range_success_ratio = {}

        # Iterate over the workload ranges
        for i, (start, end) in enumerate(workload_ranges):
            # Filter the workload values within the current range
            filtered_values = [value for value in workload_values if start <= value < end]

            # Filter the corresponding success ratio values
            filtered_success_ratio = [ratio for j, ratio in enumerate(success_ratio_values) if
                                      start <= workload_values[j] < end]

            # Calculate the mean success ratio for multiple workloads in the same range
            if len(filtered_success_ratio) > 0:
                mean_success_ratio = sum(filtered_success_ratio) / len(filtered_success_ratio)
            else:
                mean_success_ratio = float('nan')

            # Store the mean success ratio in the dictionary with the range label as the key
            range_label = f'{start}-{end}'
            range_success_ratio[range_label] = mean_success_ratio
            resource_data[range_label] = mean_success_ratio
        data.append(resource_data)
    SR_Workload = pd.DataFrame(data)

    # Set 'Resource' as the index
    SR_Workload.set_index('Resource', inplace=True)

    workload_top20 = workload_AR.copy()
    workload_top20['TotalTasks'] = workload_top20.sum(axis=1)
    workload_top20 = workload_top20.sort_values('TotalTasks', ascending=False)
    workload_top20 = workload_top20.head(20)
    workload_top20 = workload_top20.drop('TotalTasks', axis=1)
    
    workload_evaluated_top20 = workload_evaluated_AR.copy()
    workload_evaluated_top20['TotalTasks'] = workload_evaluated_top20[[col for col in workload_evaluated_top20.columns if 'success' not in col]].sum(axis=1)
    workload_evaluated_top20 = workload_evaluated_top20.sort_values('TotalTasks', ascending=False)
    workload_evaluated_top20 = workload_evaluated_top20.head(20)
    workload_evaluated_top20 = workload_evaluated_top20.drop('TotalTasks', axis=1)
    
    workload_AR = workload_AR.reset_index().rename(columns={'index': 'Resource'})
    workload_top20 = workload_top20.reset_index().rename(columns={'index': 'Resource'})
    workload_evaluated_AR = workload_evaluated_AR.reset_index().rename(columns={'index': 'Resource'})
    workload_evaluated_top20 = workload_evaluated_top20.reset_index().rename(columns={'index': 'Resource'})
    workload_sr_AR = workload_sr_AR.reset_index().rename(columns={'index': 'Resource'})
    SR_Workload = SR_Workload.reset_index().rename(columns={'index': 'Resource'})

    return workload_AR, workload_top20, workload_evaluated_AR, workload_evaluated_top20, workload_sr_AR, top_20_resources, SR_Workload


def Evaluate_Workload(workload_AR,workload_AR_Q,workload_AR_Y):
    import math
    workload_AR_R = workload_AR.copy()
    workload_AR_Q_R = workload_AR_Q.copy()
    workload_AR_Y_R =  workload_AR_Y.copy()
    
    quarter_columns_Q = workload_AR_Q_R.columns[1:]
    quarter_columns_Y = workload_AR_Y_R.columns[1:]
    quarter_columns_M = workload_AR_R.columns[1:]
    
    
    workload_AR_Q_R['Average Workload'] = workload_AR_Q_R[quarter_columns_Q].mean(axis=1).apply(math.ceil)
    workload_AR_Y_R['Average Workload'] = workload_AR_Y_R[quarter_columns_Y].mean(axis=1).apply(math.ceil)
    workload_AR_R['Average Workload'] = workload_AR_R[quarter_columns_M].mean(axis=1).apply(math.ceil)

    workload_AR_Q_R = workload_AR_Q_R.sort_values(by='Average Workload', ascending=False)
    workload_AR_Y_R = workload_AR_Y_R.sort_values(by='Average Workload', ascending=False)
    workload_AR_R = workload_AR_R.sort_values(by='Average Workload', ascending=False)

    # Add a column for the rank of the resource
    workload_AR_Q_R['Rank_per_Quarter'] = range(1, len(workload_AR_Q_R) + 1)
    workload_AR_Y_R['Rank_per_Year'] = range(1, len(workload_AR_Y_R) + 1)
    workload_AR_R['Rank_per_Month'] = range(1, len(workload_AR_R) + 1)

    # Sort the DataFrame in ascending order based on the rank
    workload_AR_Q_R = workload_AR_Q_R.sort_values(by='Rank_per_Quarter', ascending=True)
    workload_AR_Y_R = workload_AR_Y_R.sort_values(by='Rank_per_Year', ascending=True)
    workload_AR_R = workload_AR_R.sort_values(by='Rank_per_Month', ascending=True)
    
    workload_AR_Q_R = workload_AR_Q_R.reset_index(drop=True)
    workload_AR_Y_R = workload_AR_Y_R.reset_index(drop=True)
    workload_AR_R = workload_AR_R.reset_index(drop=True)
    
    return workload_AR_R,workload_AR_Q_R,workload_AR_Y_R

    

def Evaluate_Workload_SuccessRate(workload_sr_AR,workload_sr_AR_Q,workload_sr_AR_Y):
    import math
    workload_sr_AR_R = workload_sr_AR.copy()
    workload_sr_AR_Q_R  = workload_sr_AR_Q.copy()
    workload_sr_AR_Y_R = workload_sr_AR_Y.copy()
    
    workload_sr_AR_R = workload_sr_AR_R.filter(regex='^Resource|success ratio$')
    workload_sr_AR_Q_R = workload_sr_AR_Q_R.filter(regex='^Resource|success ratio$')
    workload_sr_AR_Y_R = workload_sr_AR_Y_R.filter(regex='^Resource|success ratio$')
    
    quarter_columns_Q = workload_sr_AR_Q_R.columns[1:]
    quarter_columns_Y = workload_sr_AR_Y_R.columns[1:]
    quarter_columns_M = workload_sr_AR_R.columns[1:]
    
    workload_sr_AR_Q_R[quarter_columns_Q] = np.nan_to_num(workload_sr_AR_Q_R[quarter_columns_Q], nan=0.0)
    workload_sr_AR_Y_R[quarter_columns_Y] = np.nan_to_num(workload_sr_AR_Y_R[quarter_columns_Y], nan=0.0)
    workload_sr_AR_R[quarter_columns_M] = np.nan_to_num(workload_sr_AR_R[quarter_columns_M], nan=0.0)
    
    workload_sr_AR_Q_R['Average SuccessRatio'] = workload_sr_AR_Q_R[quarter_columns_Q].mean(axis=1)
    workload_sr_AR_Y_R['Average SuccessRatio'] = workload_sr_AR_Y_R[quarter_columns_Y].mean(axis=1)
    workload_sr_AR_R['Average SuccessRatio'] = workload_sr_AR_R[quarter_columns_M].mean(axis=1)

    workload_sr_AR_Q_R = workload_sr_AR_Q_R.sort_values(by='Average SuccessRatio', ascending=False)
    workload_sr_AR_Y_R = workload_sr_AR_Y_R.sort_values(by='Average SuccessRatio', ascending=False)
    workload_sr_AR_R = workload_sr_AR_R.sort_values(by='Average SuccessRatio', ascending=False)

    # Add a column for the rank of the resource
    workload_sr_AR_Q_R['Rank_per_Quarter'] = range(1, len(workload_sr_AR_Q_R) + 1)
    workload_sr_AR_Y_R['Rank_per_Year'] = range(1, len(workload_sr_AR_Y_R) + 1)
    workload_sr_AR_R['Rank_per_Month'] = range(1, len(workload_sr_AR_R) + 1)

    # Sort the DataFrame in ascending order based on the rank
    workload_sr_AR_Q_R = workload_sr_AR_Q_R.sort_values(by='Rank_per_Quarter', ascending=True)
    workload_sr_AR_Y_R = workload_sr_AR_Y_R.sort_values(by='Rank_per_Year', ascending=True)
    workload_sr_AR_R = workload_sr_AR_R.sort_values(by='Rank_per_Month', ascending=True)
    
    workload_sr_AR_Q_R = workload_sr_AR_Q_R.reset_index(drop=True)
    workload_sr_AR_Y_R = workload_sr_AR_Y_R.reset_index(drop=True)
    workload_sr_AR_R = workload_sr_AR_R.reset_index(drop=True)
    
    return workload_sr_AR_R,workload_sr_AR_Q_R,workload_sr_AR_Y_R

def Evaluate_Workload_Ranking(workload_AR_R, workload_AR_Q_R, workload_AR_Y_R):
    
    # Merge the dataframes based on the "Resource" column
    merged_df = pd.merge(workload_AR_R[['Resource', 'Average Workload', 'Rank_per_Month']],
                         workload_AR_Y_R[['Resource', 'Average Workload', 'Rank_per_Year']],
                         on='Resource',
                         how='inner')

    merged_df = pd.merge(merged_df,
                         workload_AR_Q_R[['Resource', 'Average Workload', 'Rank_per_Quarter']],
                         on='Resource',
                         how='inner')

    # Rename the columns to indicate the period (year, quarter, month)
    merged_df = merged_df.rename(columns={'Average Workload_x': 'Average Workload_Month',
                                          'Rank_per_Month': 'Rank_Month',
                                          'Average Workload_y': 'Average Workload_Year',
                                          'Rank_per_Year': 'Rank_Year',
                                          'Average Workload': 'Average Workload_Quarter',
                                          'Rank_per_Quarter': 'Rank_Quarter'})


    # Calculate the harmonic mean of the ranks
    merged_df['Geometric Rank'] = np.power((merged_df['Rank_Month'] * merged_df['Rank_Quarter'] * merged_df['Rank_Year']), 1/3)

    # Define the desired column order
    column_order = ['Resource', 'Average Workload_Month', 'Average Workload_Quarter', 'Average Workload_Year',
                    'Rank_Month', 'Rank_Quarter', 'Rank_Year', 'Geometric Rank']

    # Reorder the columns in the merged dataframe
    merged_df = merged_df[column_order]

    # Sort the dataframe based on the desired order (e.g., rank)
    merged_df = merged_df.sort_values(by='Geometric Rank', ascending=True)

    # Add a column for the rank based on geometric mean
    merged_df['Aggregated Rank Workload'] = range(1, len(merged_df) + 1)

    # Reset the index if needed
    merged_df = merged_df.reset_index(drop=True)


    return merged_df

def Evaluate_Workload_SuccessRatio_Ranking(workload_SR_AR_R,workload_SR_AR_Q_R,workload_SR_AR_Y_R):
    
    # Merge the dataframes based on the "Resource" column
    merged_df = pd.merge(workload_SR_AR_R[['Resource', 'Average SuccessRatio', 'Rank_per_Month']],
                         workload_SR_AR_Y_R[['Resource', 'Average SuccessRatio', 'Rank_per_Year']],
                         on='Resource',
                         how='inner')

    merged_df = pd.merge(merged_df,
                         workload_SR_AR_Q_R[['Resource', 'Average SuccessRatio', 'Rank_per_Quarter']],
                         on='Resource',
                         how='inner')

    # Rename the columns to indicate the period (year, quarter, month)
    merged_df = merged_df.rename(columns={'Average SuccessRatio_x': 'Average SuccessRatio_Month',
                                          'Rank_per_Month': 'Rank_Month',
                                          'Average SuccessRatio_y': 'Average SuccessRatio_Year',
                                          'Rank_per_Year': 'Rank_Year',
                                          'Average SuccessRatio': 'Average SuccessRatio_Quarter',
                                          'Rank_per_Quarter': 'Rank_Quarter'})


    # Calculate the harmonic mean of the ranks
    merged_df['Geometric Rank'] = np.power((merged_df['Rank_Month'] * merged_df['Rank_Quarter'] * merged_df['Rank_Year']), 1/3)

    # Define the desired column order
    column_order = ['Resource', 'Average SuccessRatio_Month', 'Average SuccessRatio_Quarter', 'Average SuccessRatio_Year',
                    'Rank_Month', 'Rank_Quarter', 'Rank_Year', 'Geometric Rank']

    # Reorder the columns in the merged dataframe
    merged_df = merged_df[column_order]

    # Sort the dataframe based on the desired order (e.g., rank)
    merged_df = merged_df.sort_values(by='Geometric Rank', ascending=True)

    # Add a column for the rank based on geometric mean
    merged_df['Aggregated Rank SR'] = range(1, len(merged_df) + 1)

    # Reset the index if needed
    merged_df = merged_df.reset_index(drop=True)


    return merged_df

def Evaluate_Optimal_Workload(optimal_workload, optimal_workload_Q, optimal_workload_Y):
    merged_df = pd.concat([optimal_workload, optimal_workload_Q, optimal_workload_Y], ignore_index=True)
    mean_df = merged_df.groupby('Resource').mean()
    mean_df = mean_df.reset_index()
    mean_df['optimal_workload'] = mean_df.iloc[:, 1:].idxmax(axis=1)
    return mean_df



def All_Functions(dataframe,kpi,log,start_activity):
    dataframe = Preprocessing(dataframe)
    resources = Get_All_Resources(log)
    dataframe_good,dataframe_bad = Good_vs_Bad_log(dataframe,kpi)
    
    working_behaviour,working_behaviour_top20,working_behaviour_evaluated,working_behaviour_evaluated_top20 = Resource_Behaviour(resources,dataframe,dataframe_good,dataframe_bad,start_activity)
    weighted_success_rate,weighted_success_rate_top20 = Get_Weighted_Success_Rate(working_behaviour_evaluated)
    
    Batching_AR_AB,Batching_top20_AB,Batching_Evaluated_AR_AB,Batching_Evaluated_top20_AB,Batching_SR_AR_AB,Batching_SR_top20_AB = Activity_based_batch(dataframe,resources,dataframe_good)
    Batching_AR_TB,Batching_top20_TB,Batching_Evaluated_AR_TB,Batching_Evaluated_top20_TB,Batching_SR_AR_TB,Batching_SR_top20_TB = Time_based_batch(dataframe,resources,dataframe_good)
    Batching_AR_SB,Batching_top20_SB,Batching_Evaluated_AR_SB,Batching_Evaluated_top20_SB,Batching_SR_AR_SB,Batching_SR_top20_SB = Size_based_batch(dataframe,resources,dataframe_good)
    
    Batching_AR_AB_R,Batching_AR_TB_R,Batching_AR_SB_R = Evaluate_Batching(Batching_AR_AB,Batching_AR_TB,Batching_AR_SB)
    Batching_Evaluated = Evaluate_Batching_Ranking(Batching_AR_AB_R,Batching_AR_TB_R,Batching_AR_SB_R)
    
    Batching_SR_AR_AB_R,Batching_SR_AR_TB_R,Batching_SR_AR_SB_R = Evaluate_Batching_SuccessRate(Batching_SR_AR_AB,Batching_SR_AR_TB,Batching_SR_AR_SB)
    Batching_SuccessRate_Evaluated = Evaluate_Batching_SuccessRate_Ranking(Batching_SR_AR_AB_R,Batching_SR_AR_TB_R,Batching_SR_AR_SB_R)
    
    workload_AR,workload_top20,workload_evaluated_AR,workload_evaluated_top20,workload_sr_AR,workload_sr_top20,optimal_workload= Workload_Resource(dataframe,resources,dataframe_good)
    workload_AR_Y,workload_top20_Y,workload_evaluated_AR_Y,workload_evaluated_top20_Y,workload_sr_AR_Y,workload_sr_top20_Y,optimal_workload_Y  = Workload_Resource_Year(dataframe, resources, dataframe_good) 
    workload_AR_Q,workload_top20_Q,workload_evaluated_AR_Q,workload_evaluated_top20_Q,workload_sr_AR_Q,workload_sr_top20_Q,optimal_workload_Q  = Workload_Resource_Quarter(dataframe, resources, dataframe_good) 
    
    workload_AR_R,workload_AR_Q_R,workload_AR_Y_R = Evaluate_Workload(workload_AR,workload_AR_Q,workload_AR_Y)
    workload_SR_AR_R,workload_SR_AR_Q_R,workload_SR_AR_Y_R = Evaluate_Workload_SuccessRate(workload_sr_AR,workload_sr_AR_Q,workload_sr_AR_Y)
    
    optimal_workload_Evaluated = Evaluate_Optimal_Workload(optimal_workload, optimal_workload_Q, optimal_workload_Y)
    workload_Evaluated = Evaluate_Workload_Ranking(workload_AR_R,workload_AR_Q_R,workload_AR_Y_R)
    workload_SuccessRatio_Evaluated = Evaluate_Workload_SuccessRatio_Ranking(workload_SR_AR_R,workload_SR_AR_Q_R,workload_SR_AR_Y_R)

    #return dataframe,resources,dataframe_good,dataframe_bad,working_behaviour,working_behaviour_top20,working_behaviour_evaluated,working_behaviour_evaluated_top20, weighted_success_rate,weighted_success_rate_top20,Batching_AR_AB,Batching_top20_AB,Batching_Evaluated_AR_AB,Batching_Evaluated_top20_AB,Batching_SR_AR_AB,Batching_SR_top20_AB,Batching_AR_TB,Batching_top20_TB,Batching_Evaluated_AR_TB,Batching_Evaluated_top20_TB,Batching_SR_AR_TB,Batching_SR_top20_TB,Batching_AR_SB,Batching_top20_SB,Batching_Evaluated_AR_SB,Batching_Evaluated_top20_SB,Batching_SR_AR_SB,Batching_SR_top20_SB,Batching_AR_AB_R,Batching_AR_TB_R,Batching_AR_SB_R,Batching_SR_AR_AB_R,Batching_SR_AR_TB_R,Batching_SR_AR_SB_R,workload_AR,workload_top20,workload_evaluated_AR,workload_evaluated_top20,workload_sr_AR,workload_sr_top20,optimal_workload,workload_AR_Y,workload_top20_Y,workload_evaluated_AR_Y,workload_evaluated_top20_Y,workload_sr_AR_Y,workload_sr_top20_Y,optimal_workload_Y,workload_AR_Q,workload_top20_Q,workload_evaluated_AR_Q,workload_evaluated_top20_Q,workload_sr_AR_Q,workload_sr_top20_Q,optimal_workload_Q,workload_AR_R,workload_AR_Q_R,workload_AR_Y_R,workload_SR_AR_R,workload_SR_AR_Q_R,workload_SR_AR_Y_R,Batching_Evaluated,Batching_SuccessRate_Evaluated,workload_Evaluated,workload_SuccessRatio_Evaluated,optimal_workload_Evaluated
    #return dataframe,resources,dataframe_good,dataframe_bad,workload_AR,workload_top20,workload_evaluated_AR,workload_evaluated_top20,workload_sr_AR,workload_sr_top20,optimal_workload
    return working_behaviour,working_behaviour_top20,working_behaviour_evaluated,working_behaviour_evaluated_top20,weighted_success_rate,weighted_success_rate_top20,Batching_AR_AB,Batching_top20_AB,Batching_Evaluated_AR_AB,Batching_Evaluated_top20_AB,Batching_SR_AR_AB,Batching_SR_top20_AB,Batching_AR_TB,Batching_top20_TB,Batching_Evaluated_AR_TB,Batching_Evaluated_top20_TB,Batching_SR_AR_TB,Batching_SR_top20_TB,Batching_AR_SB,Batching_top20_SB,Batching_Evaluated_AR_SB,Batching_Evaluated_top20_SB,Batching_SR_AR_SB,Batching_SR_top20_SB,workload_AR,workload_top20,workload_evaluated_AR,workload_evaluated_top20,workload_sr_AR,workload_sr_top20,optimal_workload,workload_AR_Y,workload_top20_Y,workload_evaluated_AR_Y,workload_evaluated_top20_Y,workload_sr_AR_Y,workload_sr_top20_Y,optimal_workload_Y,workload_AR_Q,workload_top20_Q,workload_evaluated_AR_Q,workload_evaluated_top20_Q,workload_sr_AR_Q,workload_sr_top20_Q,optimal_workload_Q


def home(request):
    import pm4py
    import tempfile
    import os
    import gzip

    if request.method == 'POST':
        form = MyForm(request.POST, request.FILES)
        if form.is_valid():
            xes_file = request.FILES['xes_file']
            if xes_file.size > 0:
                # Save the uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(xes_file.read())
                    temp_file_path = temp_file.name
                    print("Temporary file path:", temp_file_path)
                
                    # Read the XES file using pm4py
                    log = pm4py.read_xes(temp_file_path)
                    df = pm4py.convert_to_dataframe(log)
                    
            
            # Remove the temporary file
            
           
            #df = pm4py.convert_to_dataframe(log)
            dropdown_choice = form.cleaned_data['dropdown']
            start_activities = pm4py.get_start_activities(log)
            for key, value in start_activities.items():
                start_activity = str(key)
                value_str = str(value)
            table11_data,table12_data,table13_data,table14_data,table15_data,table16_data,table21_data,table22_data,table23_data,table24_data,table25_data,table26_data,table221_data,table222_data,table223_data,table224_data,table225_data,table226_data,table2221_data,table2222_data,table2223_data,table2224_data,table2225_data,table2226_data, table31_data, table32_data, table33_data, table34_data, table35_data, table36_data, table37_data, table331_data,table332_data,table333_data,table334_data,table335_data,table336_data,table337_data, table3331_data,table3332_data,table3333_data,table3334_data,table3335_data,table3336_data, table3337_data = All_Functions(df,dropdown_choice,log,start_activity)
            
            os.remove(temp_file_path)
            
            # Render the output in a template
            return render(request, 'Dashboard.html', {'table11_data': table11_data, 'table12_data': table12_data, 'table13_data': table13_data, 'table14_data': table14_data, 'table15_data': table15_data, 'table16_data': table16_data,'table21_data': table21_data,'table22_data': table22_data,'table23_data': table23_data, 'table24_data': table24_data, 'table25_data': table25_data, 'table26_data': table26_data, 'table221_data': table221_data, 'table222_data': table222_data, 'table223_data': table223_data, 'table224_data': table224_data, 'table225_data': table225_data, 'table226_data': table226_data, 'table2221_data': table2221_data, 'table2222_data': table2222_data,'table2223_data': table2223_data,'table2224_data': table2224_data,'table2225_data': table2225_data,'table2226_data': table2226_data, 'table31_data': table31_data, 'table32_data': table32_data, 'table33_data': table33_data, 'table34_data': table34_data, 'table35_data': table35_data, 'table36_data': table36_data, 'table37_data': table37_data, 'table331_data': table331_data, 'table332_data': table332_data,'table333_data': table333_data,'table334_data': table334_data,'table335_data': table335_data,'table336_data': table336_data ,'table337_data': table337_data, 'table3331_data': table3331_data, 'table3332_data': table3332_data,'table3333_data': table3333_data,'table3334_data': table3334_data,'table3335_data': table3335_data,'table3336_data': table3336_data ,'table3337_data': table3337_data })
            #return render(request,'Testing_Results_V2.html')
    else:
        form = MyForm()
    return render(request, 'Home.html', {'form': form})
