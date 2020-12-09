import pandas as pd
import numpy as np

# INITIAL VALUE
my_block = 'block_0'
my_household = 'MAC003388'

# READ CSV FILE
df = pd.read_csv('hhblock_dataset/' + my_block + '.csv')
wh = pd.read_csv('weather_hourly_darksky.csv')
bh = pd.read_csv('uk_bank_holidays.csv')

df = df[df.LCLid == my_household]
df.reset_index(drop=True, inplace=True)

# TIMESTAMP REFORMAT
for i in range(len(df)):
    df.at[i, 'day'] = pd.Timestamp(df['day'][i])
for i in range(len(wh)):
    wh.at[i, 'time'] = pd.Timestamp(wh['time'][i])
for i in range(len(bh)):
    bh.at[i, 'Bank holidays'] = pd.Timestamp(bh['Bank holidays'][i])

start = pd.Timestamp('2011-12-04 00:00:00')
end = pd.Timestamp('2014-02-27 23:00:00')

df = df[(df['day'] >= start) & (df['day'] <= end)]
wh = wh[(wh['time'] >= start) & (wh['time'] <= end)]
bh = bh[(bh['Bank holidays'] >= start) & (bh['Bank holidays'] <= end)]

df.reset_index(drop=True, inplace=True)
wh.reset_index(drop=True, inplace=True)
bh.reset_index(drop=True, inplace=True)

# DROP UNUSED COLUMNS
ucols = [i for i in range(3, 50, 2)]
df.drop(df.columns[ucols], axis=1, inplace=True)

# RENAME COLUMNS
rcols = ['ID_House', 'Timestamp']
for i in range (24):
    rcols.append('Hour_' + str(i))
df.columns = rcols

# INSERT HOLIDAYS
df.insert(2, 'Holiday', np.array(['No'] * len(df)))
for i in range(len(df)):
    for j in range(len(bh)):
        if df['Timestamp'][i] == bh['Bank holidays'][j]:
            df.at[i, 'Holiday'] = 'Yes'
            break

# INSERT ACORN TYPE
df.insert(3, 'Acorn_Group', np.array(['Lavish Lifestyles'] * len(df)))

# SPLIT DATAFRAME PER HOUR
block = []
for i in range(24):
    block.append(df.loc[:, ['ID_House', 'Timestamp', 'Holiday', 'Acorn_Group']])
    for j in range(len(block[i])):
        block[i].at[j, 'Timestamp'] += pd.DateOffset(hours=i)
    block[i].insert(4, 'kWh', df['Hour_' + str(i)])
    
df = pd.DataFrame()
for i in range(24):
    df = df.append(block[i])
    
# SORTING BY TIMESTAMP
df.sort_values(by='Timestamp', inplace=True)
df.reset_index(drop=True, inplace=True)
wh.sort_values(by='time', inplace=True)
wh.reset_index(drop=True, inplace=True)

# INSERT WEATHER DATASET
df.insert(4, 'Temperature', np.array([0.0] * len(df)))
for i in range(len(df)):
    for j in range(len(wh)):
        if df['Timestamp'][i] == wh['time'][j]:
            df.at[i, 'Temperature'] = wh['temperature'][j]
            break

# SPLIT TIMESTAMP
day = np.zeros(len(df), dtype=int)
month = np.zeros(len(df), dtype=int)
year = np.zeros(len(df), dtype=int)
hour = np.zeros(len(df), dtype=int)

for i in range(len(df)):
    day[i] = df['Timestamp'][i].day
    month[i] = df['Timestamp'][i].month
    year[i] = df['Timestamp'][i].year
    hour[i] = df['Timestamp'][i].hour
    
df.insert(1, 'Day', day)
df.insert(2, 'Month', month)
df.insert(3, 'Year', year)
df.insert(4, 'Hour', hour)

df.drop('Timestamp', axis=1, inplace=True)
df.to_csv('hourlyid_dataset/' + my_household + '.csv', index=False)
