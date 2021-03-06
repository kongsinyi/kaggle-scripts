import time
import datetime
import pandas as pd
import numpy as np
import collections
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.linear_model import LogisticRegression as LR

start = time.time()
print('working on events.csv and writing to modified_events.csv...')

def blocktime(x):
    date = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    day = date.weekday()
    return day

def time_of_day(x):
    hour = int(x[11:13])
    if 0 <= hour < 6:
        tod = 'night'
    elif 6<= hour < 16:
        tod = 'day'
    elif 16<= hour <24:
        tod = 'eve'    
    return tod

events = pd.read_csv("./events.csv")
coords = list(zip(events.longitude, events.latitude))
events['coord'] = pd.Series(coords)
events['mobility'] = events.groupby(['device_id'])['coord'].transform('size')

events['dayofweek'] = events.timestamp.map(lambda x: blocktime(x))
events['time_of_day'] = events.timestamp.map(lambda x: time_of_day(x))
events['day'] = events[events.time_of_day =='day'].groupby(['device_id'])['event_id'].transform('count')
events['eve'] = events[events.time_of_day =='eve'].groupby(['device_id'])['event_id'].transform('count')
events['night'] = events[events.time_of_day =='night'].groupby(['device_id'])['event_id'].transform('count')
events['mon'] = events[events.dayofweek == 0].groupby(['device_id'])['event_id'].transform('count')
events['tues'] = events[events.dayofweek == 1].groupby(['device_id'])['event_id'].transform('count')
events['wed'] = events[events.dayofweek == 2].groupby(['device_id'])['event_id'].transform('count')
events['thurs'] = events[events.dayofweek == 3].groupby(['device_id'])['event_id'].transform('count')
events['fri'] = events[events.dayofweek == 4].groupby(['device_id'])['event_id'].transform('count')
events['sat'] = events[events.dayofweek == 5].groupby(['device_id'])['event_id'].transform('count')
events['sun'] = events[events.dayofweek == 6].groupby(['device_id'])['event_id'].transform('count')

with open('./modified_events.csv','w') as wfile0:
    events.to_csv(wfile0)
    wfile0.close()

end = time.time()
print('time elapsed: ' + str(end - start))                                                             
print('now onto phone csv and merging everything')
start = time.time()
                                                             
phone = pd.read_csv('./phone_brand_device_model.csv')
d = collections.Counter(phone.device_model)
dsort = d.most_common(len(d))
sorted_models = []
for i in range(len(dsort)):
    sorted_models.append(dsort[i][0])
phone['pop_device'] = phone.device_model.map(lambda x: sorted_models.index(x))   
merged = pd.merge(phone, events, on = 'device_id')
merged = merged.fillna(0)

end = time.time()
print('time elapsed: ' + str(end - start))                                                             
print('building training set')
start = time.time()
                                                             
train = pd.read_csv('./gender_age_train.csv')
le = LE()
train['group'] = le.fit_transform(train.group)
train_merge = pd.merge(train,merged, on = 'device_id')
train_merge = train_merge.drop_duplicates('device_id')
