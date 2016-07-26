import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn import preprocessing as pp
import pandas as pd
from sklearn.metrics import log_loss as ll
import collections
import datetime
import operator

print('open train set, score breed and colors in terms of desirability..')
start = datetime.datetime.now()
df = pd.read_csv('./train.csv')

def getbreedandcolor(df):
    df['Breeds'] = df['Breed'].map(lambda x:x.strip(" Mix").split("/"))
    df['colors'] = df['Color'].map(lambda x: x.split("/"))
    colors = list(df['colors'])
    color_list = []
    for color in colors:
        ind_animal = []
        for i in range(len(color)):
            word = color[i].split(" ")
            for j in range(len(word)):
                ind_animal.append(word[j])
        color_list.append(ind_animal)

    df['colors'] = pd.Series(color_list)
    return df
df = getbreedandcolor(df)

def compare(df,X):
    gb = df[(df.OutcomeType == 'Adoption')][X]
    bb = df[(df.OutcomeType !="Adoption")][X]
    tb = df[X]
    goodbreeds = []
    badbreeds= []
    totalbreeds = []
    for i in gb:
        for j in range(1,len(i)):
            goodbreeds.append(i[j])
    for i in bb:
        for j in range(1,len(i)):
            badbreeds.append(i[j])
    for i in tb:
        for j in range(1,len(i)):
            totalbreeds.append(i[j])

    gbd = collections.Counter(goodbreeds)
    bbd = collections.Counter(badbreeds)
    tbd = collections.Counter(totalbreeds)


    breed = []
    odds = []
    for key in tbd:
        if key not in gbd.keys():
            odd = 1 - (bbd[key]/len(badbreeds))*(len(badbreeds)/len(totalbreeds))/(tbd[key]/len(totalbreeds))

        else:
            odd = (gbd[key]/len(goodbreeds))*(len(goodbreeds)/len(totalbreeds))/(tbd[key]/len(totalbreeds))

        breed.append(key)
        odds.append(odd)

    breeddict = dict(zip(breed,odds))
    return breeddict

BREED = compare(df,'Breeds')
COLOR = compare(df,'colors')

def scoring(x,A):
    y = 0
      
    for breed in x:
        if breed in A.keys():
            y += A[breed]   
        else:           
            y += 0.5
    
    return y

def meanscoring(x,A):
    y = 0

    for breed in x:
        if breed in A.keys():
            y += A[breed]   
        else:           
            y += 0.5
    ymean = y/len(x)
    return ymean
end = datetime.datetime.now()
print('time elapsed: ',str(end - start))

print('Quantify named, outcome time, spayed, sex, breed and color..')
start = datetime.datetime.now()

def process(df):
    df['Named'] = df['Name'].isnull().map({True: 0, False : 1})
    df['Dates'] = df.DateTime.map(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
    df['Outcome_Year'] = df['DateTime'].map(lambda x:int(x[:4]))
    df['Outcome_Month'] = df['DateTime'].map(lambda x:int(x[5:7]))
    df['Outcome_Date']= df['DateTime'].map(lambda x :int(x[8:10]))
    df['Outcome_Day'] = df['Dates'].map(lambda x: datetime.date.weekday(x))
    df['Outcome_Time'] = df['DateTime'].map(lambda x: int(x[11:13]))
    df['AnimalType_code'] = df['AnimalType'].map({'Cat' :0, 'Dog':1})
    df['Spayed'] = 0
    df['Spayed'] = df['SexuponOutcome'].map({"Neutered Male" : 1,"Spayed Female" :1, "Intact Male" : 0, "Intact Female":0})
    df['Sex'] = 0
    df['Sex'] = df['SexuponOutcome'].map({"Neutered Male" : 1, "Intact Male" : 1})
    df['Breed_score'] = df['Breeds'].map(lambda x: scoring(x,BREED))
    df['Color_score'] = df['colors'].map(lambda x:meanscoring(x,COLOR))
   
    breedlist = list(df['Breed'])
    mixvec = []
    for breed in breedlist:
        if 'Mix' in breed:
            mixvec.append(1)
        else:
            mixvec.append(0)
    df['Mixed'] = pd.DataFrame(mixvec)


    ages = list(df['AgeuponOutcome'])
    agedays = []
    for i in range(len(ages)):
        try:
            a = ages[i].rstrip("s").split(" ")
            if 'year' in a[1]:
                if a[0] =='0':
                    agedays.append(182)
                else:
                    agedays.append(int(a[0])*365)
            elif a[1]== 'month':
                agedays.append(int(a[0])*30)
            elif a[1] == 'week':
                agedays.append(int(a[0])*7)
            elif a[1] =='day':
                agedays.append(int(a[0]))   
            
        except AttributeError:
            agedays.append(ages[i])
            
    df['Daysold'] = pd.DataFrame(agedays)
    df.Daysold = df.Daysold.fillna(df.Daysold.mean())
    df['logdays'] = np.log(df.Daysold)
    df['norm_age'] = pp.scale(df.logdays) 
    df['norm_outcomeyear'] = pp.scale(df.Outcome_Year)
    df['norm_outcomemonth'] = pp.scale(df.Outcome_Month)
    df['norm_outcomedate'] = pp.scale(df.Outcome_Date)
    df['norm_outcomeday'] = pp.scale(df.Outcome_Day)
    df['norm_outcometime'] = pp.scale(df.Outcome_Time)
    df['norm_breedscore'] = pp.scale(df.Breed_score)
    df['norm_colorscore'] = pp.scale(df.Color_score)
    df = df.fillna(0)
    return df

df = process(df)
df['OutcomeType_code'] = df.OutcomeType.map({"Adoption":0,"Died":1,"Euthanasia":2,"Return_to_owner":3,"Transfer":4})

end = datetime.datetime.now()
print('time elapsed: ',str(end-start))


