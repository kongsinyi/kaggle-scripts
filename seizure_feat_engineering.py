import scipy.io as sio
from scipy.signal import periodogram
from scipy import stats as stats
import numpy as np
from numpy import corrcoef as coef
import pandas as pd
from sklearn.metrics import roc_auc_score as roc
from sklearn import preprocessing as pp
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import KFold as kfold
import glob
import datetime

start = datetime.datetime.now()
def one_minute(file):
    list_of_files = []
    num_tenth_rows = np.round((len(file)/10),0)
    for i in range(10):
        minute_file = file[(i*num_tenth_rows): ((i+1)*num_tenth_rows)-1]
        if len(minute_file[minute_file.any(1)]) < 0.1*num_tenth_rows:
            pass
        else:
            list_of_files.append(minute_file[minute_file.any(1)])
    return(list_of_files)
    
def spectrum_and_entropy(file):
    max_f_list = []
    sum_list = []
    entropy_list = []
    entropy = 0
    for i in range(16):
        entropy = 0
        f, pxx = periodogram(file[:,i], fs = 400, scaling = 'density')
        max_f_list.append(f[np.argmax(pxx)])
        sum_list.append(np.sum(pxx))
        norm_pxx = pxx/np.sum(pxx)
        for j in range(len(f)):
            if pxx[j] > 0:
                entropy += -1*norm_pxx[j]*np.log(norm_pxx[j])
        entropy_list.append(entropy)

    return(max_f_list + sum_list +entropy_list)

def corr(file):
    cov = coef(np.transpose(file))
    trunc_cov = np.ravel(cov[indices]).tolist()
    return(trunc_cov)


def kurt_and_skew(array):
    kurt = []
    skew = []
    for i in range(16):
        kurt.append(stats.kurtosis(array[:,i]))
        skew.append(stats.skew(array[:,i]))
    return(kurt + skew)

def process_single_train_file(file):
 
    current_file = sio.loadmat(file)
    signal = current_file['dataStruct'][0][0][0]
    outcome = file.strip('.mat').split('\\')[-1].split('_')[-1]
    patient_id = file.strip('.mat').split('\\')[-1].split('_')[0]
    sep = ','
    one_minutes = one_minute(signal)        
    if len(one_minutes) > 0:
        for i in range(len(one_minutes)):
            s_and_e = spectrum_and_entropy(one_minutes[i])
            cov = corr(one_minutes[i])
            kurtandskew = kurt_and_skew(one_minutes[i])
            single_row = (str(outcome) + ','+ str(patient_id) + ',' + str(s_and_e).strip('[]\n') + ',' + str(cov).strip('[]\n')
                          + ',' + str(kurtandskew).strip('[]\n'))
            print(single_row, file = wfile)
            
def process_single_test_file(file):
    
    current_file = sio.loadmat(file)
    signal = current_file['dataStruct'][0][0][0]
    patient_id, file_name = file.strip('.mat').split('\\')[-1].split('_')[0:2]
   
    sep = ','
    one_minutes = one_minute(signal)        
    if len(one_minutes) > 0:
        for i in range(len(one_minutes)):
            s_and_e = spectrum_and_entropy(one_minutes[i])
            cov = corr(one_minutes[i])
            kurtandskew = kurt_and_skew(one_minutes[i])
            single_row = (str(file_name) + ','+ str(patient_id) + ',' + str(s_and_e).strip('[]\n') + ',' + str(cov).strip('[]\n')
                          + ',' + str(kurtandskew).strip('[]\n'))
            print(single_row, file = wfile)

#to start train file
col_string = 'outcome' + ',' 'patient' +','
for i in range(16):
    col_string += ('max_f_' + str(i)) + ','
for i in range(16):
    col_string += ('s_'+str(i)) + ','
for i in range(16):
    col_string += ('entropy_'+ str(i)) + ','
for i in range(120):
    col_string += ('cov_' + str(i)) + ','
for i in range(16):
    col_string += ('kurt_' + str(i)) + ','
for i in range(16):
    col_string += ('skew_' + str(i)) + ','

#to start test_file
col_string = 'file_name' + ',' 'patient' +','
for i in range(16):
    col_string += ('max_f_' + str(i)) + ','
for i in range(16):
    col_string += ('s_'+str(i)) + ','
for i in range(16):
    col_string += ('entropy_'+ str(i)) + ','
for i in range(120):
    col_string += ('cov_' + str(i)) + ','
for i in range(16):
    col_string += ('kurt_' + str(i)) + ','
for i in range(16):
    col_string += ('skew_' + str(i)) + ','
with open('./test_spec_cor_3.csv','w') as wfile:
    print(col_string.strip(''), sep = ',', file = wfile)
    wfile.close()


# for test files
files = glob.glob('./test_3/*.mat')
wfile =  open('./test_spec_cor_3.csv', 'a') 
indices = np.triu_indices(16, 1)
for i in range(len(files)):    
    try:
        process_single_test_file(files[i])
    except ValueError:   
        pass    
    if i%10 ==0:
        end = datetime.datetime.now()     
        print('done with ' + str(i) + 'time elapsed: ' + str(end-start))
wfile.close()    