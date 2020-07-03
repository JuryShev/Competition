
from functools import partial
import scipy as sp

import pandas as pd

pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import lag_plot
import numpy as np
import calendar
from tqdm import tqdm
import collections
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from joblib import Parallel
from scipy import optimize


def read_data(rows):
    print('Reading train.csv file....')
    train = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/train.csv', nrows=rows)
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    train_drop = train
    keep_id = train_drop[train_drop.type == "Assessment"]['installation_id'].drop_duplicates()
    train = pd.merge(train, keep_id, on="installation_id", how="inner")



    print('Reading test.csv file....')
    test = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/test.csv',nrows=rows)
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/train_labels.csv', nrows=rows)
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/specs.csv', nrows=rows)
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('C:/PYTHON/Kaggle/2019 Data Science Bowl/data/sample_submission.csv', nrows=rows)
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0],
                                                                          sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission


def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    list_of_type_code = list(set(train['type'].unique()).union(set(test['type'].unique())))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    type_activities = dict(zip(list_of_type_code, np.arange(len(list_of_type_code))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(
        set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    #train['type']=test['type'].map(type_activities)

    win_code = dict(zip(activities_map.values(), (4100 * np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code






def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals

def reduce_zero_time(train_s,  flag_event_train):
    num0 = 0
    num1 = 0
    num_z = 0

    day = pd.DataFrame()
    day[['date', 'time']] = train_s['timestamp'].str.split('T', expand=True)
    day[['hour', 'minut', 'sec']] = day['time'].str.split(':', expand=True)
    day[['hour']] = day[['hour']].astype(int)

    ser_hour = pd.Series(day['hour'], index=day.index)
    ser_minut = pd.Series(day['minut'], index=day.index)
    Session_time = pd.DataFrame({"game_session": [0], "long_time": [0], "ID_num0": [0], "tipe": [0]})
    Session_time.loc[[0], ['game_session']] = train_s.iloc[num0, 1]
    Session_time.loc[[0], ['ID_num0']] = num0
    Session_time.loc[[num1], ['tipe']] = train_s.iloc[0, 9]
    while (num0 < (train_s.shape[0] - 1)):
        delta_time0 = pd.to_numeric(ser_hour[num0]) * 60 + pd.to_numeric(ser_minut[num0])
        #if(train_s.iloc[num0, 9]='Assessment'):

        while (Session_time.iloc[num1, 0] == train_s.iloc[num0, 1] and num0 < (train_s.shape[0] - 1)):
            num0 = num0 + 1
                # 1 Записать время в long_time
                # 2 Прописать дельту
                # 3 расчитать время сезона

        delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(ser_minut[num0 - 1]) - delta_time0
        if (delta_time < 0):
            delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(
                ser_minut[num0 - 1]) + 1440 - delta_time0
        if (delta_time < 1):
            num_z = num_z + 1
            # train = pd.merge(train, train.loc[[num0], ['game_session']], on="game_session", how="inner")

        #print(num0)
        Session_time.loc[[num1], ['long_time']] = delta_time
        Session_time = Session_time.append({'game_session': train_s.iloc[num0, 1]}, ignore_index=True)
        num1 = num1 + 1
        Session_time.loc[[num1], ['ID_num0']] = num0
        Session_time.loc[[num1], ['tipe']] = train_s.iloc[num0, 9]

    if(flag_event_train==0):
        test_drop=Session_time[(Session_time['long_time'] > 0.4) | (Session_time['tipe'] =='Assessment')]
        test_ass_zero = Session_time[(Session_time['long_time'] < 0.4) & (Session_time['tipe'] == 'Assessment')]
        keep_ltime = test_drop.game_session.drop_duplicates()
        keep_ltime_1_1=Session_time[((Session_time['long_time'] < 100)&(Session_time['long_time'] > 0.1))|(Session_time['tipe'] =='Assessment')]
        keep_ltime_1 = Session_time[Session_time.long_time < 100]['game_session'].drop_duplicates()

        keep_ltime=pd.merge(keep_ltime, keep_ltime_1,  how="inner")
        train_s = pd.merge(train_s, keep_ltime_1_1, on="game_session", how="inner")
        train_labels_zerro=test_ass_zero.game_session.drop_duplicates()

        return train_s,train_labels_zerro
    else:
       return Session_time

def reduce_zero_time0(train,  flag_event_train):
    num0 = 0
    num1 = 0
    num_z = 0

    day = pd.DataFrame()
    day[['date', 'time']] = train['timestamp'].str.split('T', expand=True)
    day[['hour', 'minut', 'sec']] = day['time'].str.split(':', expand=True)
    day[['hour']] = day[['hour']].astype(int)

    ser_hour = pd.Series(day['hour'], index=day.index)
    ser_minut = pd.Series(day['minut'], index=day.index)
    Session_time = pd.DataFrame({"game_session": [0], "long_time": [0], "ID_num0": [0], "tipe": [0]})
    Session_time.loc[[0], ['game_session']] = train.iloc[num0, 1]
    Session_time.loc[[0], ['ID_num0']] = num0
    Session_time.loc[[num1], ['tipe']] = train.iloc[0, 9]
    while (num0 < (train.shape[0] - 1)):
        delta_time0 = pd.to_numeric(ser_hour[num0]) * 60 + pd.to_numeric(ser_minut[num0])

        while (Session_time.iloc[num1, 0] == train.iloc[num0, 1] and num0 < (train.shape[0] - 1)):
            num0 = num0 + 1
            # 1 Записать время в long_time
            # 2 Прописать дельту
            # 3 расчитать время сезона

        delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(ser_minut[num0 - 1]) - delta_time0
        if (delta_time < 0):
            delta_time = pd.to_numeric(ser_hour[num0 - 1]) * 60 + pd.to_numeric(
                ser_minut[num0 - 1]) + 1440 - delta_time0
        if (delta_time < 1):
            num_z = num_z + 1
            # train = pd.merge(train, train.loc[[num0], ['game_session']], on="game_session", how="inner")

        #print(num0)
        Session_time.loc[[num1], ['long_time']] = delta_time
        Session_time = Session_time.append({'game_session': train.iloc[num0, 1]}, ignore_index=True)
        num1 = num1 + 1
        Session_time.loc[[num1], ['ID_num0']] = num0
        Session_time.loc[[num1], ['tipe']] = train.iloc[num0, 9]

    if(flag_event_train==0):

        keep_ltime = Session_time[Session_time.long_time > 0.1]['game_session'].drop_duplicates()
        keep_ltime_1 = Session_time[Session_time.long_time < 100]['game_session'].drop_duplicates()
        keep_ltime=pd.merge(keep_ltime, keep_ltime_1,  how="inner")
        train_s = pd.merge(train, keep_ltime, on="game_session", how="inner")
        return train_s
    else:
       return Session_time

def preprocess(reduce_train, reduce_test):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')

        df['sum_event_code_count'] = df[
            [2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020,
             4021,
             4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080,
             2035,
             2040, 4090, 4220, 4095]].sum(axis=1)

        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform(
            'mean')

    features = reduce_train.loc[
        (reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns  # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in
                                                                                          assess_titles]

    return reduce_train, reduce_test, features


def get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    user_activities_count = {'Clip': 0, 'Activity': 0, 'Assessment': 0, 'Game': 0}

    # news features: time spent in each activity
    time_spent_each_act = {actv: 0 for actv in list_of_user_activities}
    event_code_count = {eve: 0 for eve in list_of_event_code}
    last_session_time_sec = 0

    accuracy_groups = {0: 0, 1: 0, 2: 0, 3: 0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []

    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]  # from Andrew

        # get current session time in seconds
        if session_type != 'Assessment':
            time_spent = int(session['game_time'].iloc[-1] / 1000)
            time_spent_each_act[activities_labels[session_title]] += time_spent

        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session) > 1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens:
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(time_spent_each_act.copy())
            features.update(event_code_count.copy())
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]  # from Andrew
            """_____________my_append________________"""
            features['game_session']=i
            session_game=user_sample[user_sample.game_session == i]
            """_________________________________________"""
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment

            """ _______________________My_correct_code____________________________________"""
            accumulated_correct_attempts += true_attempts
            accumulated_uncorrect_attempts += false_attempts

            """____________________________________________________________"""

            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            #accumulated_correct_attempts += true_attempts
            #accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2]).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy / counter if counter > 0 else 0
            accuracy = true_attempts / (true_attempts + false_attempts) if (true_attempts + false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group / counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions

            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts + false_attempts > 0:
                all_assessments.append(features)

            counter += 1

        # this piece counts how many actions was made in each event_code so far
        n_of_event_codes = collections.Counter(session['event_code'])

        for key in n_of_event_codes.keys():
            event_code_count[key] += n_of_event_codes[key]

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type
    # if test_set=True, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in train_set, all assessments are kept
    return all_assessments

###_______________Функция оценки________________________________________________
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1: TRUE
    :param a2: PREDICT
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e

class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """

    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients

        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds

        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        initial_coef=np.asarray(initial_coef)
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds

        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=[0, 1, 2, 3])

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
"""__________________________________________Основная программа____________________"""

day = pd.DataFrame()

train, test, train_labels, specs, sample_submission = read_data(rows=2000000)
# train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code=encode_title(train, test, train_labels)

day[['date', 'time']] = train['timestamp'].str.split('T', expand=True)
day[['hour', 'minut', 'sec']] = day['time'].str.split(':', expand=True)
day[['hour']] = day[['hour']].astype(int)

ser_data = pd.Series(day['date'], index=day.index)
ser_hour = pd.Series(day['hour'], index=day.index)
ser_minut = pd.Series(day['minut'], index=day.index)

print("Clear Zero train...")
train, train_times_zero = reduce_zero_time(train, flag_event_train=0)
# train_times_zero_f= pd.merge(train_labels, train_times_zero, on="game_session", how="inner")


# Session_time=reduce_zero_time(train, flag_event_train=1)

print("Clear Zero test...")
test, test_times_zero = reduce_zero_time(test, flag_event_train=0)
# Session_test_time=reduce_zero_time(test, flag_event_train=1)

###################################################################
test.to_csv('clear_test.csv', index=False)
train.to_csv('clear_rain.csv', index=False)
test_times_zero.to_csv('test_times_zero.csv', index=False)
train_times_zero.to_csv('train_times_zero.csv', index=False)

###################################################################

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(
    train, test, train_labels)
print('Post_clear_Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))
# sample_id = train[train.installation_id == "0006a69f"]
# sample_id_data = get_data(sample_id) #returns a list
# sample_df = pd.DataFrame(sample_id_data)
# train, test= preprocess(train, test)

compiled_data = []
# tqdm is the library that draws the status bar below
for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort=False)),
                                     total=train.installation_id.nunique(), desc='Installation_id', position=0):
    # user_sample is a DataFrame that contains only one installation_id

    compiled_data += get_data(user_sample)
reduce_train = pd.DataFrame(compiled_data)
del compiled_data

new_test = []
for ins_id, user_sample in tqdm(test.groupby('installation_id', sort=False), total=test.installation_id.nunique(),
                                desc='Installation_id', position=0):
    a = get_data(user_sample, test_set=True)
    new_test.append(a)

reduce_test = pd.DataFrame(new_test)

reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)
############ fill ID the zero-time array####################################################
num_id_z = []
test_zero_time = pd.DataFrame()
for id in train_times_zero:
    position_id = 0
    flag_record = 0
    #print(reduce_train.iloc[position_id, 71])
    while position_id < len(reduce_train.index) and flag_record != 1:

        if reduce_train.iloc[position_id, 71] == id:
            num_id_z.append(position_id)
            test_zero_time = test_zero_time.append(reduce_train.iloc[position_id], sort=False)

            flag_record = 1

        position_id = position_id + 1

num_id_z = np.asarray(num_id_z)
test_zero_time = test_zero_time.reindex(reduce_train.columns, axis=1)
# for drop_in in test_zero_time.index:
#     reduce_train = reduce_train.drop([drop_in])

#############################################################################################
############ fill ID the zero-time array####################################################
num_id_z_test = []
test_s_zero_time = pd.DataFrame()
for id in test_times_zero:
    position_id = 0
    flag_record = 0
    # print (reduce_train.iloc[position_id,91])
    while position_id < len(reduce_test.index) and flag_record != 1:

        if reduce_test.iloc[position_id, 71] == id:
            num_id_z_test.append(position_id)
            test_s_zero_time = test_s_zero_time.append(reduce_test.iloc[position_id], sort=False)

            flag_record = 1

        position_id = position_id + 1

num_id_z_test = np.asarray(num_id_z_test)
test_s_zero_time = test_s_zero_time.reindex(reduce_test.columns, axis=1)
for drop_in in test_s_zero_time.index:
    reduce_test = reduce_test.drop([drop_in])
flag_train=0
if reduce_test.shape[0]<1:
    flag_train=1
#############################################################################################

#############################################################################################
drop_column = ['sum_event_code_count', 'Game', '2060', '5010', '5000', '2070', '2083', '4230', '4235', '4031', '2040',
               '2050', '4050']
answer_train = np.array(reduce_train['accuracy_group'])
reduce_train = reduce_train.drop(columns='installation_id')
reduce_train = reduce_train.drop(columns='game_session')
reduce_train = reduce_train.drop(columns='accuracy_group')

# для массва тестовых сессий нулевого времени
test_zero_time_answer = np.array(test_zero_time['accuracy_group'])
test_zero_time = test_zero_time.drop(columns='installation_id')
test_zero_time = test_zero_time.drop(columns='accuracy_group')
test_zero_time = test_zero_time.drop(columns='game_session')

if flag_train==0:
    reduce_test_installation_id = np.array(reduce_test['installation_id'])
    reduce_test = reduce_test.drop(columns='installation_id')
    reduce_test = reduce_test.drop(columns='game_session')
    reduce_test = reduce_test.drop(columns='accuracy_group')

test_s_zero_time_installation_id = np.array(test_s_zero_time['installation_id'])
test_s_zero_time = test_s_zero_time.drop(columns='installation_id')
test_s_zero_time = test_s_zero_time.drop(columns='accuracy_group')
test_s_zero_time = test_s_zero_time.drop(columns='game_session')

# for drop_in in drop_column:
#    reduce_train=reduce_train.drop(columns=drop_in)
#    reduce_test = reduce_test.drop(columns=drop_in)


# reduce_train=reduce_train.drop(columns='timestampDate')

for drop_in in reduce_train:

    if drop_in != 'game_session':

        if reduce_train[drop_in].mean() < 0.1:
            reduce_train = reduce_train.drop(columns=drop_in)
            test_s_zero_time = test_s_zero_time.drop(columns=drop_in)
            if flag_train==0:
                reduce_test = reduce_test.drop(columns=drop_in)

for drop_in in reduce_train:
    if drop_in != 'game_session':
        if reduce_train[drop_in].mean() < 0.2:
            test_zero_time = test_zero_time.drop(columns=drop_in)


#######Normal_data1#########################################
# reduce_train = (reduce_train - reduce_train.mean()) / (reduce_train.max() - reduce_train.min())
# reduce_train=reduce_train.fillna(0)
# reduce_train=round(reduce_train*100)
# reduce_train=reduce_train.astype(int)
###########################################################
reduce_test_arr=0
feature_list_train_reduce = list(reduce_train.columns)
if flag_train==0:
    feature_list_test_reduce = list(reduce_test.columns)
    reduce_test_arr = np.array(reduce_test)

reduce_train_arr = np.array(reduce_train)
test_zero_time_arr = np.array(test_zero_time)


test_s_zero_time_arr = np.array(test_s_zero_time)

#######Normal_data2#########################################
test_zero_time_arr = preprocessing.scale(test_zero_time_arr)
test_s_zero_time_arr = preprocessing.scale(test_s_zero_time_arr)
reduce_train_arr = preprocessing.scale(reduce_train_arr)

#############################################################

""" __________________Method PCA___________________________________________"""
from sklearn.decomposition import PCA
pca = PCA(n_components=40)
pca.fit(reduce_train_arr)
var_ratio=pca.explained_variance_ratio_
var=pca.explained_variance_
reduce_train_arr=pca.fit_transform(reduce_train_arr)

"""__________________________________________________________________________"""

reduce_train_arr_0, reduce_train_arr_1, answer_train_0, answer_train_1 = train_test_split(
    reduce_train_arr, answer_train, test_size=0.25, random_state=42)

test_zero_time_arr_0, test_zero_time_arr_1, answer_test_zero__0, answer_test_zero__1 = train_test_split(
    test_zero_time_arr, test_zero_time_answer, test_size=0.25, random_state=42)




"""__________________________Start_train_______________________________"""
# Import the model we are using
#
# Instantiate model with 1000 decision trees


rfc = RandomForestClassifier(n_estimators=3700, random_state=100, min_samples_leaf=7)
log = LogisticRegression(penalty='l2', C=15, random_state=0, max_iter=2200)
voiting = VotingClassifier(estimators=[('rfc', rfc), ('log', log), ])
rf = RandomForestRegressor(n_estimators = 3500, random_state = 100, min_samples_leaf=5 )
# Train the model on training data
rf.fit(reduce_train_arr_0, answer_train_0)
voiting.fit(test_zero_time_arr_0, answer_test_zero__0)

predictions_b = rf.predict(reduce_train_arr_1)

optR = OptimizedRounder()
optR.fit(predictions_b.reshape(-1,),answer_train_1 )
coefficients = optR.coefficients()
predictions = optR.predict(predictions_b.reshape(-1, ), coefficients)
print("\ncoef=",coefficients)
predictions=np.around(predictions)

predictions_z = voiting.predict(test_zero_time_arr_1)

predictions_all = np.concatenate([predictions, predictions_z], axis=None)
answer_train_all = np.concatenate([answer_train_1, answer_test_zero__1], axis=None)

errors = abs(predictions - answer_train_1)
errors_z = abs(predictions_z - answer_test_zero__1)


"""____________________________________________________________________"""

"""_____________________________________Оценка обучения________________"""
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees')

answer_assessment = accuracy_score(answer_train_1, predictions)
print('\nanswer_assessment=', answer_assessment)

answer_assessment_qwk = qwk(answer_train_1, predictions)
print('\nanswer_assessment_qwk=', answer_assessment_qwk)

###########################################################
print('Mean Absolute Error:', round(np.mean(errors_z), 2), 'degrees')

answer_assessment_z = accuracy_score(answer_test_zero__1, predictions_z)
print('\nanswer_assessment_z=', answer_assessment_z)

answer_assessment_qwk_z = qwk(answer_test_zero__1, predictions_z)
print('\nanswer_assessment_qwk_z=', answer_assessment_qwk_z)
###########################################################

##########################################################

answer_assessment_all = accuracy_score(answer_train_all, predictions_all)
print('\nanswer_assessment_all=', answer_assessment_all)

answer_assessment_qwk_all = qwk(answer_train_all, predictions_all)
print('\nanswer_assessment_qwk_all=', answer_assessment_qwk_all)
###########################################################





"""____________________________________________________________________"""

"""______________________________Save Sample Sabmission________________"""
predictions_test=0
if flag_train==0:
    predictions_test = rf.predict(reduce_test_arr)


predictions_z_s = rf.predict(test_s_zero_time_arr)

######################################################
predictions_z_s[predictions_z_s <= coefficients[0]] = 0
predictions_z_s[np.where(np.logical_and(predictions_z_s > coefficients[0], predictions_z_s <= coefficients[1]))] = 1
predictions_z_s[np.where(np.logical_and(predictions_z_s > coefficients[1], predictions_z_s <= coefficients[2]))] = 2
predictions_z_s[predictions_z_s > coefficients[2]] = 3
######################################################
if flag_train==0:
    predictions_all_s = np.concatenate([ predictions_z_s,predictions_test], axis=None)
    installation_id_test = np.concatenate([test_s_zero_time_installation_id,reduce_test_installation_id], axis=None)
else:
    predictions_all_s=predictions_z_s
    installation_id_test=test_s_zero_time_installation_id
data_out = np.column_stack((installation_id_test, predictions_all_s.astype(int)))
#
#
sample_submission_test = pd.DataFrame(data_out, columns=['installation_id', 'accuracy_group'])
#
#
sample_submission_test.to_csv('submission.csv', index=False)

"""____________________________________________________________________"""
print("Finish")
