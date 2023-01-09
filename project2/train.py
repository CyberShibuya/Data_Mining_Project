import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import rfft
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier


# read sebset csv(usecols:only read subset, low_memory:不chuck简化处理)
insulin_df = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
CGM_df = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
insulin_df_2 = pd.read_csv('Insulin_patient2.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
CGM_df_2 = pd.read_csv('CGM_patient2.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

# to datetime
insulin_df['datetime'] = pd.to_datetime(insulin_df['Date'] + ' ' + insulin_df['Time'])
CGM_df['datetime'] = pd.to_datetime(CGM_df['Date'] + ' ' + CGM_df['Time'])
insulin_df_2['datetime'] = pd.to_datetime(insulin_df_2['Date'] + ' ' + insulin_df_2['Time'])
CGM_df_2['datetime'] = pd.to_datetime(CGM_df_2['Date'] + ' ' + CGM_df_2['Time'])

# Interpolate NaN in CGM data
CGM_df['Sensor Glucose (mg/dL)'] = CGM_df['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction='both', axis=0)
CGM_df_2['Sensor Glucose (mg/dL)'] = CGM_df_2['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction='both', axis=0)

# reverse row(before to after）
insulin_df = insulin_df[::-1]
insulin_df_2 = insulin_df_2[::-1]
# CGM_df = CGM_df[::-1]


def create_meal_data(i_df, c_df, num):
    # drop NaN and 0.0
    # i_df['BWZ Carb Input (grams)'].replace(0, np.nan, inplace=True)
    # df_2_5h = i_df.dropna(subset='BWZ Carb Input (grams)')
    df_2_5h = i_df[i_df['BWZ Carb Input (grams)'] > 0]
    #  reset index(dropna didn't change index and reset index will add a index from old index)
    df_2_5h = df_2_5h.reset_index().drop(columns='index')

    #  delete 无效time(time gap < 2 hours), create valid list(無法直接df裡面operate）
    valid_time_list = []
    for index, time in enumerate(df_2_5h['datetime']):  # enumerate变有序号list.(用i來range也可）
        try:
            minutes_gap = (df_2_5h['datetime'][index + 1] - time).total_seconds() / 60
            if minutes_gap >= 120:
                valid_time_list.append(time)
        except KeyError:
            break

    cgm_list = []

    for time in valid_time_list:
        # list time is not datetime any more
        start = time - timedelta(minutes=30)
        end = time + timedelta(minutes=120)
        # extract subset row CGM by between_time(by time) 而 between(by series)
        # between_time 只能比較time（datetime.time() or datetime.strftime('%H:%M:%S'),不能date! (好在mean time不跨天)
        # CGM_df['Date'] = pd.to_datetime(CGM_df['Date'])
        if num == 1:
            cgm_date_row = c_df[c_df['Date'] == time.strftime('%-m/%-d/%Y')]
        else:
            cgm_date_row = c_df[c_df['Date'] == time.strftime('%Y-%m-%d')]
        cgm_date_row = cgm_date_row.set_index("datetime")
        cgm_time_row = cgm_date_row.between_time(start.strftime('%H:%M:%S'), end.strftime('%H:%M:%S'))
        # append all value lists into one nest value list
        cgm_list.append(cgm_time_row['Sensor Glucose (mg/dL)'].values.tolist())
    return pd.DataFrame(cgm_list)  # 每組value list為一個row


meal_data = create_meal_data(insulin_df, CGM_df, 1)
meal_data_2 = create_meal_data(insulin_df_2, CGM_df_2, 0)
meal_data = meal_data.iloc[:, 0:30]  # 選每組24個value
meal_data_2 = meal_data_2.iloc[:, 0:30]


def create_no_meal_data(i_df, c_df, num):
    df_2h = i_df[i_df['BWZ Carb Input (grams)'] > 0]
    df_2h = df_2h.reset_index().drop(columns='index')

    valid_time_list = []
    for index, time in enumerate(df_2h['datetime']):
        try:
            minutes_gap = (df_2h['datetime'][index + 1] - time).total_seconds() / 60
            if minutes_gap >= 240:
                valid_time_list.append(time)
        except KeyError:
            break

    cgm_list = []
    for i, time in enumerate(valid_time_list):
        # meal time之後可有多個2h的no meal series（比如6h拆為3個no meal）
        try:
            num_series = (valid_time_list[i+1]-valid_time_list[i]).total_seconds()//7200
        except IndexError:
            break

        for j in range(int(num_series)):
            start = pd.to_datetime(time + timedelta(minutes=j*120))
            end = pd.to_datetime(time + timedelta(minutes=(j+1)*120-5))
            if num == 1:
                cgm_date_row = c_df[c_df['Date'] == time.strftime('%-m/%-d/%Y')]
            else:
                cgm_date_row = c_df[c_df['Date'] == time.strftime('%Y-%m-%d')]
            cgm_date_row = cgm_date_row.set_index("datetime")
            cgm_time_row = cgm_date_row.between_time(start.strftime('%H:%M:%S'), end.strftime('%H:%M:%S'))
            cgm_list.append(cgm_time_row['Sensor Glucose (mg/dL)'].values.tolist())
    return pd.DataFrame(cgm_list)


no_meal_data = create_no_meal_data(insulin_df, CGM_df, 1)
no_meal_data = no_meal_data.iloc[:, 0:24]
no_meal_data_2 = create_no_meal_data(insulin_df_2, CGM_df_2, 0)
no_meal_data_2 = no_meal_data_2.iloc[:, 0:24]


def create_meal_matrix(m_data):
    # clean data : drop nan >6 的row（sum isna的true(nan)）
    drop_index = m_data.isna().sum(axis=1).where(lambda x: x > 6).dropna().index
    m_data_cleaned = m_data.drop(index=drop_index).reset_index().drop(columns='index')
    m_data_cleaned = m_data_cleaned.interpolate(method='linear', axis=1)

    power_1th_list = []
    power_2nd_list = []
    HZ_1th_list = []
    HZ_2nd_list = []
    meal_feature_matrix = pd.DataFrame()

    for i in range(len(m_data_cleaned)):
        # rfft 之後是 np array
        rfft_values = abs(rfft(m_data_cleaned.iloc[i].tolist())).tolist()
        rfft_sorted = sorted(rfft_values)
        rfft_sorted.sort(reverse=True) # power 由大到小
        # choose 前三大power(magnitude震幅)
        power_1th_list.append(rfft_sorted[1])
        power_2nd_list.append(rfft_sorted[2])
        # power_3rd_list.append(rfft_sorted[3])
        # 大power對應的頻率HZ(by using python index('value'))
        HZ_1th_list.append(rfft_values.index(rfft_sorted[1]))
        HZ_2nd_list.append(rfft_values.index(rfft_sorted[2]))

    meal_feature_matrix['power_1th'] = power_1th_list
    meal_feature_matrix['power_2nd'] = power_2nd_list
    # meal_feature_matrix['power_3rd'] = power_3rd_list
    meal_feature_matrix['HZ_1th'] = HZ_1th_list
    meal_feature_matrix['HZ_2nd'] = HZ_2nd_list

    # 22:25 ???? 5:19 ????
    # 選 column index
    tm = m_data_cleaned.iloc[:, 22:25].idxmin(axis=1)
    t_max = m_data_cleaned.iloc[:, 5:19].idxmax(axis=1)

    second_differential_data = []
    standard_deviation = []

    for i in range(len(m_data_cleaned)):
        second_differential_data.append(np.diff(np.diff(m_data_cleaned.iloc[:, t_max[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(np.diff(m_data_cleaned.iloc[i])))

    meal_feature_matrix['2ndDifferential'] = second_differential_data
    meal_feature_matrix['standard_deviation'] = standard_deviation
    return meal_feature_matrix


meal_feature_matrix = create_meal_matrix(meal_data)
meal_feature_matrix_2 = create_meal_matrix(meal_data_2)
meal_feature_matrix = pd.concat([meal_feature_matrix, meal_feature_matrix_2])


# # diif_time no_diff
# diff_time_list = []
# no_diff_list = []
# for i in range(len(meal_data_cleaned)):
#     diff_time_list.append(t_max.iloc[i] - tm.iloc[i])
#     no_diff_list.append((meal_data_cleaned.iloc[i, t_max.iloc[i]] - meal_data_cleaned.iloc[i, tm.iloc[i]])
#                           / meal_data_cleaned.iloc[i, tm.iloc[i]])
# meal_feature_matrix['diff_time'] = diff_time_list
# meal_feature_matrix['no_diff'] = no_diff_list


def create_no_meal_matrix(no_m_data):
    drop_index = no_m_data.isna().sum(axis=1).where(lambda x: x > 6).dropna().index
    no_m_data_cleaned = no_m_data.drop(index=drop_index).reset_index().drop(columns='index')
    no_m_data_cleaned = no_m_data_cleaned.interpolate(method='linear', axis=1)

    power_1th_list = []
    power_2nd_list = []
    HZ_1th_list = []
    HZ_2nd_list = []
    no_meal_feature_matrix = pd.DataFrame()

    for i in range(len(no_m_data_cleaned)):
        rfft_values = abs(rfft(no_m_data_cleaned.iloc[i].tolist())).tolist()
        rfft_sorted = sorted(rfft_values)
        rfft_sorted.sort(reverse=True)
        power_1th_list.append(rfft_sorted[1])
        power_2nd_list.append(rfft_sorted[2])
        HZ_1th_list.append(rfft_values.index(rfft_sorted[1]))
        HZ_2nd_list.append(rfft_values.index(rfft_sorted[2]))

    no_meal_feature_matrix['power_1th'] = power_1th_list
    no_meal_feature_matrix['power_2nd'] = power_2nd_list
    no_meal_feature_matrix['HZ_1th'] = HZ_1th_list
    no_meal_feature_matrix['HZ_2nd'] = HZ_2nd_list

    second_differential_data = []
    standard_deviation = []

    for i in range(len(no_m_data_cleaned)):
        second_differential_data.append(np.diff(np.diff(no_m_data_cleaned.iloc[i, 0:24].tolist())).max())
        standard_deviation.append(np.std(np.diff(no_m_data_cleaned.iloc[i, 0:24])))

    no_meal_feature_matrix['2ndDifferential'] = second_differential_data
    no_meal_feature_matrix['standard_deviation'] = standard_deviation
    return no_meal_feature_matrix


no_meal_feature_matrix = create_no_meal_matrix(no_meal_data)
no_meal_feature_matrix_2 = create_no_meal_matrix(no_meal_data_2)
no_meal_feature_matrix = pd.concat([no_meal_feature_matrix, no_meal_feature_matrix_2])


# Train
std_df = StandardScaler().fit_transform(meal_feature_matrix)
std_no_df = StandardScaler().fit_transform(no_meal_feature_matrix)

pca = PCA(n_components=6)

pca.fit(std_df)
pca_df = pd.DataFrame(pca.fit_transform(std_df))

pca.fit(std_no_df)
pca_no_df = pd.DataFrame(pca.fit_transform(std_no_df))

pca_df['class'] = 1
pca_no_df['class'] = 0
total_matrix = pd.concat([pca_df, pca_no_df]).reset_index().drop(columns='index')

dataset = shuffle(total_matrix, random_state=1).reset_index().drop(columns='index')
dataset_no_class = dataset.drop(columns='class')

kf = KFold(n_splits=10)
model = DecisionTreeClassifier(criterion='entropy')
# model = MLPClassifier(activation='logistic', random_state=1, max_iter=60)

for train_index, test_index in kf.split(dataset_no_class):
    X_train, X_test = dataset_no_class.iloc[train_index], dataset_no_class.iloc[test_index]
    y_train, y_test = dataset['class'].iloc[train_index], dataset['class'].iloc[test_index]
    model.fit(X_train, y_train)

dump(model, 'Classifier.pickle')
