import pandas as pd
import numpy as np
from datetime import timedelta
import math
from scipy.fftpack import rfft
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import entropy


insulin_df = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
CGM_df = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

insulin_df['datetime'] = pd.to_datetime(insulin_df['Date'] + ' ' + insulin_df['Time'])
CGM_df['datetime'] = pd.to_datetime(CGM_df['Date'] + ' ' + CGM_df['Time'])

CGM_df['Sensor Glucose (mg/dL)'] = CGM_df['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction='both', axis=0)

insulin_df = insulin_df[::-1]
# CGM_df = CGM_df[::-1]


def get_meals(i_df, c_df):
    carb_df = i_df[i_df['BWZ Carb Input (grams)'] > 0]
    carb_df = carb_df.reset_index().drop(columns='index')

    valid_time_list = []
    b_list = []
    carb_max = carb_df["BWZ Carb Input (grams)"].max()
    carb_min = carb_df["BWZ Carb Input (grams)"].min()
    bins = math.ceil((carb_max - carb_min)/20)
    for index, time in enumerate(carb_df['datetime']):
        try:
            minutes_gap = (carb_df['datetime'][index + 1] - time).total_seconds() / 60
            if minutes_gap >= 120:
                valid_time_list.append(time)
                current_bin = math.ceil((carb_df.iloc[index]["BWZ Carb Input (grams)"] - carb_min) / 20)
                b_list.append(current_bin)
        except KeyError:
            break

    cgm_list = []

    for time in valid_time_list:
        start = time - timedelta(minutes=30)
        end = time + timedelta(minutes=120)

        cgm_date_row = c_df[c_df['Date'] == time.strftime('%-m/%-d/%Y')]

        cgm_date_row = cgm_date_row.set_index("datetime")
        cgm_time_row = cgm_date_row.between_time(start.strftime('%H:%M:%S'), end.strftime('%H:%M:%S'))

        cgm_list.append(cgm_time_row['Sensor Glucose (mg/dL)'].values.tolist())
    cgm_df = pd.DataFrame(cgm_list)
    return cgm_df, b_list, bins


meal_data, bin_list, bin_nums = get_meals(insulin_df, CGM_df)
meal_data = meal_data.iloc[:, 0:30]


# create feature matrix with a bin column
def create_meal_matrix(m_data, b_list):
    # clean data : drop nan >6 的row（sum isna的true(nan)）
    drop_index = m_data.isna().sum(axis=1).where(lambda x: x > 6).dropna().index
    drop_list = drop_index.values.tolist()
    b_dict = {}
    for i, v in enumerate(b_list):
        b_dict[i] = v
    for i in drop_list:
        del b_dict[i]
    new_b_list = list(b_dict.values())

    m_data_cleaned = m_data.drop(index=drop_index).reset_index().drop(columns='index')
    m_data_cleaned = m_data_cleaned.interpolate(method='linear', axis=1)

    power_1th_list = []
    power_2nd_list = []
    HZ_1th_list = []
    HZ_2nd_list = []
    meal_feature_matrix = pd.DataFrame()
    m_max = []
    m_min = []

    for i in range(len(m_data_cleaned)):
        rfft_values = abs(rfft(m_data_cleaned.iloc[i].tolist())).tolist()
        rfft_sorted = sorted(rfft_values)
        rfft_sorted.sort(reverse=True)
        power_1th_list.append(rfft_sorted[1])
        power_2nd_list.append(rfft_sorted[2])
        HZ_1th_list.append(rfft_values.index(rfft_sorted[1]))
        HZ_2nd_list.append(rfft_values.index(rfft_sorted[2]))
        m_min.append(min(m_data_cleaned.iloc[i]))
        m_max.append(max(m_data_cleaned.iloc[i]))

    meal_feature_matrix['power_1th'] = power_1th_list
    meal_feature_matrix['power_2nd'] = power_2nd_list
    meal_feature_matrix['HZ_1th'] = HZ_1th_list
    meal_feature_matrix['HZ_2nd'] = HZ_2nd_list

    tm = m_data_cleaned.iloc[:, 22:25].idxmin(axis=1)
    t_max = m_data_cleaned.iloc[:, 5:19].idxmax(axis=1)

    second_differential_data = []
    standard_deviation = []

    for i in range(len(m_data_cleaned)):
        second_differential_data.append(np.diff(np.diff(m_data_cleaned.iloc[:, t_max[i]:tm[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(np.diff(m_data_cleaned.iloc[i])))

    meal_feature_matrix['2ndDifferential'] = second_differential_data
    meal_feature_matrix['standard_deviation'] = standard_deviation
    return meal_feature_matrix, new_b_list, m_min, m_max


meal_feature_matrix, new_bin_list, meal_min, meal_max = create_meal_matrix(meal_data, bin_list)
meal_feature_matrix_np = meal_feature_matrix.to_numpy()
std_df = StandardScaler().fit_transform(meal_feature_matrix)

pca = PCA(n_components=6)
pca.fit(std_df)
pca_np = pca.fit_transform(std_df)


df = pd.DataFrame()
# Kmeans clustering
kmeans_c = KMeans(n_clusters=int(bin_nums), random_state=0)
kmeans_c.fit_predict(pca_np)
k_labels = list(kmeans_c.labels_)

df['kmeans_clusters'] = k_labels
df['bins'] = new_bin_list


# 2 kinds of clusters
def ground_truth_matrix(ground_truth, clusters, b_num):
    cluster_matrix = np.zeros((clusters.max()+1, b_num))
    for i, b in enumerate(ground_truth):
        c = clusters[i]
        cluster_matrix[c, b-1] += 1
    return cluster_matrix


def calculate_entropy(matrix):
    entropy_list = []
    for c in matrix:
        entropy_list.append(entropy(c, base=2))
    return entropy_list


def calculate_purity(matrix):
    purity_list = []
    for c in matrix:
        c = np.array(c)
        purity = (c / c.sum()).max()
        purity_list.append(purity)
    return purity_list


def get_sse(matrix):
    sse = []
    for i in range(len(matrix)):
        mean_i = np.array(matrix[i]).mean()
        se = []
        for j in np.array(matrix[i]):
            se.append(abs(j-mean_i)**2)
        sse.append(np.array(se).sum())
    return np.array(sse).sum()


matrix_c = ground_truth_matrix(df['bins'], df['kmeans_clusters'], bin_nums)
entropy_c = calculate_entropy(matrix_c)
purity_c = calculate_purity(matrix_c)

sum_c = np.array([c.sum() for c in matrix_c])
percent_c = sum_c / sum_c.sum()

sse_kmeans = kmeans_c.inertia_
entropy_kmeans = (np.array(entropy_c) * percent_c).sum()
purity_kmeans = (np.array(purity_c) * percent_c).sum()

# DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=3)
dbscan.fit_predict(pca_np)

df_dbscan = pd.DataFrame()
df_dbscan['dbscan_clustering'] = dbscan.labels_
df_dbscan['bins'] = new_bin_list
# Delete -1 in the df
df_dbscan = df_dbscan[df_dbscan['dbscan_clustering'] != -1]
df_dbscan = df_dbscan.reset_index().drop(columns='index')
# print(df_dbscan)
matrix_c_db = ground_truth_matrix(df_dbscan['bins'], df_dbscan['dbscan_clustering'], bin_nums)
# print(matrix_c_db)

entropy_c_db = calculate_entropy(matrix_c_db)
purity_c_db = calculate_purity(matrix_c_db)

sum_c_db = np.array([c.sum() for c in matrix_c_db])
percent_c_db = sum_c_db / sum_c_db.sum()

entropy_dbscan = (np.array(entropy_c_db) * percent_c_db).sum()
purity_dbscan = (np.array(purity_c_db) * percent_c_db).sum()
sse_dbscan = get_sse(matrix_c_db)
# sse_dbscan = 0
# print(sse_dbscan)
df_output = pd.DataFrame(
    [
        [
            sse_kmeans,
            sse_dbscan,
            entropy_kmeans,
            entropy_dbscan,
            purity_kmeans,
            purity_dbscan,
        ]
    ],
    columns=[
        "K-Means SSE",
        "DBSCAN SSE",
        "K-Means entropy",
        "DBSCAN entropy",
        "K-Means purity",
        "DBSCAN purity",
    ],
)
df_output = df_output.fillna(0)
df_output.to_csv("Result.csv", index=False, header=False)
