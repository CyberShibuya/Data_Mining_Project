import pandas as pd

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)

# Read CGMData and InsulinData csv files
df_CGM = pd.read_csv("CGMData.csv", parse_dates=[['Date', 'Time']], keep_date_col=True, low_memory=False)
df_Insulin = pd.read_csv("InsulinData.csv", parse_dates=[['Date', 'Time']], keep_date_col=True, low_memory=False)

# Find the Auto mode start time interval from InsulinData file
autoModeStart_DateTime = df_Insulin[df_Insulin['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']['Date_Time']

# Delete missing data
df_CGM.dropna(subset=['Sensor Glucose (mg/dL)'], inplace=True)

# Interpolation
# df_CGM['Sensor Glucose (mg/dL)'].interpolate(method='linear', limit_direction='backward', axis=0, inplace=True)

df_CGM['Time'] = pd.to_datetime(df_CGM['Time'])

# Divide data into Auto mode and Manual mode
df_CGM_manual = df_CGM[df_CGM['Date_Time'] < autoModeStart_DateTime.iloc[1]]
df_CGM_auto = df_CGM[df_CGM['Date_Time'] >= autoModeStart_DateTime.iloc[1]]

# Create 18 lists of manual mode
df_CGM_manual_DateGrouped = df_CGM_manual.groupby('Date')

night_hyperglycemia_manual = []
night_hyperglycemiaCritical_manual = []
night_range_manual = []
night_range_secondary_manual = []
night_hypoglycemia_level1_manual = []
night_hypoglycemia_level2_manual = []

day_hyperglycemia_manual = []
day_hyperglycemiaCritical_manual = []
night_hyperglycemiaCritical_manual2 = []
day_range_manual = []
day_range_secondary_manual = []
day_hypoglycemia_level1_manual = []
day_hypoglycemia_level2_manual = []

wholeDay_hyperglycemia_manual = []
wholeDay_hyperglycemiaCritical_manual = []
wholeDay_range_manual = []
wholeDay_range_secondary_manual = []
wholeDay_hypoglycemia_level1_manual = []
wholeDay_hypoglycemia_level2_manual = []
# len_0 = []
# len_1 = []
# len_2 = []
# len_3 = []

# Compute percentage with respect to 24 hours
count_manual_date = 0
for date, df_perDate in df_CGM_manual_DateGrouped:
    if len(df_perDate) > round(288*0.8):
        count_manual_date += 1
        # Divide into 3 different time interval(daytime, overnight, whole day)
        df_daytime = df_perDate[(df_perDate['Time'] <= pd.to_datetime("23:59:59"))
                                & (df_perDate['Time'] >= pd.to_datetime("06:00:00"))]
        df_nighttime = df_perDate[(df_perDate['Time'] < pd.to_datetime("06:00:00"))
                                  & (df_perDate['Time'] >= pd.to_datetime("00:00:00"))]
        df_wholeDaytime = df_perDate[(df_perDate['Time'] <= pd.to_datetime("23:59:59"))
                                     & (df_perDate['Time'] >= pd.to_datetime("00:00:00"))]
        # count_daytime = len(df_daytime['Sensor Glucose (mg/dL)'])
        # count_nighttime = len(df_nighttime['Sensor Glucose (mg/dL)'])
        # count_wholeDaytime = len(df_wholeDaytime['Sensor Glucose (mg/dL)'])

        day_hyperglycemia_manual.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] > 180])/len(df_daytime)*0.75)
        day_hyperglycemiaCritical_manual.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] > 250])/len(df_daytime)*0.75)
        day_range_manual.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')])/len(df_daytime)*0.75)
        day_range_secondary_manual.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')])/len(df_daytime)*0.75)
        day_hypoglycemia_level1_manual.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] < 70])/len(df_daytime)*0.75)
        day_hypoglycemia_level2_manual.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] < 54])/len(df_daytime)*0.75)

        night_hyperglycemia_manual.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] > 180]) /len(df_nighttime)*0.25)
        night_hyperglycemiaCritical_manual.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] > 250]) /len(df_nighttime)*0.25)
        night_range_manual.append(
            len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')]) /len(df_nighttime)*0.25)
        night_range_secondary_manual.append(
            len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')]) /len(df_nighttime)*0.25)
        night_hypoglycemia_level1_manual.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] < 70]) /len(df_nighttime)*0.25)
        night_hypoglycemia_level2_manual.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] < 54]) /len(df_nighttime)*0.25)

        # len_0.append(len(df_nighttime))
        # len_1.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] > 180]))
        # len_2.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')]))
        # len_3.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] < 70]))

        wholeDay_hyperglycemia_manual.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] > 180]) /len(df_wholeDaytime))
        wholeDay_hyperglycemiaCritical_manual.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] > 250]) /len(df_wholeDaytime))
        wholeDay_range_manual.append(
            len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')]) /len(df_wholeDaytime))
        wholeDay_range_secondary_manual.append(
            len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')]) /len(df_wholeDaytime))
        wholeDay_hypoglycemia_level1_manual.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] < 70]) /len(df_wholeDaytime))
        wholeDay_hypoglycemia_level2_manual.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] < 54]) /len(df_wholeDaytime))
# print(len_1)
# print(len_2)
# print(len_3)
# print(len_0)
# Create 18 lists of auto mode
df_CGM_auto_DateGrouped = df_CGM_auto.groupby('Date')

night_hyperglycemia_auto = []
night_hyperglycemiaCritical_auto = []
night_range_auto = []
night_range_secondary_auto = []
night_hypoglycemia_level1_auto = []
night_hypoglycemia_level2_auto = []

day_hyperglycemia_auto = []
day_hyperglycemiaCritical_auto = []
day_range_auto = []
day_range_secondary_auto = []
day_hypoglycemia_level1_auto = []
day_hypoglycemia_level2_auto = []

wholeDay_hyperglycemia_auto = []
wholeDay_hyperglycemiaCritical_auto = []
wholeDay_range_auto = []
wholeDay_range_secondary_auto = []
wholeDay_hypoglycemia_level1_auto = []
wholeDay_hypoglycemia_level2_auto = []

count_auto_date = 0

for date, df_perDate in df_CGM_auto_DateGrouped:
    if len(df_perDate) > round(288*0.8):
        count_auto_date += 1
        # Divide into 3 different time interval(daytime, overnight, whole day)
        df_daytime = df_perDate[(df_perDate['Time'] <= pd.to_datetime("23:59:59"))
                                & (df_perDate['Time'] >= pd.to_datetime("06:00:00"))]
        df_nighttime = df_perDate[(df_perDate['Time'] < pd.to_datetime("06:00:00"))
                                  & (df_perDate['Time'] >= pd.to_datetime("00:00:00"))]
        df_wholeDaytime = df_perDate[(df_perDate['Time'] <= pd.to_datetime("23:59:59"))
                                     & (df_perDate['Time'] >= pd.to_datetime("00:00:00"))]

        day_hyperglycemia_auto.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] > 180])/len(df_daytime)*0.75)
        day_hyperglycemiaCritical_auto.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] > 250])/len(df_daytime)*0.75)
        day_range_auto.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')])/len(df_daytime)*0.75)
        day_range_secondary_auto.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')])/len(df_daytime)*0.75)
        day_hypoglycemia_level1_auto.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] < 70])/len(df_daytime)*0.75)
        day_hypoglycemia_level2_auto.append(len(df_daytime[df_daytime['Sensor Glucose (mg/dL)'] < 54])/len(df_daytime)*0.75)

        night_hyperglycemia_auto.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] > 180]) /len(df_nighttime)*0.25)
        night_hyperglycemiaCritical_auto.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] > 250]) /len(df_nighttime)*0.25)
        night_range_auto.append(
            len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')]) /len(df_nighttime)*0.25)
        night_range_secondary_auto.append(
            len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')]) /len(df_nighttime)*0.25)
        night_hypoglycemia_level1_auto.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] < 70]) /len(df_nighttime)*0.25)
        night_hypoglycemia_level2_auto.append(len(df_nighttime[df_nighttime['Sensor Glucose (mg/dL)'] < 54]) /len(df_nighttime)*0.25)

        wholeDay_hyperglycemia_auto.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] > 180]) /len(df_wholeDaytime))
        wholeDay_hyperglycemiaCritical_auto.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] > 250]) /len(df_wholeDaytime))
        wholeDay_range_auto.append(
            len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'].between(70, 180, inclusive='both')]) /len(df_wholeDaytime))
        wholeDay_range_secondary_auto.append(
            len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'].between(70, 150, inclusive='both')]) /len(df_wholeDaytime))
        wholeDay_hypoglycemia_level1_auto.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] < 70]) /len(df_wholeDaytime))
        wholeDay_hypoglycemia_level2_auto.append(len(df_wholeDaytime[df_wholeDaytime['Sensor Glucose (mg/dL)'] < 54]) /len(df_wholeDaytime))

# Create Dataframe of the result with two modes
column_names = ['Modes',
                'Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)',
                'Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
                'Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
                'Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
                'Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
                'Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)',
                'Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)',
                'Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
                'Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
                'Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
                'Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
                'Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)',
                'WholeDay Percentage time in hyperglycemia (CGM > 180 mg/dL)',
                'WholeDay percentage of time in hyperglycemia critical (CGM > 250 mg/dL)',
                'WholeDay percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)',
                'WholeDay percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)',
                'WholeDay percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)',
                'WholeDay percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)']

df_result = pd.DataFrame(columns=column_names)
df_result['Modes'] = ['Manual Mode', 'Auto Mode']


# calculate the final percentage
def percent(list_value, count_date):
    return round(sum(list_value)/count_date*100, 2)


df_result['Daytime Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = \
    [percent(day_hyperglycemia_manual, count_manual_date), percent(day_hyperglycemia_auto, count_auto_date)]
df_result['Daytime percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = \
    [percent(day_hyperglycemiaCritical_manual, count_manual_date), percent(day_hyperglycemiaCritical_auto, count_auto_date)]
df_result['Daytime percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = \
    [percent(day_range_manual, count_manual_date), percent(day_range_auto, count_auto_date)]
df_result['Daytime percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = \
    [percent(day_range_secondary_manual, count_manual_date), percent(day_range_secondary_auto, count_auto_date)]
df_result['Daytime percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = \
    [percent(day_hypoglycemia_level1_manual, count_manual_date), percent(day_hypoglycemia_level1_auto, count_auto_date)]
df_result['Daytime percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = \
    [percent(day_hypoglycemia_level2_manual, count_manual_date), percent(day_hypoglycemia_level2_auto, count_auto_date)]

df_result['Overnight Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = \
    [percent(night_hyperglycemia_manual, count_manual_date), percent(night_hyperglycemia_auto, count_auto_date)]
df_result['Overnight percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = \
    [percent(night_hyperglycemiaCritical_manual, count_manual_date), percent(night_hyperglycemiaCritical_auto, count_auto_date)]
df_result['Overnight percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = \
    [percent(night_range_manual, count_manual_date), percent(night_range_auto, count_auto_date)]
df_result['Overnight percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = \
    [percent(night_range_secondary_manual, count_manual_date), percent(night_range_secondary_auto, count_auto_date)]
df_result['Overnight percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = \
    [percent(night_hypoglycemia_level1_manual, count_manual_date), percent(night_hypoglycemia_level1_auto, count_auto_date)]
df_result['Overnight percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = \
    [percent(night_hypoglycemia_level2_manual, count_manual_date), percent(night_hypoglycemia_level2_auto, count_auto_date)]

df_result['WholeDay Percentage time in hyperglycemia (CGM > 180 mg/dL)'] = \
    [percent(wholeDay_hyperglycemia_manual, count_manual_date), percent(wholeDay_hyperglycemia_auto, count_auto_date)]
df_result['WholeDay percentage of time in hyperglycemia critical (CGM > 250 mg/dL)'] = \
    [percent(wholeDay_hyperglycemiaCritical_manual, count_manual_date), percent(wholeDay_hyperglycemiaCritical_auto, count_auto_date)]
df_result['WholeDay percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)'] = \
    [percent(wholeDay_range_manual, count_manual_date), percent(wholeDay_range_auto, count_auto_date)]
df_result['WholeDay percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)'] = \
    [percent(wholeDay_range_secondary_manual, count_manual_date), percent(wholeDay_range_secondary_auto, count_auto_date)]
df_result['WholeDay percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)'] = \
    [percent(wholeDay_hypoglycemia_level1_manual, count_manual_date), percent(wholeDay_hypoglycemia_level1_auto, count_auto_date)]
df_result['WholeDay percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)'] = \
    [percent(wholeDay_hypoglycemia_level2_manual, count_manual_date), percent(wholeDay_hypoglycemia_level2_auto, count_auto_date)]

df_result.set_index('Modes', inplace=True)

# Write the result csv file
df_result.to_csv('Results.csv', index=False, header=False)
print(count_auto_date, count_manual_date)

