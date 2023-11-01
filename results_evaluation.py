import os
import pandas as pd
import sqlite3
import numpy as np

#data folder 
os.chdir(r'C:\Users\si_ideation\.data')

#read in data from Hilary Coon. File contains case cohort for IHC (suicide deaths with ICD codes) - 2018 -- 2022 
df_tc = pd.read_csv('true_cases_IH.csv')

#data prep for merge with other data file from Ken 
df_tc = df_tc.rename(columns = {'RedCapID':'REDCAPID'})

#data from ken with more info on cases
df_c = pd.read_csv('Case_w_empi.csv')

df_m = pd.merge(df_tc, df_c, how='left', on='REDCAPID')

df_f = df_m.drop_duplicates()

df_f = df_m.drop_duplicates(subset= 'EMPI')

df_f = df_f.rename(columns = {'CURR_PERSON_MK' : 'person_mk'} )

print(df_f['person_mk'].nunique())

#connect to sqlite db file with data
os.chdir(r'C:\si_ideation\sql')

conn = sqlite3.connect("notes_DB.db")
cur = conn.cursor()


df = pd.read_sql_query("""

SELECT *
FROM si_data_results


""", conn)

#data prep and merge with results
df['person_mk'] = df['person_mk'].astype(np.int64)

df_out = pd.merge(df_f, df, how='left', on='person_mk' )

#look at note types 
k = df_out['event_dsp'].value_counts()
print(k)


#double check the number of unique cases
df_out['NOTE_TEXT'] = df_out['NOTE_TEXT'].replace('', 'NA')
print(df_out['person_mk'].nunique())

#drop results with missing notes
df_out = df_out[df_out.NOTE_TEXT != 'NA']
print(df_out['person_mk'].nunique())

#descriptive stats for individuals. 
#we assume that for those that had at least one note -- at least one of them should've been flagged. 
df_out = df_out.dropna()
df_out['case_flg'] = df_out['case_flg'].astype(np.int64)
count_nt = df_out.groupby('person_mk')['case_flg'].sum()
d = pd.DataFrame(count_nt)
d = d.rename(columns={'case_flg': 'notes_per_person'})
d['notes_per_person'].describe()

#min -- flag for not relevant -- we assume each person should've been flagged at least once
case = df_out.groupby('person_mk')['not_relevant'].min().reset_index()
case['pred'] = 0 #predict not relavant should be 0 for each case --- should have at least one relavent case

#look at missing notes
missing_notes = case[case.isna().any(axis=1)]


case = case.dropna()
case['not_relevant'] = case['not_relevant'].astype(np.int64)
case.not_relevant.value_counts()

#results evaluation 
from sklearn.metrics import accuracy_score,classification_report
print(classification_report(case['pred'], case['not_relevant']))
# print(confusion_matrix(case['pred'], case['not_relevant']))
print(accuracy_score(case['pred'], case['not_relevant']))