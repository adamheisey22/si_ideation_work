import sqlite3
import pandas as pd
import os
import support_functions

#sql light db folder location for note types for cases and controls
os.chdir(r'C:\sql-light-db-folder-location')

#connect to the database file
conn = sqlite3.connect("notes_DB.db")
cur = conn.cursor()

#pull all data into dataframe df
df = pd.read_sql_query("""

SELECT *
FROM si_data_all


""", conn)

#filters- cleans html/taggers and returns raw text 
df_proc = support_functions.clean_html(df, 'blob_contents_decoded_no_tag')

#scrubs out the suicide ideation disclaimer statement. *there are  a couple note super elegant functions for this
df_scrub = support_functions.scrub_disclaimer(df_proc, 'NOTE_TEXT')

#srubs out the CSSR form -- functional for this analysis -- needs improvement
df_scrub2 = support_functions.cssr_scrub(df_scrub, 'NOTE_TEXT')

#df prep for output to csv or write to sql dabase file
df_out = df_scrub2.drop(columns=['blob_contents_decoded_no_tag','RAW_TEXT','TEXT_LINES'])

#write output to new table in sqlite DB file
#df_out.to_sql(name="si_data_all_clean", con=conn, if_exists='append', index = False)

#df_out.to_csv('outputfile.csv', escapechar=' ')