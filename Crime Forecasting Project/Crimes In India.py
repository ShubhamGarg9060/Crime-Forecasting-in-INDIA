import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
import mapclassify as mcl
from matplotlib.colors import Normalize
import geopandas as gdp
from statsmodels.tsa.arima.model import ARIMA

#from statsmodels.tsa.arima_model import ARIMA
#statsmodels.tsa.arima.model.ARIMA
#from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


crimes_df = pd.read_csv("01_District_wise_crimes_committed_IPC_2001_2012.csv")

crimes_main_df = crimes_df.copy()


crimes_df["TOTAL_THEFT"] = crimes_df["THEFT"] + crimes_df["OTHER THEFT"] + crimes_df["AUTO THEFT"]


fp = "india_st.shp"
map_India = gdp.read_file(fp)



#map_India['STATE'] = map_India['STATE'].replace(['Andaman & Nicobar Island', "Andhra Pradesh", 'Arunanchal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'Dadara & Nagar Havelli', 'Daman & Diu', 'NCT of Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
#       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'], 
#                                                ["A & N ISLANDS", "ANDHRA PRADESH", 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'CHANDIGARH', 'CHHATTISGARH', 'D & N HAVELI', 'DAMAN & DIU', 'DELHI UT', 'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'JHARKHAND', 'KARNATAKA', 'KERALA', 'LAKSHADWEEP', 'MADHYA PRADESH', 'MAHARASHTRA', 'MANIPUR',
#       'MEGHALAYA', 'MIZORAM', 'NAGALAND','ODISHA', 'PUDUCHERRY', 'PUNJAB', 'RAJASTHAN', 'SIKKIM', 'TAMIL NADU', 'TRIPURA', 'UTTAR PRADESH', 'UTTARAKHAND',  'WEST BENGAL'])



map_India['STATE'] = map_India['STATE'].replace(['ANDAMAN AND NICOBAR ISLANDS', 'DADRA AND NAGAR HAVELI', 'DAMAN AND DIU' , 'DELHI' , 'JAMMU AND KASHMIR', 'ORISSA', 'PONDICHERRY'], 
                                                ["A & N ISLANDS" , 'D & N HAVELI', 'DAMAN & DIU', 'DELHI UT', 'JAMMU & KASHMIR', 'ODISHA', 'PUDUCHERRY'])
  


crimes_df.head(5)


crimes_df.columns


# In[2863]:


crimes_df = crimes_df.rename(columns={'STATE/UT': 'STATE'})


# In[2864]:


crimes_df = crimes_df.rename(columns={'TOTAL IPC CRIMES': 'TOTAL'})


# In[2865]:


crimes_df['STATE'] = crimes_df['STATE'].replace(['A&N Islands', "Andhra Pradesh", 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh', 'D&N Haveli', 'Daman & Diu', 'Delhi UT', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
       'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'], 
                                                ["A & N ISLANDS", "ANDHRA PRADESH", 'ARUNACHAL PRADESH', 'ASSAM', 'BIHAR', 'CHANDIGARH', 'CHHATTISGARH', 'D & N HAVELI', 'DAMAN & DIU', 'DELHI UT', 'GOA', 'GUJARAT', 'HARYANA', 'HIMACHAL PRADESH', 'JAMMU & KASHMIR', 'JHARKHAND', 'KARNATAKA', 'KERALA', 'LAKSHADWEEP', 'MADHYA PRADESH', 'MAHARASHTRA', 'MANIPUR',
       'MEGHALAYA', 'MIZORAM', 'NAGALAND','ODISHA', 'PUDUCHERRY', 'PUNJAB', 'RAJASTHAN', 'SIKKIM', 'TAMIL NADU', 'TRIPURA', 'UTTAR PRADESH', 'UTTARAKHAND',  'WEST BENGAL'])
 


# In[2866]:


crimes_df['DISTRICT'] = crimes_df['DISTRICT'].replace(["ZZ TOTAL"], ["TOTAL"])


# In[2867]:


crimes_df['DISTRICT'] = crimes_df['DISTRICT'].replace(["SOUTH-EAST", "SOUTH-WEST", "NORTH-EAST", "NORTH-WEST",
                                                     "IGI AIRPORT", "GRP(RLY)", "STF", 'DELHI UT TOTAL'], ["SOUTH EAST", "SOUTH WEST", "NORTH EAST", "NORTH WEST",
                                                     "I.G.I. AIRPORT", "G.R.P.(RLY)", "S.T.F.", 'TOTAL'])


# In[2868]:


crimes_df['DISTRICT'] = crimes_df['DISTRICT'].replace(["HOWRAH CITY"], ["HOWRAH"])


# In[2869]:


crimes_df['DISTRICT'] = crimes_df['DISTRICT'].replace(["HOWRAH G.R.P."], ["HOWRAH"])


# In[2870]:


crimes_df['DISTRICT'] = crimes_df['DISTRICT'].replace(['G.R.P. AJMER', 'G.R.P. JODHPUR'],['G.R.P.AJMER', 'G.R.P.JODHPUR'])


# In[2871]:


crimes_df.rename(columns={'HURT/GREVIOUS HURT': 'HURT', 'KIDNAPPING & ABDUCTION': 'KIDNAPPING', 'CAUSING DEATH BY NEGLIGENCE':'DEATH_BY_NEGLIGENCE', 
                          'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY':'ASSAULT_WOMEN_MODESTY','CRUELTY BY HUSBAND OR HIS RELATIVES': 'CRUELTY_BY_HUSBAND', 
                         'CRIMINAL BREACH OF TRUST':'CRIMINAL_BREACH_OF_TRUST'}, inplace=True)


# In[2872]:


crimes_df['STATE'].unique()


# In[2873]:


crimes_df.shape


# In[2874]:


crimes_df.info()


# In[2875]:


crimes_state = crimes_df.groupby("STATE").agg({'TOTAL_THEFT':sum,'HURT':sum,'KIDNAPPING':sum,'CRUELTY_BY_HUSBAND':sum, 
                                                       'BURGLARY':sum, 'MURDER':sum,'DEATH_BY_NEGLIGENCE':sum,
                                                      'CHEATING':sum, 'RIOTS':sum, 'ASSAULT_WOMEN_MODESTY':sum ,'OTHER IPC CRIMES':sum, 'TOTAL':sum}).reset_index()
       


# In[2876]:


merged_Ind = map_India.set_index('STATE').join(crimes_state.set_index('STATE'))
merged_Ind.head()


# In[2877]:


merged_Ind.isna().sum()


# In[2878]:


crimes_distribution = crimes_df[crimes_df.DISTRICT=='TOTAL']


# In[2879]:


crimes_distribution = crimes_distribution.groupby(["YEAR"]).agg({'MURDER':sum, 'ATTEMPT TO MURDER':sum,
       'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER':sum, 'RAPE':sum, 'CUSTODIAL RAPE':sum,
       'OTHER RAPE':sum, 'KIDNAPPING':sum,
       'KIDNAPPING AND ABDUCTION OF WOMEN AND GIRLS':sum,
       'KIDNAPPING AND ABDUCTION OF OTHERS':sum, 'DACOITY':sum,
       'PREPARATION AND ASSEMBLY FOR DACOITY':sum, 'ROBBERY':sum, 'BURGLARY':sum,
        'RIOTS':sum, 'CRIMINAL_BREACH_OF_TRUST':sum,
       'CHEATING':sum, 'COUNTERFIETING':sum, 'ARSON':sum, 'HURT':sum,
       'DOWRY DEATHS':sum, 'ASSAULT_WOMEN_MODESTY':sum,
       'INSULT TO MODESTY OF WOMEN':sum, 'CRUELTY_BY_HUSBAND':sum,
       'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES':sum,
       'DEATH_BY_NEGLIGENCE':sum, 'OTHER IPC CRIMES':sum, "TOTAL_THEFT":sum}).reset_index()
       


# In[2880]:


crimes_distribution = crimes_distribution.append(crimes_distribution.sum().rename('total'))
crimes_distribution['YEAR'].replace(26091, 'Total', inplace=True)
crimes_distribution = crimes_distribution[crimes_distribution['YEAR'] == 'Total']
crimes_sum = crimes_distribution.T.reset_index()


# In[2881]:



labels = ['MURDER', 'ATTEMPT TO MURDER',
       'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER', 'RAPE', 'KIDNAPPING & ABDUCTION', 'DACOITY',
       'PREPARATION AND ASSEMBLY FOR DACOITY', 'ROBBERY', 'BURGLARY',
       'RIOTS', 'CRIMINAL BREACH OF TRUST',
       'CHEATING', 'COUNTERFIETING', 'ARSON', 'HURT/GREVIOUS HURT',
       'DOWRY DEATHS', 'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
       'INSULT TO MODESTY OF WOMEN', 'CRUELTY BY HUSBAND OR HIS RELATIVES',
       'IMPORTATION OF GIRLS FROM FOREIGN COUNTRIES',
       'CAUSING DEATH BY NEGLIGENCE', "THEFTTOTAL"]

values = [435744, 387394, 47994, 272844, 422318, 63948, 35100, 280194, 1221740,847053,
         203648, 874927, 29067, 123061, 3706545, 99285, 523894, 136643, 994067, 923, 1113381, 6310244]
fig = go.Figure(data=[go.Pie(labels=labels, values=values ,textinfo='label+percent',
                              )])
fig.update_layout(
    uniformtext_minsize= 20,
    title_text="Distribution of Crimes",
    paper_bgcolor='rgb(233,233,233)',
    autosize=False,
    width=1150,
    height=800)
fig.show()


# In[2882]:



labels = ['MURDER', 'KIDNAPPING & ABDUCTION', 'BURGLARY',
        'RIOTS',
        'CHEATING', 'HURT/GREVIOUS HURT',
        'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
        'CRUELTY BY HUSBAND OR HIS RELATIVES',
        'CAUSING DEATH BY NEGLIGENCE', "THEFTTOTAL"]

values = [435744, 422318, 1221740,847053,
          874927, 3706545, 523894, 994067, 1113381, 6310244]
fig = go.Figure(data=[go.Pie(labels=labels, values=values ,textinfo='label+percent',
                              )])
fig.update_layout(
    uniformtext_minsize= 20,
    title_text="Top 10 Crimes in India",
    paper_bgcolor='rgb(233,233,233)',
    autosize=False,
    width=1000,
    height=900)
fig.show()


# In[ ]:





# In[2883]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('TOTAL IPC CRIMES', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='TOTAL', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['TOTAL'].min(), vmax= merged_Ind["TOTAL"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2884]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('THEFT IN INDIA', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='TOTAL_THEFT', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['TOTAL_THEFT'].min(), vmax= merged_Ind["TOTAL_THEFT"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2885]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('HURTS IN INDIA', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='HURT', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['HURT'].min(), vmax= merged_Ind["HURT"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2886]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('KIDNAPPING IN INDIA', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='KIDNAPPING', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['KIDNAPPING'].min(), vmax= merged_Ind["KIDNAPPING"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2887]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('CHEATING IN INDIA', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='CHEATING', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['CHEATING'].min(), vmax= merged_Ind["CHEATING"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2888]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('BURGLARY IN INDIA', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='BURGLARY', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['BURGLARY'].min(), vmax= merged_Ind["BURGLARY"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2889]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('RIOTS IN INDIA', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='RIOTS', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['RIOTS'].min(), vmax= merged_Ind["RIOTS"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2890]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('MURDER', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='MURDER', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['MURDER'].min(), vmax= merged_Ind["MURDER"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2891]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('ASSAULT ON WOMEN MODESTY', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='ASSAULT_WOMEN_MODESTY', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['ASSAULT_WOMEN_MODESTY'].min(), vmax= merged_Ind["ASSAULT_WOMEN_MODESTY"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2892]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('CRUELTY_BY_HUSBAND', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='CRUELTY_BY_HUSBAND', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['CRUELTY_BY_HUSBAND'].min(), vmax= merged_Ind["CRUELTY_BY_HUSBAND"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2893]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('DEATH_BY_NEGLIGENCE', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='DEATH_BY_NEGLIGENCE', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['DEATH_BY_NEGLIGENCE'].min(), vmax= merged_Ind["DEATH_BY_NEGLIGENCE"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2894]:


fig, ax = plt.subplots(1, figsize=(15, 11))
ax.axis('off')
ax.set_title('OTHER IPC CRIMES', fontdict={'fontsize': '25', 'fontweight' : '3'})

# plot the figure
merged_Ind.dropna().plot(column='OTHER IPC CRIMES', cmap='PuRd', figsize=fig, scheme='equal_interval',linewidth=0.8, ax=ax, edgecolor='grey')
norm = Normalize(vmin=merged_Ind['OTHER IPC CRIMES'].min(), vmax= merged_Ind["OTHER IPC CRIMES"].max())
n_cmap = cm.ScalarMappable(norm=norm, cmap='PuRd')
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)
ax.set_axis_off()
plt.axis('equal')
plt.show()


# In[2895]:


crimes_total = crimes_df.groupby(['STATE', "YEAR"]).agg({"TOTAL": sum }).reset_index()


# In[2896]:


crimes_total


# In[2897]:


total_crime_yearwise = crimes_df.groupby(["YEAR"]).agg({"TOTAL": sum }).reset_index()
total_crime_yearwise


# In[3023]:


actual_crime=pd.read_csv('data.csv')
actual_crime.drop(['YEAR'], axis = 1, inplace = True)
#actual_crime=np.array(actual_crime)
actual_crime= actual_crime.TOTAL.astype('int64').to_numpy()
actual_crime


# In[3024]:


an_arim = ARIMA(total_crime_yearwise.TOTAL[:10].astype(np.float64).to_numpy(), order=(2,1,1))
an_model = an_arim.fit()
def get_mape(actual, predicted):
    y_actual = np.array(actual)
    y_pred = np.array(predicted)
    return np.round(np.mean(np.abs((y_actual - y_pred)/actual)) * 100,2)
forecast_an = an_model.predict(10,11)
forecast_an_df = an_model.forecast(steps=10)
predicted_crime= actual_crime[:12]
predicted_crime = np.append(np.array(predicted_crime) , np.array(forecast_an_df[:7]))
predicted_crime= predicted_crime.astype('int64')
predicted_crime


# In[3038]:


from sklearn.metrics import accuracy_score
print("Accuracy is {0:2.2f} %" .format( accuracy_score(actual_crime,predicted_crime)*100 ))


# In[ ]:





# In[3018]:


get_mape(total_crime_yearwise.TOTAL[10:12],forecast_an)


# In[3019]:


i=12
for j in range(len(forecast_an_df)):
    total_crime_yearwise.loc[i] = total_crime_yearwise.YEAR[i-1]+1
    total_crime_yearwise.TOTAL[i] = forecast_an_df[j]
    i = i+1
fig = go.Figure()
fig.add_trace(go.Scatter(x= total_crime_yearwise["YEAR"], y= total_crime_yearwise['TOTAL'],
                    name = "INDIA",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= total_crime_yearwise["YEAR"][12:], y= total_crime_yearwise['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[ ]:





# In[2945]:


#Crimes Trend in Andaman 
crime_AN_df = crimes_total[crimes_total.STATE=="A & N ISLANDS" ]
#ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_AN_df['TOTAL']),
#                                 model = "multiplicative", period=2)
#ts_plot = ts_decompose.plot()
crime_AN_df


# In[506]:


crime_AN_forecast = crime_AN_df.copy()
crime_AN_forecast.drop('STATE', axis = 1, inplace = True)


# In[507]:


crime_AN_forecast


# In[514]:


def adfull(ts):
    result = adfuller(ts, autolag=None)
    result_out = pd.Series(result[0:4], index=["Test Stats",
                                              'P-value',
                                              'lags_used',
                                              'Number of observation'])
    print(result_out)
    


# In[515]:


adfull(crime_AN_forecast.TOTAL)


# In[516]:


acf_plot = plot_acf(crime_AN_forecast.TOTAL, lags = 10)


# In[517]:


pcf_1 = plot_pacf(crime_AN_forecast.TOTAL, lags = 5, method='ywm')


# In[1045]:


an_arim = ARIMA(crime_AN_forecast.TOTAL[:10].astype(np.float64).to_numpy(), order=(1,0,0))
an_model = an_arim.fit()
#print(an_model.summary())


# In[1046]:


def get_mape(actual, predicted):
    y_actual = np.array(actual)
    y_pred = np.array(predicted)
    return np.round(np.mean(np.abs((y_actual - y_pred)/actual)) * 100,2)


# In[1047]:


forecast_an = an_model.predict(10,12)
forecast_an


# In[1048]:


forecast_an_df = an_model.forecast(steps=10)
forecast_an_df


# In[1049]:


get_mape(crime_AN_forecast.TOTAL[10:13],forecast_an)


# In[1050]:


i=12
for j in range(len(forecast_an_df)):
    crime_AN_forecast.loc[i] = crime_AN_forecast.YEAR[i-1]+1
    crime_AN_forecast.TOTAL[i] = forecast_an_df[j]
    i = i+1


# In[1052]:


fig = go.Figure()
fig.add_trace(go.Scatter(x= crime_AN_df["YEAR"], y= crime_AN_df['TOTAL'],
                    name = "A&N",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crime_AN_forecast["YEAR"][11:], y= crime_AN_forecast['TOTAL'][11:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1143]:


#trend in Andhra Pradesh
crime_AP_df = crimes_total[crimes_total.STATE=='ANDHRA PRADESH']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_AP_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1144]:


crimes_AP_forecast = crime_AP_df.copy()
crimes_AP_forecast.drop(['STATE'], axis = 1, inplace = True)


# In[1145]:


crimes_AP_forecast = crimes_AP_forecast.reset_index(drop = True)
crimes_AP_forecast


# In[1146]:


acf_plot = plot_acf(crimes_AP_forecast.TOTAL, lags = 10)


# In[1189]:


ap_arim = ARIMA(crimes_AP_forecast.TOTAL[:10].astype(np.float64).to_numpy(), order=(1,2,0))
ap_model = ap_arim.fit()
ap_model.summary()


# In[1190]:


forecast_ap = ap_model.predict(10,12)
forecast_ap
#crimes_AP_forecast.TOTAL[10:13]


# In[1191]:


get_mape(crimes_AP_forecast.TOTAL[10:13],forecast_ap)


# In[1192]:


forecast_df = ap_model.forecast(steps=10)
forecast_df


# In[1193]:


i=12
for j in range(len(forecast_df)):
    crimes_AP_forecast.loc[i] = crimes_AP_forecast.YEAR[i-1]+1
    crimes_AP_forecast.TOTAL[i] = forecast_df[j]
    i = i+1


# In[1194]:


fig = go.Figure()
fig.add_trace(go.Scatter(x= crime_AP_df['YEAR'], y= crime_AP_df['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_AP_forecast["YEAR"][11:], y= crimes_AP_forecast['TOTAL'][11:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1201]:


#ARUNACHAL PRADESH
crime_ARU_df = crimes_total[crimes_total.STATE=='ARUNACHAL PRADESH']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_ARU_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1202]:


crimes_ARU_forecast = crime_ARU_df.copy()
crimes_ARU_forecast.drop(['STATE'], axis = 1, inplace = True)


# In[1203]:


crimes_ARU_forecast = crimes_ARU_forecast.reset_index(drop = True)


# In[1204]:


acf_plot = plot_acf(crimes_ARU_forecast.TOTAL, lags = 10)


# In[1205]:


crimes_ARU_forecast


# In[1206]:


aru_arim = ARIMA(crimes_ARU_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,0,0))
aru_model = aru_arim.fit()
aru_model.summary()


# In[1207]:


forecast_aru = aru_model.predict(10,12)
forecast_aru


# In[1208]:


get_mape(crimes_AP_forecast.TOTAL[10:13],forecast_aru)


# In[1209]:


forecast_df_aru = aru_model.forecast(steps=10)
forecast_df_aru


# In[1210]:


i=12
for j in range(len(forecast_df_aru)):
    crimes_ARU_forecast.loc[i] = crimes_ARU_forecast.YEAR[i-1]+1
    crimes_ARU_forecast.TOTAL[i] = forecast_df_aru[j]
    i = i+1


# In[573]:


fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_ARU_forecast["YEAR"], y= crimes_ARU_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_ARU_forecast["YEAR"][12:], y= crimes_ARU_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[574]:


#ASSAM
crime_Assam_df = crimes_total[crimes_total.STATE=='ASSAM']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_Assam_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[575]:


def plot(dataframe):
	acf_plot = plot_acf(dataframe.TOTAL, lags = 10)
	pcf_1 = plot_pacf(dataframe.TOTAL, lags = 5 , method='ywm')
	return acf_plot, pcf_1


# In[576]:


crimes_Assam_forecast = crime_Assam_df.copy()
crimes_Assam_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_Assam_forecast = crimes_Assam_forecast.reset_index(drop = True)
plot(crimes_Assam_forecast)


# In[847]:


Assam_arim = ARIMA(crimes_Assam_forecast.TOTAL[:10].astype(np.float64).to_numpy(), order=(0,2,0))
Assam_moAssam = Assam_arim.fit()
Assam_moAssam.summary()
forecast_Assam = Assam_moAssam.predict(10,12)
forecast_Assam


# In[848]:


forecast_Assam = Assam_moAssam.predict(10,11)
forecast_Assam


# In[849]:


forecast_df_Assam = Assam_moAssam.forecast(steps=10)
forecast_df_Assam


# In[850]:


get_mape(crimes_Assam_forecast.TOTAL[10:12],forecast_Assam)


# In[851]:


i=12
for j in range(len(forecast_df_Assam)):
    crimes_Assam_forecast.loc[i] = crimes_Assam_forecast.YEAR[i-1]+1
    crimes_Assam_forecast.TOTAL[i] = forecast_df_Assam[j]
    i = i+1


# In[852]:


fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_Assam_forecast["YEAR"], y= crimes_Assam_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_Assam_forecast["YEAR"][12:], y= crimes_Assam_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[661]:


#BIHAR
crime_BIH_df = crimes_total[crimes_total.STATE=='BIHAR']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_BIH_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[662]:


crimes_BIH_forecast = crime_BIH_df.copy()
crimes_BIH_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_BIH_forecast = crimes_BIH_forecast.reset_index(drop = True)
plot(crimes_BIH_forecast)


# In[663]:


crimes_BIH_forecast


# In[834]:


BIH_arim = ARIMA(crimes_BIH_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,2,1))
BIH_mo = BIH_arim.fit()
BIH_mo.summary()


# In[835]:


forecast_BIH = BIH_mo.predict(10,12)
forecast_BIH


# In[836]:


forecast_df_BIH = BIH_mo.forecast(steps=10)
forecast_df_BIH


# In[837]:


get_mape(crimes_BIH_forecast.TOTAL[10:13],forecast_BIH)


# In[838]:


i=12
for j in range(len(forecast_df_BIH)):
    crimes_BIH_forecast.loc[i] = crimes_BIH_forecast.YEAR[i-1]+1
    crimes_BIH_forecast.TOTAL[i] = forecast_df_BIH[j]
    i = i+1


# In[839]:


fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_BIH_forecast["YEAR"], y= crimes_BIH_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_BIH_forecast["YEAR"][12:], y= crimes_BIH_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[675]:


#GOA
crime_GOA_df = crimes_total[crimes_total.STATE=='GOA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_GOA_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[676]:


crimes_GOA_forecast = crime_GOA_df.copy()
crimes_GOA_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_GOA_forecast = crimes_GOA_forecast.reset_index(drop = True)
plot(crimes_GOA_forecast)


# In[677]:


GOA_arim = ARIMA(crimes_GOA_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,1,0))
GOA_moGOA = GOA_arim.fit()
GOA_moGOA.summary()


# In[678]:


forecast_GOA = GOA_moGOA.predict(10,11)
forecast_GOA


# In[679]:


forecast_df_GOA = GOA_moGOA.forecast(steps=10)
forecast_df_GOA


# In[680]:


get_mape(crimes_GOA_forecast.TOTAL[10:13],forecast_GOA)


# In[681]:


i=12
for j in range(len(forecast_df_GOA)):
    crimes_GOA_forecast.loc[i] = crimes_GOA_forecast.YEAR[i-1]+1
    crimes_GOA_forecast.TOTAL[i] = forecast_df_GOA[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_GOA_forecast["YEAR"], y= crimes_GOA_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_GOA_forecast["YEAR"][12:], y= crimes_GOA_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[ ]:


#GUJRAT


# In[728]:


crime_GUJ_df = crimes_total[crimes_total.STATE=='GUJARAT']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_GUJ_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[729]:


crimes_GUJ_forecast = crime_GUJ_df.copy()
crimes_GUJ_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_GUJ_forecast = crimes_GUJ_forecast.reset_index(drop = True)
plot(crimes_GUJ_forecast)


# In[829]:


GUJ_arim = ARIMA(crimes_GUJ_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(0,2,1))
GUJ_moGUJ = GUJ_arim.fit()
GUJ_moGUJ.summary()


# In[830]:


forecast_GUJ = GUJ_moGUJ.predict(10,12)
forecast_GUJ


# In[831]:


forecast_df_GUJ = GUJ_moGUJ.forecast(steps=10)
forecast_df_GUJ


# In[832]:


get_mape(crimes_GUJ_forecast.TOTAL[10:13],forecast_GUJ)


# In[833]:


i=12
for j in range(len(forecast_df_GUJ)):
    crimes_GUJ_forecast.loc[i] = crimes_GUJ_forecast.YEAR[i-1]+1
    crimes_GUJ_forecast.TOTAL[i] = forecast_df_GUJ[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_GUJ_forecast["YEAR"], y= crimes_GUJ_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_GUJ_forecast["YEAR"][12:], y= crimes_GUJ_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[ ]:


#HARYANA


# In[794]:


crime_HAR_df = crimes_total[crimes_total.STATE=='HARYANA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_HAR_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[795]:


crimes_HAR_forecast = crime_HAR_df.copy()
crimes_HAR_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_HAR_forecast = crimes_HAR_forecast.reset_index(drop = True)
plot(crimes_HAR_forecast)


# In[1414]:


HAR_arim = ARIMA(crimes_HAR_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,3,1))
HAR_moHAR = HAR_arim.fit()
HAR_moHAR.summary()


# In[1415]:


forecast_HAR = HAR_moHAR.predict(10,12)
print('prediction',forecast_HAR)

forecast_df_HAR = HAR_moHAR.forecast(steps=10)
print('forecasting',forecast_df_HAR)


# In[1416]:


get_mape(crimes_HAR_forecast.TOTAL[10:13],forecast_HAR)


# In[1417]:


i=12
for j in range(len(forecast_df_HAR)):
    crimes_HAR_forecast.loc[i] = crimes_HAR_forecast.YEAR[i-1]+1
    crimes_HAR_forecast.TOTAL[i] = forecast_df_HAR[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_HAR_forecast["YEAR"], y= crimes_HAR_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_HAR_forecast["YEAR"][12:], y= crimes_HAR_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1211]:


#crime in HIMACHAL Pradesh


# In[1212]:


crime_HIM_df = crimes_total[crimes_total.STATE=='HIMACHAL PRADESH']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_HIM_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1213]:


crimes_HIM_forecast = crime_HIM_df.copy()
crimes_HIM_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_HIM_forecast = crimes_HIM_forecast.reset_index(drop = True)
plot(crimes_HIM_forecast)


# In[1390]:


HIM_arim = ARIMA(crimes_HIM_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(3,3,0))
HIM_moHIM = HIM_arim.fit()
HIM_moHIM.summary()


# In[1391]:


forecast_HIM = HIM_moHIM.predict(10,11)
print('prediction',forecast_HIM)

forecast_df_HIM = HIM_moHIM.forecast(steps=10)
print('forecasting',forecast_df_HIM)


# In[1392]:


get_mape(crimes_HIM_forecast.TOTAL[10:12],forecast_HIM)


# In[1393]:


i=12
for j in range(len(forecast_df_HIM)):
    crimes_HIM_forecast.loc[i] = crimes_HIM_forecast.YEAR[i-1]+1
    crimes_HIM_forecast.TOTAL[i] = forecast_df_HIM[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_HIM_forecast["YEAR"], y= crimes_HIM_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_HIM_forecast["YEAR"][12:], y= crimes_HIM_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1418]:


crime_JK_df = crimes_total[crimes_total.STATE=='JAMMU & KASHMIR']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_JK_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1419]:


crimes_JK_forecast = crime_JK_df.copy()
crimes_JK_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_JK_forecast = crimes_JK_forecast.reset_index(drop = True)
plot(crimes_JK_forecast)


# In[1450]:


JK_arim = ARIMA(crimes_JK_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,0,0))
JK_moJK = JK_arim.fit()
JK_moJK.summary()


# In[1451]:


forecast_JK = JK_moJK.predict(10,11)
print('prediction',forecast_JK)
forecast_df_JK = JK_moJK.forecast(steps=10)
print('forecasting',forecast_df_JK)


# In[1452]:


get_mape(crimes_JK_forecast.TOTAL[10:12],forecast_JK)

	


# In[1453]:


i=12
for j in range(len(forecast_df_JK)):
    crimes_JK_forecast.loc[i] = crimes_JK_forecast.YEAR[i-1]+1
    crimes_JK_forecast.TOTAL[i] = forecast_df_JK[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_JK_forecast["YEAR"], y= crimes_JK_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_JK_forecast["YEAR"][12:], y= crimes_JK_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1454]:


crime_JAR_df = crimes_total[crimes_total.STATE=='JHARKHAND']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_JAR_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1455]:


crimes_JAR_forecast = crime_JAR_df.copy()
crimes_JAR_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_JAR_forecast = crimes_JAR_forecast.reset_index(drop = True)
plot(crimes_JAR_forecast)


# In[1862]:


JAR_arim = ARIMA(crimes_JAR_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,1,0))
JAR_moJAR = JAR_arim.fit()
JAR_moJAR.summary()


# In[1863]:


forecast_JAR = JAR_moJAR.predict(10,12)
print('prediction',forecast_JAR)
forecast_df_JAR = JAR_moJAR.forecast(steps=10)
print('forecasting',forecast_df_JAR)


# In[1864]:


get_mape(crimes_JAR_forecast.TOTAL[10:13],forecast_JAR)


# In[1865]:


i=12
for j in range(len(forecast_df_JAR)):
    crimes_JAR_forecast.loc[i] = crimes_JAR_forecast.YEAR[i-1]+1
    crimes_JAR_forecast.TOTAL[i] = forecast_df_JAR[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_JAR_forecast["YEAR"], y= crimes_JAR_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_JAR_forecast["YEAR"][12:], y= crimes_JAR_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1580]:


crime_KAR_df = crimes_total[crimes_total.STATE=='KARNATAKA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_KAR_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1581]:


crimes_KAR_forecast = crime_KAR_df.copy()
crimes_KAR_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_KAR_forecast = crimes_KAR_forecast.reset_index(drop = True)
plot(crimes_KAR_forecast)


# In[1633]:


KAR_arim = ARIMA(crimes_KAR_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,1,0))
KAR_moKAR = KAR_arim.fit()
KAR_moKAR.summary()


# In[1634]:


forecast_KAR = KAR_moKAR.predict(10,12)
print('prediction',forecast_KAR)
forecast_df_KAR = KAR_moKAR.forecast(steps=10)
print('forecasting',forecast_df_KAR)


# In[1635]:


get_mape(crimes_KAR_forecast.TOTAL[10:13],forecast_KAR)


# In[1636]:


i=12
for j in range(len(forecast_df_KAR)):
    crimes_KAR_forecast.loc[i] = crimes_KAR_forecast.YEAR[i-1]+1
    crimes_KAR_forecast.TOTAL[i] = forecast_df_KAR[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_KAR_forecast["YEAR"], y= crimes_KAR_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_KAR_forecast["YEAR"][12:], y= crimes_KAR_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1637]:


crime_KER_df = crimes_total[crimes_total.STATE=='KERALA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_KER_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1638]:


crimes_KER_forecast = crime_KER_df.copy()
crimes_KER_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_KER_forecast = crimes_KER_forecast.reset_index(drop = True)
plot(crimes_KER_forecast)


# In[1715]:


KER_arim = ARIMA(crimes_KER_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(0,1,0))
KER_moKER = KER_arim.fit()
KER_moKER.summary()


# In[1716]:


forecast_KER = KER_moKER.predict(10,12)
print('prediction',forecast_KER)
forecast_df_KER = KER_moKER.forecast(steps=10)
print('forecasting',forecast_df_KER)


# In[1717]:


get_mape(crimes_KER_forecast.TOTAL[10:13],forecast_KER)


# In[1718]:


i=12
for j in range(len(forecast_df_KER)):
    crimes_KER_forecast.loc[i] = crimes_KER_forecast.YEAR[i-1]+1
    crimes_KER_forecast.TOTAL[i] = forecast_df_KER[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_KER_forecast["YEAR"], y= crimes_KER_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_KER_forecast["YEAR"][12:], y= crimes_KER_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1719]:


crime_MP_df = crimes_total[crimes_total.STATE=='MADHYA PRADESH']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_MP_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1720]:


crimes_MP_forecast = crime_MP_df.copy()
crimes_MP_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_MP_forecast = crimes_MP_forecast.reset_index(drop = True)
plot(crimes_MP_forecast)


# In[1785]:


MP_arim = ARIMA(crimes_MP_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,0,0))
MP_moMP = MP_arim.fit()
MP_moMP.summary()


# In[1786]:


forecast_MP = MP_moMP.predict(10,12)
print('prediction',forecast_MP)
forecast_df_MP = MP_moMP.forecast(steps=10)
print('forecasting',forecast_df_MP)


# In[1787]:


get_mape(crimes_MP_forecast.TOTAL[10:13],forecast_MP)

	


# In[1788]:


i=12
for j in range(len(forecast_df_MP)):
    crimes_MP_forecast.loc[i] = crimes_MP_forecast.YEAR[i-1]+1
    crimes_MP_forecast.TOTAL[i] = forecast_df_MP[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_MP_forecast["YEAR"], y= crimes_MP_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_MP_forecast["YEAR"][13:], y= crimes_MP_forecast['TOTAL'][13:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1789]:


crime_MAH_df = crimes_total[crimes_total.STATE=='MAHARASHTRA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_MAH_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1790]:


crimes_MAH_forecast = crime_MAH_df.copy()
crimes_MAH_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_MAH_forecast = crimes_MAH_forecast.reset_index(drop = True)
plot(crimes_MAH_forecast)


# In[1844]:


MAH_arim = ARIMA(crimes_MAH_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,2,0))
MAH_moMAH = MAH_arim.fit()
MAH_moMAH.summary()


# In[1845]:


forecast_MAH = MAH_moMAH.predict(10,12)
print('prediction',forecast_MAH)
forecast_df_MAH = MAH_moMAH.forecast(steps=10)
print('forecasting',forecast_df_MAH)


# In[1846]:


get_mape(crimes_MAH_forecast.TOTAL[10:13],forecast_MAH)


# In[1847]:


i=12
for j in range(len(forecast_df_MAH)):
    crimes_MAH_forecast.loc[i] = crimes_MAH_forecast.YEAR[i-1]+1
    crimes_MAH_forecast.TOTAL[i] = forecast_df_MAH[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_MAH_forecast["YEAR"], y= crimes_MAH_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_MAH_forecast["YEAR"][12:], y= crimes_MAH_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1849]:


crime_MAN_df = crimes_total[crimes_total.STATE=='MANIPUR']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_MAN_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1850]:


crimes_MAN_forecast = crime_MAN_df.copy()
crimes_MAN_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_MAN_forecast = crimes_MAN_forecast.reset_index(drop = True)
plot(crimes_MAN_forecast)


# In[1929]:


MAN_arim = ARIMA(crimes_MAN_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(3,0,0))
MAN_moMAN = MAN_arim.fit()
MAN_moMAN.summary()


# In[1930]:


forecast_MAN = MAN_moMAN.predict(10,12)
print('prediction',forecast_MAN)
forecast_df_MAN = MAN_moMAN.forecast(steps=10)
print('forecasting',forecast_df_MAN)


# In[1933]:


get_mape(crimes_MAN_forecast.TOTAL[10:13],forecast_MAN)


# In[1934]:


i=12
for j in range(len(forecast_df_MAN)):
    crimes_MAN_forecast.loc[i] = crimes_MAN_forecast.YEAR[i-1]+1
    crimes_MAN_forecast.TOTAL[i] = forecast_df_MAN[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_MAN_forecast["YEAR"], y= crimes_MAN_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_MAN_forecast["YEAR"][12:], y= crimes_MAN_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1935]:


crime_MEG_df = crimes_total[crimes_total.STATE=='MEGHALAYA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_MEG_df['TOTAL']),
                                 model = "multiplicative",period=2)
ts_plot = ts_decompose.plot()


# In[1936]:


crimes_MEG_forecast = crime_MEG_df.copy()
crimes_MEG_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_MEG_forecast = crimes_MEG_forecast.reset_index(drop = True)
plot(crimes_MEG_forecast)


# In[1949]:


MEG_arim = ARIMA(crimes_MEG_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(3,3,2))
MEG_moMEG = MEG_arim.fit()
MEG_moMEG.summary()


# In[1950]:


forecast_MEG = MEG_moMEG.predict(10,12)
print('prediction',forecast_MEG)
forecast_df_MEG = MEG_moMEG.forecast(steps=10)
print('forecasting',forecast_df_MEG)


# In[1952]:


get_mape(crimes_MEG_forecast.TOTAL[10:13],forecast_MEG)


# In[1953]:


i=12
for j in range(len(forecast_df_MEG)):
    crimes_MEG_forecast.loc[i] = crimes_MEG_forecast.YEAR[i-1]+1
    crimes_MEG_forecast.TOTAL[i] = forecast_df_MEG[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_MEG_forecast["YEAR"], y= crimes_MEG_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_MEG_forecast["YEAR"][13:], y= crimes_MEG_forecast['TOTAL'][13:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1954]:


crime_MIZ_df = crimes_total[crimes_total.STATE=='MIZORAM']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_MIZ_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1955]:


crimes_MIZ_forecast = crime_MIZ_df.copy()
crimes_MIZ_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_MIZ_forecast = crimes_MIZ_forecast.reset_index(drop = True)
plot(crimes_MIZ_forecast)


# In[1977]:


MIZ_arim = ARIMA(crimes_MIZ_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(3,2,0))
MIZ_moMIZ = MIZ_arim.fit()
MIZ_moMIZ.summary()


# In[1978]:


forecast_MIZ = MIZ_moMIZ.predict(10,12)
print('prediction',forecast_MIZ)
forecast_df_MIZ = MIZ_moMIZ.forecast(steps=10)
print('forecasting',forecast_df_MIZ)


# In[1979]:


get_mape(crimes_MIZ_forecast.TOTAL[10:13],forecast_MIZ)


# In[1980]:


i=12
for j in range(len(forecast_df_MIZ)):
    crimes_MIZ_forecast.loc[i] = crimes_MIZ_forecast.YEAR[i-1]+1
    crimes_MIZ_forecast.TOTAL[i] = forecast_df_MIZ[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_MIZ_forecast["YEAR"], y= crimes_MIZ_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_MIZ_forecast["YEAR"][12:], y= crimes_MIZ_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1981]:


crime_NAG_df = crimes_total[crimes_total.STATE=='NAGALAND']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_NAG_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1982]:


crimes_NAG_forecast = crime_NAG_df.copy()
crimes_NAG_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_NAG_forecast = crimes_NAG_forecast.reset_index(drop = True)
plot(crimes_NAG_forecast)


# In[1993]:


NAG_arim = ARIMA(crimes_NAG_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,1,0))
NAG_moNAG = NAG_arim.fit()
NAG_moNAG.summary()


# In[1994]:


forecast_NAG = NAG_moNAG.predict(10,12)
print('prediction',forecast_NAG)
forecast_df_NAG = NAG_moNAG.forecast(steps=10)
print('forecasting',forecast_df_NAG)


# In[1995]:


get_mape(crimes_NAG_forecast.TOTAL[10:13],forecast_NAG)


# In[1996]:


i=12
for j in range(len(forecast_df_NAG)):
    crimes_NAG_forecast.loc[i] = crimes_NAG_forecast.YEAR[i-1]+1
    crimes_NAG_forecast.TOTAL[i] = forecast_df_NAG[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_NAG_forecast["YEAR"], y= crimes_NAG_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_NAG_forecast["YEAR"][12:], y= crimes_NAG_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[1997]:


crime_ODI_df = crimes_total[crimes_total.STATE=='ODISHA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_ODI_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[1998]:


crimes_ODI_forecast = crime_ODI_df.copy()
crimes_ODI_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_ODI_forecast = crimes_ODI_forecast.reset_index(drop = True)
plot(crimes_ODI_forecast)


# In[2035]:


ODI_arim = ARIMA(crimes_ODI_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,2,1))
ODI_moODI = ODI_arim.fit()
ODI_moODI.summary()


# In[2036]:


forecast_ODI = ODI_moODI.predict(10,12)
print('prediction',forecast_ODI)
forecast_df_ODI = ODI_moODI.forecast(steps=10)
print('forecasting',forecast_df_ODI)


# In[2037]:


get_mape(crimes_ODI_forecast.TOTAL[10:13],forecast_ODI)


# In[2038]:


i=12
for j in range(len(forecast_df_ODI)):
    crimes_ODI_forecast.loc[i] = crimes_ODI_forecast.YEAR[i-1]+1
    crimes_ODI_forecast.TOTAL[i] = forecast_df_ODI[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_ODI_forecast["YEAR"], y= crimes_ODI_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_ODI_forecast["YEAR"][12:], y= crimes_ODI_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2039]:


crime_PUN_df = crimes_total[crimes_total.STATE=='PUNJAB']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_PUN_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2040]:


crimes_PUN_forecast = crime_PUN_df.copy()
crimes_PUN_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_PUN_forecast = crimes_PUN_forecast.reset_index(drop = True)
plot(crimes_PUN_forecast)


# In[2223]:


PUN_arim = ARIMA(crimes_PUN_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,1,1))
PUN_moPUN = PUN_arim.fit()
PUN_moPUN.summary()


# In[2224]:


forecast_PUN = PUN_moPUN.predict(10,12)
print('prediction',forecast_PUN)
forecast_df_PUN = PUN_moPUN.forecast(steps=10)
print('forecasting',forecast_df_PUN)


# In[2225]:


get_mape(crimes_PUN_forecast.TOTAL[10:13],forecast_PUN)


# In[2226]:


i=12
for j in range(len(forecast_df_PUN)):
    crimes_PUN_forecast.loc[i] = crimes_PUN_forecast.YEAR[i-1]+1
    crimes_PUN_forecast.TOTAL[i] = forecast_df_PUN[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_PUN_forecast["YEAR"], y= crimes_PUN_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_PUN_forecast["YEAR"][12:], y= crimes_PUN_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2227]:


crime_RAJ_df = crimes_total[crimes_total.STATE=='RAJASTHAN']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_RAJ_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2228]:


crimes_RAJ_forecast = crime_RAJ_df.copy()
crimes_RAJ_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_RAJ_forecast = crimes_RAJ_forecast.reset_index(drop = True)
plot(crimes_RAJ_forecast)


# In[2273]:


RAJ_arim = ARIMA(crimes_RAJ_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,1,1))
RAJ_moRAJ = RAJ_arim.fit()
RAJ_moRAJ.summary()


# In[2274]:


forecast_RAJ = RAJ_moRAJ.predict(10,12)
print('prediction',forecast_RAJ)
forecast_df_RAJ = RAJ_moRAJ.forecast(steps=10)
print('forecasting',forecast_df_RAJ)


# In[2275]:


get_mape(crimes_RAJ_forecast.TOTAL[10:13],forecast_RAJ)


# In[2276]:


i=12
for j in range(len(forecast_df_RAJ)):
    crimes_RAJ_forecast.loc[i] = crimes_RAJ_forecast.YEAR[i-1]+1
    crimes_RAJ_forecast.TOTAL[i] = forecast_df_RAJ[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_RAJ_forecast["YEAR"], y= crimes_RAJ_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_RAJ_forecast["YEAR"][12:], y= crimes_RAJ_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2277]:


crime_SIK_df = crimes_total[crimes_total.STATE=='SIKKIM']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_SIK_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2278]:



crimes_SIK_forecast = crime_SIK_df.copy()
crimes_SIK_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_SIK_forecast = crimes_SIK_forecast.reset_index(drop = True)
plot(crimes_SIK_forecast)


# In[2304]:


SIK_arim = ARIMA(crimes_SIK_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,1,0))
SIK_moSIK = SIK_arim.fit()
SIK_moSIK.summary()


# In[2305]:


forecast_SIK = SIK_moSIK.predict(10,12)
print('prediction',forecast_SIK)
forecast_df_SIK = SIK_moSIK.forecast(steps=10)
print('forecasting',forecast_df_SIK)


# In[2306]:


get_mape(crimes_SIK_forecast.TOTAL[10:13],forecast_SIK)


# In[2307]:


i=12
for j in range(len(forecast_df_SIK)):
    crimes_SIK_forecast.loc[i] = crimes_SIK_forecast.YEAR[i-1]+1
    crimes_SIK_forecast.TOTAL[i] = forecast_df_SIK[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_SIK_forecast["YEAR"], y= crimes_SIK_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_SIK_forecast["YEAR"][12:], y= crimes_SIK_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2308]:


crime_TN_df = crimes_total[crimes_total.STATE=='TAMIL NADU']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_TN_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2309]:


crimes_TN_forecast = crime_TN_df.copy()
crimes_TN_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_TN_forecast = crimes_TN_forecast.reset_index(drop = True)
plot(crimes_TN_forecast)


# In[2903]:


TN_arim = ARIMA(crimes_TN_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,2,0))
TN_moTN = TN_arim.fit()
TN_moTN.summary()


# In[2904]:


forecast_TN = TN_moTN.predict(10,12)
print('prediction',forecast_TN)
forecast_df_TN = TN_moTN.forecast(steps=10)
print('forecasting',forecast_df_TN)


# In[2905]:


get_mape(crimes_TN_forecast.TOTAL[10:13],forecast_TN)


# In[2906]:


i=12
for j in range(len(forecast_df_TN)):
    crimes_TN_forecast.loc[i] = crimes_TN_forecast.YEAR[i-1]+1
    crimes_TN_forecast.TOTAL[i] = forecast_df_TN[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_TN_forecast["YEAR"], y= crimes_TN_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_TN_forecast["YEAR"][12:], y= crimes_TN_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2366]:


crime_TRI_df = crimes_total[crimes_total.STATE=='TRIPURA']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_ODI_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2367]:


crimes_TRI_forecast = crime_TRI_df.copy()
crimes_TRI_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_TRI_forecast = crimes_TRI_forecast.reset_index(drop = True)
plot(crimes_TRI_forecast)


# In[2406]:


TRI_arim = ARIMA(crimes_TRI_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,1,0))
TRI_moTRI = TRI_arim.fit()
TRI_moTRI.summary()


# In[2407]:


forecast_TRI = TRI_moTRI.predict(10,12)
print('prediction',forecast_TRI)
forecast_df_TRI = TRI_moTRI.forecast(steps=10)
print('forecasting',forecast_df_TRI)


# In[2408]:


get_mape(crimes_TRI_forecast.TOTAL[10:13],forecast_TRI)


# In[2409]:


i=12
for j in range(len(forecast_df_TRI)):
    crimes_TRI_forecast.loc[i] = crimes_TRI_forecast.YEAR[i-1]+1
    crimes_TRI_forecast.TOTAL[i] = forecast_df_TRI[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_TRI_forecast["YEAR"], y= crimes_TRI_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_TRI_forecast["YEAR"][12:], y= crimes_TRI_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2410]:


crime_UP_df = crimes_total[crimes_total.STATE=='UTTAR PRADESH']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_UP_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2411]:


crimes_UP_forecast = crime_UP_df.copy()
crimes_UP_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_UP_forecast = crimes_UP_forecast.reset_index(drop = True)
plot(crimes_UP_forecast)


# In[2472]:


UP_arim = ARIMA(crimes_UP_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,2,0))
UP_moUP = UP_arim.fit()
UP_moUP.summary()


# In[2473]:


forecast_UP = UP_moUP.predict(10,12)
print('prediction',forecast_UP)
forecast_df_UP = UP_moUP.forecast(steps=10)
print('forecasting',forecast_df_UP)


# In[2474]:


get_mape(crimes_UP_forecast.TOTAL[10:13],forecast_UP)


# In[2475]:


i=12
for j in range(len(forecast_df_UP)):
    crimes_UP_forecast.loc[i] = crimes_UP_forecast.YEAR[i-1]+1
    crimes_UP_forecast.TOTAL[i] = forecast_df_UP[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_UP_forecast["YEAR"], y= crimes_UP_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_UP_forecast["YEAR"][12:], y= crimes_UP_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2476]:


crime_UK_df = crimes_total[crimes_total.STATE=='UTTARAKHAND']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_UK_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2477]:


crimes_UK_forecast = crime_UK_df.copy()
crimes_UK_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_UK_forecast = crimes_UK_forecast.reset_index(drop = True)
plot(crimes_UK_forecast)


# In[2494]:


UK_arim = ARIMA(crimes_UK_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,2,1))
UK_moUK = UK_arim.fit()
UK_moUK.summary()


# In[2495]:


forecast_UK = UK_moUK.predict(10,12)
print('prediction',forecast_UK)
forecast_df_UK = UK_moUK.forecast(steps=10)
print('forecasting',forecast_df_UK)


# In[2496]:


get_mape(crimes_UK_forecast.TOTAL[10:13],forecast_UK)


# In[2498]:


i=12
for j in range(len(forecast_df_UK)):
    crimes_UK_forecast.loc[i] = crimes_UK_forecast.YEAR[i-1]+1
    crimes_UK_forecast.TOTAL[i] = forecast_df_UK[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_UK_forecast["YEAR"], y= crimes_UK_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_UK_forecast["YEAR"][12:], y= crimes_UK_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2499]:


crime_WB_df = crimes_total[crimes_total.STATE=='WEST BENGAL']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_WB_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2500]:


crimes_WB_forecast = crime_WB_df.copy()
crimes_WB_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_WB_forecast = crimes_WB_forecast.reset_index(drop = True)
plot(crimes_WB_forecast)


# In[2529]:


WB_arim = ARIMA(crimes_WB_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,2,1))
WB_moWB = WB_arim.fit()
WB_moWB.summary()


# In[2530]:


forecast_WB = WB_moWB.predict(10,12)
print('prediction',forecast_WB)
forecast_df_WB = WB_moWB.forecast(steps=10)
print('forecasting',forecast_df_WB)


# In[2531]:


get_mape(crimes_WB_forecast.TOTAL[10:13],forecast_WB)


# In[2532]:


i=12
for j in range(len(forecast_df_WB)):
    crimes_WB_forecast.loc[i] = crimes_WB_forecast.YEAR[i-1]+1
    crimes_WB_forecast.TOTAL[i] = forecast_df_WB[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_WB_forecast["YEAR"], y= crimes_WB_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_WB_forecast["YEAR"][13:], y= crimes_WB_forecast['TOTAL'][13:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2533]:


crime_CHAN_df = crimes_total[crimes_total.STATE=='CHANDIGARH']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_CHAN_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2534]:


crimes_CHAN_forecast = crime_CHAN_df.copy()
crimes_CHAN_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_CHAN_forecast = crimes_CHAN_forecast.reset_index(drop = True)
plot(crimes_CHAN_forecast)


# In[2544]:


CHAN_arim = ARIMA(crimes_CHAN_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,1,0))
CHAN_moCHAN = CHAN_arim.fit()
CHAN_moCHAN.summary()


# In[2545]:


forecast_CHAN = CHAN_moCHAN.predict(10,12)
print('prediction',forecast_CHAN)
forecast_df_CHAN = CHAN_moCHAN.forecast(steps=10)
print('forecasting',forecast_df_CHAN)


# In[2546]:


get_mape(crimes_CHAN_forecast.TOTAL[10:13],forecast_CHAN)


# In[2547]:


i=12
for j in range(len(forecast_df_CHAN)):
    crimes_CHAN_forecast.loc[i] = crimes_CHAN_forecast.YEAR[i-1]+1
    crimes_CHAN_forecast.TOTAL[i] = forecast_df_CHAN[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_CHAN_forecast["YEAR"], y= crimes_CHAN_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_CHAN_forecast["YEAR"][12:], y= crimes_CHAN_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2548]:


crime_DNH_df = crimes_total[crimes_total.STATE=='D & N HAVELI']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_DNH_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2549]:


crimes_DNH_forecast = crime_DNH_df.copy()
crimes_DNH_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_DNH_forecast = crimes_DNH_forecast.reset_index(drop = True)
plot(crimes_DNH_forecast)


# In[2593]:


DNH_arim = ARIMA(crimes_DNH_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(3,2,1))
DNH_moDNH = DNH_arim.fit()
DNH_moDNH.summary()


# In[2594]:


forecast_DNH = DNH_moDNH.predict(10,12)
print('prediction',forecast_DNH)
forecast_df_DNH = DNH_moDNH.forecast(steps=10)
print('forecasting',forecast_df_DNH)


# In[2595]:


get_mape(crimes_DNH_forecast.TOTAL[10:13],forecast_DNH)

	


# In[2596]:


i=12
for j in range(len(forecast_df_DNH)):
    crimes_DNH_forecast.loc[i] = crimes_DNH_forecast.YEAR[i-1]+1
    crimes_DNH_forecast.TOTAL[i] = forecast_df_DNH[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_DNH_forecast["YEAR"], y= crimes_DNH_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_DNH_forecast["YEAR"][12:], y= crimes_DNH_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2597]:


crime_DD_df = crimes_total[crimes_total.STATE=='DAMAN & DIU']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_DD_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2598]:


crimes_DD_forecast = crime_DD_df.copy()
crimes_DD_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_DD_forecast = crimes_DD_forecast.reset_index(drop = True)
plot(crimes_DD_forecast)


# In[2615]:


DD_arim = ARIMA(crimes_DD_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,1,0))
DD_moDD = DD_arim.fit()
DD_moDD.summary()


# In[2616]:


forecast_DD = DD_moDD.predict(10,12)
print('prediction',forecast_DD)
forecast_df_DD = DD_moDD.forecast(steps=10)
print('forecasting',forecast_df_DD)


# In[2617]:


get_mape(crimes_DD_forecast.TOTAL[10:13],forecast_DD)


# In[2618]:


i=12
for j in range(len(forecast_df_DD)):
    crimes_DD_forecast.loc[i] = crimes_DD_forecast.YEAR[i-1]+1
    crimes_DD_forecast.TOTAL[i] = forecast_df_DD[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_DD_forecast["YEAR"], y= crimes_DD_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_DD_forecast["YEAR"][12:], y= crimes_DD_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2619]:


#DELHI


# In[2620]:


crime_DEL_df = crimes_total[crimes_total.STATE=='DELHI UT']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_DEL_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2621]:


crimes_DEL_forecast = crime_DEL_df.copy()
crimes_DEL_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_DEL_forecast = crimes_DEL_forecast.reset_index(drop = True)
crimes_DEL_forecast


# In[2736]:


DEL_arim = ARIMA(crimes_DEL_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(1,2,0))
DEL_moDEL = DEL_arim.fit()
DEL_moDEL.summary()


# In[2737]:


forecast_DEL = DEL_moDEL.predict(10,12)
print('prediction',forecast_DEL)
forecast_df_DEL = DEL_moDEL.forecast(steps=10)
print('forecasting',forecast_df_DEL)


# In[2738]:


get_mape(crimes_DEL_forecast.TOTAL[10:13],forecast_DEL)

	


# In[2739]:


i=12
for j in range(len(forecast_df_DEL)):
    crimes_DEL_forecast.loc[i] = crimes_DEL_forecast.YEAR[i-1]+1
    crimes_DEL_forecast.TOTAL[i] = forecast_df_DEL[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_DEL_forecast["YEAR"], y= crimes_DEL_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_DEL_forecast["YEAR"][12:], y= crimes_DEL_forecast['TOTAL'][12:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2740]:


crime_LAK_df = crimes_total[crimes_total.STATE=='LAKSHADWEEP']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_DEL_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2741]:


crimes_LAK_forecast = crime_LAK_df.copy()
crimes_LAK_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_LAK_forecast = crimes_LAK_forecast.reset_index(drop = True)
plot(crimes_LAK_forecast)


# In[2746]:


LAK_arim = ARIMA(crimes_LAK_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,1,0))
LAK_moLAK = LAK_arim.fit()
LAK_moLAK.summary()


# In[2747]:


forecast_LAK = LAK_moLAK.predict(10,12)
print('prediction',forecast_LAK)
forecast_df_LAK = LAK_moLAK.forecast(steps=10)
print('forecasting',forecast_df_LAK)


# In[2748]:


get_mape(crimes_LAK_forecast.TOTAL[10:13],forecast_LAK)


# In[2749]:


i=12
for j in range(len(forecast_df_LAK)):
    crimes_LAK_forecast.loc[i] = crimes_LAK_forecast.YEAR[i-1]+1
    crimes_LAK_forecast.TOTAL[i] = forecast_df_LAK[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_LAK_forecast["YEAR"], y= crimes_LAK_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_LAK_forecast["YEAR"][13:], y= crimes_LAK_forecast['TOTAL'][13:],
                    name = "Estimated",line=dict(color='blue', width=4)))


# In[2750]:


crime_PUD_df = crimes_total[crimes_total.STATE=='PUDUCHERRY']
ts_decompose = sm.tsa.seasonal_decompose(np.array(crime_PUD_df['TOTAL']),
                                 model = "multiplicative", period=2)
ts_plot = ts_decompose.plot()


# In[2751]:


crimes_PUD_forecast = crime_PUD_df.copy()
crimes_PUD_forecast.drop(['STATE'], axis = 1, inplace = True)
crimes_PUD_forecast = crimes_PUD_forecast.reset_index(drop = True)
plot(crimes_PUD_forecast)


# In[2815]:


PUD_arim = ARIMA(crimes_PUD_forecast.TOTAL[0:10].astype(np.float64).to_numpy(), order=(2,0,0))
PUD_moPUD = PUD_arim.fit()
PUD_moPUD.summary()


# In[2816]:


forecast_PUD = PUD_moPUD.predict(10,12)
print('prediction',forecast_PUD)
forecast_df_PUD = PUD_moPUD.forecast(steps=10)
print('forecasting',forecast_df_PUD)


# In[2817]:


get_mape(crimes_PUD_forecast.TOTAL[10:13],forecast_PUD)


# In[2818]:


i=12
for j in range(len(forecast_df_PUD)):
    crimes_PUD_forecast.loc[i] = crimes_PUD_forecast.YEAR[i-1]+1
    crimes_PUD_forecast.TOTAL[i] = forecast_df_PUD[j]
    i = i+1

fig = go.Figure()
fig.add_trace(go.Scatter(x= crimes_PUD_forecast["YEAR"], y= crimes_PUD_forecast['TOTAL'],
                    name = "Actual",line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x= crimes_PUD_forecast["YEAR"][13:], y= crimes_PUD_forecast['TOTAL'][13:],
                    name = "Estimated",line=dict(color='blue', width=4)))

