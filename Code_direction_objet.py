#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np

data = pd.read_csv('weatherAUS.csv', sep = ',', header = 0)


# In[7]:


class TrWindDirection():
    """ TrWindDirection class """

    POINTS  = { "N" : 1,  "NNE" : 2,  "NE" : 3,  "ENE" : 4,
                "E" : 5,  "ESE" : 6,  "SE" : 7,  "SSE" : 8,
                "S" : 9,  "SSW" : 10, "SW" : 11, "WSW" : 12,
                "W" : 13, "WNW" : 14, "NW" : 15, "NNW" : 16 }

    COLUMNS = [ "WindGustDir", "WindDir9am", "WindDir3pm" ]

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ replace wind direction with numeric values """
        df = data.copy()
        df.loc[:, __class__.COLUMNS]    = df.loc[:, __class__.COLUMNS].replace(__class__.POINTS)
        df.loc[:, "WindGustDir_sin"]    = df.loc[:, "WindGustDir"].apply(lambda x : np.sin(np.pi * x / 8) if not np.isnan(x) else x)
        df.loc[:, "WindGustDir_cos"]    = df.loc[:, "WindGustDir"].apply(lambda x : np.cos(np.pi * x / 8) if not np.isnan(x) else x)
        df.loc[:, "WindDir9am_sin"]     = df.loc[:, "WindDir9am"].apply(lambda x : np.sin(np.pi * x / 8) if not np.isnan(x) else x)
        df.loc[:, "WindDir9am_cos"]     = df.loc[:, "WindDir9am"].apply(lambda x : np.cos(np.pi * x / 8) if not np.isnan(x) else x)
        df.loc[:, "WindDir3pm_sin"]     = df.loc[:, "WindDir3pm"].apply(lambda x : np.sin(np.pi * x / 8) if not np.isnan(x) else x)
        df.loc[:, "WindDir3pm_cos"]     = df.loc[:, "WindDir3pm"].apply(lambda x : np.cos(np.pi * x / 8) if not np.isnan(x)  else x)
        df = df.drop(__class__.COLUMNS, axis = 1)

        col = ['9am', '3pm', 'gust']
        
        for i in col:
            df = self.replace_middle(df, i)
            df = self.replace_nowind(df, i)
            df = self.replace_sameday(df, i)
            df = self.replace_mode(df, i)
            df = self.replace_clust_gust(df, i)
            
        return df

    def replace_middle(self, df, col):

        end = len(df) - 1
    
        if col == '9am':
            dir1sin = "WindDir9am_sin"
            dir1cos = "WindDir9am_cos"
            dir2sin = "WindDir3pm_sin"
            dir2cos = "WindDir3pm_cos"
            speed = 'WindSpeed9am'
            gust = False
        elif col == '3pm':
            dir1sin = "WindDir3pm_sin"
            dir1cos = "WindDir3pm_cos"
            dir2sin = "WindDir9am_sin"
            dir2cos = "WindDir9am_cos"
            speed = 'WindSpeed3pm'
            gust = False
        elif col == 'gust':
            dir1sin = "WindGustDir_sin"
            dir1cos = "WindGustDir_cos"
            dir2sin = "WindGustDir_sin"
            dir2cos = "WindGustDir_cos"
            speed = ''
            gust = True
        
        isn1 = df[dir1sin].isna().sum()
        isn2 = df[dir1cos].isna().sum()
        print("Number of NaN's in column ", col, " before 'replace middle': ", isn1 ,"-", isn2)
        
        
        # For each line in df except the last to avoid index out of bound
        
        for i in range (0, end):

            if col == '9am':
                plsmin = i-1
                plsmini = i
                cond = True
                cond2 = df.loc[i, speed] != 0
            elif col == '3pm':
                plsmin = i+1
                plsmini = i
                cond = True
                cond2 = df.loc[i, speed] != 0
            elif col == 'gust':
                plsmin = i+1
                plsmini = i-1
                if i != 0:
                    cond = df.loc[i, 'Location'] == df.loc[i-1, 'Location']
                else:
                    cond = False
                cond2 = True
                
            # i should not be 0 to avoid index out of bound when reading the precedent line
            # and the location should be identical to location in precedent line to avoid
            # shift between cities
    
            if i != 0 and df.loc[i, 'Location'] == df.loc[plsmin, 'Location'] and cond:
                
                # There should be a Nan in this line and windspeed not 0.
                # Zero windspeed is delt with in another function
                
                if pd.isna(df.loc[i, dir1sin]) and cond2:
                    
                    # No NaN at 3 pm and no NaN at 3pm the day before and neither is 0,
                    # that is no direction
                    
                    if pd.notna(df.loc[plsmini, dir2sin]) and pd.notna(df.loc[plsmin, dir2sin]) and df.loc[plsmini, dir2sin] != 0 and df.loc[plsmin, dir2sin] != 0:
                        
                        # Calculate middle between direction at 3pm the day before and 3pm the
                        # same day.
    
                        vec1 = [df.loc[plsmini, dir2sin], df.loc[plsmini, dir2cos]]
                        vec2 = [df.loc[plsmin, dir2sin], df.loc[plsmin, dir2cos]]

                        vec_sum = np.add(vec1, vec2)

                        res = vec_sum/(np.sqrt(vec_sum[0]**2 + vec_sum[1]**2))                        
                        
                        #print("Res: ", i, "***", res)
                        
                        # Replave NaN with the middle value
                        
                        df.at[i, dir1sin] = res[0]
                        df.at[i, dir1cos] = res[1]
                        
                        #print(df.loc[i, 'WindDir9am'])    
        
        isn1 = df[dir1sin].isna().sum()
        isn2 = df[dir1cos].isna().sum()
        print("Number of NaN's in column ",  col, " after 'replace middle': ", isn1 ,"-", isn2)
                        
        return df

    def replace_nowind (self, df, col):
    
        end = len(df)
    
        if col == '9am':
            dir1sin = "WindDir9am_sin"
            dir1cos = "WindDir9am_cos"
            speed = 'WindSpeed9am'
            go = True
        elif col == '3pm':
            dir1sin = "WindDir3pm_sin"
            dir1cos = "WindDir3pm_cos"
            speed = 'WindSpeed3pm'
            go = True
        elif col == 'gust':
            dir1sin = ""
            dir1cos = ""
            speed = ''
            go = False
                
        if go == True:
            
            for i in range (0, end):
                
                # If direction is NaN and wind speed is 0
                
                if pd.isna(df.loc[i, dir1sin]) and df.loc[i, speed] == 0:
                    
                    # Replace wind direction with 0
                    
                    df.at[i, dir1sin] = 0
                    df.at[i, dir1cos] = 0
                    
                    #print(df.loc[i, 'WindDir9am'])
            
            isn1 = df[dir1sin].isna().sum()
            isn2 = df[dir1cos].isna().sum()
            print("Number of NaN's in column ",  col, " after 'replace nowind': ", isn1 ,"-", isn2)
            
        return(df)

    def replace_sameday (self, df, col):
    
        end = len(df)

        if col == '9am':
            dir1sin = "WindDir9am_sin"
            dir1cos = "WindDir9am_cos"
            dir2sin = "WindDir3pm_sin"
            dir2cos = "WindDir3pm_cos"
            go = True
        elif col == '3pm':
            dir1sin = "WindDir3pm_sin"
            dir1cos = "WindDir3pm_cos"
            dir2sin = "WindDir9am_sin"
            dir2cos = "WindDir9am_cos"
            go = True
        elif col == 'gust':
            dir1sin = ""
            dir1cos = ""
            dir2sin = ""
            dir2cos = ""
            go = False

        if go == True:
        
            for i in range (0, end):
                    
                # If direction at 9am is NaN and direction at 3pm is not
                
                if pd.isna(df.loc[i, dir1sin]) and pd.notna(df.loc[i, dir2sin]):
                        
                    # Set wind direction at 9am equal to the wind direction at 3pm (same day)

                    df.at[i, dir1sin] = df.loc[i, dir2sin]
                    df.at[i, dir1cos] = df.loc[i, dir2cos]
                                                
                    #print(df.loc[i, 'WindDir9am'])
            
            isn1 = df[dir1sin].isna().sum()
            isn2 = df[dir1cos].isna().sum()
            print("Number of NaN's in column ",  col, " after 'replace same day': ", isn1 ,"-", isn2)
            
        return df

    def replace_mode (self, df, col):
    
        end = len(df)

        if col == '9am':
            dir1sin = "WindDir9am_sin"
            dir1cos = "WindDir9am_cos"
        elif col == '3pm':
            dir1sin = "WindDir3pm_sin"
            dir1cos = "WindDir3pm_cos"
        elif col == 'gust':
            dir1sin = "WindDir3pm_sin"
            dir1cos = "WindDir3pm_cos"
        
        # Mode for the direction in the same city and hour. If double mode, the first is chosen.
        
        mod_sin = df.groupby(df['Location'])[dir1sin].agg(lambda x: pd.Series.mode(x)[0]).to_frame()
        mod_cos = df.groupby(df['Location'])[dir1cos].agg(lambda x: pd.Series.mode(x)[0]).to_frame()
        
        mod_sin = mod_sin.reset_index(level=['Location'])
        mod_cos = mod_cos.reset_index(level=['Location'])
        
        for i in range (0, end):
            
            # If direction is NaN replacew with mode.
            
            if pd.isna(df.loc[i, dir1sin]):                
                df.at[i, dir1sin] = mod_sin[mod_sin['Location'] == df.loc[i, 'Location']][dir1sin]
                df.at[i, dir1cos] = mod_cos[mod_cos['Location'] == df.loc[i, 'Location']][dir1cos]
                
                #print("9am", df.loc[i, 'WindDir9am'])
        
        isn1 = df[dir1sin].isna().sum()
        isn2 = df[dir1cos].isna().sum()
        print("Number of NaN's in column ",  col, " after 'replace mode': ", isn1 ,"-", isn2)
            
        return df

    def replace_clust_gust(self, df, col):
    
        end = len(df)
    
        if col == '9am':
            go = False
        elif col == '3pm':
            go = False
        elif col == 'gust':
            go = True
    
        if go == True:
            
            dict = {'Albury': [4], 'BadgerysCreek': [6], 'Cobar': [4], 'CoffsHarbour': [2], 'Moree': [4],
               'Newcastle': [5], 'NorahHead': [2], 'NorfolkIsland': [2], 'Penrith': [6], 'Richmond': [6],
               'Sydney': [5], 'SydneyAirport': [5], 'WaggaWagga': [4], 'Williamtown': [5],
               'Wollongong': [5], 'Canberra': [7], 'Tuggeranong': [7], 'MountGinini': [8], 'Ballarat': [7],
               'Bendigo': [6], 'Sale': [6], 'MelbourneAirport': [6], 'Melbourne': [6], 'Mildura': [4],
               'Nhil': [4], 'Portland': [6], 'Watsonia': [6], 'Dartmoor': [6], 'Brisbane': [3], 'Cairns': [1],
               'GoldCoast': [2], 'Townsville': [1], 'Adelaide': [5], 'MountGambier': [6], 'Nuriootpa': [6],
               'Woomera': [4], 'Albany': [6], 'Witchcliffe': [5], 'PearceRAAF': [5], 'PerthAirport': [5],
               'Perth': [5], 'SalmonGums': [4], 'Walpole': [6], 'Hobart': [7], 'Launceston': [7],
               'AliceSprings': [3], 'Darwin': [1], 'Katherine': [1], 'Uluru': [3]}
               
            df['Climate_Type'] = df['Location']
            
            df['Climate_Type'] = df['Climate_Type'].replace(dict)
            
            
            #display(df.head())
            
            # Mode for the direction in the same climate zone. If double mode, the first is chosen.
            
            mod_sin = df.groupby(df['Climate_Type'])['WindGustDir_sin'].agg(lambda x: pd.Series.mode(x, dropna=True)[0]).to_frame()
            mod_cos = df.groupby(df['Climate_Type'])['WindGustDir_cos'].agg(lambda x: pd.Series.mode(x, dropna=True)[0]).to_frame()
            
            mod_sin = mod_sin.reset_index(level=['Climate_Type'])
            mod_cos = mod_cos.reset_index(level=['Climate_Type'])
            
            #display(mod_gust)
            
            for i in range (0, end):
                
                # If direction is NaN replacew with mode for climate type.
                
                if pd.isna(df.loc[i, 'WindGustDir_sin']):                
                    df.at[i, 'WindGustDir_sin'] = mod_sin[mod_sin['Climate_Type'] == df.loc[i, 'Climate_Type']]['WindGustDir_sin']
                    df.at[i, 'WindGustDir_cos'] = mod_cos[mod_cos['Climate_Type'] == df.loc[i, 'Climate_Type']]['WindGustDir_cos']
            
        
            isn1 = df['WindGustDir_sin'].isna().sum()
            isn2 = df['WindGustDir_cos'].isna().sum()
            print("Number of NaN's in column ",  col, " after 'replace climate cluster': ", isn1 ,"-", isn2)
            
            df = df.drop('Climate_Type', axis=1)
            
        return df




# In[8]:


dirw = TrWindDirection()

df_res = dirw.transform(data)




# In[ ]:




