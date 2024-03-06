""" transformers module """

################################################################################

# IMPORTATION DES MODULES

from sklearn.base   import TransformerMixin

################################################################################

# DEFINITION DES CONSTANTES

EMPTY_DATA = "Empty data"

################################################################################

class TrDate(TransformerMixin):
    """ TrRain class """

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ add date units to dataframe """
        df          = data.copy()
        df["Year"]  = data["Date"].dt.year
        df["Month"] = data["Date"].dt.month
        return df

################################################################################

class TrRain(TransformerMixin):
    """ TrRain class """

    RAIN    = { "No" : 0, "Yes" : 1 }

    COLUMNS = [ "RainToday", "RainTomorrow" ]

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ replace rain values with numeric values """
        data.loc[:, __class__.COLUMNS] = data.loc[:, __class__.COLUMNS].replace(__class__.RAIN)
        return data

################################################################################

class TrSeasons(TransformerMixin):
    """ Seasons class """

    SEASONS = { 12 : 1, 1  : 1, 2  : 1,
                3  : 2, 4  : 2, 5  : 2,
                6  : 3, 7  : 3, 8  : 3,
                9  : 4, 10 : 4, 11 : 4 }

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ build season column """
        data["Season"] = data["Month"].apply(lambda x : __class__.SEASONS[x])
        return data

################################################################################

class TrSubsetNaN(TransformerMixin):
    """ SubsetNaN class """

    def __init__(self, n):
        """ constructor """
        self.n      = n

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ build cities list with least percentage NaN values """
        x_mean_nan           = data.groupby(["Location"]).apply(lambda x : x.notna().mean())
        x_mean_nan["MEAN"]   = x_mean_nan.mean(axis = 1)
        cities_nan           = x_mean_nan["MEAN"].sort_values(ascending = False)
        return data[data["Location"].isin(cities_nan.index[0:self.n])]

################################################################################

# Nouvelle définition des points cardinaux ajustée pour respecter les conventions mathématiques
POINTS = {
    "N": 4, "NNE": 3, "NE": 2, "ENE": 1,
    "E": 0, "ESE": 15, "SE": 14, "SSE": 13,
    "S": 12, "SSW": 11, "SW": 10, "WSW": 9,
    "W": 8, "WNW": 7, "NW": 6, "NNW": 5
}

class TrWindDirection(TransformerMixin):
    """ Transformer class for wind direction encoding with trigonometric functions. """
    
    COLUMNS = ["WindGustDir", "WindDir9am", "WindDir3pm"]

    def __init__(self):
        """ Constructor """
    
    def fit(self, data, y=None):
        """ Fit method, not used as we don't learn from the data """
        return self

    def transform(self, data):
        """ Replace wind direction with trigonometric encoding """
        for col in self.COLUMNS:
            # Encode wind direction using the adjusted POINTS dictionary
            data[col] = data[col].replace(POINTS)
            
            # Check for NaN values and apply trigonometric transformation only to non-NaN values
            valid_indices = data[col].notna()
            
            # Convert degrees to radians and calculate sine and cosine components
            data.loc[valid_indices, f"{col}_sin"] = np.sin(np.pi * data.loc[valid_indices, col] / 8)
            data.loc[valid_indices, f"{col}_cos"] = np.cos(np.pi * data.loc[valid_indices, col] / 8)
        return data

################################################################################

class TrZonesRainfall(TransformerMixin):
    """ TrZonesRainfall class """

    ZONES = { 50   : 1, 100  : 2, 200  : 3,
              300  : 4, 400  : 5, 600  : 6,
              1000 : 7, 1500 : 8, 2000 : 9,
              3000 : 10 }

    def __init__(self):
        """ constructor """

    def fit(self, data):
        """ fit method """
        if data is None:
            raise ValueError(EMPTY_DATA)
        return self

    def transform(self, data):
        """ build zones based on average annual rainfall """
        df          = data.copy()
        df["Zone"]  = 0
        cities      = data["Location"].unique()
        rain        = data.groupby(["Location", "Year"])["Rainfall"].sum()

        for city in cities:
            year_rainfall = rain.get(city)
            for year, rain_val in year_rainfall.items():
                for rain_limit, zone in __class__.ZONES.items():
                    if rain_val < rain_limit:
                        df.loc[(df["Location"] == city) & (df["Year"] == year), "Zone"] = zone
                        break
        return df

################################################################################
