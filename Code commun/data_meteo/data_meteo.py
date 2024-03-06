""" datameteo module """

from sklearn.pipeline   import Pipeline

import numpy            as np
import pandas           as pd


class DataMeteo:
    """ DataMeteo class """

    INITIAL_DATA = None

    def __init__(self, file):
        """ constructor """
        if __class__.INITIAL_DATA is None:
            __class__.INITIAL_DATA = pd.read_csv(file)
        self.data = __class__.INITIAL_DATA.copy()

    def build_dataset(self, transformers):
        """ build new dataset """
        pipeline    = Pipeline(steps = transformers)
        self.data   = pipeline.fit_transform(self.data)

    def change_type_columns(self, columns, new_type):
        """ change type of columns """
        for col in columns:
            self.data[col] = self.data[col].astype(new_type)

    def convert_to_datetime(self, col):
        """ convert Date column to datetime type """
        self.data[col] = pd.to_datetime(self.data[col])

    def delete_columns(self, columns):
        """ delete columns """
        clean_data = self.data.copy()
        return clean_data.drop(columns, axis = 1)

    def display_info_data(self, n):
        """ display basic info about dataframe """
        print(f"Shape : {self.data.shape}.\n")
        print(f"Info  : {self.data.info()}.\n")
        print(self.data.head(n))

    def display_percentage_nan(self):
        """ display percentage of NaN values in entire dataset """
        p = self.data.isna().sum().sum() / (self.data.shape[0] * self.data.shape[1]) * 100
        print(f"\nMissing values percentage : {p:.2f}%.")

    def fillna_data(self, fun, *args):
        """ fill nan values with fun function """
        cat_df          = self.data.select_dtypes(["object", "datetime"]).reset_index(drop = True)
        numeric_data    = self.data.select_dtypes(np.number).reset_index(drop = True)
        numeric_data    = fun(numeric_data, *args)
        self.data       = pd.concat([cat_df, numeric_data], axis = 1)

    def reset_data(self):
        """ reinitialize data with initial dataframe """
        self.data = __class__.INITIAL_DATA.copy()
