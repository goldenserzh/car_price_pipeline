import re
import pandas as pd




class Preprocessing:

  BASE_COLORS = ['Black', 'White', 'Gray', 'Silver', 'Blue', 'Red',
                   'Green', 'Gold', 'Brown', 'Orange', 'Beige', 'Yellow']

  FUEL_MAP = {
        'flex': 'Flex Fuel', 'flex fuel': 'Flex Fuel',
        'e85': 'Flex Fuel', 'ethanol': 'Flex Fuel',
        'petrol': 'Gasoline'
    }

  MISSING_VALUES = ['-', '–', '', 'None', 'none', 'NULL', 'N/A', 'na']

  def __init__(self, df:pd.DataFrame, is_linear: bool = True) -> None:
      self.df = df
      self.is_linear = is_linear

  def parse_engine(self, feature_str):

    power_match = re.search(r'(\d+\.?\d*)HP', feature_str)
    power = float(power_match.group(1)) if power_match else None

    displacement_match = re.search(r'(\d+\.?\d*)L', feature_str)
    displacement = float(displacement_match.group(1)) if displacement_match else None

    cylinders = None
    cylinders_match = re.search(r'(\d+)\s*Cylinder', feature_str)
    if cylinders_match:
        cylinders = int(cylinders_match.group(1))
    else:
        v_match = re.search(r'V(\d+)', feature_str)
        if v_match:
            cylinders = int(v_match.group(1))

    fuel_match = re.search(r'(Gasoline|Diesel|Electric|Petrol|Hybrid|Flex|Flex Fuel|E85|Ethanol)', feature_str, re.IGNORECASE)
    if fuel_match:
        fuel = fuel_match.group(1).capitalize()
        if fuel.lower() in ['Flex', 'Flex fuel', 'E85', 'Ethanol']:
            fuel = 'Flex Fuel'
        elif fuel.lower() == 'Petrol':
            fuel = 'Gasoline'
    else:
        fuel = None
    return pd.Series([power, displacement, cylinders, fuel])

  def parse_transmission(self, trans_str):
    if pd.isna(trans_str):
        return pd.Series([None, None])

    trans_str = str(trans_str)

    is_automatic = 1

    manual_keywords = ['M/T', 'Manual']
    if any(keyword in trans_str for keyword in manual_keywords):
        is_automatic = 0

    if 'CVT' in trans_str:
        is_automatic = 1

    speed_match = re.search(r'(\d+)-?\s*(Speed|A/T|M/T|Automatic)', trans_str)

    if speed_match:
        speeds = int(speed_match.group(1))
    elif 'A/T' in trans_str and 'Speed' not in trans_str:
        speeds = 4
    elif 'M/T' in trans_str and 'Speed' not in trans_str:
        speeds = 5
    elif 'Automatic' in trans_str and 'Speed' not in trans_str:
        speeds = 4
    elif 'CVT' in trans_str:
        speeds = 0
    else:
        speeds = None

    return pd.Series([is_automatic, speeds])


  def extract_color(self, name_feature: str):
      base_colors = ['Black', 'White', 'Gray', 'Silver', 'Blue', 'Red',
                'Green', 'Gold', 'Brown', 'Orange', 'Beige', 'Yellow']

      self.df[name_feature] = self.df[name_feature].apply(
          lambda x: next((c for c in base_colors if c.lower() in str(x).lower()), 'Other')
      )
      return self.df

  def detect_missing(self):
    categor_feature = self.df.select_dtypes(['object']).columns.to_list()
    numerical_feature = self.df.select_dtypes([np.number]).columns.to_list()

    for col in categor_feature:
      self.df[col] = self.df[col].fillna('Missing')

    for col in numerical_feature:
        self.df[col] = self.df[col].fillna(
            self.df.groupby(['brand', 'model'])[col].transform('median')
        )

        self.df[col] = self.df[col].fillna(
            self.df.groupby('brand')[col].transform('median')
        )

        if self.df[col].isna().any():
            global_median = self.df[col].median()
            self.df[col] = self.df[col].fillna(global_median)
    return self.df

  def make_preprocessing(self) -> pd.DataFrame:
    self.df = self.df.drop("id", axis=1)
    self.df[['power_hp', 'engine_volume_l', 'cylinders_count', 'type_fuel']] = self.df['engine'].apply(self.parse_engine)
    self.df = self.df.drop('engine', axis=1)
    self.df['accident'] = self.df['accident'].apply(lambda x: 1 if pd.notna(x) and x != "None reported" else 0)
    self.df['clean_title'] = self.df['clean_title'].apply(lambda x: 1 if x == 'Yes' else 0)
    self.df[['is_automatic', 'gears_count']] = self.df['transmission'].apply(self.parse_transmission)
    self.df = self.df.drop('transmission', axis=1)
    self.df['model'] = self.df['model'].str.split().str[0]
    self.df = self.extract_color('ext_col')
    self.df = self.extract_color('int_col')
    self.df['fuel_type'] = self.df['fuel_type'].replace(['-', '–', '', 'None', 'none', 'NULL'], np.nan)
    self.df = self.detect_missing()
    return self.df
