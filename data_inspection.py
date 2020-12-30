import pandas as pd
import numpy as np

from source.utils import get_household_complete_data


dataset_paths = {
    'household_info': 'datasets/London/informations_households.csv',
    'acorn_groups': 'datasets/London/acorn_details.csv',
    'weather_daily': 'datasets/London/weather_daily_darksky.csv',
    'weather_hourly': 'datasets/London/weather_hourly_darksky.csv',
    'holidays': 'datasets/London/uk_bank_holidays.csv',
    'daily_block': 'datasets/London/daily_dataset/daily_dataset/',
    'hh_block': 'datasets/London/halfhourly_dataset/halfhourly_dataset/'
}


households_info_dataset = pd.read_csv(dataset_paths['household_info'])
# acorn_groups = pd.read_csv(dataset_paths['acorn_groups'], encoding="ISO-8859-1")
# weather_daily = pd.read_csv(dataset_paths['weather_daily'])
# weather_hourly = pd.read_csv(dataset_paths['weather_hourly'])
# uk_holidays = pd.read_csv(dataset_paths['holidays'])

target_block = 'block_0'

daily_dataset = pd.read_csv(dataset_paths['daily_block'] + target_block + '.csv')
hourly_dataset = pd.read_csv(dataset_paths['hh_block'] + target_block + '.csv')

household_ids = list(np.unique(hourly_dataset['LCLid']))

print(len(np.unique(households_info_dataset['LCLid'])))
exit()

x, y = get_household_complete_data(hourly_dataset, household_ids[2])

print(x.shape, y.shape)
