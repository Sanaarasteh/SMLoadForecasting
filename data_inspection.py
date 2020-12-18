import pandas as pd

import matplotlib.pyplot as plt


dataset = pd.read_csv('datasets/AppliancesEnergy/energydata_complete.csv')

# Getting the attributes
attributes = dataset.columns
print(attributes)

# Getting the number of instances
print(dataset.shape)

# Check if there is any missing values
print(dataset.isnull().any())

# Dataset statistical summary
print(dataset.describe())

plt.subplot(3, 3, 1)
plt.plot(dataset['T1'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 2)
plt.plot(dataset['T2'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 3)
plt.plot(dataset['T3'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 4)
plt.plot(dataset['T4'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 5)
plt.plot(dataset['T5'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 6)
plt.plot(dataset['T6'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 7)
plt.plot(dataset['T7'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 8)
plt.plot(dataset['T8'], dataset['Appliances'], 'bo')
plt.subplot(3, 3, 9)
plt.plot(dataset['T9'], dataset['Appliances'], 'bo')
plt.show()

plt.plot(dataset['T_out'], dataset['Appliances'], 'bo')
plt.show()


