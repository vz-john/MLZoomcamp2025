#!/usr/bin/python3
import pandas as pd
import numpy 

pandasVersion = pd.__version__
csvFile = pd.read_csv(open("car_fuel_efficiency.csv"))
countRow = csvFile.shape[0]
fuelTypes = len(csvFile['fuel_type'].unique())
missingValues = csvFile.isna().any().sum()
maxEficiencyFuelAsia = (csvFile[csvFile['origin'] == 'Asia'])['fuel_efficiency_mpg'].max()
medianHorsepower = csvFile['horsepower'].median()
modeHorsepower = csvFile['horsepower'].value_counts().idxmax()
newMedianHorsepower = (csvFile['horsepower'].fillna(value=modeHorsepower)).median()
if medianHorsepower < newMedianHorsepower:
    change=("Yes, it increased")
elif medianHorsepower > newMedianHorsepower:
    change("Yes, it decreased")
elif medianHorsepower == newMedianHorsepower:
    change=("No")
else:
    change=("Error")

x = numpy.asarray((csvFile[csvFile['origin'] == 'Asia'])[['vehicle_weight', 'model_year']].iloc[:7])
tx = numpy.transpose(x)
xtxInverse = numpy.linalg.inv(numpy.matmul(tx, x))
y = [1100, 1300, 800, 900, 1000, 1100, 1200]
w = numpy.sum(numpy.multiply(numpy.matmul(xtxInverse, tx), y))

print("Q1. Pandas version:")
print("  Version {}".format(pandasVersion))
print("Q2. Records count")
print("  How many records are in the dataset? {}".format(countRow))
print("Q3. Fuel types")
print("  How many fuel types are presented in the dataset? {}".format(fuelTypes))
print("Q4. Missing values")
print("  How many columns in the dataset have missing values? {}".format(missingValues))
print("Q5. Max fuel efficiency")
print("  What's the maximum fuel efficiency of cars from Asia? {}".format(maxEficiencyFuelAsia))
print("Q6. Median value of horsepower")
print("  1. Find the median value of horsepower column in the dataset. {}".format(medianHorsepower))
print("  2. Next, calculate the most frequent value of the same horsepower column. {}".format(modeHorsepower))
print("  3. Use fillna method to fill the missing values in horsepower column with the most frequent value from the previous step.")
print("  4. Now, calculate the median value of horsepower once again. {} ({})".format(change, newMedianHorsepower))
print("Q7. Sum of weights "+str(w))
