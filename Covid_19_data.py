import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv('covid_19_data.csv')
df.drop(['SNo', 'Last Update'], axis=1, inplace=True)
df.rename(columns={'ObservationDate': 'Date', 'Province/State': 'Province', 'Country/Region': 'Country'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.fillna('NA')

imputer = SimpleImputer(strategy='constant')
df2 = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df3 = df2.groupby(['Country', 'Date'])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
df4 = df3.groupby(['Date'])[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
countries = df3['Country'].unique()


def countries_data():
    for index in range(0, len(countries)):
        country = df3[df3['Country'] == countries[index]].reset_index()
        plt.scatter(np.arange(0, len(country)), country['Confirmed'], color='b', label='Confirmed')
        plt.scatter(np.arange(0, len(country)), country['Recovered'], color='g', label='Recovered')
        plt.scatter(np.arange(0, len(country)), country['Deaths'], color='r', label='Deaths')
        plt.title(countries[index])
        plt.xlabel('Days since the first suspect')
        plt.ylabel('Number of cases')
        plt.legend()
        plt.show()


def world_data():
    c = df4
    plt.scatter(np.arange(0, len(c)), c['Confirmed'], color='blue', label='Confirmed')
    plt.scatter(np.arange(0, len(c)), c['Recovered'], color='green', label='Recovered')
    plt.scatter(np.arange(0, len(c)), c['Deaths'], color='red', label='Deaths')
    plt.title('World')
    plt.xlabel('Days since the first suspect')
    plt.ylabel('Number of cases')
    plt.legend()
    plt.show()
