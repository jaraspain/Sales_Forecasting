import pandas as pd
import matplotlib.pyplot as plt

def extract():
    ## Extraction of XLSX file and putting it into suitable dataframes
    ## Stored in Data folder
    xls = pd.ExcelFile('Data/1990–2019_PC_by_country_EU+EFTA.xlsx')
    dataframes = []
    #x = []
    #y = []

    for i in range (1990, 2009):
        df = pd.read_excel(xls, str(i))
        print('Extracted {}'.format(i))
        #y.append(df.iloc[16,13])
        #x.append(str(i))
        #print(df.iloc[16,13])
        df = df.iloc[1:17, 1:13]
        df = df.transpose()
        dataframes.append(df)
        #print(df)
    '''
    plt.figure(figsize=(12,12))
    plt.plot(x,y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.xticks(x, rotation='vertical')
    plt.title('EU-15 Car Sales over 1990-2008', fontsize = 20)
    plt.xlabel('Year', fontsize= 18)
    plt.ylabel('Sales', fontsize = 18)
    plt.savefig('Plot1.jpg')
    '''
    xf = pd.concat(dataframes)
    #print(xf)
    return xf

def describe_data():
    return extract().describe().transpose()
def save_df(dataframe):
    x = 1990
    for i in range(len(dataframe)):
        df = dataframe[i]
        df.to_csv('Dataframes/'+str(x)+'.csv')
        x += 1
    return None

def plot_monthly(dataframes):
    x = []
    y_1 = []
    y_2 = []
    y_3 = []
    y_4 = []
    y_avg = []
    t = 1990
    months = [1,4,7,11]
    for i in range(len(dataframes)):
        y_1.append((dataframes[i].iloc[-1, 1]))
        y_2.append(dataframes[i].iloc[-1, 4])
        y_3.append(dataframes[i].iloc[-1, 7])
        y_4.append(dataframes[i].iloc[-1, 11])
        x.append(str(t))
        y_avg.append(dataframes[i].iloc[-1,1:].mean())
        t+=1
    plt.figure(figsize=(12, 12))
    plt.plot(x, y_1,color='yellow', marker='o', linestyle='dashed', linewidth=1, markersize=8, label = 'Month Sales for Jan')
    plt.plot(x, y_2, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=8,
             label='Month Sales for Apr')
    plt.plot(x, y_3, color='orange', marker='o', linestyle='dashed', linewidth=1, markersize=8,
             label='Month Sales for July')
    plt.plot(x, y_4, color='red', marker='o', linestyle='dashed', linewidth=1, markersize=8,
             label='Month Sales for Nov')
    plt.plot(x, y_avg, color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=10, label = 'Avg Monthly Sales')
    plt.xticks(x, rotation='vertical')
    plt.legend()
    plt.title('Monthly Sales Over 1990-2008 for months ', fontsize=20)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Sales', fontsize=18)
    plt.savefig('Plot3.jpg')
    return None
