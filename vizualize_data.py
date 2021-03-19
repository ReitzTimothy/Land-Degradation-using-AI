import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mpl_dates
from datetime import datetime , timedelta
from matplotlib import ticker


def read_data(filename):
    year_data = np.load(filename)
    return year_data

def graphyears(number_of_years,start_year,dates):
    Months = ['January','March','May','July','September','November','January']
    labels=[]
    fig , ax = plt.subplots(figsize=(40,4))
    for i in range(0,number_of_years):
        data=read_data(str(start_year)+'year.npy')
        labels.append(str(start_year))
        ax.plot(dates[0],data[0],linewidth=2.5)


        start_year += 1
    ax.set_xticklabels(Months)
    ax.legend(labels,fancybox=True, shadow=True)
    ax.grid()
    plt.title('Precipitation over a year')
    plt.savefig('Year.png')
    plt.show()

def get_dates_for_graphing(startYear=1111):
    startDay = '01'
    endDay = '01'
    dateslist = []

    startDate = str(startYear) + '-'+'01'+'-' + startDay
    endDate = str(startYear+1) + '-' +'01'+'-'+ endDay

    dateslist.append(np.arange(startDate , endDate , dtype='datetime64[D]'))
    dateslist = np.array(dateslist)
    return dateslist
def make_dates():
    dates=[]
    Months=['01','02','03','04','05','06','07','08','09','10','11','12']
    thirtydaymonths=['09','04','06','11']
    for Month in Months:
        if Month in thirtydaymonths:
            for days in range(1,31):
                dates.append(Month+'-'+str(days))
        elif Month == '02':
            for days in range(1,29):
                dates.append(Month+'-'+str(days))
        else:
            for days in range(1,32):
                dates.append(Month+'-'+str(days))
    dates=np.array([dates])
    return dates

def main():
    number_of_years=input("Enter how many Years you want to graph: ")
    start_year=input('Please enter a starting year: ')
    #dates=make_dates()
    dates = get_dates_for_graphing()
    print(dates.shape)
    graphyears(int(number_of_years),int(start_year),dates)
if __name__ == '__main__':
    main()
