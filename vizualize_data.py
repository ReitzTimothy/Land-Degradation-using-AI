import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.dates as mpl_dates
from datetime import datetime , timedelta
from matplotlib import ticker


def read_data(filename):
    year_data = np.load(filename)
    return year_data

def graphyears(number_of_years,start_year):

    Months = ['January','March','May','July','September','November','January']
    labels=[]

    dates = get_dates_for_graphing()
    fig, ax = plt.subplots(figsize=(17,4))

    for counter in range(0,number_of_years):
        data=read_data(str(start_year)+'year.npy')
        labels.append(str(start_year))
        ax.plot(dates[0],data[0],linewidth=2.5)


        start_year += 1
    ax.set_xticklabels(Months)
    ax.legend(labels,fancybox=True, shadow=True)
    ax.grid()
    plt.title('Precipitation over a year')
    plt.xlabel('Months')
    plt.ylabel('Total precipitation')
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


def make_dates_for_year():

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

def average_month(year_Data):
    Months = ['01' , '02' , '03' , '04' , '05' , '06' , '07' , '08' , '09' , '10' , '11' , '12']
    thirtydaymonths = ['09' , '04' , '06' , '11']
    averages= []
    count=0
    for Month in Months:
        total=0
        if Month in thirtydaymonths:
            for day in range(0 , 30):
                total+=year_Data[0][count]
                count+=1
            average= total / 30
            averages.append(average)
        elif Month == '02':
            for day in range(0 , 28):
                total += year_Data[0][count]
                count += 1
            average = total / 29
            averages.append(average)
        else:
            for day in range(0 , 31):
                total += year_Data[0][count]
                count += 1
            average = total / 31
            averages.append(average)
    averages = np.array([averages])
    return averages

def graph_month_average(number_of_years,start_year):

    Months = ['January' ,'Febuary', 'March' ,'April', 'May' ,'June', 'July' ,'August', 'September' ,'October', 'November' ,'December']
    labels = []
    dates = make_months_for_graphing()
    fig, ax = plt.subplots(figsize=(12.5 , 4))
    for counter in range(0 , number_of_years):
        data = read_data(str(start_year) + 'year.npy')
        data=average_month(data)
        labels.append(str(start_year))
        ax.plot(dates[0] , data[0] , linewidth=2.5)
        start_year += 1
    ax.set_xticklabels(Months)
    ax.legend(labels , fancybox=True , shadow=True)
    ax.grid()
    plt.xlabel('Months')
    plt.ylabel('Total precipitation')
    plt.title('Average precipitation by month over a year')
    plt.savefig('Month_Average.png')
    plt.show()

def make_months_for_graphing():
    Months = ['01' , '02' , '03' , '04' , '05' , '06' , '07' , '08' , '09' , '10' , '11' , '12']
    Months = np.array([Months])
    return Months

def average_year(year_Data):
    total = 0
    for index in range(0,len(year_Data[0])):
        total+=year_Data[0][index]
    return (total/len(year_Data[0]))



def make_years_for_graphing(number_of_years,start_year):
    years=[]
    for i in range (number_of_years):
        years.append(start_year)
        start_year+=1
    years = np.array(years)
    return years

def graph_average_of_years(number_of_years,start_year):
    Months = ['January', 'Febuary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    labels = []
    dates = make_years_for_graphing(number_of_years,start_year)
    year_data=[]
    fig, ax = plt.subplots(figsize=(12.5, 4))
    for counter in range(0, number_of_years):
        data = read_data(str(start_year) + 'year.npy')
        year_data.append(average_year(data))
        labels.append(str(start_year))
        start_year += 1
    year_data=np.array(year_data)
    print(dates,year_data)
    ax.plot(dates, year_data, linewidth=2.5)
    ax.grid()
    plt.locator_params(axis='x', nbins=number_of_years)
    plt.xlabel('Years')
    plt.ylabel('Total precipitation')
    ax.legend(['Average'], fancybox=True, shadow=True)
    plt.title('Average precipitation by year')
    plt.savefig('Year_average.png')
    plt.show()

def main():


    number_of_years=int(input("Enter how many Years you want to graph: "))
    start_year=int(input('Please enter a starting year: '))
    graph_average_of_years(number_of_years,start_year)
    graphyears(number_of_years,start_year)
    graph_month_average(number_of_years,start_year)

if __name__ == '__main__':
    main()
