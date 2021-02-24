import ee
import datetime
import folium
import numpy as np


# Define a method for displaying Earth Engine image tiles to folium map.
def add_ee_layer(self, eeImageObject, visParams, name):
  mapID = ee.Image(eeImageObject).getMapId(visParams)
  folium.raster_layers.TileLayer(
    tiles = mapID['tile_fetcher'].url_format,
    attr = "Map Data &copy; <a href='https://earthengine.google.com/'>Google Earth Engine</a>",
    name = name,
    overlay = True,
    control = True
  ).add_to(self)




#create a folium map and save it to the same directory as the script
def makeMapFromImage(filename, imageOverlay, layerName, startLoc, startZoom):
    #Create a folium map centered on Columbia
    m = folium.Map(location=startLoc,zoom_start=startZoom)

    #Earth engine visualization parameters for the layer we will overlay on the map
    visParams = {'palette':['D4394A', 'F66C45', 'FCAF62', 'FFE18B', 'E7F598', 'AADBA4','63C1A3', '3180BA'], 'gain':[.1], 'opacity':.8}

    #mask overlay so that 0 values are not displayed
    imageOverlay=imageOverlay.updateMask(imageOverlay)


    #Overlay the image from the earth engine dataset on the folium map
    m.add_ee_layer(imageOverlay, visParams, layerName)

    # Add a layer control panel to the map.
    m.add_child(folium.LayerControl())

    #Save the map to an HTML file 
    m.save(filename+".html")


#Iterate over images in a list and output a numpy array of total precipitation in a specified region
def getTotalPrecipitationForRegion(imlist, region, scale):
    listSize = imlist.length().getInfo()
    out = np.empty(shape=listSize)
    for i in range(listSize):
        tot = ee.Image(imlist.get(i)).reduceRegion(reducer = ee.Reducer.mean(), geometry = region, scale = scale, maxPixels = 1e9);
        out[i] = (tot.getInfo()['precipitation'])
        if i%10 == 0:
            print("aggregating precipitation ",i,"/",listSize)
    return out
    
#Iterate over each year and get total precipitation for region
#TODO: make this output a 2d numpy array instead of printing to the console
def listDailyPrecipitationTotalsForYear(region,dataset):

    #Convert the dataset into a list of earth engine image objects and get the first one from the list.  This is inefficient so use filter() when you can
    datalist = dataset.toList(dataset.size())

    #agregate the total rainfall for the area into a numpy where each entry is the aggreagate of the rainfall in each image in the list
    totals = getTotalPrecipitationForRegion(datalist, region, 200)
    return totals


def get_dataset(startDate,endDate):
    dataset = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filter(ee.Filter.date(startDate , endDate))
    return dataset

def select_data(dataset,data):
    dataout = dataset.select(data)
    return dataout

def viualize_data(dataset):
    print("this is your code: ")
    print(dataset)



def main():


#Get an authentication token from google, do every time if running on the cloud, do first time only if running local
#ee.Authenticate()
# Add EE drawing method to folium.
    folium.Map.add_ee_layer = add_ee_layer
#Initialize the earth engine API
    ee.Initialize()





#Geographic area to use
    geoArea = ee.Geometry.Rectangle(-79.55,12.43,-65.46,-4.86)

#Years to loop over and print data
    startDay = '01-01'
    endDay = '01-01'
    startYear = 1989
    endYear = 1990
    startDate = str(startYear)+'-'+startDay
    endDate = str(endYear+1)+'-'+endDay

    list=[]

    # Loops though the year listing daily totals
    for year in range(startYear , endYear):
        print("Year: " + str(year))

        # Date range to filter dataset on
        startDate = str(year) + '-' + startDay
        endDate = str(year + 1) + '-' + endDay

        # Get our dataset from earth engine and filter it on a date range
        dataset = get_dataset(startDate,endDate)
        precipitation = select_data(dataset,'precipitation')
        temp=listDailyPrecipitationTotalsForYear(geoArea,precipitation)
        list.append(temp)
    print(temp)



#Create overlays for the images and clip them to the area we want to analyze (uses lat/long coords)
#precipitationOverlay = ee.Image(datalist.get(0)).clip(geoArea)

#make map out of the overlay
#makeMapFromImage("map", precipitationOverlay, "Precipitation", [4.1156735, -72.9301367], 5)














#Things I still need to figure out
    #How to make a smooth gradient color transition instead of color pallette
    #How to normalize the dataset ranges to be between 0 and 1
    #How to load the background map in grayscale to avoid polluting the overlay with green from forested areas while keeping opacity






#Example of date time conversion between earth engine and python formats
# ee_date = ee.Date('2020-01-01')
# py_date = datetime.datetime.utcfromtimestamp(ee_date.getInfo()['value']/1000.0)

# py_date = datetime.datetime.utcnow()
# ee_date = ee.Date(py_date)

if __name__ == '__main__':
    main()