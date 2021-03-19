import ee
import datetime
import folium
import numpy as np
import geemap
import pandas as pd
import tensorflow as tf
import Model
import sklearn
from sklearn.preprocessing import MinMaxScaler

#Get an authentication token from google, do every time if running on the cloud, do first time only if running local
#ee.Authenticate()

#Initialize the earth engine API
ee.Initialize()



#Geographic area to use for rectangular input
geoArea = ee.Geometry.Rectangle(-80,13,-62,-5)
#Scale to use when converting earth engine data into pixels (may be CHIRPS specific, IDK)
imScale = 200









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

# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

#create a folium map and save it to the same directory as the script
def make_map_from_image(filename, imageOverlay, layerName, startLoc, startZoom):
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
def get_total_precipitation_for_region(imlist, region, scale):
    listSize = imlist.length().getInfo()
    out = np.empty(shape=listSize)
    for i in range(listSize):
        tot = ee.Image(imlist.get(i)).reduceRegion(reducer = ee.Reducer.sum(), geometry = region, scale = scale, maxPixels = 1e9);
        out[i] = (tot.getInfo()['precipitation'])
        if i%10 == 0:
            print("aggregating precipitation ",i,"/",listSize)
    return out
    
#Iterate over each year and get total precipitation for region.  For leap years the last day is truncated
def list_daily_precipitation_totals_for_year_range(startYear, endYear, region, scale):
    startDay = '01-01'
    endDay = '01-01'
    
    output = np.empty([endYear-startYear, 365])
    
    for year in range(startYear, endYear):
        print("Year: "+str(year))

        #Date range to filter dataset on
        startDate = str(year)+'-'+startDay
        endDate = str(year+1)+'-'+endDay

        #Get our dataset from earth engine and filter it on a date range
        dataset = get_dataset(startDate, endDate);
        precipitation = dataset.select('precipitation');
        

        #Convert the dataset into a list of earth engine image objects and get the first one from the list.  This is inefficient so use filter() when you can
        datalist = precipitation.toList(dataset.size())
        
        #truncate the last item if its a leap year
        if datalist.length().getInfo() > 365:
            datalist = datalist.remove(datalist.get(365))

        #agregate the total rainfall for the area into a numpy array
        output[year-startYear] = get_total_precipitation_for_region(datalist, region, scale)
        
    return output

#Get pixel values as numpy array from CHIRPS images for a certain date range
def get_precipitation_maps_for_range(startDate, endDate, geoArea, scale):
    print("Begin loading data from EE...")
    arrList = []
    data = get_dataset(startDate, endDate)
    l = data.toList(data.size())
    
    
    
    for i in range(l.size().getInfo()):
        if i%100 == 0:
            print(i)
        im = ee.Image(l.get(i))
        arr = geemap.ee_to_numpy(im,region = geoArea, default_value = 0)
        arrList.append(arr)
    
    output = np.concatenate(arrList, axis = 2)
    print("Done")
    return output

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


    

        
    
    
#Ex 1) Total up precipitation across the region and print it


    #Years to loop over and print data
    startYear = 2000
    endYear = 2002

    #precipVals = list_daily_precipitation_totals_for_year_range(startYear, endYear, geoArea, imScale)
    #print(precipVals)




#Ex 2) Displaying data on a folium map, in this case precipitation for a single day

    #get the data for a single day
    dataset = get_dataset('1989-02-01','1989-02-02')
    datalist = dataset.toList(dataset.size())

    #Create overlays for the images and clip them to the area we want to analyze (uses lat/long coords)
    precipitationOverlay = ee.Image(datalist.get(0)).clip(geoArea)

    #make map out of the overlay
    make_map_from_image("map", precipitationOverlay, "Precipitation", [4.1156735, -72.9301367], 5)
    
    

#Ex 3) UNFINISHED   Train a model to make predictions based on the previous day only
    width = 360
    height = 360
    batchsz = 32
    
    
    #Load map data into numpy array for training
    maps = get_precipitation_maps_for_range('1989-01-01', '1990-01-01', geoArea, imScale)

    #Copy the data into input and expected output.  In this case its just a day's image for input, and the following day's image for expected output
    inp = np.empty(shape = (maps.shape[2]-1, maps.shape[0], maps.shape[1], 1))
    out = np.empty(shape = (maps.shape[2]-1, maps.shape[0], maps.shape[1],  1))
    
    
    #Iterate over the data and create input and output arrays
    samples = maps.shape[2]-1
    for i in range(samples):
        inp[i,:,:,0] = maps[:,:,i]
        out[i,:,:,0] = maps[:,:,i+1]
    
    
    print(inp.shape)
    print(out.shape)
    
    #Normalize the data.  It is stored sequentially in a 4D array so all the dimensions except samples must be flattened and reshaped after normalization
    norm_in = MinMaxScaler().fit(inp.reshape((samples,-1)))
    inp = norm_in.transform(inp.reshape((samples,-1))).reshape(samples,width,height,1)
    out = norm_in.transform(out.reshape((samples,-1))).reshape(samples,width,height,1)
    
    
    #Create Tensorflow model based on an autoencoder for simple predictions
    model = Model.getAutoEncoder((360, 360, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    print(model.summary())
    
    #Fit the model to the data and save it
    model.fit(inp, out, batch_size=batchsz, steps_per_epoch=(int)(samples/batchsz), epochs=30, max_queue_size = 1, shuffle = True)
    #model.save('AutoEncoder.model')


#Ex: 4 Save entire CHIRPS DAILY dataset until 01/01/2021 in numpy file

    maps = get_precipitation_maps_for_range('1981-01-01', '2021-01-01', geoArea, imScale)
    np.save("H:\\Datasets\\ChirpsDaily", maps)















#Things I still need to figure out
    #How to make a smooth gradient color transition instead of color pallette
    #How to load the background map in grayscale to avoid polluting the overlay with green from forested areas while keeping opacity






#Example of date time conversion between earth engine and python formats
# ee_date = ee.Date('2020-01-01')
# py_date = datetime.datetime.utcfromtimestamp(ee_date.getInfo()['value']/1000.0)

# py_date = datetime.datetime.utcnow()
# ee_date = ee.Date(py_date)

if __name__ == '__main__':
    main()