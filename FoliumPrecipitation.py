import ee
import datetime
import folium
import numpy as np
import geemap
import pandas as pd
import DataProcessing
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

#Get an authentication token from google, do every time if running on the cloud, do first time only if running local
#ee.Authenticate()

#Initialize the earth engine API
ee.Initialize()



#Geographic area to use for rectangular input
geoArea = ee.Geometry.Rectangle(-80,13,-62,-5)
#Scale to use when converting earth engine data into pixels
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

#Save entire CHIRPS DAILY dataset within date range in numpy file
def download_data_as_nparray(start_date, end_date, filename):
    maps = get_precipitation_maps_for_range(start_date, end_date, geoArea, imScale)
    np.save("H:\\Datasets\\ChirpsDaily\\"+filename, maps)

#Load the numpy file for dataset, normalize, and split it into batches for training.  Then save batched and normalized data to folder
def batch_numpy_file_to_folder(input_filename, save_folder_name, batch_size):
    fp = "H:\\Datasets\\ChirpsDaily"
    maps = np.load(fp+"\\"+input_filename)
    width = 360
    height = 360
    

    norm_in = MinMaxScaler().fit(maps.reshape((maps.shape[2],-1)))
    maps = norm_in.transform(maps.reshape((maps.shape[2],-1))).reshape(width,height, maps.shape[2])
    print(maps[:,:,0])
    
    
    num_batches = int((maps.shape[2]-1)/batch_size)
    print(num_batches)
    for b in range(num_batches):
        print(b)
        inp = np.empty(shape = (batch_size, maps.shape[0], maps.shape[1], 1))
        out = np.empty(shape = (batch_size, maps.shape[0], maps.shape[1],  1))
        for i in range(batch_size):
            inp[i,:,:,0] = maps[:,:,b*batch_size+i]
            out[i,:,:,0] = maps[:,:,b*batch_size+i+1]
        
        np.save(fp+"\\PreBatched\\"+save_folder_name+"\\input"+str(b), inp)
        np.save(fp+"\\PreBatched\\"+save_folder_name+"\\output"+str(b), out)
    test = np.load(fp+"\\PreBatched\\"+save_folder_name+"\\input0.npy")
    print(test.shape)

def get_dataset(startDate,endDate):
    dataset = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filter(ee.Filter.date(startDate , endDate))
    return dataset

def select_data(dataset,data):
    dataout = dataset.select(data)
    return dataout

def viualize_data(dataset):
    print("this is your code: ")
    print(dataset)




    

#Examples   
 
#Ex 1) Total up precipitation across the region and print it
def ex1():

    startYear = 2000
    endYear = 2002

    precipVals = list_daily_precipitation_totals_for_year_range(startYear, endYear, geoArea, imScale)
    print(precipVals)

#Ex 2) Displaying data on a folium map, in this case precipitation for a single day
def ex2():
    #get the data for a single day
    dataset = get_dataset('1989-02-01','1989-02-02')
    datalist = dataset.toList(dataset.size())

    #Create overlays for the images and clip them to the area we want to analyze (uses lat/long coords)
    precipitationOverlay = ee.Image(datalist.get(0)).clip(geoArea)

    #make map out of the overlay
    make_map_from_image("map", precipitationOverlay, "Precipitation", [4.1156735, -72.9301367], 5)

#Ex 3) Load data from api and train a model to make predictions based on the previous day only.  Requires no local files but takes a long time and is limited by RAM capacity
def ex3():
    import tensorflow as tf
    import Model

    width = 360
    height = 360
    batchsz = 32
    
    
    #Load map data into numpy array for training. 
    maps = get_precipitation_maps_for_range('1989-01-01', '1990-01-01', geoArea, imScale)
    print("dataset loaded to memory shape ", maps.shape)



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
    model.save('AutoEncoder.model')
    
#Ex 4) Train single day autoencoder model on pre batched numpy files
def ex4():
    import tensorflow as tf
    import Model
    
    epochs = 5

    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    batch_count = int(len(os.listdir(data_path))/2)
    print(batch_count)
    
    model = Model.getAutoEncoder((360, 360, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    for e in range(epochs):
        print("Epoch "+str(e))
        for b in range(batch_count):
            inp = np.load(data_path+"Train\\"+"input"+str(b)+".npy")
            out = np.load(data_path+"Train\\"+"output"+str(b)+".npy")
            
            if b %(batch_count/8) == 0:
                print(model.train_on_batch(inp, out, return_dict = True))
            else:
                model.train_on_batch(inp, out, return_dict = True)
    #Note that the values that will be printed out are not the overall loss for the epoch, rather they are the loss for that specific batch.  So it will fluctuate a bit as it finds batches that it performs better or worse on    
        
    model.save('AutoEncoder.model')


#Ex 5) Load model and make a prediction from the test set
def ex5(test_file_number):
    import tensorflow.keras as keras

    model = keras.models.load_model('AutoEncoder.model')
    
    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    inp = np.load(data_path+"Test\\"+"input"+str(test_file_number)+".npy")
    out = np.load(data_path+"Test\\"+"output"+str(test_file_number)+".npy")
    
    pred = model.predict_on_batch(inp)
    f, im = plt.subplots(2,2)
    im[0,0].set_title("Input Data")
    im[0,0].imshow(inp[0,:,:,0])
    im[1,1].set_title("Ground Truth")
    im[1,1].imshow(out[0,:,:,0])
    im[1,0].set_title("Next Day Prediction")
    im[1,0].imshow(pred[0,:,:,0])
    
    plt.show()
    

def main():
    ex5(0)
    ex5(1)
    ex5(2)
    

    

        
    
    






    
    


    
    
    


















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