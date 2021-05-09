import ee
import datetime
import folium
import numpy as np
import geemap
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow.keras as keras
import tensorflow as tf
import Model
from tensorflow.keras.losses import Loss as Loss
from datetime import date, timedelta
import GANModel


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
def batch_numpy_file_to_folder(input_filename, save_folder, batch_size, sequence_length):
    fp = "H:\\Datasets\\ChirpsDaily"
    maps = np.load(fp+"\\"+input_filename)
    width = 360
    height = 360
    

    norm_in = MinMaxScaler().fit(maps.reshape((maps.shape[2],-1)))
    maps = norm_in.transform(maps.reshape((maps.shape[2],-1))).reshape(width,height, maps.shape[2])
    print(maps[:,:,0])
    
    
    num_batches = int((maps.shape[2]-sequence_length*2)/batch_size)
    print(num_batches)
    for b in range(num_batches):
        print(b)
        inp = np.empty(shape = (batch_size, sequence_length, maps.shape[0], maps.shape[1]))
        out = np.empty(shape = (batch_size, 1, maps.shape[0], maps.shape[1]))
        for i in range(batch_size):
            for s in range(sequence_length):
                inp[i,s,:,:] = maps[:,:,b*batch_size+i+s]
            out[i,0,:,:] = maps[:,:,b*batch_size+i+sequence_length+s]
        
        np.save(fp+save_folder+"\\input"+str(b), inp)
        np.save(fp+save_folder+"\\output"+str(b), out)
    test = np.load(fp+save_folder+"\\input0.npy")
    print(test.shape)
    return norm_in
    
def batch_numpy_file_to_folder_with_normalizer(input_filename, save_folder, batch_size, sequence_length, normalizer):
    fp = "H:\\Datasets\\ChirpsDaily"
    maps = np.load(fp+"\\"+input_filename)
    width = 360
    height = 360
    

    maps = normalizer.transform(maps.reshape((maps.shape[2],-1))).reshape(width,height, maps.shape[2])
    print(maps[:,:,0])
    
    
    num_batches = int((maps.shape[2]-sequence_length*2)/batch_size)
    print(num_batches)
    for b in range(num_batches-1):
        print(b)
        inp = np.empty(shape = (batch_size, sequence_length, maps.shape[0], maps.shape[1]))
        out = np.empty(shape = (batch_size, 1, maps.shape[0], maps.shape[1]))
        for i in range(batch_size):
            for s in range(sequence_length):
                inp[i,s,:,:] = maps[:,:,b*batch_size+i+s]
            out[i,0,:,:] = maps[:,:,b*batch_size+i+sequence_length+s]
        
        np.save(fp+save_folder+"\\input"+str(b), inp)
        np.save(fp+save_folder+"\\output"+str(b), out)
    test = np.load(fp+save_folder+"\\input0.npy")
    print(test.shape)
    return normalizer

def get_dataset(startDate,endDate):
    dataset = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filter(ee.Filter.date(startDate , endDate))
    return dataset

def select_data(dataset,data):
    dataout = dataset.select(data)
    return dataout

def viualize_data(dataset):
    print("this is your code: ")
    print(dataset)

def get_data_stats():
    data = np.load("H:\\Datasets\\ChirpsDaily\\ChirpsDaily.npy")
    avg = np.average(data, 2)
    std = np.std(data, axis = 2)
    #display_single_image_plot(avg)
    #display_single_image_plot(std)
    return (avg, std)

def on_click(event):
    print(str([event.xdata,event.ydata]))

def display_single_image_plot(im):
    plt.imshow(im)
    plt.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def get_split_averages_on_data(num_splits):
    data = np.load("H:\\Datasets\\ChirpsDaily\\ChirpsDaily.npy")
    splits = []
    split_len = (int)(data.shape[2]/num_splits)

    for s in range(num_splits):
        splits.append(np.average(data[:,:,s*split_len:(s+1)*split_len], axis = 2))
    return splits
    
def get_split_stdevs_on_data(num_splits):
    data = np.load("H:\\Datasets\\ChirpsDaily\\ChirpsDaily.npy")
    splits = []
    split_len = (int)(data.shape[2]/num_splits)

    for s in range(num_splits):
        splits.append(np.std(data[:,:,s*split_len:(s+1)*split_len], axis = 2))
    return splits

def visualize_splits_vs_avg(splits, avg, width, height):
    f, im = plt.subplots(width,height)

    for i in range(len(splits)):
        dat = np.subtract(splits[i], avg)
        n = colors.TwoSlopeNorm(vcenter = 0)
        if height > 1:
            im[i%width, int(i/width)].imshow(dat, cmap = 'seismic', norm = n)
        else:
            im[i].imshow(dat, cmap = 'seismic', norm = n)
    
    plt.show()

def visualize_splits_vs_prev(splits, width, height):
    f, im = plt.subplots(width,height)

    for i in range(len(splits)):
        n = colors.TwoSlopeNorm(vcenter = 0)
        if(i>0):
            dat = np.subtract(splits[i], splits[i-1])
        else:
            dat = splits[i]
        if height > 1:
            im[i%width, int(i/width)].imshow(dat, cmap = 'seismic', norm = n)
        else:
            im[i].imshow(dat, cmap = 'seismic', norm = n)
    
    plt.show()   

def visualize_splits_vs_first(splits, width, height):
    f, im = plt.subplots(width,height)

    for i in range(len(splits)):
        n = colors.TwoSlopeNorm(vcenter = 0)
        dat = np.subtract(splits[i], splits[0])
        if height > 1:
            im[i%width, int(i/width)].imshow(dat, cmap = 'seismic', norm = n)
        else:
            im[i].imshow(dat, cmap = 'seismic', norm = n)
    
    plt.show()    
    
def generate_empty_data(batch_size):
    dat = np.zeros(shape = (batch_size,1,360,360))
    #dat[int(batch_size/2):batch_size,:,:,:] = 1
    np.save("H:\\Datasets\\ChirpsDaily\\PreBatched\\EmptyData\\empty", dat)

def getDateVector(start_date, days, sequence_len, ):
    data = np.zeros((days, sequence_len, 13))
    for i in range(days):
        for s in range(sequence_len):
            date = start_date+timedelta(days = i+s)
            data[i,s,(date.month-1)] = 1
            data[i,s,12] = date.year-start_date.year/40
    date = start_date+timedelta(days = days)
    return (data, date)




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
    
    epochs = 3

    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    print(batch_count)
    
    model = tf.keras.models.load_model('AutoEncoder.model') #Model.getAutoEncoder((360, 360, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(model.summary())
    
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


#Ex 5) Load autoencoder model and make a prediction from the test set
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
    
#Ex 6) Train sequential autoencoder model on pre batched files - Deprecated, data format changed
def ex6(sequence_length):
    import tensorflow as tf
    import Model
    
    epochs = 3
    sequence_length = 5
    
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    validation_batch_count = int(len(os.listdir(data_path+"Validation\\"))/2)
    print(batch_count)
    
    model = Model.getSequentialAutoEncoder((360, 360, sequence_length))#tf.keras.models.load_model('SequentialAutoEncoder.model') 
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(model.summary())
    
    for e in range(epochs):
        print("Epoch "+str(e))
        epoch_loss = 0;
        validation_loss = 0;
        for b in range(batch_count):
            inputs = []
            inp = np.load(data_path+"Train\\"+"input"+str(b)+".npy")
            for i in range(inp.shape[3]):
                inputs.append(inp[:,:,:,i])
            out = np.load(data_path+"Train\\"+"output"+str(b)+".npy")
            batch_loss = model.train_on_batch(inputs, out, return_dict = False)
            epoch_loss = epoch_loss+batch_loss
            if b %(batch_count/8) == 0:
                print("...")
        for v in range(validation_batch_count):
            split_inp = []
            inp = np.load(data_path+"Validation\\"+"input"+str(v)+".npy")
            for i in range(inp.shape[3]):
                split_inp.append(inp[:,:,:,i])
            o = np.load(data_path+"Validation\\"+"output"+str(v)+".npy")
            v_loss = model.test_on_batch(split_inp, o, return_dict = False)
            validation_loss = validation_loss+v_loss
            
        epoch_loss = epoch_loss/batch_count
        validation_loss = validation_loss/validation_batch_count
        print("validation loss "+str(validation_loss))
        print("loss "+str(epoch_loss))
        
    model.save('SequentialAutoEncoder.model')

#Ex 7) Load sequentialautoencoder model and make a prediction from the test set - Deprecated, data format changed
def ex7(test_file_number):
    import tensorflow.keras as keras

    model = keras.models.load_model('SequentialAutoEncoder.model')
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    inputs = []
    inp = np.load(data_path+"Validation\\"+"input"+str(test_file_number)+".npy")
    for i in range(inp.shape[3]):
        inputs.append(inp[:,:,:,i])
    out = np.load(data_path+"Validation\\"+"output"+str(test_file_number)+".npy")
    
    pred = model.predict_on_batch(inputs)
    f, im = plt.subplots(7)
    for i in range(inp.shape[3]):
        im[i].set_title("Input Data"+str(i))
        im[i].imshow(inp[0,:,:,i])
    im[5].set_title("Next Day Prediction")
    im[5].imshow(pred[0,:,:,0])
    im[6].set_title("Ground Truth")
    im[6].imshow(out[0,:,:,0])
    
    plt.show()
    
#Ex 8) Train single day 1-1 autoencoder model on pre batched numpy files
def ex8():
    import tensorflow as tf
    import Model
    
    epochs = 5
    zeros_frequency = 100

    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    print(batch_count)
    
    ensemble = Model.getSameDayAutoEncoder((360,360,1))
    
    
    model = ensemble[0] 
    model.compile(optimizer='adam', loss='mae')
    print(model.summary())
    
    zeros = np.reshape(np.load(data_path+"EmptyData\\"+"empty.npy"), (-1,360,360,1))
    zerout = np.empty(zeros.shape)
    
    for e in range(epochs):
        print("Epoch "+str(e))
        tot_loss = 0;
        for b in range(batch_count):
            inp = np.reshape(np.load(data_path+"Train\\"+"input"+str(b)+".npy"), (-1,360,360,1))
            out = np.reshape(np.load(data_path+"Train\\"+"input"+str(b)+".npy"), (-1,360,360,1))
            
            if(b%zeros_frequency == 0):
                print(str(model.train_on_batch(zeros, zeros, return_dict = True)) +" on noise")
            tot_loss = tot_loss + model.train_on_batch(inp, out, return_dict = False)
            if b %(batch_count/8) == 0:
                print(tot_loss)
    #Note that the values that will be printed out are not the overall loss for the epoch, rather they are the loss for that specific batch.  So it will fluctuate a bit as it finds batches that it performs better or worse on    
        print(tot_loss)
    model.save('SameDayAutoEncoder.model')
    ensemble[1].save('encoder.model')
    ensemble[2].save('decoder.model')
    
#Ex 9) Load single day 1-1 autoencoder model and make a prediction from the test set
def ex9(test_file_number):

    model = keras.models.load_model('AutoEncoder.model')
    
    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    inp = np.reshape(np.load(data_path+"Test\\"+"input"+str(test_file_number)+".npy"), (-1,360,360,1))
    out = np.reshape(np.load(data_path+"Test\\"+"input"+str(test_file_number)+".npy"), (-1,360,360,1))
    
    pred = model.predict_on_batch(inp)
    f, im = plt.subplots(2,2)
    im[0,0].set_title("Input Data")
    im[0,0].imshow(inp[0,:,:,0])
    im[1,1].set_title("Ground Truth")
    im[1,1].imshow(out[0,:,:,0])
    im[1,0].set_title("Next Day Prediction")
    im[1,0].imshow(pred[0,:,:,0])
    
    plt.show()

#Ex 10) Load pre-encoded model and train the forecasting bottleneck - Deprecated, data format changed
def ex10(sequence_length):
    import tensorflow as tf
    import Model
    
    epochs = 3
    sequence_length = 5
    
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    validation_batch_count = int(len(os.listdir(data_path+"Validation\\"))/2)
    print(batch_count)
    
    model = Model.getSequentialPreEncoded((360, 360, sequence_length))#tf.keras.models.load_model('SequentialPreEncoded.model') 
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(model.summary())
    
    for e in range(epochs):
        print("Epoch "+str(e))
        epoch_loss = 0;
        validation_loss = 0;
        for b in range(batch_count):
            inputs = []
            inp = np.load(data_path+"Train\\"+"input"+str(b)+".npy")
            for i in range(inp.shape[3]):
                inputs.append(inp[:,:,:,i])
            out = np.load(data_path+"Train\\"+"output"+str(b)+".npy")
            batch_loss = model.train_on_batch(inputs, out, return_dict = False)
            epoch_loss = epoch_loss+batch_loss
            if b %(batch_count/8) == 0:
                print("...")
        for v in range(validation_batch_count):
            split_inp = []
            inp = np.load(data_path+"Validation\\"+"input"+str(v)+".npy")
            for i in range(inp.shape[3]):
                split_inp.append(inp[:,:,:,i])
            o = np.load(data_path+"Validation\\"+"output"+str(v)+".npy")
            v_loss = model.test_on_batch(split_inp, o, return_dict = False)
            validation_loss = validation_loss+v_loss
            
        epoch_loss = epoch_loss/batch_count
        validation_loss = validation_loss/validation_batch_count
        print("validation loss "+str(validation_loss))
        print("loss "+str(epoch_loss))
        
    model.save('SequentialPreEncoded.model')

#Ex 11) Load pre-encoded model and make prediction - Deprecated, data format changed
def ex11(test_file_number):
    import tensorflow.keras as keras

    model = keras.models.load_model('SequentialAutoEncoder.model')
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    inputs = []
    inp = np.load(data_path+"Validation\\"+"input"+str(test_file_number)+".npy")
    for i in range(inp.shape[3]):
        inputs.append(inp[:,:,:,i])
    out = np.load(data_path+"Validation\\"+"output"+str(test_file_number)+".npy")
    
    pred = model.predict_on_batch(inputs)
    f, im = plt.subplots(7)
    for i in range(inp.shape[3]):
        im[i].set_title("Input Data"+str(i))
        im[i].imshow(inp[0,:,:,i])
    im[5].set_title("Next Day Prediction")
    im[5].imshow(pred[0,:,:,0])
    im[6].set_title("Ground Truth")
    im[6].imshow(out[0,:,:,0])
    
    plt.show()


class CustomLoss(Loss):
    @tf.function
    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        diff = y_pred-y_true
        loss = keras.backend.square(diff)+keras.backend.abs(diff)*100#keras.backend.abs(diff+tf.math.divide(diff, 2))

        loss = keras.backend.min(loss, axis=0)+keras.backend.max(loss, axis=0)#keras.backend.sum(loss, axis=0)#      
        return loss
        
class RMSE(Loss):
    @tf.function
    def call(self, y_true, y_pred):
        diff = tf.squeeze(y_pred) - y_true
        return keras.backend.sqrt(keras.backend.mean(keras.backend.square(diff)))

#Ex 12) Load pre-encoded GRU model and train the forecasting bottleneck
def ex12(sequence_length):
    loss = CustomLoss()
    keras.losses.loss = loss
    epochs = 2
    sequence_length = 5
    
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    validation_batch_count = int(len(os.listdir(data_path+"Validation\\"))/2)
    print(batch_count)
    
    
    
    model =  Model.getModel((sequence_length, 360, 360, 1))
    
    # l = keras.models.load_model('MultiEncoder.model', compile = False)
    # model.set_weights(l.get_weights())
    # l = None


    model.compile(optimizer='adam', loss='mae')
    print(model.summary())
    
    for e in range(epochs):
        print("Epoch "+str(e))
        epoch_loss = 0;
        validation_loss = 0;
        train_date = date(1981,1,1)
        validation_date = date(2021,1,1)
        for b in range(batch_count):
            batch_loss = 0;
            inp = np.load(data_path+"Train\\"+"input"+str(b)+".npy")
            out = np.load(data_path+"Train\\"+"output"+str(b)+".npy")
            date_vec = getDateVector(train_date, inp.shape[0], sequence_length)
            train_date = date_vec[1]
            
            for bs in range(inp.shape[0]):
                batch_loss = batch_loss+model.train_on_batch([inp[None,bs],np.zeros((1,1)), date_vec[0][None, bs]], out[None, bs,0,:,:], return_dict = False)
            
            #batch_loss = model.train_on_batch([inp,np.zeros((inp.shape[0],1)), date_vec[0]], out[:,0,:,:], return_dict = False)
            epoch_loss = epoch_loss+batch_loss
            if b %(batch_count/8) == 0:
                print("...")#+ str(batch_loss))
        for v in range(validation_batch_count):
            inp = np.load(data_path+"Validation\\"+"input"+str(v)+".npy")
            o = np.load(data_path+"Validation\\"+"output"+str(v)+".npy")
            date_vec = getDateVector(validation_date, inp.shape[0], sequence_length)
            validation_date = date_vec[1]
            
            #v_loss = model.test_on_batch([inp,np.zeros((inp.shape[0],1)), date_vec[None, 0]], o[:,0,:,:], return_dict = False)
            #validation_loss = validation_loss+v_loss
            
        epoch_loss = epoch_loss/batch_count
        #validation_loss = validation_loss/validation_batch_count
        #print("validation loss "+str(validation_loss))
        print("loss "+str(epoch_loss))
        
    model.save('MultiEncoder.model')
    
#Ex 13) Load pre-encoded GRU model and make prediction
def ex13(test_file_number):
    validation_date = date(2021,1,1)+timedelta(days=test_file_number)
    loss = CustomLoss()
    keras.losses.loss = loss
    model = keras.models.load_model('MultiEncoder.model')
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    inp = np.load(data_path+"Train\\"+"input"+str(test_file_number)+".npy")
    out = np.load(data_path+"Train\\"+"output"+str(test_file_number)+".npy")
    
    date_vec = getDateVector(validation_date, inp.shape[0], 5)
    
    pred = model.predict_on_batch([inp,np.zeros((inp.shape[0],1)), date_vec[0]])
    f, im = plt.subplots(1,inp.shape[1]+2)
    
    
    for i in range(inp.shape[1]):
        im[i].set_title("Input Data"+str(i))
        im[i].imshow(inp[0,i,:,:])
    im[inp.shape[1]].set_title("Prediction")
    im[inp.shape[1]].imshow(pred[0,0,:,:])
    im[inp.shape[1]+1].set_title("Ground Truth")
    im[inp.shape[1]+1].imshow(out[0,0,:,:])
    plt.show()

#Ex 14) Make covariance matrix for all training data for autoencoder latent space
def ex14():
    import seaborn as sn

    encoder = keras.models.load_model('encoder.model')
    decoder = keras.models.load_model('decoder.model')
    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    latent = np.empty((0,1))
        
    for b in range(batch_count):
        inp = np.reshape(np.load(data_path+"Train\\"+"input"+str(b)+".npy"), (-1,360,360,1))
        l = encoder.predict_on_batch(inp)
        latent = np.append(latent, l, axis = 0)
            
    cov_mat = np.cov(latent, rowvar = False)
    sn.heatmap(cov_mat, annot=False, fmt='g')
    plt.show()



#Train variational autoencoder
def ex15():
    import tensorflow as tf
    import VAE
    vae = VAE.VAE()
    
    epochs = 5
    zeros_frequency = 100

    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    
    ensemble = vae.get_VAE((360,360,1))
    
    
    model = ensemble[0] 
    model.compile(optimizer='adam', loss=vae.vae_loss, metrics = [vae.vae_r_loss, vae.vae_kl_loss])
    print(model.summary())
    print(type(vae.mu))
    
    zeros = np.reshape(np.load(data_path+"EmptyData\\"+"empty.npy"), (-1,360,360,1))
    zerout = np.empty(zeros.shape)
    
    for e in range(epochs):
        print("Epoch "+str(e))
        tot_loss = 0;
        for b in range(batch_count):
            inp = np.reshape(np.load(data_path+"Train\\"+"input"+str(b)+".npy"), (-1,360,360,1))
            out = np.reshape(np.load(data_path+"Train\\"+"input"+str(b)+".npy"), (-1,360,360,1))
            
            if(b%zeros_frequency == 0):
                print(str(model.train_on_batch(zeros, zeros, return_dict = True)) +" on noise")
            tot_loss = tot_loss + model.train_on_batch(inp, out, return_dict = False)[0]
            if b %(batch_count/8) == 0:
                print(tot_loss)
    #Note that the values that will be printed out are not the overall loss for the epoch, rather they are the loss for that specific batch.  So it will fluctuate a bit as it finds batches that it performs better or worse on    
        print(tot_loss)
    model.save('SameDayAutoEncoder.model')
    ensemble[1].save('encoder.model')
    ensemble[2].save('decoder.model')

#Train multiencoder using VAE instead of regular autoencoder
def ex16(sequence_length):
    import VAE
    vae = VAE.VAE()
    
    loss = RMSE()
    keras.losses.loss = loss
    epochs = 5
    sequence_length = 5
    
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    validation_batch_count = int(len(os.listdir(data_path+"Validation\\"))/2)
    print(batch_count)
    
    
    
    model =  Model.getModelWithVAE((sequence_length, 360, 360, 1), 16)
    
    l = keras.models.load_model('MultiEncoder.model', compile = False, custom_objects={'vae_sampling': vae.vae_sampling})
    model.set_weights(l.get_weights())
    l = None


    model.compile(optimizer='adadelta', loss=loss)
    print(model.summary())
    
    for e in range(epochs):
        print("Epoch "+str(e))
        epoch_loss = 0;
        validation_loss = 0;
        train_date = date(1981,1,1)
        validation_date = date(2021,1,1)
        for b in range(batch_count):
            batch_loss = 0;
            inp = np.load(data_path+"Train\\"+"input"+str(b)+".npy")
            out = np.load(data_path+"Train\\"+"output"+str(b)+".npy")
            date_vec = getDateVector(train_date, inp.shape[0], sequence_length)
            train_date = date_vec[1]
            
            for bs in range(inp.shape[0]):
                batch_loss = batch_loss+model.train_on_batch([inp[None,bs],np.zeros((1,1)), date_vec[0][None, bs]], out[None, bs,0,:,:], return_dict = False)
            
            #batch_loss = model.train_on_batch([inp,np.zeros((inp.shape[0],1)), date_vec[0]], out[:,0,:,:], return_dict = False)
            epoch_loss = epoch_loss+batch_loss
            if b %(batch_count/8) == 0:
                print("...")#+ str(batch_loss))
        for v in range(validation_batch_count):
            inp = np.load(data_path+"Validation\\"+"input"+str(v)+".npy")
            o = np.load(data_path+"Validation\\"+"output"+str(v)+".npy")
            date_vec = getDateVector(validation_date, inp.shape[0], sequence_length)
            validation_date = date_vec[1]
            
            #v_loss = model.test_on_batch([inp,np.zeros((inp.shape[0],1)), date_vec[None, 0]], o[:,0,:,:], return_dict = False)
            #validation_loss = validation_loss+v_loss
            
        epoch_loss = epoch_loss/batch_count
        #validation_loss = validation_loss/validation_batch_count
        #print("validation loss "+str(validation_loss))
        print("loss "+str(epoch_loss))
        
    model.save('MultiEncoder.model')

#predict with VAE encoder
def ex17(test_file_number):
    import VAE
    vae = VAE.VAE()
    validation_date = date(2021,1,1)+timedelta(days=test_file_number)
    loss = RMSE()
    keras.losses.loss = loss
    model = keras.models.load_model('MultiEncoder.model', custom_objects={'vae_sampling': vae.vae_sampling, 'RMSE': RMSE})
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    inp = np.load(data_path+"Train\\"+"input"+str(test_file_number)+".npy")
    out = np.load(data_path+"Train\\"+"output"+str(test_file_number)+".npy")
    
    date_vec = getDateVector(validation_date, inp.shape[0], 5)
    
    pred = model.predict_on_batch([inp,np.zeros((inp.shape[0],1)), date_vec[0]])
    f, im = plt.subplots(1,inp.shape[1]+2)
    
    
    for i in range(inp.shape[1]):
        im[i].set_title("Input Data"+str(i))
        im[i].imshow(inp[0,i,:,:])
    im[inp.shape[1]].set_title("Prediction")
    im[inp.shape[1]].imshow(pred[0,0,:,:])
    im[inp.shape[1]+1].set_title("Ground Truth")
    im[inp.shape[1]+1].imshow(out[0,0,:,:])
    plt.show()

#Ex 18) Make covariance matrix for all training data for autoencoder latent space
def ex18():
    import seaborn as sn
    import VAE
    vae = VAE.VAE()

    encoder = keras.models.load_model('encoder.model', custom_objects={'vae_sampling': vae.vae_sampling})
    decoder = keras.models.load_model('decoder.model')
    data_path = "H:\\Datasets\\ChirpsDaily\\PreBatched\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    latent = np.empty((0,16))
        
    for b in range(batch_count):
        inp = np.reshape(np.load(data_path+"Train\\"+"input"+str(b)+".npy"), (-1,360,360,1))
        l = encoder.predict_on_batch(inp)
        latent = np.append(latent, l, axis = 0)
            
    cov_mat = np.cov(latent, rowvar = False)
    sn.heatmap(cov_mat, annot=True, fmt='g')
    plt.show()


losses = []
critic_losses = []
#Ex 19) Train prediction model with a discriminator
def ex19(sequence_length):
    import VAE
    import Critic
    
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1)
    critic_optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1)
    latent_size = 32
    
    vae = VAE.VAE() #Just need the VAE to use its lambda function
    critic = Critic.Critic((latent_size))
    
    #load encoder for generating critic data
    encoder = vae.get_VAE((360, 360, 1))[1]
    e = keras.models.load_model('encoder.model', compile = False, custom_objects={'vae_sampling': vae.vae_sampling})
    encoder.set_weights(e.get_weights())
    e = None
    
    epochs = 5
    sequence_length = 5
    
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    batch_count = int(len(os.listdir(data_path+"Train\\"))/2)
    validation_batch_count = int(len(os.listdir(data_path+"Validation\\"))/2)
    print(batch_count)
    
    model = GANModel.getGANModel((sequence_length, 360, 360, 1), latent_size)
    
    # l = keras.models.load_model('gan.model', compile = False, custom_objects={'vae_sampling': vae.vae_sampling})
    # model.set_weights(l.get_weights())
    # l = None
    # c = keras.models.load_model('critic.model', compile = False)
    # critic.model.set_weights(c.get_weights())
    # c = None

    critic.model.compile(optimizer=critic_optimizer, loss='binary_crossentropy')
    model.compile(optimizer=optimizer, loss=model.gan_loss, metrics = [])
    print(model.summary())
    
    for e in range(epochs):
        print("Epoch "+str(e))
        
        epoch_loss = 0;
        validation_loss = 0;
        train_date = date(1981,1,1)
        validation_date = date(2021,1,1)
        
        #train critic/discriminator
        
        for b in range(batch_count):
        
            for layer in critic.model.layers:
                layer.trainable = True
            
            inp = np.load(data_path+"Train\\"+"input"+str(b)+".npy")
        
            date_info = getDateVector(train_date, inp.shape[0], sequence_length)
            train_date = date_info[1]
            
            generated_data = model.predict_on_batch([inp, date_info[0]])[1]
            real = np.load(data_path+"Train\\"+"output"+str(b)+".npy")
            real_data = encoder.predict_on_batch(real[:,0,:,:])
            critic_batch_loss = critic.train_critic_on_batch(generated_data, real_data)
            critic_losses.append(critic_batch_loss)
            if b %(batch_count/8) == 0:
                print("Critic Loss "+str(critic_batch_loss))
        
            for layer in critic.model.layers:
                layer.trainable = False
            model.set_critic(critic.model)
    
    
    
    
        
        
            batch_loss = 0;
            inp = np.load(data_path+"Train\\"+"input"+str(b)+".npy")
            out = np.load(data_path+"Train\\"+"output"+str(b)+".npy")
            
            #stochastic gradient descent
            for bs in range(inp.shape[0]):
                l = model.train_on_batch([inp[None,bs], date_info[0][None, bs]], [out[None, bs,0,:,:], np.zeros(shape = (1))], return_dict = False)
                batch_loss = batch_loss+l[0]
            losses.append(batch_loss)
            
            #batch_loss = model.train_on_batch([inp, date_info[0]], out[:,0,:,:], return_dict = False)
            epoch_loss = epoch_loss+batch_loss
            if b %(batch_count/8) == 0:
                print("...")#+ str(batch_loss))
        for v in range(validation_batch_count):
            inp = np.load(data_path+"Validation\\"+"input"+str(v)+".npy")
            o = np.load(data_path+"Validation\\"+"output"+str(v)+".npy")
            date_info = getDateVector(validation_date, inp.shape[0], sequence_length)
            validation_date = date_info[1]
            
            #v_loss = model.test_on_batch([inp,np.zeros((inp.shape[0],1)), date_info[None, 0]], o[:,0,:,:], return_dict = False)
            #validation_loss = validation_loss+v_loss
            
        epoch_loss = epoch_loss/batch_count
        #validation_loss = validation_loss/validation_batch_count
        #print("validation loss "+str(validation_loss))
        print("loss "+str(epoch_loss))
        
    model.save('gan.model')
    critic.model.save('critic.model')
    plt.plot(losses)
    plt.plot(critic_losses)
    plt.show()

#predict with gan
def ex20(test_file_number):
    import VAE
    vae = VAE.VAE()
    validation_date = date(2021,1,1)+timedelta(days=test_file_number)

    model = keras.models.load_model('gan.model', custom_objects={'vae_sampling': vae.vae_sampling, 'gan_loss': None})
    encoder = keras.models.load_model('encoder.model', custom_objects={'vae_sampling': vae.vae_sampling})
    
    data_path = "H:\\Datasets\\ChirpsDaily\\Sequential\\"
    inp = np.load(data_path+"Train\\"+"input"+str(test_file_number)+".npy")
    out = np.load(data_path+"Train\\"+"output"+str(test_file_number)+".npy")
    
    date_vec = getDateVector(validation_date, inp.shape[0], 5)
    
    p = model.predict_on_batch([inp, date_vec[0]])
    enc = encoder.predict_on_batch(out[:,0,:,:])
    pred = p[0]
    lat_vec = p[1]
    print(p[1])
    print(enc)
    print("")
    f, im = plt.subplots(1,inp.shape[1]+2)
    
    
    for i in range(inp.shape[1]):
        im[i].set_title("Input Data"+str(i))
        im[i].imshow(inp[0,i,:,:])
    im[inp.shape[1]].set_title("Prediction")
    im[inp.shape[1]].imshow(pred[0,0,:,:])
    im[inp.shape[1]+1].set_title("Ground Truth")
    im[inp.shape[1]+1].imshow(out[0,0,:,:])
    plt.show()
    
    

def main():
    # norm = batch_numpy_file_to_folder("ChirpsDaily.npy", "\\Sequential\\Train", 32, 5)
    # batch_numpy_file_to_folder_with_normalizer("ChirpsDaily2021.npy", "\\Sequential\\Validation", 1, 5, norm)
    # generate_empty_data(64)
    
    #Train variational autoencoder
    # ex15()
    # ex9(0)
    # ex9(15)
    # ex9(30)
    # ex9(45)
    
    #Train Latent Predictor
    ex19(5)
    ex20(0)
    ex20(32)
    ex20(64)
    ex20(96)
    ex20(97)
    
    #Calculate and visualize dataset analysis
    # stats = get_data_stats()
    # splits = get_split_averages_on_data(40)
    # visualize_splits_vs_avg(splits, stats[0],10,4)
    # visualize_splits_vs_prev(splits, 10,4)
    # visualize_splits_vs_first(splits, 10,4)
    # splits = get_split_stdevs_on_data(40)
    # visualize_splits_vs_avg(splits, stats[1],10,4)
    
    #VAE analysis
    # ex18()
    
    

        
    
    






    
    


    
    
    


















#Things I still need to figure out with foluium
    #How to make a smooth gradient color transition instead of color pallette
    #How to load the background map in grayscale to avoid polluting the overlay with green from forested areas while keeping opacity






#Example of date time conversion between earth engine and python formats
# ee_date = ee.Date('2020-01-01')
# py_date = datetime.datetime.utcfromtimestamp(ee_date.getInfo()['value']/1000.0)

# py_date = datetime.datetime.utcnow()
# ee_date = ee.Date(py_date)

if __name__ == '__main__':
    main()