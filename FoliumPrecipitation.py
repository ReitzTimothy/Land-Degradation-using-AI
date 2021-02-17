import ee
import datetime
import folium


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





#Get an authentication token from google, do every time if running on the cloud, do first time only if running local
#ee.Authenticate()

#Initialize the earth engine API
ee.Initialize()






#Date range to filter dataset on
startDate = '2018-05-01'
endDate = '2018-05-03'
#Geographic area to use
geoArea = ee.Geometry.Rectangle(-79.55,12.43,-65.46,-4.86)





#Get our dataset from earth engine and filter it on a date range
dataset = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filter(ee.Filter.date(startDate, endDate));
precipitation = dataset.select('precipitation');

#Convert the dataset into a list of earth engine image objects and get the first one from the list.  This is inefficient so use filter() when you can
datalist = precipitation.toList(dataset.size())

#Print the length of the list.  Our dataset is daily images so it should match the date range we filtered on
print(len(datalist.getInfo()))

#Create a folium map centered on Columbia
m = folium.Map(location=[4.1156735, -72.9301367],zoom_start=5)

#Earth engine visualization parameters for the layer we will overlay on the map
visParams = {'palette':['D4394A', 'F66C45', 'FCAF62', 'FFE18B', 'E7F598', 'AADBA4','63C1A3', '3180BA'], 'gain':[.1], 'opacity':.8}

#Create overlays for the images and clip them to the area we want to analyze (uses lat/long coords)
precipitationOverlay = ee.Image(datalist.get(0)).clip(geoArea)
precipitationOverlay=precipitationOverlay.updateMask(precipitationOverlay)


#Overlay the image from the earth engine dataset on the folium map
m.add_ee_layer(precipitationOverlay, visParams, 'Precipitation')

# Add a layer control panel to the map.
m.add_child(folium.LayerControl())

#Save the map to an HTML file 
m.save("map.html")



#Things I still need to figure out
    #How to make a smooth gradient color transition instead of color pallette
    #How to normalize the dataset ranges to be between 0 and 1
    #How to load the background map in grayscale to avoid polluting the overlay with green from forested areas while keeping opacity






#Example of date time conversion between earth engine and python formats
# ee_date = ee.Date('2020-01-01')
# py_date = datetime.datetime.utcfromtimestamp(ee_date.getInfo()['value']/1000.0)

# py_date = datetime.datetime.utcnow()
# ee_date = ee.Date(py_date)