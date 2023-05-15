library(sf)
library(raster)

ran150 = st_read("random_points_150.shp")

ran150_latlong = st_transform(ran150, 4326)
# outputting the transformed layer
ran150_latlong$latitude = st_coordinates(ran150_latlong)[,2]
ran150_latlong$longitude = st_coordinates(ran150_latlong)[,1]
write.csv(ran150_latlong[,c(70,71)], "random_points_150_WGS84.csv")


dpeth = raster("/Users/ej/MASTER/Master research/Covariates/Shangguan/BDTICM_M_1km_ll.tif")

ran150$absdep = extract(dpeth, ran150)
