library(sf)
library(raster)
library(terra)

ran300 = st_read("random_points_300.shp")

ran300_latlong = st_transform(ran300, 4326)
# outputting the transformed layer
ran300_latlong$latitude = st_coordinates(ran300_latlong)[,2]
ran300_latlong$longitude = st_coordinates(ran300_latlong)[,1]

luse1 = raster("./Covariates/E060N20_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif")
luse2 = raster("./Covariates/E080N20_PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif")
luse = merge(luse1, luse2)


LKAadm0 = st_read("LKA_adm0.shp")
#plgn = st_transform(plgn, "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")
#plgn = vect(plgn)

ml = crop(luse, LKAadm0, mask = TRUE)
writeRaster(ml, "SLK_landuse_copernicus.tif")
ran300_latlong$luse = extract(ml, ran300_latlong)

ran300_latlong_luse = ran300_latlong[-which(ran300_latlong$luse==50),]
ran300_latlong_luse = ran300_latlong[-which(ran300_latlong$luse==70),]
ran300_latlong_luse = ran300_latlong[-which(ran300_latlong$luse==80),]
ran300_latlong_luse = ran300_latlong[-which(ran300_latlong$luse==200),]

write.csv(ran300_latlong_luse, "random_points_300_WGS84.csv")
st_write(ran300_latlong_luse, "ran300_latlong_luse.shp")

ran300_latlong_luse_utm44 = st_transform(ran300_latlong_luse, 32644)
st_write(ran300_latlong_luse_utm44, "ran300_latlong_luse_utm44.shp")

# we use only utm44 for mapping purposes only the real analysis for GAN is going to be on latlong

