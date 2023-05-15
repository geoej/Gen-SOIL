library(sf)
library(terra)


# data description https://www.umb.edu/spectralmass/terra_aqua_modis/v006/mcd34a4_nbar_product

ran150 = st_read("random_points_150.shp")
ran150 = vect(ran150)



### batch

tiles = read.csv("/Users/ej/MASTER/Master research/Covariates/MCD43A4/tiles.csv")

for (i in 1:nrow(tiles)){
  n1 = list.files("/Users/ej/MASTER/Master research/Covariates/MCD43A4/", as.character(tiles[i,1]))[1]
  n2 = list.files("/Users/ej/MASTER/Master research/Covariates/MCD43A4/", as.character(tiles[i,2]))[1]
  
  m1 = sds(paste0("/Users/ej/MASTER/Master research/Covariates/MCD43A4/",n1))
  m2 = sds(paste0("/Users/ej/MASTER/Master research/Covariates/MCD43A4/",n2))
  
  #m7 = terra::merge(m1[[7]], m2[[7]])
  m14 = terra::merge(m1[[14]], m2[[14]])
  
  m14 = project(m14, "EPSG:32644")
  ran150$albedo = extract(m14, ran150)
  ran150[,paste(tiles[i,"start"])] = extract(m14, ran150)[2]
}

a = summary(ran150$`17/12/2020`)
for (i in tiles$start){
  a = rbind(a, summary(ran150[,i]))
}


sapply(ran150[,c(70:94)], max)
sapply(ran150[,c(70:94)], min)
       

###### all 600
library(tibble)
lists = list.files("/Users/ej/MASTER/Master research/Covariates/MCD43A4/600/", "hdf")
df <- tibble(name= "name", min = 0, max = 0)


for (i in lists){
  m1 = sds(paste0("/Users/ej/MASTER/Master research/Covariates/MCD43A4/600/",i))
  m14 = project(m1[[14]], "EPSG:32644")
  ext = extract(m14, ran150)
  df = df %>% add_row(name= i, min = min(ext[,2], na.rm = T), 
                      max = max(ext[,2],na.rm = T))
}
df= df[-1,]

min(df$min)
max(df$max)

# which layers have the lowest and highest possible albedo?
df[which(df$min == min(df$min)),] # MCD43A4.A2021082.h26v08.061.2021091045620.hdf
df[which(df$max == max(df$max)),] # MCD43A4.A2021245.h25v08.061.2021254053029.hdf

# images are from 2 September 2021
#minlayer = sds("/Users/ej/MASTER/Master research/Covariates/MCD43A4/600/MCD43A4.A2021082.h26v08.061.2021091045620.hdf")
complayer = sds("/Users/ej/MASTER/Master research/Covariates/MCD43A4/600/MCD43A4.A2021245.h26v08.061.2021254054145.hdf")
maxlayer = sds("/Users/ej/MASTER/Master research/Covariates/MCD43A4/600/MCD43A4.A2021245.h25v08.061.2021254053029.hdf")

#minlayer = minlayer[[14]]
complayer = complayer[[14]]
maxlayer = maxlayer[[14]]

all = merge(complayer, maxlayer)
plot(all)
#crs(minlayer, describe=TRUE)

LKAadm0 = st_read("LKA_adm0.shp")
plgn = st_transform(plgn, "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")
plgn = vect(plgn)

ml = crop(all, plgn, mask = TRUE)
plot(ml)
writeRaster(ml, "albedo.tif")

values = terra::extract(ml, plgn, xy = TRUE)
write.csv(values, "albedo_values.csv")
subs = values[which(values$Nadir_Reflectance_Band7>=0.381 &
                      values$Nadir_Reflectance_Band7<0.39),]
nrow(subs)
write.csv(subs, "pseudo_coordinates.csv")


subpoints = st_as_sf(subs, coords = c("x","y"),
                     crs = st_crs("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"))
#plgn = st_transform(subpoints, 4326)
plot(subpoints)

subpoints$ID = c(1:nrow(subpoints))
subpoints$sand = 70
subpoints$OC = 0
subpoints$pH = 7

st_write(subpoints, "pseudo_coordinates.shp", append=FALSE)

# plotting
lkk = st_read("LKA_adm0.shp")
lkk = st_transform(plgn, "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs")

plot(lkk[,"ID"])
plot(subpoints[,"ID"], add = T, col = "red")

#subs = values[which(values$Nadir_Reflectance_Band7<0.1),]

# 
# cl = cells(ml, plgn)
# coords = xyFromCell(ml, cl[,2])
# 
# 
# geom(plgn)
