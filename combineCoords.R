library(sf)
library(sp)
library(leaflet)

library(readxl)
combined_coordinates <- read_excel("combined_coordinates.xlsx", 
                                   sheet = "Sheet2")
#View(combined_coordinates)

dats = combined_coordinates
coordinates(dats) <- ~ Longitude + Latitude

plot(dats)

dats = st_as_sf(dats)

st_crs(dats) = 4326
dats$id2 = c(1:nrow(dats))

leaflet(data = dats) %>% addTiles() %>%
  addMarkers(st_coordinates(dats)[,1], 
             st_coordinates(dats)[,2],popup = dats$id2)

st_write(dats, "nc.shp")


datsproject = st_transform(dats, 32644)
st_write(datsproject, "nc_utm44.shp")

slk = st_read("LKA_adm0.shp")
slkpolygonproject = st_transform(slk, 32644)
st_write(slkpolygonproject, "LKA_adm0utm44.shp")


diff = st_read("difference.shp")
diffproject = st_transform(diff, 32644)
st_write(diffproject, "diff_utm44.shp")
