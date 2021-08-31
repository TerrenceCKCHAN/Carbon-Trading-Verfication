from owslib.wcs import WebCoverageService

wcs = WebCoverageService('https://maps.isric.org/mapserv?map=/map/ocs.map',
                         version='2.0.1')

cov_id = 'ocs_0-30cm_mean'
ocs_0_30 = wcs.contents[cov_id]
ocs_0_30.supportedFormats

subsets = [('X', 1293000, 2427000), ('Y', 4561000, 5408000)]

crs = "http://www.opengis.net/def/crs/EPSG/0/4326"

response = wcs.getCoverage(
    identifier=[cov_id], 
    crs=crs,
    subsets=subsets, 
    resx=250, resy=250, 
    format=ocs_0_30.supportedFormats[0]
)

with open('data/SoilGrids2.0_ocs_0-30cm_mean.tif', 'wb') as file:
    file.write(response.read())