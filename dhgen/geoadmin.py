# DHgeN: a Python module for generating District Heating Networks layouts.

# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
# Written by Giuseppe Peronato <Giuseppe.Peronato@idiap.ch>

# This file is part of DHgeN.

# DHgeN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# DHgeN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DHgeN. If not, see <http://www.gnu.org/licenses/>

"""
Get geodata using GeoAmin API.

https://api3.geo.admin.ch


@author: gperonato
"""
import requests
import time
from shapely.geometry import shape
import pandas as pd
import geopandas as gpd
import logging
logging.basicConfig(level=logging.INFO)

URL="http://api3.geo.admin.ch/rest/services/api/MapServer/"

# Sample test parameters
BBOX = [2573626,1161163,2574216,1161873]  # Broc area
SERVICE = "identify"
PARAMETERS = dict(
layers="all:ch.swisstopo.swisstlm3d-strassen", # identify
# layer="ch.bfs.gebaeude_wohnungs_register", # find
# searchText="CH287752308752", # find
# searchField="egrid", # find
geometry=",".join([str(x) for x in BBOX]), # identify
geometryType="esriGeometryEnvelope", # identify
returnGeometry=True,
geometryFormat="geojson",
offset=0, # identify
tolerance=0, #identify
sr=2056)


def dict2pars(parameters: dict):
    """Create the query string from the parameters."""
    string = ""
    for i, key in enumerate(parameters.keys()):
        string += key
        string += "="
        string += str(parameters[key])
        if i < len(parameters)-1:
            string += "&"
    string = string.replace(" ","")
    string = string.replace("'",'"')
    return string


def geoadmin_query(parameters, service="identify"):
    """Construct query from parameters."""
    url = URL+service+"?"+dict2pars(parameters)
    logging.debug(url)
    r = requests.get(url)
    df = pd.DataFrame()
    if r.status_code == 200:
        results = r.json()["results"]
        if len(results) > 0:
            df = pd.DataFrame(r.json()["results"])
            properties = pd.json_normalize(df['properties'])
            df = pd.concat([df,properties],axis=1)
            if "geometry" in df.columns:
                df["geometry"] = df.apply(lambda x: shape(x.geometry), axis=1)
        else:
            logging.debug("No records.")
            logging.debug(r.text)
    else:
        logging.warning("Failed retrieving, status code:",r.status_code)
    time.sleep(1)
    return df

def geoadmin_to_gdf(parameters: dict=PARAMETERS, delist=True, service = SERVICE):
    """Query iteratively over all records."""
    pars = parameters.copy()
    dfs = []
    total_records = 0
    records = 1 # should be > 0 to start
    if "layers" in pars.keys():
        logging.info("Querying layers {}".format(pars["layers"]))
    if "layer" in pars.keys():
        logging.info("Querying layer {}".format(pars["layer"]))
    while records > 0:
        if total_records > 0:
            pars["offset"] = total_records
        df = geoadmin_query(pars,service)
        dfs.append(df)
        records = df.shape[0]
        total_records += records
        logging.info("Retrieved {} records in this call. Total records: {}".format(records,
                                                                                   total_records))
    gdf = gpd.GeoDataFrame(pd.concat(dfs,ignore_index=True),
                             crs="EPSG:{}".format(parameters["sr"]))
    
    if delist:
    # Transform lists into strings, so that it can be saved
        def delist(mylist):
            """List to string."""
            if mylist and len(mylist) > 0:
                return [str(s) for s in mylist]
            else:
                return ""
            
        hasList = (gdf.applymap(type) == list).any()
        for col in hasList.loc[hasList].index:
            gdf[col] = gdf.apply(lambda x:",".join(delist(x[col])),axis=1)
            
    return gdf

if __name__ == "__main__":
    gdf = geoadmin_to_gdf(PARAMETERS,service=SERVICE)