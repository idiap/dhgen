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
Run and plot examples for Broc area.

This file is conceived to test the module
in an area of the municipality of Broc (FR),
where a DHN is currently in operation.

@author: gperonato
"""
import os
import geopandas as gpd
from dhgen import nx_to_gdf, delisting, generate_dhn, setup_graph, apply_models, geoadmin_to_gdf

BOUNDS = [2573626,1161155,2574255,1161894] # Broc

# Limit comparison to buildings whose primary energy source
# for space heating or DHW according to public RegBL data is District Heating
# i.e., "genh1" == 7580 | "genw1" == 7580
# https://www.housing-stat.ch/fr/help/42.html
# The following buildings have been filtered out on Sept 1, 2022
LIMIT_EGIDS = [1509812,191945103,1509902,1509907,3091085,1509998,1509996,
               1509904,191749229,191360895,1509848,1509837,1509817,1509867,
               1509850,1509833,1509834,1509883,1509880,1509845,1509869,1509868,
               190352168,1509843,1509820,1509835,1509875,1509898,1509995,
               1510003,1510002,1509909,1509910,1509896,1509900,1509905,1509858,
               1509852,190849489,1509901,1509860,1509911,1509899,1509862,
               1509864,1509830,1509849,1509776,1509824,1509796,1509846,
               191962701,1510228,1509844,1509798,1509816,190637250,1509876,
               1509881,191360891,1509841,1509821,1510302,1509825,1509873,
               1509882,1509809,1509877,1509855,1509763,191360890,1509870,
               1509857,1509838,1509887,1509863,1509822,191360894,1509779,
               1509842,191360835,1510248,191360893,1509826,1509778,191360892,
               1509797,1509728,1509789,1509764,1509866,1509765]

HEATING_STATIONS = [(2573634,1161667)]

GRID_SIZE = {"width": 25,
             "height": 25,
             "rotation": 42}
        
ROAD_WEIGHT = 0.5

SEED = 42

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # Case 1a
    print("Generating a grid-based layout...")
    dhn_grid_graph = setup_graph(BOUNDS,
                        grid_size=GRID_SIZE,
                        heating_stations=HEATING_STATIONS,
                        limit_egids = LIMIT_EGIDS,
                        )
    grid = nx_to_gdf(dhn_grid_graph)[1]
    dhn_grid = apply_models(dhn_grid_graph,
                        model="husek",
                        seed = SEED
                        )
    print("")
    
    
    # Case 1b
    print("Generating layout using geodata")
    dhn = generate_dhn(BOUNDS,
                        grid_size=GRID_SIZE,
                        heating_stations=HEATING_STATIONS,
                        limit_egids = LIMIT_EGIDS,
                        add_roads=True,
                        add_footprints=True,
                        model="husek",
                        road_weight=ROAD_WEIGHT,
                        seed = SEED,
                        )
    print("")
    
    
    # Create GDF
    n, p = nx_to_gdf(dhn)
    n_grid, p_grid = nx_to_gdf(dhn_grid)
    
    # Assing crs
    p.crs = "EPSG:2056"
    p_grid.crs = "EPSG:2056"
    
    
    # Save GPKG
    delisting(p).to_file("broc.gpkg",layer="pipes",driver="GPKG")
    delisting(n).to_file("broc.gpkg",layer="nodes",driver="GPKG")
    delisting(p_grid).to_file("broc.gpkg",layer="pipes_grid",driver="GPKG")
    delisting(n_grid).to_file("broc.gpkg",layer="nodes_grid",driver="GPKG")
    
    
    # Read data already retrieved from the GeoAdmin API
    regbl = gpd.read_file("GeoAdmin_cache.gpkg",layer=0)
    regbl.egid = regbl.egid.astype(int)
    footprints = gpd.read_file("GeoAdmin_cache.gpkg",layer="all:ch.swisstopo.vec25-gebaeude")
    roads = gpd.read_file("GeoAdmin_cache.gpkg",layer="all:ch.swisstopo.swisstlm3d-strassen")
        
    # Compare results with existing network.
    # Data is retrieved from a private database.
    if os.path.exists(".env"):
        from decouple import config
        from sqlalchemy import create_engine
        

        con = create_engine('postgresql://{}:{}@{}/{}'.format(config("USERNAME"),
                                                                config("PASSWORD"),
                                                                config("SERVER"),
                                                                config("DB")
                                                                ))

        substations = gpd.GeoDataFrame.from_postgis("SELECT * from district_heating.substations_mod2",
                                                con=con)
        substations = substations.to_crs("EPSG:2056")  
        network_points = gpd.GeoDataFrame.from_postgis("SELECT * from district_heating.network_points",
                                                con=con)
        network_points = network_points.to_crs("EPSG:2056")  
        pipes = gpd.GeoDataFrame.from_postgis("SELECT * from district_heating.pipes",
                                              con=con)
        pipes = pipes.to_crs("EPSG:2056")
        
        # Select only warm pipes from database pipes
        pipes = pipes.loc[pipes.startpoint.isin(network_points.loc[network_points["warmth"] == "warm",
                                                                   "npid"])]
        
        # Create input data based on substations point geometry and data from REGBL
        substations_corr = substations.copy()
        
        input_data = substations_corr.merge(regbl,on="egid")
        
        input_data = input_data.merge(network_points[["npid","geom"]],
                                      left_on="NPID_in",
                                      right_on="npid")

        input_data = input_data.drop(["geometry","geom_x"],axis=1)
        input_data = input_data.rename({"geom_y":"geometry","egid_right":"egid"},axis=1)
        input_data = gpd.GeoDataFrame(input_data)
        input_data = input_data.explode().reset_index()
        

        
        # Create input data file
        input_data.to_file("input_data.gpkg", layer="substations",driver="GPKG")
        footprints.to_file("input_data.gpkg", layer="footprints",driver="GPKG")
        roads.to_file("input_data.gpkg", layer="roads",driver="GPKG")
        
        
        
        
        # Case 2a
        print("Generating layout using actual substation data...")
        dhn_db = generate_dhn(BOUNDS,
                               grid_size=GRID_SIZE,
                                geometry_file_path="input_data.gpkg",
                                heating_stations=HEATING_STATIONS,
                                add_roads=True,
                                add_footprints=True,
                                model="husek",
                                road_weight = ROAD_WEIGHT,
                                seed = SEED,
                                )
        
        # Create gdf
        n_db, p_db = nx_to_gdf(dhn_db)        
        
        # Assing GDF
        p_db.crs = "EPSG:2056"
        n_db.crs ="EPSG:2056"
        

        delisting(p_db).to_file("broc.gpkg",layer="pipes_db",driver="GPKG")
        delisting(n_db).to_file("broc.gpkg",layer="nodes_db",driver="GPKG")
    
        
        # Plotting hiding the position of the substation
        substation_buffer = gpd.GeoDataFrame(geometry=substations.buffer(1))
        
        # Removing intersection with substations buffer
        pipes_anon = pipes.loc[~pipes.pid.isin(gpd.sjoin(pipes,substation_buffer)["pid"]),:]        
        # Removing terminal pipes
        p_db_anon = p_db.loc[~p_db["end"].isin(n_db.loc[n_db.fillna(False).isConnected,"node"].values),:].copy()
        p_db_anon = p_db_anon.loc[~p_db_anon["start"].isin(n_db.loc[n_db.fillna(False).isConnected,"node"].values),:]

        delisting(pipes_anon).to_file("broc_anon.gpkg",driver="GPKG",layer="pipes_rwb")
        delisting(p_db_anon).to_file("broc_anon.gpkg",driver="GPKG",layer="pipes_db")
        
        
        
        
        # Case 2b
        roads_to_avoid = [
            # 19819482,19819505,
            20728434,19819591]
        roads_mod = roads.copy()
        if "weight" in roads_mod.columns:
            roads_mod = roads_mod.drop("weight",axis=1)
        roads_mod.loc[roads_mod.id.isin(roads_to_avoid),"weight"] = 10

        
        footprints_mod = footprints.copy()        
        input_data.to_file("input_data_mod.gpkg", layer="substations",driver="GPKG")
        footprints_mod.to_file("input_data_mod.gpkg", layer="footprints",driver="GPKG")
        roads_mod.to_file("input_data_mod.gpkg", layer="roads",driver="GPKG")

        
        print("Generating layout using actual substation data with custom weights...")
        dhn_db_mod = generate_dhn(BOUNDS,
                               grid_size=GRID_SIZE,
                                geometry_file_path="input_data_mod.gpkg",
                                heating_stations=HEATING_STATIONS,
                                add_roads=True,
                                add_footprints=True,
                                model="husek",
                                road_weight = ROAD_WEIGHT,
                                seed = SEED,
                                )

        # Create gdf
        n_db_mod, p_db_mod = nx_to_gdf(dhn_db_mod)        
        
        # Assing GDF
        p_db_mod.crs = "EPSG:2056"
        n_db_mod.crs ="EPSG:2056"
        

        delisting(p_db_mod).to_file("broc.gpkg",layer="pipes_db_mod",driver="GPKG")
        delisting(n_db_mod).to_file("broc.gpkg",layer="nodes_db_mod",driver="GPKG")
        
    

        p_db_mod_anon = p_db_mod.loc[~p_db_mod["end"].isin(n_db_mod.loc[n_db_mod.fillna(False).isConnected,"node"].values),:].copy()
        p_db_mod_anon = p_db_mod_anon.loc[~p_db_mod_anon["start"].isin(n_db_mod.loc[n_db_mod.fillna(False).isConnected,"node"].values),:]
        
        delisting(p_db_mod_anon).to_file("broc_anon.gpkg",driver="GPKG",layer="pipes_db_mod")
        
        roads_mod["weight"] = roads_mod["weight"].fillna(ROAD_WEIGHT)
        
        
    # Set weight for all records
    roads["weight"] = ROAD_WEIGHT
        
        
    # Get railway geometry for plotting
    parameters = PARAMETERS = dict(
    layers="all:ch.bav.schienennetz", # identify
    geometry=",".join([str(x) for x in BOUNDS]), # identify
    geometryType="esriGeometryEnvelope", # identify
    returnGeometry=True,
    geometryFormat="geojson",
    offset=0, # identify
    tolerance=0, #identify
    sr=2056)
    
    rail = geoadmin_to_gdf(parameters)
    rail = rail.loc[rail.nombrevoies > 0,:] # only tracks, no stations
    
    
    
    # Plots
    # 1A
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15.8/2.54, 7.5))
    axes = axes.reshape(1,4)[0]
    # grid.plot(ax=axes[0],color="gray",linewidth=0.5)
    p_grid.plot(ax=axes[0],color="#000080ff",linewidth=0.5)
    n_grid.loc[n_grid.isConnected.fillna(False),:].plot(ax=axes[0],
                                                        color="#000080ff",
                                                        markersize=1)
    n_grid.loc[n_grid.isHeatingStation.fillna(False),:].plot(ax=axes[0],
                                                        color="red",
                                                        markersize=1)
    
    # 1B
    rail.plot(ax=axes[1],color = "#ccccccff",linestyle="dashed",linewidth=1)
    roads.plot(ax=axes[1],linewidth=1.2,color="#b2b2b2ff")
    footprints.plot(ax=axes[1],color="#ccccccff")
    p.plot(ax=axes[1],color="#000080ff",linewidth=0.5)
    n.loc[n.isConnected.fillna(False),:].plot(ax=axes[1],
                                              color="#000080ff",
                                              markersize=1)
    n_grid.loc[n_grid.isHeatingStation.fillna(False),:].plot(ax=axes[1],
                                                        color="red",
                                                        markersize=1)
    n.loc[n.isHeatingStation.fillna(False),:].plot(ax=axes[1],
                                                   color="red",
                                                   markersize=1)
    
    if os.path.exists(".env"):
    # 2A
        rail.plot(ax=axes[2],color = "#ccccccff",linestyle="dashed",linewidth=1)
        roads.plot(ax=axes[2],color = "#b2b2b2ff",linewidth=1.2)
        footprints.plot(ax=axes[2],color="#ccccccff")
        pipes_anon.plot(ax=axes[2],color="red",linewidth=0.7)
        p_db_anon.plot(ax=axes[2],color="#000080ff",linewidth=0.5)
        
        # 2B
        rail.plot(ax=axes[3],color = "#ccccccff",linestyle="dashed",linewidth=1)
        roads_mod.loc[roads_mod["weight"] == 0.5,:].plot(ax=axes[3],
                                                         color="#b2b2b2ff",
                                                         linewidth=1.2)
        roads_mod.loc[roads_mod["weight"] == 10,:].plot(ax=axes[3],
                                                        color="#1a1a1aff",
                                                        linestyle="dotted",
                                                        linewidth=1.2)
        footprints_mod.plot(ax=axes[3],color="#ccccccff")
        pipes_anon.plot(ax=axes[3],color="red",linewidth=0.7)
        p_db_mod_anon.plot(ax=axes[3],color="#000080ff",linewidth=0.5)
    
    for a in range(4):
        axes[a].set_xticks([])
        axes[a].set_yticks([])
        axes[a].set_xlim(BOUNDS[0]-20,BOUNDS[2]+20)
        axes[a].set_ylim(BOUNDS[1]-20,BOUNDS[3]+20)
    
    if not os.path.esists("plots"):
        os.mkdir("plots")
        
    plt.savefig("plots/comparison.pdf", bbox_inches='tight')
    
    plt.show()
    
    
    
    # Reports
    print("")
    
    def calculate(pipes_to_compare, nodes_to_compare=[]):
        """ Utility to calculate statistics for different variants """
        
        print("Length", round(pipes_to_compare.length.sum()), "m")
        print("N of pipes", pipes_to_compare.shape[0])
        if os.path.exists(".env"):
            print("Difference with ref network",round((pipes_to_compare.length.sum()-pipes.length.sum())/pipes.length.sum()*100), "%")
            if len(nodes_to_compare) > 0:
                peak_power = abs(nodes_to_compare.loc[nodes_to_compare.fillna(False).isHeatingStation,"peak"].values[0])
                print("Peak power", round(peak_power/1000,2), "MW")
                print("Difference with ref network",
                      int(round(((peak_power-substations.power.sum())/substations.power.sum())*100, 0)), "%")
            else:
                print("Peak power", round(substations.power.sum()/1000,2), "MW")
        print("")
    
    if os.path.exists(".env"):
        print("Reference")  
        calculate(pipes)
    print("RegBL")
    calculate(p_grid,n_grid)
    print("RegBL + roads + footprints")
    calculate(p,n)
    
    if os.path.exists(".env"):
        print("DB")
        calculate(p_db,n_db)
        print("DB mod")
        calculate(p_db_mod,n_db_mod)
        

