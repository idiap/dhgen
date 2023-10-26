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
Generate DHN layouts from input geodata.

@author: gperonato
"""

from . import geoadmin, size

import geopandas as gpd
import pandas as pd
import networkx as nx
import numpy as np
from shapely.geometry import box, Point, LineString, Polygon, MultiPoint
from math import atan2, degrees, tan, radians
from shapely import affinity
from networkx.algorithms import approximation as approx
import os
import logging
import signal
import time
import subprocess
import uuid
import fiona
import pickle

logging.basicConfig(level=logging.INFO)
REPOSITORY_DIR = os.path.dirname(os.getcwd())

GRID_SIZE = {"width": 25,
             "height": 25,
             "rotation": 0}


# Default parameters used by functions
ROAD_WEIGHT = 0.3
GEOADMIN_CACHE_FILE = "GeoAdmin_cache.gpkg"
T_EXT = -7 # Payerne (CH) - SIA:2028 (2015)
T_INT = 20

def nx_grid(bounds, width, height, rotation=0):
    """Create spatialized nx grid"""
    # Original method by Mativane and Ric
    # https://gis.stackexchange.com/a/316460
    # CC BY-SA 4.0
    xmin, ymin, xmax, ymax = bounds
    if rotation != 0:
        bbox = box(*bounds)
        center = bbox.centroid
        rotated_box = affinity.rotate(bbox,rotation,center)
        xmin, ymin, xmax, ymax = [*rotated_box.bounds]
        
    cols = list(np.arange(xmin, xmax + width, height))
    rows = list(np.arange(ymin, ymax + height, height))
    
    gridn = nx.grid_2d_graph(len(cols), len(rows))
    
    # Check for node intersection (and related edges)
    for node in gridn.nodes:
        point = Point(cols[node[0]],rows[node[1]])
        if rotation != 0:
            point = affinity.rotate(point,rotation,center)
        gridn.nodes[node].update({"geometry":point})
    return gridn


def gdf_to_nx(gdf, add_edge_attributes = False):
    """Create a nx from a line gdf"""
    points = []
    lines = gdf.explode(index_parts=True)
    i = 0
    net = nx.Graph()
    for l, line in lines.iterrows():
        line_pts = []
        for p, pt in enumerate(line["geometry"].coords):
            point = Point(pt[0],pt[1])
            line_pts.append(point)
            if point not in points:
                points.append(point)
                net.add_node(i,geometry=point)
                i += 1
            if p > 0:
                attributes = line.to_dict() if add_edge_attributes else {}
                attributes = {key: attributes[key] for key in attributes if key != "geometry"}
                net.add_edge(points.index(line_pts[-2]),points.index(line_pts[-1]),**attributes)
    return net


def flow_to_nx(network, flowdict):
    """Add information from flowdict to graph"""
    for origin in flowdict.keys():
        for destination in flowdict[origin].keys():
            try:
                network.edges[(origin,destination)]["power"] = flowdict[origin][destination]
            except:
                pass
    return network


def add_node_attributes_from_network(network, network_to_copy):
    """Add information from other network"""
    data = network_to_copy.nodes(data=True)     
    for node in network.nodes:
        network.nodes[node].update(data[node])
    return network

def add_edge_attributes_from_network(network, network_to_copy):
    """Add information from other network"""   
    for edge in network.edges:
        network.edges[edge[0],edge[1]].update(network_to_copy.edges[edge[0],edge[1]])
    return network

def add_edge_length(network, to_round=True):
    """Add length from nodes geometry"""      
    for edge in network.edges:
        if "geometry" not in network.nodes[edge[0]].keys():
            print(edge,network.nodes[edge[0]])
        pt1 = network.nodes[edge[0]]["geometry"]
        pt2 = network.nodes[edge[1]]["geometry"]
        dist = pt1.distance(pt2)
        if dist == 0:
            logging.warning("Edge {}-{} has null length".format(edge[0],edge[1]))
        if to_round:
            dist = int(round(dist))
        network.edges[edge]["length_m"] = dist
    return network

def add_power(regbl, t_ext=T_EXT, t_int=T_INT, to_round=True, limit=True):
    """Simple physical model for building peak power"""
    deltaT = t_int - t_ext
    GBAUP_to_UValue = {8011: 0.94, # before 1919 (Perez, 2014)
                       8012: 0.94, # 1919-1945 (Perez, 2014)
                       8013: 1.35, # 1946-1960 (Perez, 2014)
                       8014: 1.07, # 1961-1970 (Perez, 2014)
                       8015: 0.88, # 1971-1980 (Perez, 2014)
                       8016: 0.90, # 1981-1985 (Perez, 2014)
                       8017: 0.90, # 1986-1990 (Perez, 2014)
                       8018: 0.69, # 1991-1995 (Perez, 2014)
                       8019: 0.69, # 1996-2000 (Perez, 2014)
                       8020: 0.51, # 2001-2005 (Perez, 2014)
                       8021: 0.51, # 2006-2010 (Perez, 2014)
                       8022: 0.20, # 2011-2015 (SIA 380/1:2009)
                       8023: 0.20 # after 2015 (SIA 380/1:2009)
                         }
    # Benchmark values by SuisseEnergie
    # https://pubdb.bfe.admin.ch/fr/publication/download/2781
    GKLAS_to_DHW_power = {1110: 2, # Single-family
                          1121: 3, # Multi-family (2)
                          1122: 3, # Multi-family (>= 3)
                          1130: 3, # Community buildings
                         } 
    regbl["gbaup"] = pd.to_numeric(regbl["gbaup"]).astype("Int64")
    regbl["gklas"] = pd.to_numeric(regbl["gklas"]).astype("Int64")
    regbl["nfloors"] = pd.to_numeric(regbl["gastw"])
    regbl.loc[regbl["nfloors"].isnull() & ~regbl["egid"].isnull(),"nfloors"] = 1.999 # Default value
    regbl["footprint_area"] = pd.to_numeric(regbl["garea"])
    regbl.loc[regbl["footprint_area"].isnull() & ~regbl["egid"].isnull(),"footprint_area"] = 49.999 # Default value
    regbl["floor_area"] = (regbl["footprint_area"] * regbl["nfloors"])
    
    def get_heat_power(regbl, deltaT=deltaT, GBAUP_to_UValue=GBAUP_to_UValue):
        wall_area = 4 * np.sqrt(regbl["footprint_area"]) * regbl["nfloors"] * 3
        roof_area = regbl["footprint_area"]
        ground_area = regbl["footprint_area"] 
        envelope_area = wall_area + roof_area + ground_area
        return (envelope_area * deltaT * GBAUP_to_UValue.get(regbl["gbaup"],
                                                             1.35 # Default value
                                                             ))/1000
    
    def get_dhw_power(regbl,GKLAS_to_DHW_power=GKLAS_to_DHW_power):
        return (regbl["floor_area"] * GKLAS_to_DHW_power.get(regbl["gklas"],
                                                             1 # Default value
                                                             ))/1000
    
    regbl["peak_heat"] = regbl.apply(get_heat_power,axis=1)
    regbl["peak_dhw"] = regbl.apply(get_dhw_power,axis=1)
    regbl["peak"] = (regbl["peak_heat"] + regbl["peak_dhw"]).fillna(0)
    
    if limit:
    # Limit results to recommended benchmark values by SuisseEnergie
    # https://pubdb.bfe.admin.ch/fr/publication/download/2781
        regbl.loc[regbl["peak"]/regbl["floor_area"] > (80/1000), "peak"] = (80 * regbl["floor_area"])/1000
        regbl.loc[regbl["peak"]/regbl["floor_area"] < (25/1000), "peak"] = (25 * regbl["floor_area"])/1000
    
    if to_round:
        regbl["peak"] = regbl["peak"].round().astype(int)
    return regbl

def get_angle(p1: Point, p2: Point) -> float:
    """Get the angle of this line with the horizontal axis."""
    # Utility function adapted from Roberto Boghetti's pairing tool
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    theta = atan2(dy, dx)
    angle = degrees(theta)  # angle is in (-180, 180]
    if angle < 0:
        angle = 360 + angle
    return angle
    
def get_perpendicular(nodes, edges, pt,
                      max_distance=50,
                      acceptable_angle_diff = 3,
                      weigh_connections=False,
                      connect_only_to_main=False):
    """Get perpendicular from a point to the closest graph edge"""

    def get_angles_difference(a, b):
        # Utility function adapted from Roberto Boghetti's pairing tool
        x = a - b
        x = (x + 180) % 360 - 180
        return x
    
    def get_intersection(pt1, angle1, pt2, angle2):
        m1 = tan(radians(angle1))
        m2 = tan(radians(angle2))
        q1 = pt1.y - m1*pt1.x
        q2 = pt2.y - m2*pt2.x
        x = (q1 - q2) / (m2 - m1)
        y = m1 * ((q1 - q2) / (m2 - m1)) + q1
        return Point(x,y)
    
    closest_nodes = list(nodes.loc[nodes["distance"] < max_distance,"node"].values)
    
    closest_edges = []
    for edge in edges:
        if edge[0] in closest_nodes or edge[1] in closest_nodes:
            my_dict = {"start": edge[0],"end": edge[1]}
            my_dict["weight"] = edge[2].get("weight",
                                            1) # default weight
            my_dict["isTerminal"] = edge[2].get("isTerminal",False)
            closest_edges.append(my_dict)
                
    closest_edges = pd.DataFrame(closest_edges)
    
    if connect_only_to_main:
        closest_edges = closest_edges.loc[~closest_edges["isTerminal"].fillna(False),:]

    if len(closest_edges) == 0:
        raise Warning("No edges within {} m to point {}".format(max_distance,pt))
    closest_edges["p1"] = closest_edges.merge(nodes, right_on="node",left_on="start", how="left")["geometry"]
    closest_edges["p2"] = closest_edges.merge(nodes, right_on="node",left_on="end", how="left")["geometry"]
    closest_edges = closest_edges.loc[~closest_edges["p1"].isnull(),:]
    closest_edges = closest_edges.loc[~closest_edges["p2"].isnull(),:]
    closest_edges["angle"] = closest_edges.apply(lambda x: get_angle(x.p1,x.p2),axis=1)
    dfs = []
    # Loop over angles close to 90˚
    for angle in list(range(90-acceptable_angle_diff,90+acceptable_angle_diff+1,acceptable_angle_diff)) if acceptable_angle_diff != 0 else [90]:
        cl_ed = closest_edges.copy()
        cl_ed["angle_diff"] = cl_ed.apply(lambda x: get_angles_difference(x.angle,angle),axis=1)
        cl_ed["geometry"] = cl_ed.apply(lambda x: LineString([x.p1, x.p2]),axis=1)
        cl_ed["int"] = cl_ed.apply(lambda x: get_intersection(pt.geometry,
                                                              x.angle_diff,
                                                              x.p1,
                                                              x.angle),
                                                   axis=1)
        dfs.append(cl_ed)
    closest_edges = pd.concat(dfs,axis=0,ignore_index=True)
    
    # Filter only points with the intersection falling within the edge
    closest_edges["int_on_edge"] = closest_edges.apply(lambda x: x.geometry.distance(x.int) < 1e-4,axis=1)
    closest_edges = closest_edges.loc[closest_edges["int_on_edge"],:]
        
    # Get only the closest intersection points
    closest_edges["dist"] = closest_edges.apply(lambda x: pt.geometry.distance(x.int),axis=1)
    if not weigh_connections:
        closest_edges["weight"] = 1 # overwrite weight
    closest_edges["dist_weighted"] = closest_edges["dist"] * closest_edges["weight"]
    closest_edges = closest_edges.sort_values("dist_weighted")
        
    closest_edges = gpd.GeoDataFrame(closest_edges)
    return closest_edges.iloc[0]


def add_gdf_to_nx(graph, gdf,
                  connect_all=False,
                  max_connections=1, 
                  max_degree_increase=1,
                  max_distance=200, 
                  prefix=-1,
                  mode = "perpendicular",
                  weigh_connections = False,
                  connect_only_to_main = True
                  ):
    """
    Connect point gdf to an existing NetworkX graph.

    Parameters
    ----------
    graph : nx.Graph
        Networkx graph to modify.
    gdf : gpd.GeoDataFrame
        GeoDataFrame of points to add to the graph.
    connect_all : boolean, optional
        DESCRIPTION. The default is False.
    max_connections :  int, optional
        Degree of added node. Not used when perpendicular. The default is 1.
    max_degree_increase : int, optional
        Degree of existing node minus the original degree. Not used when perpendicular.
        The default is 2.
    max_distance : int, optional
        Max distance of added node to existing node. The default is 100.
    prefix : int, optional
        Prefix in tuple for added node. The default is -1.
    mode : str, optional
        If not "perpendicular" look for closest point. The default is perpendicular.
    weigh_connections : boolean, optional
        In perpendicular mode, weigh the distances using the weight of the target edge. The default is False.
    connect_only_to_main : boolean, optional
        In perpendicular mode, connect only to edges belong to the original graph. The default is False.
    Returns
    -------
    graph : nx.Graph
        Graph including the new nodes and edges.

    """
    # Create a geodataframe from nodes of the original graph
    nodes = pd.DataFrame(graph.nodes(data=True),columns=["node","attributes"])
    nodes = nodes.join(pd.DataFrame(nodes.pop('attributes').values.tolist()),rsuffix="_")
    nodes = gpd.GeoDataFrame(nodes)
    nodes = nodes.set_index("node")
    nodes["original_degree"] = pd.Series(dict(graph.degree).values(),
                                         index=dict(graph.degree).keys())
    nodes = nodes.reset_index()
    
    
    # Prepare points to add
    gdf = gdf.copy() # Prevent SettingWithCopyError in case gdf was sliced
    gdf["node"] = [(prefix,n) for n in gdf.index]
    if connect_all:
        # Look for closest in all points: might create unconnected graph
        pts = pd.concat([nodes,gdf],
                        ignore_index=True)
    else:
        # Look for closest to only existing nodes
        pts = nodes
        
    # Iterate over points to add
    for n, point in gdf.iterrows():
        # Check first whether the geometry to add is already in the graph
        if point["geometry"] in nodes["geometry"]:
            logging.warning(f'Connecting node {point["node"]} with existing geometry') 
            #TODO avoid creating node with the same geometry
            
        # Add each point to the graph            
        graph.add_node(point["node"],
                       **point.to_dict())
        
        # Look for potential connections to create edges
        if mode == "perpendicular":
            # Get nodes and edges at every cycle as they change
            nodes = pd.DataFrame(graph.nodes(data=True),columns=["node","attributes"])
            nodes = nodes.join(pd.DataFrame(nodes.pop('attributes').values.tolist()),
                               rsuffix="_")
            nodes = gpd.GeoDataFrame(nodes)
            nodes["distance"] = nodes.distance(point.geometry)
            nodes = nodes.sort_values("distance")
            edges = graph.edges(data=True)
            closest_edge = get_perpendicular(nodes,
                                             edges,
                                             point,
                                             max_distance,
                                             weigh_connections=weigh_connections,
                                             connect_only_to_main=connect_only_to_main)
                        
            int_node_found = False
            # Add intersection node
            if closest_edge["int"] in nodes["geometry"]:
                # The intersection node is already in the graph
                int_node = nodes.loc[nodes["geometry"] == closest_edge["int"],"node"].values[0]
                int_node_found = True
                # But no edge is connected to that: we can safely use to split the existing edge
                if graph.degree(int_node) != 0:
                    logging.warning(f"Creating intersection {int_node} with existing geometry")
                    #TODO avoid creating node with the same geometry
                    int_node_found = False
            if not int_node_found:
                # We need to add the node and its perpendicular edge
                int_node = "perpint_" + str(uuid.uuid4())
                graph.add_node(int_node,geometry=closest_edge["int"])
                # Add perpendicular connection
                graph.add_edge(point["node"],int_node,
                                isTerminal=True # this is used by connect_only_to_main
                                )
            # The closest edge is split
            edge_attributes = graph.edges[closest_edge["start"],closest_edge["end"]]
            edge_attributes =  {key: edge_attributes[key] for key in edge_attributes if key != "geometry"}
            edge_attributes["weight"] = closest_edge["weight"]
    
            # Add first half of new edge
            graph.add_edge(closest_edge["start"],int_node,
                           **edge_attributes)
            # Add second half of new edge
            graph.add_edge(int_node,closest_edge["end"],
                           **{key: edge_attributes[key] for key in edge_attributes if key != "isTerminal"})
            # Rmove original edge
            graph.remove_edge(closest_edge["start"],closest_edge["end"])
            
   
        else:
            n_connections = 0
            while n_connections <= max_connections:
                distances = pts.distance(point["geometry"]).sort_values()
                distances = distances.loc[distances<= max_distance]
                closest_pts = pts.loc[distances.index[1:],:] #start from 1: 0 is the same point
                for c, closest_pt in closest_pts.iterrows():
                    current_degree = graph.degree[closest_pt["node"]]
                    if current_degree - closest_pt["original_degree"] < max_degree_increase:
                        graph.add_edge(closest_pt["node"],
                                       point["node"])
                        n_connections += 1
                        break
    return graph


def densify_geometry(line_geometry, step):
     """Densify line geometry"""
     # Oeiginal Solution by Nick Pucino. Including suggestion of adding the last step.
     # https://gis.stackexchange.com/a/373279
     # CC BY-SA 4.0
     
     length_m=line_geometry.length # get the length
 
     xy=[] # to store new tuples of coordinates
     for distance_along_old_line in np.arange(0,
                                              int(length_m),
                                              step): 
 
         point = line_geometry.interpolate(distance_along_old_line) # interpolate a point every step along the old line
         xp,yp = point.x, point.y # extract the coordinates
 
         xy.append((xp,yp)) # and store them in xy list
         
     # Modification of the original solution to add the last point
     xy.append((line_geometry.coords[-1][0],line_geometry.coords[-1][1]))
     if len(xy) == 0  or len(xy) > 1:
         new_line=LineString(xy) # Here, we finally create a new line with densified points.
         return new_line
     else:
         logging.warning("Found invalid line geometry... skipping")


def connect_graphs(n1, n2, max_distance=25, add_length=True):
    """Connect two graphs by adding edbe between closest nodes."""
    n1_pts = pd.DataFrame(n1.nodes(data=True),columns=["node","attributes"])
    n1_pts = n1_pts.join(pd.DataFrame(n1_pts.pop('attributes').values.tolist()),rsuffix="_")
    n1_pts = gpd.GeoDataFrame(n1_pts)
    n3 = n1.copy()
    for n, attrs in n2.nodes(data=True):
        n3.add_node(n, **attrs)
        distances = n1_pts.distance(attrs["geometry"]).sort_values()
        distances = distances.loc[distances <= max_distance]
        if len(distances) > 1:
            closest_pt = n1_pts.loc[distances.index[1],"node"]
            n3.add_edge(n, closest_pt)
            if add_length:
                n3.edges[n,closest_pt]["length_m"] = distances.iloc[1]
    
    for n1, n2,  attrs in n2.edges(data=True):
        n3.add_edge(n1, n2, **attrs)
        
    return n3

def intersect_graphs(n1, n2):
    """Connect two graphs by finding their intersecting nodes"""

    def create_intersections(graph, name, gdfs, all_inter, all_pts):
        # Loop over edges containing an intersection
        for line in all_inter["{}_line".format(name)].unique():
            # Getting the intersection points
            inters = all_inter.loc[all_inter["{}_line".format(name)] == line,:]
            # Getting the start and end points
            start = gdfs[1].loc[line, "start"]
            end = gdfs[1].loc[line, "end"]

            start_end = pd.DataFrame([{"name": start,
                          "geometry" : graph.nodes[start]["geometry"],
                          "x": graph.nodes[start]["geometry"].x,
                          "y": graph.nodes[start]["geometry"].y
                          },
                          {"name": end,
                           "geometry" : graph.nodes[end]["geometry"],
                           "x": graph.nodes[end]["geometry"].x,
                           "y": graph.nodes[end]["geometry"].y
                                        }
                          ])
            # Sort all points
            inters = inters.sort_values(["x","y"])
            start_end = start_end.sort_values(["x","y"])
            
            # Get attributes of original edge
            attributes = graph.edges[start,end]
            attributes = {key: attributes[key] for key in attributes if key != "geometry"}
 
            # Create the new edges and nodes
            counter = 0
            for _int, inter in inters.iterrows():
                if inter["geometry"] in all_pts["geometry"]:
                    # Skipping as the same geometry exists in the graph
                    pass
                else:
                    graph.add_node(inter["name"],geometry=inter["geometry"])
                    # New edges
                    if counter == 0:
                        graph.remove_edge(start,end) # Remove the original edge
                        graph.add_edge(start_end.iloc[0]["name"],inter["name"],
                                    geometry = LineString([start_end.iloc[0]["geometry"],
                                                          inter["geometry"]]),
                                                          **attributes
                                    )
                    if counter < len(inters) -1:
                        graph.add_edge(inters.iloc[counter]["name"],inters.iloc[counter+1]["name"],
                                    geometry = LineString([inters.iloc[counter]["geometry"],
                                                          inters.iloc[counter+1]["geometry"]]),
                                                          **attributes
                                    )
                    if counter == len(inters) -1:
                        graph.add_edge(inter["name"],start_end.iloc[1]["name"],
                                    geometry = LineString([inter["geometry"],
                                                          start_end.iloc[1]["geometry"]]),
                                                          **attributes
                                    )
                        break
                    counter += 1 
        return graph
                
    n1_pts, n1_lines = nx_to_gdf(n1)
    n2_pts, n2_lines = nx_to_gdf(n2)
    all_pts = pd.concat([n1_pts,n2_pts])
    all_inter = []
    logging.warning('Warnings in shapely intersection method are currently ignored...')
    for n2i, n2_seg in n2_lines.iterrows():
        # Suppress warnings in Shapely for empty intersections until fixed
        # "RuntimeWarning: invalid value encountered in intersection"
        initial_settings = np.seterr()
        np.seterr(invalid="ignore")
        inter = n1_lines.intersection(n2_seg.geometry)
        # Reset initial error settings
        np.seterr(**initial_settings)
        inter = inter[~inter.is_empty] 
        for n1i, inte in pd.DataFrame(inter).iterrows():
            inter_dict = {}
            if type(inte.values[0]) == Point:
                inter_dict["geometry"] = inte.values[0]
                inter_dict["n1_line"] = n1i
                inter_dict["n2_line"] = n2i
                all_inter.append(inter_dict)
            
    all_inter = pd.DataFrame(all_inter)
    all_inter["x"] = all_inter.geometry.apply(lambda x: x.x)
    all_inter["y"] = all_inter.geometry.apply(lambda x: x.y)
    all_inter["name"] = all_inter.geometry.apply(lambda x: "netint_"+str(uuid.uuid4()))
    
    n1 = create_intersections(n1,"n1",[n1_pts, n1_lines], all_inter, all_pts)
    n2 = create_intersections(n2,"n2",[n2_pts, n2_lines], all_inter, all_pts)
    
    n3 = nx.compose(n1,n2)
    
    Gcc = sorted(nx.connected_components(n3), key=len, reverse=True)
    if len(Gcc) > 1:
        with open('subgraphs.p', 'wb') as f:
            pickle.dump(n3, f)
        logging.warning("{} subgraphs... keeping only the largest one. A pickled nx graph 'subgraphs.p' has been saved.".format(len(Gcc)))
        n3 = n3.subgraph(Gcc[0])
    return nx.Graph(n3)

def weigh_edges(network, weight,
                original_attribute = "length_m",
                weighted_attribute = "weighted_attr",
                to_round=True,
                replace=False):
    """Weight edge attribute."""
    for edge in network.edges():
        if original_attribute in network.edges()[edge]:
            if weighted_attribute in network.edges()[edge] and not replace:
                logging.debug("Attribute {} existing for edge {}: skipping.".format(weighted_attribute,edge))
            else:
                network.edges()[edge][weighted_attribute] = network.edges()[edge][original_attribute] * weight
                if to_round:
                    network.edges()[edge][weighted_attribute] = int(round(network.edges()[edge][weighted_attribute]))
    return network

def get_geoadmin_api(api_parameters):
    """Retrieve data from GeoAdmin API and save local file"""
    gdf = geoadmin.geoadmin_to_gdf(api_parameters)
    gdf.to_file(GEOADMIN_CACHE_FILE,layer=api_parameters["layers"],driver="GPKG")
    return gdf

def api_parameters_to_geometry(api_parameters):
    """Create shapely geometry from geometry in API parameters"""
    if api_parameters["geometryType"] == "esriGeometryEnvelope" and len(api_parameters["geometry"]) > 0:
        bounds = [float(x) for x in api_parameters["geometry"].split(",")]
        polygon = box(*bounds)
    elif api_parameters["geometryType"] == "esriGeometryPolygon":
        polygon = Polygon([[float(x) for x in lst] for lst in api_parameters["geometry"]["rings"][0]])
    else:
        polygon = None
    return polygon
        
def import_substations(api_parameters,gpkg_path=GEOADMIN_CACHE_FILE,limit_egids=[],layer=None):
    """Load substations data"""
    substations = import_geometry(api_parameters,gpkg_path)
    
    is_dupl = substations.geometry.duplicated()
    if is_dupl.sum() > 0:
        logging.warning(f"{is_dupl.sum()} substations have the same geometry:"
                        "an offset is applied")
        substations.loc[is_dupl,"geometry"] = substations.loc[is_dupl].translate(xoff=0.5,
                                                                                 yoff=0.5)

    substations["isConnected"] = True
    substations["isHeatingStation"] = False

    if "egid" in substations.columns:
        substations.egid = pd.to_numeric(substations.egid)
    
    if len(limit_egids) > 0:
        substations = substations.loc[substations.egid.isin(limit_egids),:]

    return substations

def import_geometry(api_parameters,gpkg_path=GEOADMIN_CACHE_FILE):
    """Load geodata from either file or API"""
    polygon = api_parameters_to_geometry(api_parameters)
    layer = api_parameters["layers"]
    if os.path.exists(gpkg_path) and layer in fiona.listlayers(gpkg_path):
        geometry = gpd.read_file(gpkg_path, layer=layer)
        if polygon:
            geometry = geometry.loc[geometry.intersects(polygon),:]
        if geometry.shape[0] == 0:
            geometry = get_geoadmin_api(api_parameters)
    else:
        geometry = get_geoadmin_api(api_parameters)
        
    # Make sure that in GeoAdmin geodata order is preserved to enforce reproducibility
    if "id" in geometry.columns:
        geometry = geometry.sort_values("id").reset_index()
        
    return geometry

def create_stations(substations, heating_stations=[]):
    """Concatenate heating stations and substations in a single gdf"""
    # Add power to heating substations
    if "peak" not in substations.columns:
        substations = add_power(substations)
    
    if len(heating_stations) == 0:
        heating_stations = [box(*substations.total_bounds).centroid] # at the center
        logging.warning("Heating station missing... adding one station at the center of the substations area")
    heating_stations = gpd.GeoDataFrame(geometry = [Point(x) for x in heating_stations],
                                        crs=substations.crs)
    heating_stations["isConnected"] = True
    heating_stations["isHeatingStation"] = True
    
    # Concatenate points
    pts = pd.concat([substations,
                     heating_stations],
                    ignore_index=True
                    )
    
    # Add power to heating station
    pts.loc[pts["isHeatingStation"],"peak"] = -pts["peak"].sum()
    return pts

def create_road_network(roads, weight, densify_step = 5):
    """Densify road geometry and create a road network assigning a weight."""
    roads_dense = roads.copy().explode(index_parts=True).reset_index()
    logging.info("Densifying road geometry")
    roads_dense["geometry"] = roads_dense["geometry"].apply(densify_geometry,
                                                              step=densify_step)
    if "weight" in roads_dense.columns:
        no_weight = roads_dense.weight.isnull()
        logging.info("Using custom weight for {} out of {} segments".format(roads_dense.shape[0]-no_weight.sum(),
                                                                            roads_dense.shape[0]))
        roads_dense.loc[no_weight,"weight"] = weight
    else:
        roads_dense["weight"] = weight
    
    # Condense all columns in a dict on properties columns if not existing already
    if "properties" not in roads_dense.columns:
        selected = roads_dense.loc[:,
                                    [x for x in roads_dense.columns if x not in ["properties","weight","geometry"]]]
        roads_dense["properties"] = pd.Series(selected.transpose().to_dict(orient='dict'))
    
    logging.info("Creating a graph from roads gdf")
    roadsn = gdf_to_nx(roads_dense[["properties","weight","geometry"]],
                         add_edge_attributes=True)
        
    return roadsn

def add_pos_from_point(G):
    """Add pos from shapely Point geometry attribute."""
    for node,attrs in G.nodes(data=True):
        point = attrs["geometry"]
        G.nodes[node]["pos"] = (point.x, point.y)
    return G

def delisting(orig_df):
    """Any list or tuple to string."""
    df = orig_df.copy()
    def delist(mylist):
        """List to string."""
        if (type(mylist) == list or type(mylist) == tuple) and len(mylist) > 0:
            return [str(s) for s in mylist]
        else:
            return [str(mylist)]
    hasList = (df.drop("geometry",axis=1).applymap(type) == list).any()
    hasTuple = (df.drop("geometry",axis=1).applymap(type) == tuple).any()
    for col in hasList.loc[hasList].index:
        df[col] = df.apply(lambda x:",".join(delist(x[col])),axis=1)
    for col in hasTuple.loc[hasTuple].index:
        df[col] = df.apply(lambda x:",".join(delist(x[col])),axis=1)           
    return df

def nx_to_gdf(network, pts=None):
    """Convert nx to gdf"""    
    if pts: # Use geometry from a point gdf
        df = pd.DataFrame(network.edges(data=True),columns=["pt1","pt2","attributes"])
        df["geometry"] = df.apply(lambda x: LineString([pts.loc[x.pt1,"geometry"],
                                                        pts.loc[x.pt2,"geometry"]]),axis=1)
        df = df.join(pd.DataFrame(df.pop('attributes').values.tolist()))
        edges = gpd.GeoDataFrame(df)  
        return edges
    else: # Use geometry from the nodes attributes
        nodes = pd.DataFrame(network.nodes(data=True),columns=["node","attributes"])
        nodes = nodes.join(pd.DataFrame(nodes.pop('attributes').values.tolist()),rsuffix="_")
        nodes = gpd.GeoDataFrame(nodes)
        edges = []
        for edge in network.edges(data=True):
            edge_dict = {"start":edge[0], "end": edge[1],
                         "geometry":LineString([nodes.loc[nodes.node == edge[0]]["geometry"].values[0],
                                     nodes.loc[nodes.node == edge[1]]["geometry"].values[0]])}
            edge_dict.update(edge[2])
            edges.append(edge_dict)
        edges = gpd.GeoDataFrame(edges)
        return nodes, edges
    
def apply_husek(graph, wait_solution=5, seed=None):
    """Apply the Steiner tree heuristics by Hušek et al.
    https://github.com/goderik01/PACE2018/blob/master/Readme.pdf"""
    
    nodes = pd.DataFrame(index=range(len(graph.nodes)),columns=["name"])
    edge_list = nx.to_pandas_edgelist(graph)

    for n, node in enumerate(graph.nodes):
        nodes.loc[n,"name"] = node

    # Dicts to map the tuple to the index
    nodes_map = dict((str(nodes.to_dict()["name"][k]), k) for k in nodes.to_dict()["name"])
    nodes_map_inv = dict((k, nodes.to_dict()["name"][k]) for k in nodes.to_dict()["name"])
    
    nodes_data = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index').reset_index()
    edge_list["source_"] = edge_list["source"].astype(str).replace(nodes_map)
    edge_list["target_"] = edge_list["target"].astype(str).replace(nodes_map)
    
    terminals = nodes_data.loc[nodes_data.isConnected.fillna(False), :]

    solution_fn = "network.ost"
    if not os.path.exists(solution_fn):
        solution_fn = "network.ost.temp"
        file = "SECTION Graph\n"
        file += "Nodes {}\n".format(len(nodes))
        file += "Edges {}\n".format(len(edge_list))
        for e, edge in edge_list.iterrows():
            file += "E {} {} {}\n".format(edge.source_+1, edge.target_+1, edge.priority)
        file += "END\n\n\n"
        file += "SECTION Terminals\n"
        file += "Terminals {}\n".format(len(terminals))
        for t, terminal in terminals.iterrows():
            file += "T {}\n".format(t+1)
        file += "END\n"
        file += "EOF"
        
        # Read the input .ost file
        with open("input.gr", "w") as f:
            f.write(file)
        infile = open("input.gr", "r")
        outfile = open(solution_fn, "w")
        # Perform the model
        logging.info("Running model for {} seconds".format(wait_solution))
        command = ["star_contractions_test"]
        if seed != None:
            logging.info("Random seed set to {}".format(seed))
            command.extend(["-s", str(seed)])
        p = subprocess.Popen(command,
                             stdin=infile,
                             stdout=outfile,
                             stderr=subprocess.DEVNULL, # do not print
                             preexec_fn=os.setpgrp,
                             )
        time.sleep(wait_solution)
        os.killpg(p.pid, signal.SIGTERM)
        p.wait()
        logging.info("Waiting for results...")
        time.sleep(2+len(nodes)//1000)
        infile.close()
        outfile.close()
        
    with open(solution_fn) as f:
        content = f.read().splitlines()
    edges = []
    logging.info("Tree weight: {}".format(content[0].split("VALUE ")[1]))
    for l in content[1:]:
        edges.append([nodes_map_inv[int(x)-1] for x in l.split(" ")])
    return nx.from_edgelist(edges)
        
def fours_to_threes(G, distance=2):
    """Replace 4-degree nodes with 2 3-degree nodes"""

    fours = []
    for node in G.nodes:
        if G.degree[node] == 4:
            fours.append(node)
    

    for node in fours:
        # Get edges starting from the central node
        nodes, edges_ = nx_to_gdf(nx.edge_subgraph(G,G.edges(node)))
        if "weight" not in edges_.columns:
            edges_["weight"] = 1
        edges_["startc"] = edges_["start"]
        edges_["endc"] = edges_["end"]
        
        # Get start and end nodes for all edges oriented outwards from the node
        to_be_inverted = []
        for e, edge in edges_.iterrows():
            if edge["start"] != node:
                to_be_inverted.append(e)
        edges_.loc[to_be_inverted,"startc"] = edges_.loc[to_be_inverted,"end"]
        edges_.loc[to_be_inverted,"endc"] = edges_.loc[to_be_inverted,"start"]
        
        # Calculate angle using edges oriented outwards        
        edges_["p1c"] = edges_.merge(nodes[["geometry","node"]], right_on="node",left_on="startc", how="left")["geometry_y"].values
        edges_["p2c"] = edges_.merge(nodes[["geometry","node"]], right_on="node",left_on="endc", how="left")["geometry_y"].values
        edges_["anglec"] = edges_.apply(lambda x: get_angle(x.p1c,x.p2c),axis=1)
        # Sort edges by ascending angle, two consecutive ones will be used
        edges_ = edges_.sort_values(by="anglec")
        # Use weights to decide which one is a prioritary axis (not to be moved)
        weights = edges_.copy().sort_values(by="weight")["weight"].fillna(1)
        # Here is the edge to be moved
        edge1 = edges_.iloc[list(edges_.index).index(weights.index[0])-1] 
        # Here is the edge used as axis for the translation
        edge2 = edges_.loc[weights.index[0]] 
        
        # Calculate geometry of the new node on the translation axis
        point = LineString([edge2["p1c"],edge2["p2c"]]).interpolate(distance) 
        
        # Add new node
        new_node = "cross_" + str(uuid.uuid4())
        
        # Check length of new edge
        actual_distance = LineString([point,
                                      G.nodes[node]["geometry"]]).length
        if actual_distance < distance:
            logging.warning("The new node {} can be moved only by {} m".format(new_node,
                                                                               actual_distance))
        G.add_node(new_node,
                   **G.nodes[node]) # add attributes from old node
        G.nodes[new_node]["geometry"] = point # replace geometry
        
        # The second edge is now split into 2 edges
        edge2_attributes = G.edges[edge2["start"],edge2["end"]]
        edge2_attributes = {key: edge2_attributes[key] for key in edge2_attributes if key != "geometry"}
        G.add_edge(edge2["endc"],new_node,
                   **edge2_attributes)
        G.add_edge(new_node,node,
                   **edge2_attributes)
        # This is the new first edge
        edge1_attributes = G.edges[edge1["start"],edge1["end"]]
        edge1_attributes = {key: edge1_attributes[key] for key in edge1_attributes if key != "geometry"}
        G.add_edge(edge1["endc"],new_node,
                   **edge1_attributes) # add attributes from old edge
        
        # Remove original edges
        G.remove_edge(edge1["start"],edge1["end"])
        G.remove_edge(edge2["start"],edge2["end"])
        
    return G
        
def remove_intersecting_edges(G, geometry):
    """Remove edges intersecting with input geometry"""
    edges_gdf = nx_to_gdf(G)[1]
    edges_gdf["intersects"] = edges_gdf["geometry"].intersects(geometry)
    edges_to_remove = list(edges_gdf.loc[edges_gdf["intersects"],["start","end"]].to_records(index=False))
    edges_to_remove = [[x[0], x[1]] for x in edges_to_remove]
    G.remove_edges_from(edges_to_remove)
    return G

def voronoi_nx(pts_gdf, bounds):
    """Create a voronoi graph"""
    from geovoronoi import voronoi_regions_from_coords
    G = nx.Graph()
    mybox = box(*bounds)
    region_polys, region_pts = voronoi_regions_from_coords(pts_gdf["geometry"], mybox)
    for r in region_polys:
        points = region_polys[r].exterior.coords
        for p, pt in enumerate(points[:-1]):
            G.add_node((pt[0],pt[1]),geometry=Point(pt))
            G.add_edge((pt[0],pt[1]),(points[p+1][0],points[p+1][1]))
    return G

def net_to_gpkg(network, path):
    """Export GPKG with nodes and pipes"""
    nodes, pipes = nx_to_gdf(network)
    delisting(nodes).to_file(path,driver="GPKG",layer="nodes")
    delisting(pipes).to_file(path,driver="GPKG",layer="pipes")
    print("Exported to {}".format(path))

def setup_graph(analysis_area=[],
                 grid_size = GRID_SIZE,
                 road_weight = ROAD_WEIGHT,
                 heating_stations=[],
                 limit_egids=[],
                 add_roads=False,
                 add_footprints=False,
                 grid_type = "fishnet",
                 connect_only_to_main=False,
                 geometry_file_path = None,              
                 ):
    """Generate graph of pipes to be used to find the best layout"""
    
    parameters_api = geoadmin.PARAMETERS

    if any(isinstance(i, list) for i in analysis_area):
        bounds = Polygon(analysis_area).bounds
        parameters_api["geometry"] = {"rings": [[[float(x) for x in lst] for lst in analysis_area]]}
        parameters_api["geometryType"] = "esriGeometryPolygon"
        isPolygon=True
    else:
        bounds = analysis_area
        parameters_api["geometry"] = ",".join([str(float(x)) for x in bounds])
        isPolygon = False
        
    # Create substations and heating stations
    # Using the GeoAdmin API, but checking if data is already locally available
    if not geometry_file_path:
        gpkg_file_path = GEOADMIN_CACHE_FILE
        parameters_api["layers"] = "all:ch.bfs.gebaeude_wohnungs_register"
    # Using the GPKG file provided by the user
    else:
        parameters_api["layers"] = "substations"
        gpkg_file_path = geometry_file_path

    substations = import_substations(parameters_api, gpkg_file_path, limit_egids)
    stations = create_stations(substations, heating_stations)
        
    # Create Graph    
    logging.info('Creating graph')
    if len(bounds) == 0:
        logging.info('Using bounds from input (sub)stations')
        bounds = stations.total_bounds
        
    if add_footprints:
        # Create a polygon of the analysis area with the footprints as holes (i.e., what should be kept) 
        if isPolygon:
            inside = Polygon(analysis_area)
        else:
            inside = MultiPoint(list(stations.geometry)).convex_hull.buffer((grid_size["height"]+grid_size["width"])/2).simplify(1.0)
        parameters_footprints = parameters_api.copy()
        parameters_footprints["layers"] = "all:ch.swisstopo.vec25-gebaeude"
        if geometry_file_path:
            parameters_footprints["layers"] = "footprints"
        footprints = import_geometry(parameters_footprints,gpkg_file_path)
        inside = inside.difference(footprints.unary_union)
    
    if grid_type == "fishnet":
        graph = nx_grid(bounds,**grid_size)
        if add_footprints:
            logging.info("Intersect with footprints")
            points, lines = nx_to_gdf(graph)
            points["intersecting"] = points.intersects(inside)
            nodes_to_remove =  points.loc[~points["intersecting"], "node"].tolist()
            edges_to_remove = []
            edges_to_remove.extend([x for x in graph.edges if x[0] in nodes_to_remove or x[1] in nodes_to_remove])
            graph.remove_nodes_from(nodes_to_remove)
            graph.remove_edges_from([[x[0], x[1]] for x in edges_to_remove])
            lines["within"] = lines.within(inside)
            edges_to_remove = lines.loc[~lines["within"],["start","end"]].values.tolist()
            graph.remove_edges_from(edges_to_remove)
    elif grid_type == "voronoi":
        if len(limit_egids) > 0:
            buildings = create_stations(import_substations(parameters_api),
                                               heating_stations=heating_stations)
        else:
            buildings = stations
        graph = voronoi_nx(buildings,bounds)
        logging.info('Removing intersectinge edges')
        graph = remove_intersecting_edges(graph,footprints.unary_union)
    else:
        raise AttributeError("Unknown grid_type {}".format(grid_type))
    
    if not nx.is_connected(graph) and not add_roads and add_footprints:
        raise ValueError("Grid network is not connected. You should add a road network and/or not consider footprints.")
    elif not nx.is_connected(graph):
        logging.warning("Grid network is not connected")
    
    # Connect graph of roads
    if add_roads:
        parameters_roads = parameters_api.copy()
        parameters_roads["layers"] = "all:ch.swisstopo.swisstlm3d-strassen"
        if geometry_file_path:
            parameters_roads["layers"] = "roads"
        roads = import_geometry(parameters_roads,gpkg_file_path)
        roadsn = create_road_network(roads,road_weight,
                                         densify_step=min(10,(grid_size["height"]+grid_size["width"])/2))
        if not nx.is_connected(roadsn):
            logging.warning("Road network is not connected")
        logging.info('Adding roads to graph')
        graph = intersect_graphs(graph, roadsn)
        assert nx.is_connected(graph) == True
    
    logging.info('Adding stations to graph')
    
    graph = add_gdf_to_nx(graph,stations.loc[stations["isConnected"] & ~stations["isHeatingStation"],:],
                          max_distance = 2*(grid_size["height"]+grid_size["width"]), # 4 times grid size
                          weigh_connections=True,
                          connect_only_to_main = connect_only_to_main,
                          )
    graph = add_gdf_to_nx(graph,stations.loc[stations["isHeatingStation"],:],
                          max_distance = 2*(grid_size["height"]+grid_size["width"]), # 4 times grid size
                          weigh_connections=True,
                          connect_only_to_main = connect_only_to_main,
                          )

        
    assert nx.is_connected(graph) == True
    
    return graph


def apply_models(graph,
                 model="husek",
                 wait_solution=5,
                 DT=30,
                 maxPressureLosses=250,
                 seed=None
                 ):
    """Generate DHN layout using Steiner tree and Simplex models."""

    # Create steiner tree of heating (sub)stations
    logging.info('Creating Steiner tree')
    # Add attributes for Steiner tree
    graph = add_edge_length(graph, to_round=False)
    graph = weigh_edges(graph, 1, "length_m", "length_rounded")
    for edge in graph.edges:
        graph.edges[edge]["priority"] = int(round(graph.edges[edge].get("weight",1) * graph.edges[edge]["length_m"]  * 100))
    
    if model == "husek":
        st = apply_husek(graph,
                         wait_solution,
                         seed
                         )
    elif model == "kou" or model == "mehlhorn":
        if int(nx.__version__[0]) >= 3:
            kwargs = {"method": model}
        else:
            kwargs = {}
            if model == "mehlhorn":
                logging.warning("Melhorn method not available for NetworkX<3.0: using Kou.")
  
        st = approx.steinertree.steiner_tree(graph,
                                             [x for x,y in graph.nodes(data=True) if y.get('isConnected',False)==True],
                                             weight="priority",
                                             **kwargs
                                             )
        st = nx.Graph(st) # unfreeze graph
    else:
        raise ValueError("Model '{}' is not supported.".format(model))
    logging.info("Removing 4-degree nodes")
    st = add_node_attributes_from_network(st,graph)
    st = add_edge_attributes_from_network(st,graph)
    
    graph3 = fours_to_threes(st)
    
    # Create digraph starting from heating station
    # Warning: only works with one heating heating
    network = nx.dfs_tree(graph3,
                          source=[x for x,y in graph3.nodes(data=True) if y.get('isHeatingStation',False)==True][0])
    
    # Add attributes to network
    network = add_node_attributes_from_network(network,graph3)
    network = add_edge_length(network,to_round=False)
    
    
    # Simplex model for power flow
    logging.info('Calculating simplex')
    flowCost, flowDict = nx.network_simplex(network,
                                            demand = 'peak',
                                            capacity = 'capacity', #not implemented
                                            weight = 'length_rounded')
    network = flow_to_nx(network, flowDict)
    
    # Get DN
    for edge in network.edges(data=True):
        edge[2]["DN"] = size.getDN(edge[2]["power"],
                                   DT=DT,
                                   maxPressureLosses=maxPressureLosses)
        
    return network

def generate_dhn(analysis_area,
                 grid_size = GRID_SIZE,
                 road_weight = ROAD_WEIGHT,
                 heating_stations=[],
                 limit_egids=[],
                 geometry_file_path = None,
                 add_roads=False,
                 add_footprints=False,
                 model="networkx",
                 grid_type = "fishnet",
                 wait_solution=5,
                 connect_only_to_main=False,
                 DT=30,
                 maxPressureLosses=250,
                 seed=None
                 ):
    """Setup graph and apply models."""
    
    graph = setup_graph(analysis_area,
                     grid_size,
                     road_weight,
                     heating_stations,
                     limit_egids,
                     add_roads,
                     add_footprints,
                     grid_type,
                     connect_only_to_main,
                     geometry_file_path,
                     )

    network = apply_models(graph,
                            model,
                            wait_solution,
                            DT,
                            maxPressureLosses,
                            seed)
    
    
    return network
 