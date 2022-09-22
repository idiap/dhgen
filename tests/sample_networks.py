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
Run unittests on sample networkx.

@author: gperonato
"""

import os
import sys
import unittest
import shutil
from dhgen import *

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
GEOADMIN_CACHE_FILE = "GeoAdmin_cache.gpkg"

class SimpleNetworkTestCase(unittest.TestCase):

    def test_generate_dhn(self):
        """Test "generate_dhn.py"."""
        
        if os.path.exists(GEOADMIN_CACHE_FILE):
            os.remove(GEOADMIN_CACHE_FILE)
            
        path = os.path.join(BASE_DIR, "examples")
        sys.path.append(path)
        import generate_dhn
        
        stations, pipes = nx_to_gdf(generate_dhn.dhn)
        
        self.assertEqual(len(pipes),110)
        self.assertAlmostEqual(pipes.length.sum(), 1162, delta=0.5)
        self.assertEqual(abs(stations["peak"].min()),545)
        
    def test_generate_dhn_file(self):
        """Test "generate_dhn_file.py"."""
    
        path = os.path.join(BASE_DIR, "examples")
        sys.path.append(path)
        import generate_dhn_file
    
        stations, pipes = nx_to_gdf(generate_dhn_file.dhn)
    
        self.assertEqual(len(pipes),136)
        self.assertAlmostEqual(pipes.length.sum(),961,delta=0.5)
        self.assertEqual(abs(stations["peak"].min()),390)
        
    def test_broc_eurosun(self):
        """Test EuroSun example with roads"""
                
        path = os.path.join(BASE_DIR, "examples", "EuroSun")
        sys.path.append(path)
        import broc_comparison
        # Copy the cache file to be used
        shutil.copy(os.path.join(path,GEOADMIN_CACHE_FILE),GEOADMIN_CACHE_FILE)
        dhn = generate_dhn(broc_comparison.BOUNDS,
                            grid_size=broc_comparison.GRID_SIZE,
                            heating_stations=broc_comparison.HEATING_STATIONS,
                            limit_egids = broc_comparison.LIMIT_EGIDS,
                            add_roads=True,
                            add_footprints=True,
                            model="husek",
                            road_weight=broc_comparison.ROAD_WEIGHT,
                            seed=broc_comparison.SEED
                            ) 
    
        stations, pipes = nx_to_gdf(dhn)
    
        self.assertEqual(len(pipes),553)
        self.assertAlmostEqual(pipes.length.sum(),4371,delta=0.5)
        self.assertEqual(abs(stations["peak"].min()),2628)
        

unittest.main()