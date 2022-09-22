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
Test automated DHN generation using a custom GPKG file.

@author: gperonato
"""

from dhgen import *
import os
BOUNDS = [788808,
          5798949,
          789112,
          5799229]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = 42 # for test reproducibility

dhn = generate_dhn(BOUNDS,
                    add_footprints=False,
                    add_roads=True,
                    road_weight = 0.1,
                    model="husek",
                    wait_solution=1,
                    geometry_file_path=os.path.join(BASE_DIR,"sample_data","sample_input.gpkg"),
                    seed=SEED)

stations, pipes = nx_to_gdf(dhn)
pipes.plot(column="power",legend=True, legend_kwds={'label': "kW"})

