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

# You should have r["eceived a copy of the GNU General Public License
# along with DHgeN. If not, see <http://www.gnu.org/licenses/>

"""
Test automated DHN generation using Swiss open data.

@author: gperonato
"""

from dhgen import *

BOUNDS = [2573625,1161442,2573825,1161642] # 200x200-m area in Broc (FR) - CH
HEATING_STATIONS = [(2573626,1161664)]

dhn = generate_dhn(BOUNDS,
                    heating_stations=HEATING_STATIONS,
                    add_footprints=False,
                    add_roads=False,
                    model="networkx",
                    )

stations, pipes = nx_to_gdf(dhn)
pipes.plot(column="power",legend=True, legend_kwds={'label': "kW"})