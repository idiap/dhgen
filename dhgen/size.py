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
Size the DHN piping based on reference data.

@author: gperonato
"""


def getDN(power, #peak power in kW
          DT=30, # K
          maxPressureLosses=250 # Pa/m
          ):
    # Values provided by RWB
    if DT== 30 and maxPressureLosses==250:
        if  power  <= 29:
            return [20,125]
        if  power  <= 53:
            return [25,125]
        if  power <= 121:
            return [32,140]
        if  power  <= 179:
            return [40,140]
        if  power <= 333:
            return [50,160]
        if  power  <= 652:
            return [65,180]
        if  power  <= 994:
            return [80,200]
        if  power  <= 1975:
            return [100,250]
        if  power  <= 3398:
            return [125,280]
        if  power  <= 6600:
            return [150,315]
        if  power  <= 11146:
            return [200,400]
        if  power  <= 20224:
            return [250,500]
        if  power  <= 31796:
            return [300,560]
        if  power  > 31796:
            return [350,630]
    else:
        raise ValueError("DT and maxPressureLosses values not supported at the moment.")