# EuroSun

This directory contains the code to run the experiments shown in the paper
presented at the EuroSun 2022 conference.

Geodata retrieved on September 1st, 2022 from the GeoAdmin API are
available in the repository to enforce the reproducibility of the results.
By default, the `GeoAdmin_cache.gpkg` file is used.

The DHgeN parameter `limit_egids` is set so as to connect only buildings
whose energy source for space heating or domestic hot water (DHW)
is listed as *district heating* in the Swiss Federal Register of Buildings
and Dwellings (RegBL).

Data from the actual network presented in the paper are not available here, 
but the script shows how the comparison is done.

