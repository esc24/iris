# (C) British Crown Copyright 2010 - 2012, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
A collection of routines which create standard Cubes for test purposes.

"""

import os.path
import warnings

import numpy

from iris.cube import Cube
import iris.aux_factory
import iris.coords
import iris.coords as icoords
import iris.tests as tests
from iris.coord_systems import LatLonCS, GeoPosition, SpheroidDatum, PrimeMeridian


def lat_lon_cube():
    """
    Returns a cube with a latitude and longitude suitable for testing saving to PP/NetCDF etc.
    
    """
    cube = Cube(numpy.arange(12, dtype=numpy.int32).reshape((3, 4)))
    cs = LatLonCS(None, 'pm', GeoPosition(90, 0), 0)
    cube.add_dim_coord(iris.coords.DimCoord(points=numpy.array([-1, 0, 1], dtype=numpy.int32), standard_name='latitude', units='degrees', coord_system=cs), 0)
    cube.add_dim_coord(iris.coords.DimCoord(points=numpy.array([-1, 0, 1, 2], dtype=numpy.int32), standard_name='longitude', units='degrees', coord_system=cs), 1)
    return cube


def global_pp():
    """
    Returns a two-dimensional cube derived from PP/aPPglob1/global.pp.

    The standard_name and unit attributes are added to compensate for the
    broken STASH encoding in that file.

    """
    def callback_global_pp(cube, field, filename):
        cube.standard_name = 'air_temperature'
        cube.units = 'K'
    path = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
    cube = iris.load_strict(path, callback=callback_global_pp)
    return cube

def simple_pp():
    filename = tests.get_data_path(['PP', 'simple_pp', 'global.pp'])   # Differs from global_pp()
    cube = iris.load_strict(filename)
    return cube


def simple_1d(with_bounds=True):
    """
    Returns an abstract, one-dimensional cube.

    >>> print simple_1d()
    thingness                           (foo: 11)
         Dimension coordinates:
              foo                           x

    >>> print `simple_1d().data`
    [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]

    """
    cube = Cube(numpy.arange(11, dtype=numpy.int32))
    cube.long_name = 'thingness'
    cube.units = '1'
    points = numpy.arange(11, dtype=numpy.int32) + 1
    bounds = numpy.column_stack([numpy.arange(11, dtype=numpy.int32), numpy.arange(11, dtype=numpy.int32) + 1])
    coord = iris.coords.DimCoord(points, long_name='foo', units='1', bounds=bounds)
    cube.add_dim_coord(coord, 0)
    return cube


def simple_2d(with_bounds=True):
    """
    Returns an abstract, two-dimensional, optionally bounded, cube.

    >>> print simple_2d()
    thingness                           (bar: 3; foo: 4)
         Dimension coordinates:
              bar                           x       -
              foo                           -       x

    >>> print `simple_2d().data`
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]


    """
    cube = Cube(numpy.arange(12, dtype=numpy.int32).reshape((3, 4)))
    cube.long_name = 'thingness'
    cube.units = '1'
    y_points = numpy.array([  2.5,   7.5,  12.5])
    y_bounds = numpy.array([[0, 5], [5, 10], [10, 15]], dtype=numpy.int32)
    y_coord = iris.coords.DimCoord(y_points, long_name='bar', units='1', bounds=y_bounds if with_bounds else None)
    x_points = numpy.array([ -7.5,   7.5,  22.5,  37.5])
    x_bounds = numpy.array([[-15, 0], [0, 15], [15, 30], [30, 45]], dtype=numpy.int32)
    x_coord = iris.coords.DimCoord(x_points, long_name='foo', units='1', bounds=x_bounds if with_bounds else None)

    cube.add_dim_coord(y_coord, 0)
    cube.add_dim_coord(x_coord, 1)
    return cube


def simple_2d_w_multidim_coords(with_bounds=True):
    """
    Returns an abstract, two-dimensional, optionally bounded, cube.

    >>> print simple_2d_w_multidim_coords()
    thingness                           (*ANONYMOUS*: 3; *ANONYMOUS*: 4)
         Auxiliary coordinates:
              bar                                   x               x
              foo                                   x               x

    >>> print `simple_2d().data`
    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11]]

    """
    cube = simple_3d_w_multidim_coords(with_bounds)[0, :, :]
    cube.remove_coord('wibble')
    cube.data = numpy.arange(12, dtype=numpy.int32).reshape((3, 4))
    return cube


def simple_3d_w_multidim_coords(with_bounds=True):
    """
    Returns an abstract, two-dimensional, optionally bounded, cube.

    >>> print simple_3d_w_multidim_coords()
    thingness                           (wibble: 2; *ANONYMOUS*: 3; *ANONYMOUS*: 4)
         Dimension coordinates:
              wibble                           x               -               -
         Auxiliary coordinates:
              bar                              -               x               x
              foo                              -               x               x

    >>> print simple_3d_w_multidim_coords().data
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
     [[12 13 14 15]
      [16 17 18 19]
      [20 21 22 23]]]

    """
    cube = Cube(numpy.arange(24, dtype=numpy.int32).reshape((2, 3, 4)))
    cube.long_name = 'thingness'
    cube.units = '1'
    
    y_points = numpy.array([[  2.5,   7.5,  12.5,  17.5],
                            [ 10. ,  17.5,  27.5,  42.5],
                            [ 15. ,  22.5,  32.5,  50. ]])
    y_bounds = numpy.array([[[0, 5], [5, 10], [10, 15], [15, 20]], 
                            [[5, 15], [15, 20], [20, 35], [35, 50]],
                            [[10, 20], [20, 25], [25, 40], [40, 60]]], dtype=numpy.int32)
    y_coord = iris.coords.AuxCoord(points=y_points, long_name='bar', units='1', bounds=y_bounds if with_bounds else None)
    x_points = numpy.array([[ -7.5,   7.5,  22.5,  37.5],
                            [-12.5,   4. ,  26.5,  47.5],
                            [  2.5,  14. ,  36.5,  44. ]])
    x_bounds = numpy.array([[[-15, 0], [0, 15], [15, 30], [30, 45]], 
                            [[-25, 0], [0, 8], [8, 45], [45, 50]],
                            [[-5, 10], [10, 18],  [18, 55], [18, 70]]], dtype=numpy.int32)
    x_coord = iris.coords.AuxCoord(points=x_points, long_name='foo', units='1', bounds=x_bounds if with_bounds else None)
    wibble_coord = iris.coords.DimCoord(numpy.array([ 10.,  30.], dtype=numpy.float32), long_name='wibble', units='1')

    cube.add_dim_coord(wibble_coord, [0])
    cube.add_aux_coord(y_coord, [1, 2])
    cube.add_aux_coord(x_coord, [1, 2])
    return cube


def track_1d(duplicate_x=False):
    """
    Returns a one-dimensional track through two-dimensional space.

    >>> print track_1d()
    air_temperature                     (y, x: 11)
         Dimensioned coords:
              x -> x
              y -> y
         Single valued coords:

    >>> print `track_1d().data`
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    """
    cube = Cube(numpy.arange(11, dtype=numpy.int32), standard_name='air_temperature', units='K')
    bounds = numpy.column_stack([numpy.arange(11, dtype=numpy.int32), numpy.arange(11, dtype=numpy.int32) + 1])
    pts = bounds[:, 1]
    coord = iris.coords.AuxCoord(pts, long_name='x', units='1', bounds=bounds)
    cube.add_aux_coord(coord, [0])
    if duplicate_x:
        coord = iris.coords.AuxCoord(pts, long_name='x', units='1', bounds=bounds)
        cube.add_aux_coord(coord, [0])
    coord = iris.coords.AuxCoord(pts * 2, long_name='y', units='1', bounds=bounds * 2)
    cube.add_aux_coord(coord, 0)
    return cube


def simple_2d_w_multidim_and_scalars():
    data = numpy.arange(50, dtype=numpy.int32).reshape((5, 10))
    cube = iris.cube.Cube(data, long_name='test 2d dimensional cube', units='meters')

    # DimCoords
    dim1 = iris.coords.DimCoord(numpy.arange(5, dtype=numpy.float32) * 5.1 + 3.0, long_name='dim1', units='meters')
    dim2 = iris.coords.DimCoord(numpy.arange(10, dtype=numpy.int32), long_name='dim2', units='meters',
                                bounds=numpy.arange(20, dtype=numpy.int32).reshape(10, 2))

    # Scalars
    an_other = iris.coords.AuxCoord(3.0, long_name='an_other', units='meters')
    yet_an_other = iris.coords.DimCoord(23.3, standard_name='air_temperature', long_name='custom long name', units='K')
    
    # Multidim
    my_multi_dim_coord = iris.coords.AuxCoord(numpy.arange(50, dtype=numpy.int32).reshape(5, 10), 
                                              long_name='my_multi_dim_coord', units='1', 
                                              bounds=numpy.arange(200, dtype=numpy.int32).reshape(5, 10, 4))

    cube.add_dim_coord(dim1, 0)
    cube.add_dim_coord(dim2, 1)
    cube.add_aux_coord(an_other)
    cube.add_aux_coord(yet_an_other)
    cube.add_aux_coord(my_multi_dim_coord, [0, 1])

    return cube


def hybrid_height():
    """
    Returns a two-dimensional (Z, X), hybrid-height cube.

    >>> print hybrid_height()
    TODO: Update!
    air_temperature                     (level_height: 3; *ANONYMOUS*: 4)
         Dimension coordinates:
              level_height                           x               -
         Auxiliary coordinates:
              model_level_number                     x               -
              sigma                                  x               -
              surface_altitude                       -               x
         Derived coordinates:
              altitude                               x               x

    >>> print hybrid_height().data
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
    
    """
    data = numpy.arange(12).reshape((3, 4))

    orography = icoords.AuxCoord([10, 25, 50, 5], standard_name='surface_altitude', units='m')
    model_level = icoords.AuxCoord([2, 1, 0], standard_name='model_level_number')
    level_height = icoords.DimCoord([100, 50, 10], long_name='level_height',
                                    units='m', attributes={'positive': 'up'},
                                    bounds=[[150, 75], [75, 20], [20, 0]])
    sigma = icoords.AuxCoord([0.8, 0.9, 0.95], long_name='sigma',
                             bounds=[[0.7, 0.85], [0.85, 0.97], [0.97, 1.0]])
    hybrid_height = iris.aux_factory.HybridHeightFactory(level_height, sigma, orography)

    cube = iris.cube.Cube(data, standard_name='air_temperature', units='K',
                          dim_coords_and_dims=[(level_height, 0)],
                          aux_coords_and_dims=[(orography, 1), (model_level, 0), (sigma, 0)],
                          aux_factories=[hybrid_height])
    return cube


def realistic_4d():
    """
    Returns a realistic 4d cube.
    
    >>> print repr(realistic_4d())
    <iris 'Cube' of air_potential_temperature (time: 6; model_level_number: 70; grid_latitude: 100; grid_longitude: 100)>

    """
    # the stock arrays were created in Iris 0.8 with:
#    >>> fname = iris.sample_data_path('PP', 'COLPEX', 'theta_and_orog_subset.pp')
#    >>> theta = iris.load_strict(fname, 'air_potential_temperature')
#    >>> for coord in theta.coords():
#    ...  print coord.name, coord.has_points(), coord.has_bounds(), coord.units
#    ... 
#    grid_latitude True True degrees
#    grid_longitude True True degrees
#    level_height True True m
#    model_level True False 1
#    sigma True True 1
#    time True False hours since 1970-01-01 00:00:00
#    source True False no_unit
#    forecast_period True False hours
#    >>> arrays = []    
#    >>> for coord in theta.coords():
#    ...  if coord.has_points(): arrays.append(coord.points)
#    ...  if coord.has_bounds(): arrays.append(coord.bounds)
#    >>> arrays.append(theta.data)
#    >>> arrays.append(theta.coord('sigma').coord_system.orography.data)
#    >>> numpy.savez('stock_arrays.npz', *arrays)
    
    data_path = os.path.join(os.path.dirname(__file__), 'stock_arrays.npz')
    r = numpy.load(data_path)
    # sort the arrays based on the order they were originally given. The names given are of the form 'arr_1' or 'arr_10'
    _, arrays =  list(zip(*sorted(iter(r.items()), key=lambda item: int(item[0][4:]))))
    
    lat_pts, lat_bnds, lon_pts, lon_bnds, level_height_pts, \
    level_height_bnds, model_level_pts, sigma_pts, sigma_bnds, time_pts, \
    _source_pts, forecast_period_pts, data, orography = arrays
    
    
    ll_cs = LatLonCS(SpheroidDatum(label='spherical', semi_major_axis=6371229.0, semi_minor_axis=6371229.0, 
                                   flattening=0.0, units='m'), PrimeMeridian(label='Greenwich', value=0.0), 
                     GeoPosition(latitude=37.5, longitude=177.5), 
                     0.0)
    
    lat = icoords.DimCoord(lat_pts, standard_name='grid_latitude', units='degrees', 
                           bounds=lat_bnds, coord_system=ll_cs)
    lon = icoords.DimCoord(lon_pts, standard_name='grid_longitude', units='degrees',
                           bounds=lon_bnds, coord_system=ll_cs)
    level_height = icoords.DimCoord(level_height_pts, long_name='level_height', 
                                    units='m', bounds=level_height_bnds, 
                                    attributes={'positive': 'up'})
    model_level = icoords.DimCoord(model_level_pts, standard_name='model_level_number', 
                                   units='1', attributes={'positive': 'up'})
    sigma = icoords.AuxCoord(sigma_pts, long_name='sigma', units='1', bounds=sigma_bnds)
    orography = icoords.AuxCoord(orography, standard_name='surface_altitude', units='m')
    time = icoords.DimCoord(time_pts, standard_name='time', units='hours since 1970-01-01 00:00:00')
    forecast_period = icoords.DimCoord(forecast_period_pts, standard_name='forecast_period', units='hours')
    
    hybrid_height = iris.aux_factory.HybridHeightFactory(level_height, sigma, orography)

    cube = iris.cube.Cube(data, standard_name='air_potential_temperature', units='K',
                          dim_coords_and_dims=[(time, 0), (model_level, 1), (lat, 2), (lon, 3)],
                          aux_coords_and_dims=[(orography, (2, 3)), (level_height, 1), (sigma, 1),
                                               (forecast_period, None)],
                          attributes={'source': 'Iris test case'},
                          aux_factories=[hybrid_height])
    return cube


def realistic_4d_no_derived():
    """
    Returns a realistic 4d cube without hybrid height
    
    >>> print repr(realistic_4d())
    <iris 'Cube' of air_potential_temperature (time: 6; model_level_number: 70; grid_latitude: 100; grid_longitude: 100)>

    """
    cube = realistic_4d()
    
    # TODO determine appropriate way to remove aux_factory from a cube
    cube._aux_factories = []

    return cube

def realistic_4d_w_missing_data():
    data_path = os.path.join(os.path.dirname(__file__), 'stock_mdi_arrays.npz')
    data_archive = numpy.load(data_path)
    data = numpy.ma.masked_array(data_archive['arr_0'], mask=data_archive['arr_1'])

    # sort the arrays based on the order they were originally given. The names given are of the form 'arr_1' or 'arr_10'
    
    ll_cs = LatLonCS(SpheroidDatum(label='spherical', semi_major_axis=6371229.0, semi_minor_axis=6371229.0, 
                                   flattening=0.0, units='m'), PrimeMeridian(label='Greenwich', value=0.0), 
                     GeoPosition(latitude=0.0, longitude=90.0),
                     0.0)
    
    lat = iris.coords.DimCoord(numpy.arange(20, dtype=numpy.float32), standard_name='grid_latitude', 
                               units='degrees', coord_system=ll_cs)
    lon = iris.coords.DimCoord(numpy.arange(20, dtype=numpy.float32), standard_name='grid_longitude',
                               units='degrees', coord_system=ll_cs)
    time = iris.coords.DimCoord([1000., 1003., 1006.], standard_name='time', 
                                units='hours since 1970-01-01 00:00:00')
    forecast_period = iris.coords.DimCoord([0.0, 3.0, 6.0], standard_name='forecast_period', units='hours')
    pressure = iris.coords.DimCoord(numpy.array([  800.,   900.,  1000.], dtype=numpy.float32), 
                                    long_name='pressure', units='hPa')

    cube = iris.cube.Cube(data, long_name='missing data test data', units='K',
                          dim_coords_and_dims=[(time, 0), (pressure, 1), (lat, 2), (lon, 3)],
                          aux_coords_and_dims=[(forecast_period, 0)],
                          attributes={'source':'Iris test case'})
    return cube


def global_grib2():
    path = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
    cube = iris.load_strict(path)
    return cube
