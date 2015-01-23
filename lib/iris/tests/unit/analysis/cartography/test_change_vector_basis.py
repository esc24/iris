# (C) British Crown Copyright 2015, Met Office
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
"""Unit tests for the :func:`iris.analysis.cartography.project` function."""

from __future__ import (absolute_import, division, print_function)

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.analysis.cartography import change_vector_basis
from iris.cube import Cube
from iris.coords import DimCoord, AuxCoord
import iris.coord_systems



def uv_cubes(shape=(5, 6)):
    """Return u, v cubes with a grid in a rotated pole CRS."""
    cs = iris.coord_systems.RotatedGeogCS(grid_north_pole_latitude=37.5,
                                          grid_north_pole_longitude=177.5)
    x = np.linspace(311.9, 391.1, shape[1])
    y = np.linspace(-23.6, 24.8, shape[0])

    x2d, y2d = np.meshgrid(x, y)
    u = 10 * (2 * np.cos(2 * np.deg2rad(x2d) + 3 * np.deg2rad(y2d + 30)) ** 2)
    v = 20 * np.cos(6 * np.deg2rad(x2d))

    lon = DimCoord(x, standard_name='grid_longitude', units='degrees',
                   coord_system=cs)
    lat = DimCoord(y, standard_name='grid_latitude', units='degrees',
                   coord_system=cs)

    u_cube = Cube(u, standard_name='x_wind', units='m/s')
    v_cube = Cube(v, standard_name='y_wind', units='m/s')
    for cube in (u_cube, v_cube):
        cube.add_dim_coord(lat.copy(), 0)
        cube.add_dim_coord(lon.copy(), 1)

    return u_cube, v_cube


class TestPrerequisites(tests.IrisTest):
    def test_different_coord_systems(self):
        u, v = uv_cubes()
        v.coord('grid_latitude').coord_system = iris.coord_systems.GeogCS(1)
        with self.assertRaisesRegexp(ValueError, 'Coordinates differ between '
                'u and v cubes'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())

    def test_different_xy_coord_systems(self):
        u, v = uv_cubes()
        u.coord('grid_latitude').coord_system = iris.coord_systems.GeogCS(1)
        v.coord('grid_latitude').coord_system = iris.coord_systems.GeogCS(1)
        with self.assertRaisesRegexp(ValueError, 'Coordinate systems of x '
                'and y coordinates differ'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())

    def test_different_shape(self):
        u, _ = uv_cubes(shape=(10, 20))
        _, v = uv_cubes(shape=(11, 20))
        with self.assertRaisesRegexp(ValueError, 'same shape'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())

    def test_xy_dimensionality(self):
        u, v = uv_cubes()
        # Replace 1d lat with 2d lat.
        x = u.coord('grid_longitude').points
        y = u.coord('grid_latitude').points
        x2d, y2d = np.meshgrid(x, y)
        lat_2d = AuxCoord(y2d, 'grid_latitude', units='degrees',
                          coord_system=u.coord('grid_latitude').coord_system)
        for cube in (u, v):
            cube.remove_coord('grid_latitude')
            cube.add_aux_coord(lat_2d.copy(), (0, 1))

        with self.assertRaisesRegexp(ValueError, 'x and y coordinates must have '
                'the same number of dimensions'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())

    def test_dim_mapping(self):
        u, v = uv_cubes(shape=(3,3))
        v.transpose()
        with self.assertRaisesRegexp(ValueError, 'Dimension mapping'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())


class TestAnalyticalComparison(tests.IrisTest):
    pass


class TestRotatedToOSGB(tests.IrisTest):
    def test_name(self):
        pass

    def test_new_coords(self):
        pass

    def test_orig_coords(self):
        pass

    def test_magnitude_preservation(self):
        pass

    def test_u_values(self):
        pass

    def test_v_values(self):
        pass

    def test_2d_coords(self):
        pass


if __name__ == "__main__":
    tests.main()
