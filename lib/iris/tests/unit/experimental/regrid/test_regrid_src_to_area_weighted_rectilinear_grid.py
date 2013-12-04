# (C) British Crown Copyright 2013, Met Office
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
Test function
:func:`iris.experimental.regrid_src_to_area_weighted_rectilinear_grid`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import copy

import numpy as np
import numpy.ma as ma

import iris
import iris.coords
import iris.cube
from iris.experimental.regrid \
    import regrid_src_to_area_weighted_rectilinear_grid as regrid


class Test_regrid_src_to_area_weighted_rectilinear_grid(tests.IrisTest):
    def setUp(self):
        # Source cube.
        points = np.array([[10, 10, 10, 10],
                           [30, 30, 30, 30],
                           [30, 30, 30, 30]])
        src_y = iris.coords.AuxCoord(points,
                                     standard_name='latitude',
                                     units='degrees')

        data = ma.arange(1, 13, dtype=np.float).reshape(3, 4)
        attributes = dict(wibble='wobble')
        bibble = iris.coords.DimCoord([1], long_name='bibble')
        self.src = iris.cube.Cube(data,
                                  standard_name='air_temperature',
                                  units='K',
                                  aux_coords_and_dims=[(src_y, (0, 1)),
                                                       (bibble, None)],
                                  attributes=attributes)

        # Source cube x coordinates.
        points = np.array([[10, 20, 30, 200],
                           [110, 120, 180, 190],
                           [200, 205, 210, 220]])
        self.src_x_positive = iris.coords.AuxCoord(points,
                                                   standard_name='longitude',
                                                   units='degrees')
        points = np.array([[-180, -170, -160, -150],
                           [-180, -175, -170, -165],
                           [-160, -155, -150, -145]])
        self.src_x_negative = iris.coords.AuxCoord(points,
                                                   standard_name='longitude',
                                                   units='degrees')

        # Area weights cube.
        self.area = iris.cube.Cube(np.asarray(data) * 10)

        # Target grid cube.
        grid_y = iris.coords.DimCoord([10, 30],
                                      standard_name='latitude',
                                      units='degrees',
                                      bounds=[[0, 20], [20, 40]])
        self.grid = iris.cube.Cube(np.zeros((2, 2)),
                                   dim_coords_and_dims=[(grid_y, 0)])

        # Target grid cube x coordinates.
        self.grid_x_positive = iris.coords.DimCoord([190, 210],
                                                    standard_name='longitude',
                                                    units='degrees',
                                                    bounds=[[180, 200],
                                                            [200, 220]])

    def _weighted_mean(self, points):
        points = np.asarray(points, dtype=np.float)
        weights = points * 10
        numerator = denominator = 0
        for point, weight in zip(points, weights):
            numerator += point * weight
            denominator += weight
        return numerator / denominator

    def _expected_cube(self, data):
        cube = iris.cube.Cube(data)
        cube.metadata = copy.deepcopy(self.src)
        grid_x = self.grid.coord('longitude')
        grid_y = self.grid.coord('latitude')
        cube.add_dim_coord(grid_x.copy(), self.grid.coord_dims(grid_x))
        cube.add_dim_coord(grid_y.copy(), self.grid.coord_dims(grid_y))
        src_x = self.src.coord('longitude')
        src_y = self.src.coord('latitude')
        for coord in self.src.aux_coords:
            if coord is not src_x and coord is not src_y:
                if not self.src.coord_dims(coord):
                    cube.add_aux_coord(coord)
        return cube

    def test_aligned_src_x_positive(self):
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.grid.add_dim_coord(self.grid_x_positive, 1)
        result = regrid(self.src, self.area, self.grid)
        data = np.array([0,
                         self._weighted_mean([4]),
                         self._weighted_mean([7, 8]),
                         self._weighted_mean([9, 10, 11])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)

    def test_aligned_src_x_positive_mask(self):
        self.src.add_aux_coord(self.src_x_positive, (0, 1))
        self.src.data[([1, 2, 2], [3, 0, 2])] = ma.masked
        self.grid.add_dim_coord(self.grid_x_positive, 1)
        result = regrid(self.src, self.area, self.grid)
        data = np.array([0,
                         self._weighted_mean([4]),
                         self._weighted_mean([7]),
                         self._weighted_mean([10])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)

    def test_misaligned_src_x_negative(self):
        self.src.add_aux_coord(self.src_x_negative, (0, 1))
        self.grid.add_dim_coord(self.grid_x_positive, 1)
        result = regrid(self.src, self.area, self.grid)
        data = np.array([self._weighted_mean([1, 2]),
                         self._weighted_mean([3, 4]),
                         self._weighted_mean([5, 6, 7, 8]),
                         self._weighted_mean([9, 10, 11, 12])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)

    def test_misaligned_src_x_negative_mask(self):
        self.src.add_aux_coord(self.src_x_negative, (0, 1))
        self.src.data[([0, 0, 1, 1, 2, 2],
                       [1, 3, 1, 3, 1, 3])] = ma.masked
        self.grid.add_dim_coord(self.grid_x_positive, 1)
        result = regrid(self.src, self.area, self.grid)
        data = np.array([self._weighted_mean([1]),
                         self._weighted_mean([3]),
                         self._weighted_mean([5, 7]),
                         self._weighted_mean([9, 11])]).reshape(2, 2)
        expected = self._expected_cube(data)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    tests.main()
