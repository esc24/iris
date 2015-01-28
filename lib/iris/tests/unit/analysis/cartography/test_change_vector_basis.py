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

import cartopy.crs as ccrs
from iris.analysis.cartography import change_vector_basis, unrotate_pole
from iris.cube import Cube
from iris.coords import DimCoord, AuxCoord
import iris.coord_systems


def uv_cubes(shape=(5, 6), x=None, y=None):
    """Return u, v cubes with a grid in a rotated pole CRS."""
    cs = iris.coord_systems.RotatedGeogCS(grid_north_pole_latitude=37.5,
                                          grid_north_pole_longitude=177.5)
    if x is None:
        x = np.linspace(311.9, 391.1, shape[1])
    if y is None:
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


def uv_cubes_3d(ref_cube, n_realization=3):
    """
    Return 3d u, v cubes with a grid in a rotated pole CRS.

    Based on the given 2d cube, with added leading dim * 'n_realization'.

    """
    cs = ref_cube.coord(axis='x').coord_system
    x = ref_cube.coord(axis='x').points
    y = ref_cube.coord(axis='y').points
    x2d, y2d = np.meshgrid(x, y)
    u = 10 * (2 * np.cos(2 * np.deg2rad(x2d) + 3 * np.deg2rad(y2d + 30)) ** 2)
    v = 20 * np.cos(6 * np.deg2rad(x2d))
    # Multiply slices by factor to give variation over 0th dim.
    factor = np.arange(1, n_realization + 1).reshape(n_realization, 1, 1)
    u = factor * u
    v = factor * v
    lon = DimCoord(x, standard_name='grid_longitude', units='degrees',
                   coord_system=cs)
    lat = DimCoord(y, standard_name='grid_latitude', units='degrees',
                   coord_system=cs)
    realization = DimCoord(np.arange(n_realization), 'realization')
    u_cube = Cube(u, standard_name='x_wind', units='m/s')
    v_cube = Cube(v, standard_name='y_wind', units='m/s')
    for cube in (u_cube, v_cube):
        cube.add_dim_coord(realization.copy(), 0)
        cube.add_dim_coord(lat.copy(), 1)
        cube.add_dim_coord(lon.copy(), 2)
    return u_cube, v_cube


class TestPrerequisites(tests.IrisTest):
    def test_different_coord_systems(self):
        u, v = uv_cubes()
        v.coord('grid_latitude').coord_system = iris.coord_systems.GeogCS(1)
        with self.assertRaisesRegexp(
                ValueError, 'Coordinates differ between u and v cubes'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())

    def test_different_xy_coord_systems(self):
        u, v = uv_cubes()
        u.coord('grid_latitude').coord_system = iris.coord_systems.GeogCS(1)
        v.coord('grid_latitude').coord_system = iris.coord_systems.GeogCS(1)
        with self.assertRaisesRegexp(
                ValueError,
                'Coordinate systems of x and y coordinates differ'):
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

        with self.assertRaisesRegexp(
                ValueError,
                'x and y coordinates must have the same number of dimensions'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())

    def test_dim_mapping(self):
        u, v = uv_cubes(shape=(3, 3))
        v.transpose()
        with self.assertRaisesRegexp(ValueError, 'Dimension mapping'):
            change_vector_basis(u, v, iris.coord_systems.OSGB())


class TestAnalyticComparison(tests.IrisTest):
    def test_rotated_to_true(self):
        u_rot, v_rot = uv_cubes()
        test_cs = iris.coord_systems.GeogCS(1.0)
        u_true, v_true = change_vector_basis(u_rot, v_rot, test_cs)

        # Calculate "exact" results to compare.
        # (equations for rotated-pole transformation : cf. UMDP015)
        cs_rot = u_rot.coord(axis='x').coord_system
        pole_lat = cs_rot.grid_north_pole_latitude
        pole_lon = cs_rot.grid_north_pole_longitude

        # Work out the rotation angles.
        lambda_angle = np.radians(pole_lon - 180.0)
        phi_angle = np.radians(90.0 - pole_lat)

        # Get the locations in true lats+lons.
        rotated_lons = u_rot.coord(axis='x').points
        rotated_lats = u_rot.coord(axis='y').points
        rotated_lons_2d, rotated_lats_2d = np.meshgrid(
            rotated_lons, rotated_lats)
        trueLongitude, trueLatitude = unrotate_pole(rotated_lons_2d,
                                                    rotated_lats_2d,
                                                    pole_lon,
                                                    pole_lat)

        # Calculate inter-coordinate transform coefficients.
        cos_rot = (np.cos(np.radians(rotated_lons_2d)) *
                   np.cos(np.radians(trueLongitude) - lambda_angle) +
                   np.sin(np.radians(rotated_lons_2d)) *
                   np.sin(np.radians(trueLongitude) - lambda_angle) *
                   np.cos(phi_angle))
        sin_rot = -((np.sin(np.radians(trueLongitude) - lambda_angle) *
                     np.sin(phi_angle))
                    / np.cos(np.radians(rotated_lats_2d)))

        # Matrix-multiply to rotate the vectors.
        u_ref = u_rot.data * cos_rot - v_rot.data * sin_rot
        v_ref = v_rot.data * cos_rot + u_rot.data * sin_rot

        # Check that all the numerical results are fairly close to these.
        self.assertArrayAllClose(u_true.data, u_ref, rtol=1e-5, atol=0.0005)
        self.assertArrayAllClose(v_true.data, v_ref, rtol=1e-5, atol=0.0005)


class TestRotatedToOSGB(tests.IrisTest):
    # Define some coordinate ranges for the uv_cubes 'standard' RotatedPole
    # system, that exceed the OSBG margins, but not by "too much".
    _rp_x_min, _rp_x_max = -5.0, 5.0
    _rp_y_min, _rp_y_max = -5.0, 15.0

    def _uv_cubes_OSGB(self, shape=(5, 6)):
        # Make test cubes suitable for transforming to OSGB, as the standard
        # 'uv_cubes' result goes too far outside, leading to errors.
        ny, nx = shape
        x = np.linspace(self._rp_x_min, self._rp_x_max, nx)
        y = np.linspace(self._rp_y_min, self._rp_y_max, ny)
        return uv_cubes(x=x, y=y, shape=shape)

    def test_name(self):
        u, v = self._uv_cubes_OSGB()
        u.rename('bob')
        v.rename('alice')
        ut, vt = change_vector_basis(u, v, iris.coord_systems.OSGB())
        self.assertEqual(ut.name(), 'transformed_' + u.name())
        self.assertEqual(vt.name(), 'transformed_' + v.name())

    def test_new_coords(self):
        nx = 6
        ny = 5
        u, v = self._uv_cubes_OSGB((ny, nx))
        ut, vt = change_vector_basis(u, v, iris.coord_systems.OSGB())
        # x, y values taken from uv_cubes().
        x = np.linspace(self._rp_x_min, self._rp_x_max, nx)
        y = np.linspace(self._rp_y_min, self._rp_y_max, ny)
        x2d, y2d = np.meshgrid(x, y)
        src_crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
        tgt_crs = ccrs.OSGB()
        xyz_tran = tgt_crs.transform_points(src_crs, x2d, y2d)

        points = xyz_tran[..., 0].reshape(x2d.shape)
        expected_x = AuxCoord(points,
                              standard_name='projection_x_coordinate',
                              units='m',
                              coord_system=iris.coord_systems.OSGB())
        self.assertEqual(ut.coord('projection_x_coordinate'), expected_x)
        self.assertEqual(vt.coord('projection_x_coordinate'), expected_x)

        points = xyz_tran[..., 1].reshape(y2d.shape)
        expected_y = AuxCoord(points,
                              standard_name='projection_y_coordinate',
                              units='m',
                              coord_system=iris.coord_systems.OSGB())
        self.assertEqual(ut.coord('projection_y_coordinate'), expected_y)
        self.assertEqual(vt.coord('projection_y_coordinate'), expected_y)

    def test_orig_coords(self):
        u, v = self._uv_cubes_OSGB()
        ut, vt = change_vector_basis(u, v, iris.coord_systems.OSGB())
        self.assertEqual(u.coord('grid_latitude'), ut.coord('grid_latitude'))
        self.assertEqual(v.coord('grid_latitude'), vt.coord('grid_latitude'))
        self.assertEqual(u.coord('grid_longitude'), ut.coord('grid_longitude'))
        self.assertEqual(v.coord('grid_longitude'), vt.coord('grid_longitude'))

    def test_magnitude_preservation(self):
        u, v = self._uv_cubes_OSGB()
        ut, vt = change_vector_basis(u, v, iris.coord_systems.OSGB())
        orig_sq_mag = u.data**2 + v.data**2
        res_sq_mag = ut.data**2 + vt.data**2
        self.assertArrayAllClose(orig_sq_mag, res_sq_mag, rtol=5e-4)

    def test_data_values(self):
        u, v = self._uv_cubes_OSGB()
        # Slice out 4 points that lie in and outside OSGB extent.
        u = u[1:3, 3:5]
        v = v[1:3, 3:5]
        ut, vt = change_vector_basis(u, v, iris.coord_systems.OSGB())
        # Values precalculated and checked.
        expected_ut_data = np.array([[0.16285514,  0.35323639],
                                     [1.82650698,  2.62455840]])
        expected_vt_data = np.array([[19.88979966,  19.01921346],
                                     [19.88018847,  19.01424281]])
        # Compare u and v data values against previously calculated values.
        self.assertArrayAllClose(ut.data, expected_ut_data, rtol=1e-5)
        self.assertArrayAllClose(vt.data, expected_vt_data, rtol=1e-5)

    def test_nd_data(self):
        u2d, y2d = self._uv_cubes_OSGB()
        u, v = uv_cubes_3d(u2d)
        u = u[:, 1:3, 3:5]
        v = v[:, 1:3, 3:5]
        ut, vt = change_vector_basis(u, v, iris.coord_systems.OSGB())
        # Values precalculated and checked (as test_data_values above),
        # then scaled by factor [1, 2, 3] along 0th dim (see uv_cubes_3d()).
        expected_ut_data = np.array([[0.16285514,  0.35323639],
                                     [1.82650698,  2.62455840]])
        expected_vt_data = np.array([[19.88979966,  19.01921346],
                                     [19.88018847,  19.01424281]])
        factor = np.array([1, 2, 3]).reshape(3, 1, 1)
        expected_ut_data = factor * expected_ut_data
        expected_vt_data = factor * expected_vt_data
        # Compare u and v data values against previously calculated values.
        self.assertArrayAlmostEqual(ut.data, expected_ut_data)
        self.assertArrayAlmostEqual(vt.data, expected_vt_data)


class TestRoundTrip(tests.IrisTest):
    def test_rotated_to_unrotated(self):
        # Check ability to use 2d coords as input.
        u, v = uv_cubes()
        ut, vt = change_vector_basis(u, v, iris.coord_systems.GeogCS(6371229))
        # Remove  grid lat and lon, leaving 2d projection coords.
        ut.remove_coord('grid_latitude')
        vt.remove_coord('grid_latitude')
        ut.remove_coord('grid_longitude')
        vt.remove_coord('grid_longitude')
        # Change back.
        orig_cs = u.coord('grid_latitude').coord_system
        res_u, res_v = change_vector_basis(ut, vt, orig_cs)
        # Check data values - limited accuracy due to numerical approx.
        self.assertArrayAlmostEqual(res_u.data, u.data, decimal=3)
        self.assertArrayAlmostEqual(res_v.data, v.data, decimal=3)
        # Check coords locations.
        x2d, y2d = np.meshgrid(u.coord('grid_longitude').points,
                               u.coord('grid_latitude').points)
        # Shift longitude from 0 to 360 -> -180 to 180.
        x2d = np.where(x2d > 180, x2d - 360, x2d)
        res_x = res_u.coord('projection_x_coordinate',
                            coord_system=orig_cs).points
        res_y = res_u.coord('projection_y_coordinate',
                            coord_system=orig_cs).points
        self.assertArrayAlmostEqual(res_x, x2d)
        self.assertArrayAlmostEqual(res_y, y2d)
        res_x = res_v.coord('projection_x_coordinate',
                            coord_system=orig_cs).points
        res_y = res_v.coord('projection_y_coordinate',
                            coord_system=orig_cs).points
        self.assertArrayAlmostEqual(res_x, x2d)
        self.assertArrayAlmostEqual(res_y, y2d)


if __name__ == "__main__":
    tests.main()
