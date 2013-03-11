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


# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

from functools import wraps
import types
import warnings

import matplotlib.pyplot as plt
import numpy as np

import iris
import iris.coords as coords
import iris.plot as iplt
import iris.quickplot as qplt
import iris.symbols
import iris.tests.stock
import iris.tests.test_mapping as test_mapping


def simple_cube():
    cube = iris.tests.stock.realistic_4d()
    cube = cube[:, 0, 0, :]
    cube.coord('time').guess_bounds()
    return cube


class TestSimple(tests.GraphicsTest):
    def test_points(self):
        cube = simple_cube()
        qplt.contourf(cube)
        self.check_graphic()

    def test_bounds(self):
        cube = simple_cube()
        qplt.pcolor(cube)
        self.check_graphic()


class TestMissingCoord(tests.GraphicsTest):
    def _check(self, cube):
        qplt.contourf(cube)
        self.check_graphic()

        qplt.pcolor(cube)
        self.check_graphic()

    def test_no_u(self):
        cube = simple_cube()
        cube.remove_coord('grid_longitude')
        self._check(cube)

    def test_no_v(self):
        cube = simple_cube()
        cube.remove_coord('time')
        self._check(cube)

    def test_none(self):
        cube = simple_cube()
        cube.remove_coord('grid_longitude')
        cube.remove_coord('time')
        self._check(cube)


@iris.tests.skip_data
class TestMissingCS(tests.GraphicsTest):
    @iris.tests.skip_data
    def test_missing_cs(self):
        cube = tests.stock.simple_pp()
        cube.coord("latitude").coord_system = None
        cube.coord("longitude").coord_system = None
        qplt.contourf(cube)
        qplt.plt.gca().coastlines()
        self.check_graphic()


class TestHybridHeight(tests.GraphicsTest):
    def setUp(self):
        self.cube = iris.tests.stock.realistic_4d()[0, :15, 0, :]

    def _check(self, plt_method, test_altitude=True):
        plt_method(self.cube)
        self.check_graphic()

        plt_method(self.cube, coords=['level_height', 'grid_longitude'])
        self.check_graphic()

        plt_method(self.cube, coords=['grid_longitude', 'level_height'])
        self.check_graphic()

        if test_altitude:
            plt_method(self.cube, coords=['grid_longitude', 'altitude'])
            self.check_graphic()

            plt_method(self.cube, coords=['altitude', 'grid_longitude'])
            self.check_graphic()

    def test_points(self):
        self._check(qplt.contourf)

    def test_bounds(self):
        self._check(qplt.pcolor, test_altitude=False)

    def test_orography(self):
        qplt.contourf(self.cube)
        iplt.orography_at_points(self.cube)
        iplt.points(self.cube)
        self.check_graphic()

        coords = ['altitude', 'grid_longitude']
        qplt.contourf(self.cube, coords=coords)
        iplt.orography_at_points(self.cube, coords=coords)
        iplt.points(self.cube, coords=coords)
        self.check_graphic()

        # TODO: Test bounds once they are supported.
        with self.assertRaises(NotImplementedError):
            qplt.pcolor(self.cube)
            iplt.orography_at_bounds(self.cube)
            iplt.outline(self.cube)
            self.check_graphic()


# Caches test cubes so subsequent calls are faster.
def cache(fn, cache={}):
    def inner(*args, **kwargs):
        key = fn.__name__
        if key not in cache:
            cache[key] = fn(*args, **kwargs)
        return cache[key]
    return inner


@cache
def _load_4d_testcube():
    # Load example 4d data (TZYX).
    test_cube = iris.tests.stock.realistic_4d()
    # Replace forecast_period coord with a multi-valued version.
    time_coord = test_cube.coord('time')
    n_times = len(time_coord.points)
    time_dims = test_cube.coord_dims(time_coord)
    # Make up values to roughly match older testdata.
    points = np.linspace((1 + 1.0 / 6), 2.0, n_times)
    # NOTE: this must be a DimCoord
    #  - an equivalent AuxCoord produces different plots.
    new_forecast_coord = iris.coords.DimCoord(
        points=points,
        standard_name='forecast_period',
        units=iris.unit.Unit('hours')
    )
    new_forecast_coord.guess_bounds(0.0)
    test_cube.remove_coord('forecast_period')
    test_cube.add_aux_coord(new_forecast_coord, time_dims)
    # Heavily reduce dimensions for faster testing.
    test_cube = test_cube[:, :10, :15, :20]
    return test_cube


@cache
def _load_4d_testcube_no_bounds():
    test_cube = _load_4d_testcube().copy()

    # Remove bounds from all coords that have them.
    test_cube.coord('grid_latitude').bounds = None
    test_cube.coord('grid_longitude').bounds = None
    test_cube.coord('forecast_period').bounds = None
    test_cube.coord('level_height').bounds = None
    test_cube.coord('sigma').bounds = None

    return test_cube


def _time_series(src_cube):
    # Until we have plotting support for multiple axes on the same dimension,
    # remove the time coordinate and its axis.
    cube = src_cube.copy()
    cube.remove_coord('time')
    return cube


def _date_series(src_cube):
    # Until we have plotting support for multiple axes on the same dimension,
    # remove the forecast_period coordinate and its axis.
    cube = src_cube.copy()
    cube.remove_coord('forecast_period')
    return cube


class SliceMixin(object):
    """Mixin class providing tests for each 2-dimensional permutation of axes.

    Requires self.draw_method to be the relevant plotting function,
    and self.results to be a dictionary containing the desired test results."""

    def test_yx(self):
        cube = self.wind[0, 0, :, :]
        self.draw_method(cube)
        self.check_graphic()

    def test_zx(self):
        cube = self.wind[0, :, 0, :]
        self.draw_method(cube)
        self.check_graphic()

    def test_tx(self):
        cube = _time_series(self.wind[:, 0, 0, :])
        self.draw_method(cube)
        self.check_graphic()

    def test_zy(self):
        cube = self.wind[0, :, :, 0]
        self.draw_method(cube)
        self.check_graphic()

    def test_ty(self):
        cube = _time_series(self.wind[:, 0, :, 0])
        self.draw_method(cube)
        self.check_graphic()

    def test_tz(self):
        cube = _time_series(self.wind[:, :, 0, 0])
        self.draw_method(cube)
        self.check_graphic()


class TestContour(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.contour routine."""
    def setUp(self):
        self.wind = _load_4d_testcube()
        self.draw_method = iplt.contour


class TestContourf(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.contourf routine."""
    def setUp(self):
        self.wind = _load_4d_testcube()
        self.draw_method = iplt.contourf


class TestPcolor(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.pcolor routine."""
    def setUp(self):
        self.wind = _load_4d_testcube()
        self.draw_method = iplt.pcolor


class TestPcolormesh(tests.GraphicsTest, SliceMixin):
    """Test the iris.plot.pcolormesh routine."""
    def setUp(self):
        self.wind = _load_4d_testcube()
        self.draw_method = iplt.pcolormesh


def check_warnings(method):
    """
    Decorator that adds a catch_warnings and filter to assert
    the method being decorated issues a UserWarning.

    """
    @wraps(method)
    def decorated_method(self, *args, **kwargs):
        # Force reset of iris.coords warnings registry to avoid suppression of
        # repeated warnings. warnings.resetwarnings() does not do this.
        if hasattr(coords, '__warningregistry__'):
            coords.__warningregistry__.clear()

        # Check that method raises warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                return method(self, *args, **kwargs)
    return decorated_method


def ignore_warnings(method):
    """
    Decorator that adds a catch_warnings and filter to suppress
    any warnings issues by the method being decorated.

    """
    @wraps(method)
    def decorated_method(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return method(self, *args, **kwargs)
    return decorated_method


class CheckForWarningsMetaclass(type):
    """
    Metaclass that adds a further test for each base class test
    that checks that each test raises a UserWarning. Each base
    class test is then overriden to ignore warnings in order to
    check the underlying functionality.

    """
    def __new__(cls, name, bases, local):
        def add_decorated_methods(attr_dict, target_dict, decorator):
            for key, value in attr_dict.items():
                if (isinstance(value, types.FunctionType) and
                        key.startswith('test')):
                    new_key = '_'.join((key, decorator.__name__))
                    if new_key not in target_dict:
                        wrapped = decorator(value)
                        wrapped.__name__ = new_key
                        target_dict[new_key] = wrapped
                    else:
                        raise RuntimeError('A attribute called {!r} '
                                           'already exists.'.format(new_key))

        def override_with_decorated_methods(attr_dict, target_dict,
                                            decorator):
            for key, value in attr_dict.items():
                if (isinstance(value, types.FunctionType) and
                        key.startswith('test')):
                    target_dict[key] = decorator(value)

        # Add decorated versions of base methods
        # to check for warnings.
        for base in bases:
            add_decorated_methods(base.__dict__, local, check_warnings)

        # Override base methods to ignore warnings.
        for base in bases:
            override_with_decorated_methods(base.__dict__, local,
                                            ignore_warnings)

        return type.__new__(cls, name, bases, local)


class TestPcolorNoBounds(tests.GraphicsTest, SliceMixin):
    """
    Test the iris.plot.pcolor routine on a cube with coordinates
    that have no bounds.

    """
    __metaclass__ = CheckForWarningsMetaclass

    def setUp(self):
        self.wind = _load_4d_testcube_no_bounds()
        self.draw_method = iplt.pcolor


class TestPcolormeshNoBounds(tests.GraphicsTest, SliceMixin):
    """
    Test the iris.plot.pcolormesh routine on a cube with coordinates
    that have no bounds.

    """
    __metaclass__ = CheckForWarningsMetaclass

    def setUp(self):
        self.wind = _load_4d_testcube_no_bounds()
        self.draw_method = iplt.pcolormesh


class Slice1dMixin(object):
    """Mixin class providing tests for each 1-dimensional permutation of axes.

    Requires self.draw_method to be the relevant plotting function,
    and self.results to be a dictionary containing the desired test results."""

    def test_x(self):
        cube = self.wind[0, 0, 0, :]
        self.draw_method(cube)
        self.check_graphic()

    def test_y(self):
        cube = self.wind[0, 0, :, 0]
        self.draw_method(cube)
        self.check_graphic()

    def test_z(self):
        cube = self.wind[0, :, 0, 0]
        self.draw_method(cube)
        self.check_graphic()

    def test_t(self):
        cube = _time_series(self.wind[:, 0, 0, 0])
        self.draw_method(cube)
        self.check_graphic()

    def test_t_dates(self):
        cube = _date_series(self.wind[:, 0, 0, 0])
        self.draw_method(cube)
        plt.gcf().autofmt_xdate()
        plt.xlabel('Phenomenon time')

        self.check_graphic()


class TestPlot(tests.GraphicsTest, Slice1dMixin):
    """Test the iris.plot.plot routine."""
    def setUp(self):
        self.wind = _load_4d_testcube()
        self.draw_method = iplt.plot


class TestQuickplotPlot(tests.GraphicsTest, Slice1dMixin):
    """Test the iris.quickplot.plot routine."""
    def setUp(self):
        self.wind = _load_4d_testcube()
        self.draw_method = qplt.plot


_load_cube_once_cache = {}


def load_cube_once(filename, constraint):
    """Same syntax as load_cube, but will only load a file once,

    then cache the answer in a dictionary.

    """
    global _load_cube_once_cache
    key = (filename, str(constraint))
    cube = _load_cube_once_cache.get(key, None)

    if cube is None:
        cube = iris.load_cube(filename, constraint)
        _load_cube_once_cache[key] = cube

    return cube


class LambdaStr(object):
    """Provides a callable function which has a sensible __repr__."""
    def __init__(self, repr, lambda_fn):
        self.repr = repr
        self.lambda_fn = lambda_fn

    def __call__(self, *args, **kwargs):
        return self.lambda_fn(*args, **kwargs)

    def __repr__(self):
        return self.repr


@iris.tests.skip_data
class TestPlotCoordinatesGiven(tests.GraphicsTest):
    def setUp(self):
        filename = tests.get_data_path(('PP', 'COLPEX',
                                        'theta_and_orog_subset.pp'))
        self.cube = load_cube_once(filename, 'air_potential_temperature')

        self.draw_module = iris.plot
        self.contourf = LambdaStr('iris.plot.contourf',
                                  lambda cube, *args, **kwargs:
                                  iris.plot.contourf(cube, *args, **kwargs))
        self.contour = LambdaStr('iris.plot.contour',
                                 lambda cube, *args, **kwargs:
                                 iris.plot.contour(cube, *args, **kwargs))
        self.points = LambdaStr('iris.plot.points',
                                lambda cube, *args, **kwargs:
                                iris.plot.points(cube, c=cube.data,
                                                 *args, **kwargs))
        self.plot = LambdaStr('iris.plot.plot',
                              lambda cube, *args, **kwargs:
                              iris.plot.plot(cube, *args, **kwargs))

        self.results = {'yx': ([self.contourf, ['grid_latitude',
                                                'grid_longitude']],
                               [self.contourf, ['grid_longitude',
                                                'grid_latitude']],
                               [self.contour, ['grid_latitude',
                                               'grid_longitude']],
                               [self.contour, ['grid_longitude',
                                               'grid_latitude']],
                               [self.points, ['grid_latitude',
                                              'grid_longitude']],
                               [self.points, ['grid_longitude',
                                              'grid_latitude']],),
                        'zx': ([self.contourf, ['model_level_number',
                                                'grid_longitude']],
                               [self.contourf, ['grid_longitude',
                                                'model_level_number']],
                               [self.contour, ['model_level_number',
                                               'grid_longitude']],
                               [self.contour, ['grid_longitude',
                                               'model_level_number']],
                               [self.points, ['model_level_number',
                                              'grid_longitude']],
                               [self.points, ['grid_longitude',
                                              'model_level_number']],),
                        'tx': ([self.contourf, ['time', 'grid_longitude']],
                               [self.contourf, ['grid_longitude', 'time']],
                               [self.contour, ['time', 'grid_longitude']],
                               [self.contour, ['grid_longitude', 'time']],
                               [self.points, ['time', 'grid_longitude']],
                               [self.points, ['grid_longitude', 'time']],),
                        'x': ([self.plot, ['grid_longitude']],),
                        'y': ([self.plot, ['grid_latitude']],)
                        }

    def draw(self, draw_method, *args, **kwargs):
        draw_fn = getattr(self.draw_module, draw_method)
        draw_fn(*args, **kwargs)
        self.check_graphic()

    def run_tests(self, cube, results):
        for draw_method, coords in results:
            draw_method(cube, coords=coords)
            try:
                self.check_graphic()
            except AssertionError, err:
                self.fail('Draw method %r failed with coords: %r. '
                          'Assertion message: %s' % (draw_method, coords, err))

    def test_yx(self):
        test_cube = self.cube[0, 0, :, :]
        self.run_tests(test_cube, self.results['yx'])

    def test_zx(self):
        test_cube = self.cube[0, :15, 0, :]
        self.run_tests(test_cube, self.results['zx'])

    def test_tx(self):
        test_cube = self.cube[:, 0, 0, :]
        self.run_tests(test_cube, self.results['tx'])

    def test_x(self):
        test_cube = self.cube[0, 0, 0, :]
        self.run_tests(test_cube, self.results['x'])

    def test_y(self):
        test_cube = self.cube[0, 0, :, 0]
        self.run_tests(test_cube, self.results['y'])

    def test_badcoords(self):
        cube = self.cube[0, 0, :, :]
        draw_fn = getattr(self.draw_module, 'contourf')
        self.assertRaises(ValueError, draw_fn, cube,
                          coords=['grid_longitude', 'grid_longitude'])
        self.assertRaises(ValueError, draw_fn, cube,
                          coords=['grid_longitude', 'grid_longitude',
                                  'grid_latitude'])
        self.assertRaises(iris.exceptions.CoordinateNotFoundError, draw_fn,
                          cube, coords=['grid_longitude', 'wibble'])
        self.assertRaises(ValueError, draw_fn, cube, coords=[])
        self.assertRaises(ValueError, draw_fn, cube,
                          coords=[cube.coord('grid_longitude'),
                                  cube.coord('grid_longitude')])
        self.assertRaises(ValueError, draw_fn, cube,
                          coords=[cube.coord('grid_longitude'),
                                  cube.coord('grid_longitude'),
                                  cube.coord('grid_longitude')])

    def test_non_cube_coordinate(self):
        cube = self.cube[0, :, :, 0]
        pts = -100 + np.arange(cube.shape[1]) * 13
        x = coords.DimCoord(pts, standard_name='model_level_number',
                            attributes={'positive': 'up'})
        self.draw('contourf', cube, coords=['grid_latitude', x])


class TestSymbols(tests.GraphicsTest):
    def test_cloud_cover(self):
        iplt.symbols(range(10), [0] * 10, [iris.symbols.CLOUD_COVER[i]
                                           for i in range(10)], 0.375)
        self.check_graphic()


class Test_map_common(tests.IrisTest):
    def setUp(self):
        self.bounded_cube = tests.stock.lat_lon_cube()
        self.bounded_cube.coord("latitude").guess_bounds()
        self.bounded_cube.coord("longitude").guess_bounds()

    def test_boundmode_multidim(self):
        # Test exception translation.
        # We can't get contiguous bounded grids from multi-d coords.
        cube = self.bounded_cube
        cube.remove_coord("latitude")
        cube.add_aux_coord(coords.AuxCoord(points=cube.data,
                                           standard_name='latitude',
                                           units='degrees'), [0, 1])
        with self.assertRaises(ValueError):
            iplt._map_common("pcolormesh", None, coords.BOUND_MODE, cube, None)

    def test_boundmode_4bounds(self):
        # Test exception translation.
        # We can only get contiguous bounded grids with 2 bounds per point.
        cube = self.bounded_cube
        lat = coords.AuxCoord.from_coord(cube.coord("latitude"))
        lat.bounds = np.array([lat.points, lat.points + 1,
                               lat.points + 2, lat.points + 3]).transpose()
        cube.remove_coord("latitude")
        cube.add_aux_coord(lat, 0)
        with self.assertRaises(ValueError):
            iplt._map_common("pcolormesh", None, coords.BOUND_MODE, cube, None)


if __name__ == "__main__":
    tests.main()
