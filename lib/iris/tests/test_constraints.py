# (C) British Crown Copyright 2010 - 2013, Met Office
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
Test the constrained cube loading mechanism.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import iris
import iris.tests.stock as stock


SN_AIR_POTENTIAL_TEMPERATURE = 'air_potential_temperature'
SN_SPECIFIC_HUMIDITY = 'specific_humidity'


# TODO: Workaround, pending #1262
def workaround_pending_1262(cubes):
    """Reverse the cube if sigma was chosen as a dim_coord."""
    for i, cube in enumerate(cubes):
        ml = cube.coord("model_level_number").points
        if ml[0] > ml[1]:
            cubes[i] = cube[::-1]


class TestSimple(tests.IrisTest):
    slices = iris.cube.CubeList(stock.realistic_4d().slices(['grid_latitude', 'grid_longitude']))

    def test_constraints(self):
        constraint = iris.Constraint(model_level_number=10)
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 6)

        constraint = iris.Constraint(model_level_number=[10, 22])
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 2 * 6)

        constraint = iris.Constraint(model_level_number=lambda c: ( c > 30 ) | (c <= 3))
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 43 * 6)

        constraint = iris.Constraint(coord_values={'model_level_number': lambda c: c > 1000})
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 0)

        constraint = (iris.Constraint(model_level_number=10) &
                      iris.Constraint(time=347922.))
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 1)

        constraint = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 70 * 6)

    def test_mismatched_type(self):
        constraint = iris.Constraint(model_level_number='aardvark')
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 0)

    def test_cell(self):
        cell = iris.coords.Cell(10)
        constraint = iris.Constraint(model_level_number=cell)
        sub_list = self.slices.extract(constraint)
        self.assertEqual(len(sub_list), 6)


class TestMixin(object):
    """
    Mix-in class for attributes & utilities common to the "normal" and "strict" test cases.

    """
    def setUp(self):
        self.dec_path = tests.get_data_path(['PP', 'globClim1', 'dec_subset.pp'])
        self.theta_path = tests.get_data_path(['PP', 'globClim1', 'theta.pp'])

        self.humidity = iris.Constraint(SN_SPECIFIC_HUMIDITY)
        self.theta = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)

        # Coord based constraints
        self.level_10 = iris.Constraint(model_level_number=10)
        self.level_22 = iris.Constraint(model_level_number=22)

        # Value based coord constraint
        self.level_30 = iris.Constraint(model_level_number=30)
        self.level_gt_30_le_3 = iris.Constraint(model_level_number=lambda c: ( c > 30 ) | (c <= 3))
        self.invalid_inequality = iris.Constraint(coord_values={'model_level_number': lambda c: c > 1000})

        # bound based coord constraint
        self.level_height_of_model_level_number_10 = iris.Constraint(level_height=1900)
        self.model_level_number_10_22 = iris.Constraint(model_level_number=[10, 22])

        # Invalid constraints
        self.pressure_950 = iris.Constraint(model_level_number=950)

        self.lat_30 = iris.Constraint(latitude=30)
        self.lat_gt_45 = iris.Constraint(latitude=lambda c: c > 45)


class RelaxedConstraintMixin(TestMixin):
    @staticmethod
    def fixup_sigma_to_be_aux(cubes):
        # XXX Fix the cubes such that the sigma coordinate is always an AuxCoord. Pending gh issue #18
        if isinstance(cubes, iris.cube.Cube):
            cubes = [cubes]

        for cube in cubes:
            sigma = cube.coord('sigma')
            sigma = iris.coords.AuxCoord.from_coord(sigma)
            cube.replace_coord(sigma)

    def assertCML(self, cubes, filename):
        filename = "%s_%s.cml" % (filename, self.suffix)
        tests.IrisTest.assertCML(self, cubes, ('constrained_load', filename))

    def load_match(self, files, constraints):
        raise NotImplementedError()  # defined in subclasses

    def test_single_atomic_constraint(self):
        cubes = self.load_match(self.dec_path, self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'all_10')

        cubes = self.load_match(self.dec_path, self.theta)
        self.assertCML(cubes, 'theta')

        cubes = self.load_match(self.dec_path, self.model_level_number_10_22)
        self.fixup_sigma_to_be_aux(cubes)
        workaround_pending_1262(cubes)
        self.assertCML(cubes, 'all_ml_10_22')

        # Check that it didn't matter that we provided sets & tuples to the model_level
        for constraint in [iris.Constraint(model_level_number=set([10, 22])), iris.Constraint(model_level_number=tuple([10, 22]))]:
            cubes = self.load_match(self.dec_path, constraint)
            self.fixup_sigma_to_be_aux(cubes)
            workaround_pending_1262(cubes)
            self.assertCML(cubes, 'all_ml_10_22')

    def test_string_standard_name(self):
        cubes = self.load_match(self.dec_path, SN_AIR_POTENTIAL_TEMPERATURE)
        self.assertCML(cubes, 'theta')

        cubes = self.load_match(self.dec_path, [SN_AIR_POTENTIAL_TEMPERATURE])
        self.assertCML(cubes, 'theta')

        cubes = self.load_match(self.dec_path, iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE))
        self.assertCML(cubes, 'theta')

        cubes = self.load_match(self.dec_path, iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE, model_level_number=10))
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')

    def test_latitude_constraint(self):
        cubes = self.load_match(self.theta_path, self.lat_30)
        self.assertCML(cubes, 'theta_lat_30')

        cubes = self.load_match(self.theta_path, self.lat_gt_45)
        self.assertCML(cubes, 'theta_lat_gt_30')

    def test_single_expression_constraint(self):
        cubes = self.load_match(self.theta_path, self.theta & self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')

        cubes = self.load_match(self.theta_path, self.level_10 & self.theta)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')

    def test_dual_atomic_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta, self.level_10])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_and_all_10')

    def test_dual_repeated_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta, self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_and_theta')

    def test_dual_expression_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta & self.level_10, self.level_gt_30_le_3 & self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10_and_theta_level_gt_30_le_3')

    def test_invalid_constraint(self):
        cubes = self.load_match(self.theta_path, self.pressure_950)
        self.assertCML(cubes, 'pressure_950')

        cubes = self.load_match(self.theta_path, self.invalid_inequality)
        self.assertCML(cubes, 'invalid_inequality')

    def test_inequality_constraint(self):
        cubes = self.load_match(self.theta_path, self.level_gt_30_le_3)
        self.assertCML(cubes, 'theta_gt_30_le_3')


class StrictConstraintMixin(RelaxedConstraintMixin):
    def test_single_atomic_constraint(self):
        cubes = self.load_match(self.theta_path, self.theta)
        self.assertCML(cubes, 'theta')

        cubes = self.load_match(self.theta_path, self.level_10)
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_10')

    def test_invalid_constraint(self):
        with self.assertRaises(iris.exceptions.ConstraintMismatchError):
            self.load_match(self.theta_path, self.pressure_950)

    def test_dual_atomic_constraint(self):
        cubes = self.load_match(self.dec_path, [self.theta, self.level_10 & self.theta])
        self.fixup_sigma_to_be_aux(cubes)
        self.assertCML(cubes, 'theta_and_theta_10')


@iris.tests.skip_data
class TestCubeLoadConstraint(RelaxedConstraintMixin, tests.IrisTest):
    suffix = 'load_match'

    def load_match(self, files, constraints):
        cubes = iris.load(files, constraints)
        if not isinstance(cubes, iris.cube.CubeList):
            raise Exception("NOT A CUBE LIST! " + str(type(cubes)))
        return cubes


@iris.tests.skip_data
class TestCubeListConstraint(RelaxedConstraintMixin, tests.IrisTest):
    suffix = 'load_match'

    def load_match(self, files, constraints):
        cubes = iris.load(files).extract(constraints)
        if not isinstance(cubes, iris.cube.CubeList):
            raise Exception("NOT A CUBE LIST! " + str(type(cubes)))
        return cubes


@iris.tests.skip_data
class TestCubeLoadStrictConstraint(StrictConstraintMixin, tests.IrisTest):
    suffix = 'load_strict'

    def load_match(self, files, constraints):
        cubes = iris.load_strict(files, constraints)
        return cubes


@iris.tests.skip_data
class TestCubeListStrictConstraint(StrictConstraintMixin, tests.IrisTest):
    suffix = 'load_strict'

    def load_match(self, files, constraints):
        cubes = iris.load(files).extract_strict(constraints)
        return cubes


@iris.tests.skip_data
class TestCubeExtract(TestMixin, tests.IrisTest):
    def setUp(self):
        TestMixin.setUp(self)
        self.cube = iris.load_cube(self.theta_path)

    def test_attribute_constraint(self):
        # there is no my_attribute attribute on the cube, so ensure it returns None
        cube = self.cube.extract(iris.AttributeConstraint(my_attribute='foobar'))
        self.assertIsNone(cube)

        orig_cube = self.cube
        # add an attribute to the cubes
        orig_cube.attributes['my_attribute'] = 'foobar'

        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute='foobar'))
        self.assertCML(cube, ('constrained_load', 'attribute_constraint.cml'))

        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute='not me'))
        self.assertIsNone(cube)

        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute=lambda val: val.startswith('foo')))
        self.assertCML(cube, ('constrained_load', 'attribute_constraint.cml'))

        cube = orig_cube.extract(iris.AttributeConstraint(my_attribute=lambda val: not val.startswith('foo')))
        self.assertIsNone(cube)

        cube = orig_cube.extract(iris.AttributeConstraint(my_non_existant_attribute='hello world'))
        self.assertIsNone(cube)

    def test_standard_name(self):
        r = iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE)
        self.assertTrue(self.cube.extract(r).standard_name, SN_AIR_POTENTIAL_TEMPERATURE)

        r = iris.Constraint('wibble')
        self.assertEqual(self.cube.extract(r), None)

    def test_empty_data(self):
        # Ensure that the process of WHERE does not load data if there was empty data to start with...
        self.assertNotEquals(None, self.cube._data_manager)

        self.assertNotEquals(None, self.cube.extract(self.level_10)._data_manager)

        self.assertNotEquals(None, self.cube.extract(self.level_10).extract(self.level_10)._data_manager)

    def test_non_existant_coordinate(self):
        # Check the behaviour when a constraint is given for a coordinate which does not exist/span a dimension
        self.assertEqual(self.cube[0, :, :].extract(self.level_10), None)

        self.assertEqual(self.cube.extract(iris.Constraint(wibble=10)), None)


@iris.tests.skip_data
class TestConstraints(TestMixin, tests.IrisTest):
    def test_constraint_expressions(self):
        rt = repr(self.theta)
        rl10 = repr(self.level_10)

        rt_l10 = repr(self.theta & self.level_10)
        self.assertEqual(rt_l10, "ConstraintCombination(%s, %s, <built-in function __and__>)" % (rt, rl10))

    def test_string_repr(self):
        rt = repr(iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE))
        self.assertEqual(rt, "Constraint(name='%s')" % SN_AIR_POTENTIAL_TEMPERATURE)

        rt = repr(iris.Constraint(SN_AIR_POTENTIAL_TEMPERATURE, model_level_number=10))
        self.assertEqual(rt, "Constraint(name='%s', coord_values={'model_level_number': 10})" % SN_AIR_POTENTIAL_TEMPERATURE)

    def test_number_of_raw_cubes(self):
        # Test the constraints generate the correct number of raw cubes.
        raw_cubes = iris.load_raw(self.theta_path)
        self.assertEqual(len(raw_cubes), 38)

        raw_cubes = iris.load_raw(self.theta_path, [self.level_10])
        self.assertEqual(len(raw_cubes), 1)

        raw_cubes = iris.load_raw(self.theta_path, [self.theta])
        self.assertEqual(len(raw_cubes), 38)

        raw_cubes = iris.load_raw(self.dec_path, [self.level_30])
        self.assertEqual(len(raw_cubes), 4)

        raw_cubes = iris.load_raw(self.dec_path, [self.theta])
        self.assertEqual(len(raw_cubes), 38)


class TestBetween(tests.IrisTest):
    def run_test(self, function, numbers, results):
        for number, result in zip(numbers, results):
            self.assertEqual(function(number), result)

    def test_le_ge(self):
        function = iris.util.between(2, 4)
        numbers = [1, 2, 3, 4, 5]
        results = [False, True, True, True, False]
        self.run_test(function, numbers, results)

    def test_lt_gt(self):
        function = iris.util.between(2, 4, rh_inclusive=False, lh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, False, True, False, False]
        self.run_test(function, numbers, results)

    def test_le_gt(self):
        function = iris.util.between(2, 4, rh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, True, True, False, False]
        self.run_test(function, numbers, results)

    def test_lt_ge(self):
        function = iris.util.between(2, 4, lh_inclusive=False)
        numbers = [1, 2, 3, 4, 5]
        results = [False, False, True, True, False]
        self.run_test(function, numbers, results)


class TestTimeConstraint(tests.IrisTest):
    def setUp(self):
        # Data with a variety of time coordinates
        # Hourly: 2010-03-09 9am to 2010-03-10 9pm
        self.hourly_cube_points = iris.load_strict(tests.get_data_path(['PP', 'uk4', 'uk4par09.pp']), 'air_temperature')
        # Daily Dec 87 - Nov 88, 360 day
        self.daily_cube_bounds = iris.load_strict(tests.get_data_path(['PP', 'ian_edmond', 'abcza_pa19591997_daily_29.b.pp']),
                                                  'precipitation_flux')
        # Dec, Jan, Feb every year 2091 - 2095, 360 day
        self.yearly_djf_cube_bounds = iris.load_strict(tests.get_data_path(['PP', 'ian_edmond', 'aaxzc_n10r13xy.b.pp']))
        # Yearly with bounds spanning the year, Sep 74 to Sep 96, 360 day
        self.yearly_cube_bounds = iris.load_strict(tests.get_data_path(['PP', 'ian_edmond', 'model.b.pp']))
        # Yearly with points at Midnight March 1st from Mar 1861 to Mar 2100, 360 day
        self.yearly_cube_points = iris.load_strict(tests.get_data_path(['PP', 'ian_edmond', 'HadCM2_ts_SAT_ann_18602100.b.pp']))
        # Points every ten minutes on 2009-09-09 from 17:10 to 18:00
        self.minutes_cube_points = iris.load_strict(tests.get_data_path(['PP', 'COLPEX', 'theta_and_orog_subset.pp']),
                                                    'air_potential_temperature') #  Every ten minutes: 2009-09-09 17:10 to 18:00

    def test_empty(self):
        # all
        cube = self.hourly_cube_points.extract(iris.TimeConstraint())
        self.assertEqual(self.hourly_cube_points, cube)
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint())
        self.assertEqual(self.daily_cube_bounds, cube)
        cube = self.yearly_djf_cube_bounds.extract(iris.TimeConstraint())
        self.assertEqual(self.yearly_djf_cube_bounds, cube)
        cube = self.yearly_cube_bounds.extract(iris.TimeConstraint())
        self.assertEqual(self.yearly_cube_bounds, cube)
        cube = self.yearly_cube_points.extract(iris.TimeConstraint())
        self.assertEqual(self.yearly_cube_points, cube)
        cube = self.minutes_cube_points.extract(iris.TimeConstraint())
        self.assertEqual(self.minutes_cube_points, cube)

    def test_year(self):
        # yearly_cube_bounds, yearly_cube_points
        cube = self.yearly_cube_bounds.extract(iris.TimeConstraint(year=1990))
        self.assertCML(cube, ('constrained_load', 'year_bnds.cml'))
        cube = self.yearly_cube_bounds.extract(iris.TimeConstraint(year=2000))
        self.assertIsNone(cube)
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(year=1990))
        self.assertCML(cube, ('constrained_load', 'year_pts.cml'))
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(year=1850))
        self.assertIsNone(cube)
        cube = self.yearly_cube_bounds.extract(iris.TimeConstraint(year=[1990,1995,2000]))
        self.assertCML(cube, ('constrained_load', 'multiyear.cml'))

    def test_month(self):
        # daily_cube_bounds, yearly_cube_bounds, yearly_cube_points, hourly_cube_points, yearly_djf_cube_bounds
        months = ('January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December')
        for i, month in enumerate(months, start=1):
            # Numerical
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(month=i))
            self.assertCML(cube, ('constrained_load', month.lower() + '_bnds.cml'))
            # Short name e.g 'Feb'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(month=month[0:3]))
            self.assertCML(cube, ('constrained_load', month.lower() + '_bnds.cml'))
            # Long name e.g. 'February'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(month=month))
            self.assertCML(cube, ('constrained_load', month.lower() + '_bnds.cml'))
            # Lower case short name e.g. 'feb'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(month=month[0:3].lower()))
            self.assertCML(cube, ('constrained_load', month.lower() + '_bnds.cml'))
            # Lower case long name e.g. 'february'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(month=month.lower()))
            self.assertCML(cube, ('constrained_load', month.lower() + '_bnds.cml'))

        with self.assertRaises(ValueError):
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(month='misspelled'))
        with self.assertRaises(ValueError):
            subcube = self.daily_cube_bounds.extract(iris.TimeConstraint(month=0))

        cube = self.hourly_cube_points.extract(iris.TimeConstraint(month='Mar'))
        self.assertCML(cube, ('constrained_load', 'month_pts.cml'))

        cube = self.yearly_cube_bounds.extract(iris.TimeConstraint(year=1990, month='Feb'))
        self.assertCML(cube, ('constrained_load', 'year_month_bnds_1.cml'))
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(year=1988, month='Feb'))
        self.assertCML(cube, ('constrained_load', 'year_month_bnds_2.cml'))
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(year=1990, month='Mar'))
        self.assertCML(cube, ('constrained_load', 'year_month_pts.cml'))
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(year=1990, month='Feb'))
        self.assertIsNone(cube)        
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(year=1987, month='Feb'))
        self.assertIsNone(cube)        
        cube = self.yearly_djf_cube_bounds.extract(iris.TimeConstraint(month=['Dec', 'Jan', 'Feb']))
        self.assertCML(cube, ('constrained_load', 'multimonth.cml'))

    def test_day(self):
        # daily_cube_bounds, yearly_cube_bounds, yearly_cube_points
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(day=23))
        self.assertCML(cube, ('constrained_load', 'day_bnds.cml'))
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(day=[1,2,3,4,5]))
        self.assertCML(cube, ('constrained_load', 'multiday.cml'))

        for x in xrange(10):
            day = random.randint(1,28)
            cube = self.yearly_cube_bounds.extract(iris.TimeConstraint(day=day))
            self.assertCML(cube, ('constrained_load', 'day_random.cml'))    # Should match every cell as bounds span more than a mont

        cube = self.yearly_cube_points.extract(iris.TimeConstraint(day=1))
        self.assertCML(cube, ('constrained_load', 'day_pts.cml'))
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(day=23))
        self.assertIsNone(cube)

        # leap year - need a gregorian calendar - ostia data would suffice but is over 800MB
        #cube = iris.load_strict(tests.get_data_path(['PP', 'ostia', 'ostia_sst_200604_201009_N216.pp']), 
        #                                            iris.TimeConstraint(month='Feb', day=29))
        #self.assertCML(cube, ('constrained_load', 'leap_year.cml'))
        
    def test_season(self):
        # daily_cube_bounds, yearly_djf_cube_bounds, yearly_cube_points
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season='djf'))
        self.assertCML(cube, ('constrained_load', 'djf_bnds.cml'))            
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season='mam'))
        self.assertCML(cube, ('constrained_load', 'mam_bnds.cml'))            
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season='jja'))
        self.assertCML(cube, ('constrained_load', 'jja_bnds.cml'))            
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season='son'))
        self.assertCML(cube, ('constrained_load', 'son_bnds.cml'))

        # No change to data containing only djf
        cube = self.yearly_djf_cube_bounds.extract(iris.TimeConstraint(season='djf'))
        self.assertEqual(self.yearly_djf_cube_bounds, cube)                
        
        # Data refers only to March 1st 0:00:00 
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(season='djf'))
        self.assertIsNone(cube)
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(season='mam'))
        self.assertCML(cube, ('constrained_load', 'mam_pts.cml'))
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(season='jja'))
        self.assertIsNone(cube)
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(season='son'))
        self.assertIsNone(cube)

        with self.assertRaises(ValueError):
            cube = self.yearly_cube_points.extract(iris.TimeConstraint(season='summer'))
        with self.assertRaises(ValueError):
            cube = self.yearly_cube_points.extract(iris.TimeConstraint(season='ABC'))

    def test_season_year(self):
        # daily_cube_bounds, yearly_cube_points
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season_year=1988))
        self.assertEqual(self.daily_cube_bounds, cube)
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season_year=1988, season='djf'))
        self.assertCML(cube, ('constrained_load', 'djf_year_bnds.cml'))
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season_year=1987))
        self.assertIsNone(cube)
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(season_year=1989))
        self.assertIsNone(cube)
        cube = self.yearly_cube_points.extract(iris.TimeConstraint(season_year=[1980, 1985, 1990], season='mam'))
        self.assertCML(cube, ('constrained_load', 'mam_season_year_pts.cml'))

    def test_time(self):
        # minutes_cube_points, daily_cube_bounds, hourly_cube_points
        cube = self.minutes_cube_points.extract(iris.TimeConstraint(hour=17))
        self.assertCML(cube, ('constrained_load', 'hour.cml'))        
        cube = self.minutes_cube_points.extract(iris.TimeConstraint(minute=20))
        self.assertCML(cube, ('constrained_load', 'minute.cml'))                
        cube = self.minutes_cube_points.extract(iris.TimeConstraint(hour=17, minute=20, second=0))
        self.assertCML(cube, ('constrained_load', 'hour_min_sec.cml'))                
        cube = self.minutes_cube_points.extract(iris.TimeConstraint(hour=17, minute=20, second=23))
        self.assertIsNone(cube)
        # Data has bounds that span a day so should match any time
        cube = self.daily_cube_bounds.extract(iris.TimeConstraint(hour=17, minute=20, second=10, microsecond=200))
        self.assertEqual(self.daily_cube_bounds, cube)
        # 24 hour clock
        cube = self.hourly_cube_points.extract(iris.TimeConstraint(hour=0))
        self.assertCML(cube, ('constrained_load', 'hour_1.cml'))
        cube = self.hourly_cube_points.extract(iris.TimeConstraint(hour=9))
        self.assertCML(cube, ('constrained_load', 'hour_2.cml'))
        cube = self.hourly_cube_points.extract(iris.TimeConstraint(hour=21))
        self.assertCML(cube, ('constrained_load', 'hour_3.cml'))
        

class TestTimePeriodConstraint(tests.IrisTest):
    def setUp(self):
        
    def test_year:
        # Bounds
        cube = self.yearly_cube_bounds.extract(iris.TimePeriodConstraint(start_year=1990))
        self.assertCML(cube, ('constrained_load', 'start_year_bnds.cml'))        
        cube = self.yearly_cube_bounds.extract(iris.TimePeriodConstraint(end_year=1995))                
        self.assertCML(cube, ('constrained_load', 'end_year_bnds.cml'))        
        cube = self.yearly_cube_bounds.extract(iris.TimePeriodConstraint(start_year=1990, end_year=1995))
        self.assertCML(cube, ('constrained_load', 'startend_year_bnds.cml'))        
        cube = self.yearly_cube_bounds.extract(iris.TimePeriodConstraint(start_year=1990, end_year=1990))
        self.assertCML(cube, ('constrained_load', 'start_year_equal_bnds.cml'))        

        # Points
        cube = self.yearly_cube_points.extract(iris.TimePeriodConstraint(start_year=1990))
        self.assertCML(cube, ('constrained_load', 'start_year_bnds.cml'))        
        cube = self.yearly_cube_points.extract(iris.TimePeriodConstraint(end_year=1995))                
        self.assertCML(cube, ('constrained_load', 'end_year_bnds.cml'))        
        cube = self.yearly_cube_points.extract(iris.TimePeriodConstraint(start_year=1990, end_year=1995))
        self.assertCML(cube, ('constrained_load', 'startend_year_bnds.cml'))        
        cube = self.yearly_cube_points.extract(iris.TimePeriodConstraint(start_year=1990, end_year=1990))
        self.assertIsNone(cube)

    def test_month:
        months = ('January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December')
        for i, month in enumerate(months, start=1):
            # Numerical
            cube = self.daily_cube_bounds.extract(iris.TimePeriodConstraint(start_month=i))
            self.assertCML(cube, ('constrained_load', 'start_' + month.lower() + '_bnds.cml'))
            cube = self.daily_cube_bounds.extract(iris.TimePeriodConstraint(end_month=i))
            self.assertCML(cube, ('constrained_load', 'end_' + month.lower() + '_bnds.cml'))
            # Short name e.g 'Feb'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(start_month=month[0:3]))
            self.assertCML(cube, ('constrained_load', 'start_' + month.lower() + '_bnds.cml'))
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(end_month=month[0:3]))
            self.assertCML(cube, ('constrained_load', 'end_' + month.lower() + '_bnds.cml'))
            # Long name e.g. 'February'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(start_month=month))
            self.assertCML(cube, ('constrained_load', 'start_' + month.lower() + '_bnds.cml'))
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(end_month=month))
            self.assertCML(cube, ('constrained_load', 'end_' + month.lower() + '_bnds.cml'))
            # Lower case short name e.g. 'feb'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(start_month=month[0:3].lower()))
            self.assertCML(cube, ('constrained_load', 'start_' + month.lower() + '_bnds.cml'))
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(end_month=month[0:3].lower()))
            self.assertCML(cube, ('constrained_load', 'end_' + month.lower() + '_bnds.cml'))
            # Lower case long name e.g. 'february'
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(start_month=month.lower()))
            self.assertCML(cube, ('constrained_load', 'start_' + month.lower() + '_bnds.cml'))
            cube = self.daily_cube_bounds.extract(iris.TimeConstraint(end_month=month.lower()))
            self.assertCML(cube, ('constrained_load', 'end_' + month.lower() + '_bnds.cml'))

        # Over the end of the year
        cube = self.daily_cube_bounds.extract(iris.TimePeriodConstraint(start_month='Feb', end_month='Dec'))
        self.assertCML(cube, ('constrained_load', 'startend_month_bnds_1.cml'))
        cube = self.daily_cube_bounds.extract(iris.TimePeriodConstraint(start_month='Dec', end_month='Feb'))
        self.assertCML(cube, ('constrained_load', 'startend_month_bnds_2.cml'))
        yearly_djf_cube_bounds.extract(iris.TimePeriodConstraint(start_month='Dec', end_month='Mar'))
        self.assertEqual(cube, yearly_djf_cube_bounds)

        # Year and month
        cube = self.yearly_cube_points.extract(iris.TimePeriodConstraint(start_year=1890, start_month='Nov', end_year=1900, end_month='Feb'))
        self.assertCML(cube, ('constrained_load', 'startend_year_month_bnds.cml'))
        cube = self.yearly_cube_points.extract(iris.TimePeriodConstraint(start_month='Nov', end_month='Feb'))
        self.assertIsNone(cube)
        
    def test_day:
        # Bounds
        cube = daily_cube_bounds.extract(iris.TimePeriodConstraint(start_day=5))
        self.assertCML(cube, ('constrained_load', 'start_day_bnds.cml'))        
        cube = daily_cube_bounds.extract(iris.TimePeriodConstraint(end_day=5))
        self.assertCML(cube, ('constrained_load', 'end_day_bnds.cml'))        
        cube = self.daily_cube_bounds.extract(iris.TimePeriodConstraint(start_day=5, end_day=10))
        self.assertCML(cube, ('constrained_load', 'startend_day_bnds_1.cml'))        
        cube = self.daily_cube_bounds.extract(iris.TimePeriodConstraint(start_day=25, end_day=5))
        self.assertCML(cube, ('constrained_load', 'startend_day_bnds_2.cml'))        
        # Points
        cube = hourly_cube_points.extract(iris.TimePeriodConstraint(start_day=8))
        self.assertEqual(self.hourly_cube_points, cube)
        cube = hourly_cube_points.extract(iris.TimePeriodConstraint(start_day=9))
        self.assertCML(cube, ('constrained_load', 'start_day_pts_1.cml'))
        cube = hourly_cube_points.extract(iris.TimePeriodConstraint(start_day=10))
        self.assertCML(cube, ('constrained_load', 'start_day_pts_2.cml'))
        cube = hourly_cube_points.extract(iris.TimePeriodConstraint(start_day=11))
        self.assertIsNone(cube)
        cube = hourly_cube_points.extract(iris.TimePeriodConstraint(end_day=9))
        self.assertIsNone(cube)
        cube = hourly_cube_points.extract(iris.TimePeriodConstraint(end_day=10))
        self.assertCML(cube, ('constrained_load', 'start_day_pts_2.cml'))
        cube = hourly_cube_points.extract(iris.TimePeriodConstraint(end_day=11))
        self.assertEqual(self.hourly_cube_points, cube)

        # Month and day
        cube = self.daily_cube_bounds.extract(iris.TimePeriodConstraint(start_month='Mar', start_day=5, end_month='Mar', end_day=10))
        self.assertCML(cube, ('constrained_load', 'startend_month_day_bnds.cml'))

    def test_ymd:
        
        pass
        pass
    def test_over_period_end:
        pass
    def test_period_combinations:
        pass
    def test_time_and_period_combinations:
        pass


if __name__ == "__main__":
    tests.main()
