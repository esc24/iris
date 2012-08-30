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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os

import numpy

import iris
import iris.coords
import iris.coord_systems
import iris.unit
import iris.tests.pp as pp
import iris.util
import stock


def itab_callback(cube, field, filename):
    cube.add_aux_coord(iris.coords.AuxCoord([field.lbrel], long_name='MOUMHeaderReleaseNumber', units='no_unit')) 
    cube.add_aux_coord(iris.coords.AuxCoord([field.lbexp], long_name='ExperimentNumber(ITAB)', units='no_unit')) 


@iris.tests.skip_data
class TestPPSave(tests.IrisTest, pp.PPTest):
    def test_pp_save_rules(self):
        # Test pp save rules without user rules.

        #read
        in_filename = tests.get_data_path(('PP', 'simple_pp', 'global.pp'))
        cubes = iris.load(in_filename, callback=itab_callback)

        reference_txt_path = tests.get_result_path(('cube_to_pp', 'simple.txt'))
        with self.cube_save_test(reference_txt_path, reference_cubes=cubes) as temp_pp_path:
            iris.save(cubes, temp_pp_path)

    def test_user_pp_save_rules(self):
        # Test pp save rules with user rules.
        
        #create a user rules file
        user_rules_filename = iris.util.create_temp_filename(suffix='.txt')
        try:
            with open(user_rules_filename, "wt") as user_rules_file:
                user_rules_file.write("IF\ncm.standard_name == 'air_temperature'\nTHEN\npp.lbuser[3] = 9222")
            iris.fileformats.pp.add_save_rules(user_rules_filename)
            try:
                #read pp
                in_filename = tests.get_data_path(('PP', 'simple_pp', 'global.pp'))
                cubes = iris.load(in_filename, callback=itab_callback)

                reference_txt_path = tests.get_result_path(('cube_to_pp', 'user_rules.txt'))
                with self.cube_save_test(reference_txt_path, reference_cubes=cubes) as temp_pp_path:
                    iris.save(cubes, temp_pp_path)

            finally:
                iris.fileformats.pp.reset_save_rules()
        finally:
            os.remove(user_rules_filename)

    def test_pp_append_singles(self):
        # Test pp append saving - single cubes.
        
        # load 2 arrays of >2D cubes
        cube = stock.simple_pp()
        
        reference_txt_path = tests.get_result_path(('cube_to_pp', 'append_single.txt'))
        with self.cube_save_test(reference_txt_path, reference_cubes=[cube, cube]) as temp_pp_path:
            iris.save(cube, temp_pp_path)                # Create file
            iris.save(cube, temp_pp_path, append=True)   # Append to file

        reference_txt_path = tests.get_result_path(('cube_to_pp', 'replace_single.txt'))
        with self.cube_save_test(reference_txt_path, reference_cubes=cube) as temp_pp_path:
            iris.save(cube, temp_pp_path)                # Create file
            iris.save(cube, temp_pp_path)                # Replace file

    def test_pp_append_lists(self):
        # Test pp append saving - lists of cubes.
        
        # Locate the first 4 files from the analysis dataset
        names = ['2008120%d1200__qwqu12ff.initanl.pp' % i for i in range(1, 5)]
        prefix = ['PP', 'trui', 'air_temp_init']
        paths = [tests.get_data_path(prefix + [name]) for name in names]

        # Grab the first two levels from each file
        cubes = [iris.load_strict(path, callback=itab_callback) for path in paths]
        cubes = [cube[:2] for cube in cubes]

        reference_txt_path = tests.get_result_path(('cube_to_pp', 'append_multi.txt'))
        with self.cube_save_test(reference_txt_path, reference_cubes=cubes) as temp_pp_path:
            iris.save(cubes[:2], temp_pp_path)
            iris.save(cubes[2:], temp_pp_path, append=True)

        reference_txt_path = tests.get_result_path(('cube_to_pp', 'replace_multi.txt'))
        with self.cube_save_test(reference_txt_path, reference_cubes=cubes[2:]) as temp_pp_path:
            iris.save(cubes[:2], temp_pp_path)
            iris.save(cubes[2:], temp_pp_path)

    def add_coords_to_cube_and_test(self, coord1, coord2):
        # a wrapper for creating arbitrary 2d cross-sections and run pp-saving tests
        dataarray = numpy.arange(16, dtype='>f4').reshape(4,4)
        cm = iris.cube.Cube(data=dataarray)

        cm.add_dim_coord(coord1, 0)
        cm.add_dim_coord(coord2, 1)
        cm.assert_valid()
        
        # TODO: This is the desired line of code...
        # reference_txt_path = tests.get_result_path(('cube_to_pp', '%s.%s.pp.txt' % (coord1.name(), coord2.name())))
        # ...but this is required during the CF change, to maintain the original filename.
        coord1_name = coord1.name().replace("air_", "")
        coord2_name = coord2.name().replace("air_", "") 
        reference_txt_path = tests.get_result_path(('cube_to_pp', '%s.%s.pp.txt' % (coord1_name, coord2_name))) 

        # test with name
        with self.cube_save_test(reference_txt_path, reference_cubes=cm, 
                field_coords=[coord1.name(), coord2.name()]) as temp_pp_path:
            iris.save(cm, temp_pp_path, field_coords=[coord1.name(), coord2.name()])
        # test with coord
        with self.cube_save_test(reference_txt_path, reference_cubes=cm, 
                field_coords=[coord1, coord2]) as temp_pp_path:
            iris.save(cm, temp_pp_path, field_coords=[coord1, coord2])

    def test_non_standard_cross_sections(self):
        #ticket #1037, the five variants being dealt with are
        #    'pressure.latitude',
        #    'depth.latitude',
        #    'eta.latitude',
        #    'pressure.time',
        #    'depth.time',

        f = fakePPEnvironment()

        self.add_coords_to_cube_and_test(
            iris.coords.DimCoord(f.z, long_name='air_pressure', units='hPa', bounds=f.z_bounds),
            iris.coords.DimCoord(f.y, standard_name='latitude', units='degrees', bounds=f.y_bounds, coord_system=f.horiz_coord_system()))
            
        self.add_coords_to_cube_and_test(
            iris.coords.DimCoord(f.z, long_name='depth', units='m', bounds=f.z_bounds),
            iris.coords.DimCoord(f.y, standard_name='latitude', units='degrees', bounds=f.y_bounds, coord_system=f.horiz_coord_system()))
            
        self.add_coords_to_cube_and_test(
            iris.coords.DimCoord(f.z, long_name='eta', units='1', bounds=f.z_bounds),
            iris.coords.DimCoord(f.y, standard_name='latitude', units='degrees', bounds=f.y_bounds, coord_system=f.horiz_coord_system()))
            
        self.add_coords_to_cube_and_test(
            iris.coords.DimCoord(f.z, long_name='air_pressure', units='hPa', bounds=f.z_bounds),
            iris.coords.DimCoord(f.y, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.y_bounds))
            
        self.add_coords_to_cube_and_test(
            iris.coords.DimCoord(f.z, standard_name='depth', units='m', bounds=f.z_bounds),
            iris.coords.DimCoord(f.y, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.y_bounds))

            
class fakePPEnvironment(object):
    ''' fake a minimal PP environment for use in cross-section coords, as in PP save rules '''
    y = [1,2,3,4]
    z = [111,222,333,444]
    y_bounds = [[0.9,1.1], [1.9,2.1], [2.9,3.1], [3.9,4.1]]
    z_bounds = [[110.9,111.1], [221.9,222.1], [332.9,333.1], [443.9,444.1]]

    def horiz_coord_system(self):
        """Return a LatLonCS for this PPField.

        Returns:
            A LatLonCS with the appropriate earth shape, meridian and pole position.
        """
        return iris.coord_systems.LatLonCS(
                   iris.coord_systems.SpheroidDatum("spherical", 6371229.0, 
                        flattening=0.0, units=iris.unit.Unit('m')),
                   iris.coord_systems.PrimeMeridian(label="Greenwich", value=0.0),
                   iris.coord_systems.GeoPosition(90.0, 0.0), 0.0)


@iris.tests.skip_data
class TestPPSaveRules(tests.IrisTest):  
    def lbproc_from_pp(self, filename):
        # Gets the lbproc field from the ppfile
        pp_file = iris.fileformats.pp.load(filename)
        field = pp_file.next()
        return field.lbproc

    def test_pp_save_rules(self):
        # Test single process flags
        for _, process_desc in iris.fileformats.pp.LBPROC_PAIRS[1:]:
            # Get basic cube and set process flag manually
            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = (process_desc,)
    
            # Save cube to pp
            temp_filename = iris.util.create_temp_filename(".pp")
            iris.save(ll_cube, temp_filename)
     
            # Check the lbproc is what we expect
            self.assertEquals(self.lbproc_from_pp(temp_filename), iris.fileformats.pp.lbproc_map[process_desc])

            os.remove(temp_filename)

        # Test mutiple process flags
        multiple_bit_values = ((128, 64), (4096, 1024), (8192, 1024))
        
        # Maps lbproc value to the process flags that should be created
        multiple_map = {sum(bits) : [iris.fileformats.pp.lbproc_map[bit] for bit in bits] for bits in multiple_bit_values}

        for lbproc, descriptions in multiple_map.iteritems():
            ll_cube = stock.lat_lon_cube()
            ll_cube.attributes["ukmo__process_flags"] = descriptions
            
            # Save cube to pp
            temp_filename = iris.util.create_temp_filename(".pp")
            iris.save(ll_cube, temp_filename)
            
            # Check the lbproc is what we expect
            self.assertEquals(self.lbproc_from_pp(temp_filename), lbproc)

            os.remove(temp_filename)


if __name__ == "__main__":
    tests.main()
