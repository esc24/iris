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
Test pickling of Iris objects.

"""


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import pickle
import io


import iris


class TestPickle(tests.IrisTest):
    def pickle_then_unpickle(self, obj):
        """Returns a generator of ("cpickle protocol number", object) tuples.""" 
        for protocol in range(1 + pickle.HIGHEST_PROTOCOL):
            str_buffer = io.StringIO()
            pickle.dump(obj, str_buffer, protocol)
            
            # move the str_buffer back to the start and reconstruct
            str_buffer.seek(0)
            reconstructed_obj = pickle.load(str_buffer)
            
            yield protocol, reconstructed_obj
    
    @iris.tests.skip_data
    def test_cube_pickle(self):
        cube = iris.load_strict(tests.get_data_path(('PP', 'globClim1', 'theta.pp')))
        self.assertCML(cube, ('cube_io', 'pickling', 'theta.cml'), checksum=False)
        
        for _, recon_cube in self.pickle_then_unpickle(cube):
            self.assertNotEqual(recon_cube._data_manager, None)
            self.assertEqual(cube._data_manager, recon_cube._data_manager)
            self.assertCML(recon_cube, ('cube_io', 'pickling', 'theta.cml'), checksum=False)

    @iris.tests.skip_data                    
    def test_cubelist_pickle(self):
        cubelist = iris.load(tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog_subset.pp')))
        single_cube = cubelist[0]
                
        self.assertCML(cubelist, ('cube_io', 'pickling', 'cubelist.cml'))
        self.assertCML(single_cube, ('cube_io', 'pickling', 'single_cube.cml'))
        
        for _, reconstructed_cubelist in self.pickle_then_unpickle(cubelist):
            self.assertCML(reconstructed_cubelist, ('cube_io', 'pickling', 'cubelist.cml'))
            self.assertCML(reconstructed_cubelist[0], ('cube_io', 'pickling', 'single_cube.cml'))
            
            for cube_orig, cube_reconstruct in zip(cubelist, reconstructed_cubelist):
                self.assertArrayEqual(cube_orig.data, cube_reconstruct.data)
                self.assertEqual(cube_orig, cube_reconstruct)
            
    def test_picking_equality_misc(self):
        items_to_test = [
                        iris.unit.Unit("hours since 2007-01-15 12:06:00", calendar=iris.unit.CALENDAR_STANDARD),
                        iris.unit.as_unit('1'),
                        iris.unit.as_unit('meters'),
                        iris.unit.as_unit('no-unit'),
                        iris.unit.as_unit('unknown')
                        ]

        for orig_item in items_to_test:
            for protocol, reconstructed_item in self.pickle_then_unpickle(orig_item):
                fail_msg = ('Items are different after pickling at protocol %s.'
                           '\nOrig item: %r\nNew item: %r' % (protocol, orig_item, reconstructed_item)
                            )
                self.assertEqual(orig_item, reconstructed_item, fail_msg)


if __name__ == "__main__":
    tests.main()
