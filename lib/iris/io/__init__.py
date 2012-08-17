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
Provides an interface to manage URI scheme support in iris.

"""

import glob
import os.path
import types
import re
import warnings
import collections

import iris.fileformats
import iris.fileformats.dot
import iris.cube
import iris.exceptions
           

NO_CUBE = 'NOCUBE'
"""Used by callbacks to specify that the given cube should not be loaded."""
CALLBACK_DEPRECATION_MSG = "Callback functions with a return value are deprecated."


# Saving routines, indexed by file extension. 
class SaversDict(dict):
    """A dictionary that can only have string keys with no overlap."""
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("key is not a string")
        if key in list(self.keys()):
            raise ValueError("A saver already exists for", key)
        for k in list(self.keys()):
            if k.endswith(key) or key.endswith(k):
                raise ValueError("key %s conflicts with existing key %s" % (key, k))
        dict.__setitem__(self, key, value)


_savers = SaversDict()


def select_data_path(resources_subdir, rel_path):
    """
    Given a resource subdirectory and the resource file's relative path return
    the fully qualified path of a resource.
    
    The function checks to see whether :const:`iris.config.MASTER_DATA_REPOSITORY`
    exists, in which case it will check that each file also exists
    on :const:`iris.config.DATA_REPOSITORY`, providing a warning and a printed
    message indicating how to resolve the divergent folders. The resultant path
    will always use the :const:`iris.config.DATA_REPOSITORY` configuration value.
    
    Args:
    
    * resources_subdir
        The name of the subdirectory found in :const:`iris.config.DATA_REPOSITORY`
    * rel_path
        The tuple representing the relative path from the resources_subdir of the desired resource.
           
    """
    if iris.config.DATA_REPOSITORY is None:
        raise Exception('No data repository has been configured.')
    MASTER_REPO = iris.config.MASTER_DATA_REPOSITORY
    DATA_REPO = os.path.join(iris.config.DATA_REPOSITORY, resources_subdir)
    
    path_on_slave = os.path.join(DATA_REPO, *rel_path) 
    
    if MASTER_REPO and os.path.isdir(MASTER_REPO):
        path_on_master = os.path.join(MASTER_REPO, *rel_path)
        master_files = set((fname.replace(MASTER_REPO + os.sep, '', 1) for fname in glob.iglob(path_on_master)))
        slave_files = set((fname.replace(DATA_REPO + os.sep, '', 1) for fname in glob.iglob(path_on_slave)))
        
        if slave_files != master_files:
            all_files = slave_files | master_files
            
            for file_in_master_not_in_slave in (all_files - slave_files):
                master_file = os.path.join(MASTER_REPO, file_in_master_not_in_slave)
                slave_file = os.path.join(DATA_REPO, file_in_master_not_in_slave)
                print('; File exists at %s which does not exist at %s' % (master_file, slave_file))
                if DATA_REPO.startswith(MASTER_REPO):
                    print('ln -s %s %s' % (master_file, slave_file))
                else:
                    print('cp %s %s' % (master_file, slave_file))
                
            for file_in_slave_not_in_master in (all_files - master_files):
                master_file = os.path.join(MASTER_REPO, file_in_slave_not_in_master)
                slave_file = os.path.join(DATA_REPO, file_in_slave_not_in_master)
                print('; File exists at %s which does not exist at %s' % (slave_file, master_file))
                print('rm -rf %s' % os.path.join(DATA_REPO, file_in_slave_not_in_master))

    return path_on_slave


def run_callback(callback, cube, field, filename):
    """
    Runs the callback mechanism given the appropriate arguments.
    
    Args:
    
    * callback:
        A function to add metadata from the originating field and/or URI which obeys the following rules:
            1. Function signature must be: ``(cube, field, filename)``
            2. Must not return any value - any alterations to the cube must be made by reference
            3. If the cube is to be rejected the callback must raise an :class:`iris.exceptions.IgnoreCubeException`
   
    .. note:: 
        It is possible that this function returns None for certain callbacks, the caller of this 
        function should handle this case.
        
    """
    #call the custom uri cm func, if provided, for every loaded cube
    if callback is None:
        return cube
    
    try:
        result = callback(cube, field, filename) #  Callback can make changes to cube by reference
    except iris.exceptions.IgnoreCubeException:
        return None
    else: 
        if result is not None:
            #raise TypeError("Callback functions must have no return value.") # no deprecation support method
            
            if isinstance(result, iris.cube.Cube):
                # no-op
                result = result
            elif result == NO_CUBE:
                result = None
            else: # Invalid return type, raise exception
                raise TypeError("Callback function returned an unhandled data type.")
            
            # Warn the user that callbacks that return something are deprecated
            warnings.warn(CALLBACK_DEPRECATION_MSG)
            return result
            
        else:
            return cube


def decode_uri(uri, default='file'):
    r'''
    Decodes a single URI into scheme and scheme-specific parts.
    
    In addition to well-formed URIs, it also supports bare file paths.
    Both Windows and UNIX style paths are accepted.

    .. testsetup::
    
        from iris.io import *

    Examples:
        >>> from iris.io import decode_uri
        >>> print decode_uri('http://www.thing.com:8080/resource?id=a:b')
        ('http', '//www.thing.com:8080/resource?id=a:b')
        
        >>> print decode_uri('file:///data/local/dataZoo/...')
        ('file', '///data/local/dataZoo/...')
        
        >>> print decode_uri('/data/local/dataZoo/...')
        ('file', '/data/local/dataZoo/...')
        
        >>> print decode_uri('file:///C:\data\local\dataZoo\...')
        ('file', '///C:\\data\\local\\dataZoo\\...')
        
        >>> print decode_uri('C:\data\local\dataZoo\...')
        ('file', 'C:\\data\\local\\dataZoo\\...')
        
        >>> print decode_uri('dataZoo/...')
        ('file', 'dataZoo/...')

    '''
    # Catch bare UNIX and Windows paths
    i = uri.find(':')
    if i == -1 or re.match('[a-zA-Z]:', uri):
        scheme = default
        part = uri
    else:
        scheme = uri[:i]
        part = uri[i + 1:]

    return scheme, part


def load_files(filenames, callback):
    """
    Takes a list of filenames which may also be globs, and optionally a callback function, and returns a generator of Cubes from the given files.
    
    .. note:: 
        Typically, this function should not be called directly; instead, the intended interface for loading is :func:`iris.load`.
    
    """
    # Remove any hostname component - currently unused
    filenames = [os.path.expanduser(fn[2:] if fn.startswith('//') else fn) for fn in filenames]
    
    # Try to expand all filenames as globs       
    glob_expanded = {fn : sorted(glob.glob(fn)) for fn in filenames}
    
    # If any of the filenames or globs expanded to an empty list then raise an error
    if not all(glob_expanded.values()):
        raise IOError("One or more of the files specified did not exist %s." % 
        ["%s expanded to %s" % (pattern, expanded if expanded else "empty") for pattern, expanded in glob_expanded.items()])
    
    # Create default dict mapping iris format handler to its associated filenames
    handler_map = collections.defaultdict(list)
    for fn in sum([x for x in glob_expanded.values()], []):
        with open(fn, 'rb') as fh:         
            handling_format_spec = iris.fileformats.FORMAT_AGENT.get_spec(os.path.basename(fn), fh)
            handler_map[handling_format_spec].append(fn)
    
    # Call each iris format handler with the approriate filenames
    for handling_format_spec, fnames in handler_map.items():
        for cube in handling_format_spec.handler(fnames, callback):
            yield cube


def _check_init_savers():
    # TODO: Raise a ticket to resolve the cyclic import error that requires
    # us to initialise this on first use. Probably merge io and fileformats.
    if "pp" not in _savers:
        _savers.update({"pp": iris.fileformats.pp.save,
                        "nc": iris.fileformats.netcdf.save,
                        "dot": iris.fileformats.dot.save,
                        "dotpng": iris.fileformats.dot.save_png,
                        "grib2": iris.fileformats.grib.save_grib2})


def add_saver(file_extension, new_saver):
    """
    Add a custom saver to the Iris session.

    Args:

        * file_extension - A string such as "pp" or "my_format".
        * new_saver      - A function of the form ``my_saver(cube, target)``.
        
    See also :func:`iris.io.save`

    """
    # Make sure it's a func with 2+ args
    if not hasattr(new_saver, "__call__") or new_saver.__code__.co_argcount < 2:
        raise ValueError("Saver routines must be callable with 2+ arguments.")
    
    # Try to add this saver. Invalid keys will be rejected.
    _savers[file_extension] = new_saver


def find_saver(filespec):
    """
    Find the saver function appropriate to the given filename or extension.

    Args:

        * filespec - A string such as "my_file.pp" or "PP".

    Returns:
        A save function or None.
        Save functions can be passed to :func:`iris.io.save`.
    
    """
    _check_init_savers()
    matches = [ext for ext in _savers if filespec.lower().endswith('.' + ext) or
                                         filespec.lower() == ext]
    # Multiple matches could occur if one of the savers included a '.':
    #   e.g. _savers = {'.dot.png': dot_png_saver, '.png': png_saver}
    if len(matches) > 1:
        fmt = "Multiple savers found for %r: %s"
        matches = ', '.join(map(repr, matches))
        raise ValueError(fmt % (filespec, matches))
    return _savers[matches[0]] if matches else None 


def save(source, target, saver=None, **kwargs):
    """
    Save one or more Cubes to file (or other writable).
    
    A custom saver can be provided, or Iris can select one based on filename.

    Args:

        * source    - A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or sequence of cubes.
        * target    - A filename (or writable, depending on file format).
                      When given a filename or file, Iris can determine the file format.

    Kwargs:

        * saver     - Optional. Specifies the save function to use.
                      If omitted, Iris will attempt to determine the format.
    
                      This keyword can be used to implement a custom save format (see below).
                      Function form must be: ``my_saver(cube, target)`` plus any custom keywords.
                      It is assumed that a saver will accept an ``append`` keyword if it's file format
                      can handle multiple cubes. See also :func:`iris.io.add_saver`.

    All other keywords are passed through to the saver function.

    Examples::

        # Save a cube to PP
        iris.io.save(my_cube, "myfile.pp")
        iris.io.save(my_cube_list, "myfile.pp", append=True)

        # Save a cube to a custom file format and provide a format-specific argument.
        # The walk keyword is passed through to my_spam_format.save.
        import my_spam_format
        iris.io.save(my_cube, "my_file.spam", saver=my_spam_format.save, walk="silly")

        # Add a custom file format to the Iris session and save a cube list.
        # When saving a cube list, Iris passes an append keyword to the saver.
        iris.io.add_saver(".spam", my_spam_format.save)            
        iris.io.save(my_cube_list, "myfile.spam", walk="silly")

        # Get help on the PP saver, for example to see it's accepted keywords.
        help(iris.io.find_saver("pp"))
        
        # Create and display a PNG image of a DOT graph representation of a cube.
        iris.io.save(my_cube, "my_file.dotpng", launch=True)

        # Save a cube to netCDF, defaults to NETCDF4 file format
        iris.io.save(my_cube, "myfile.nc")

        # Save a cube to netCDF using NETCDF3 file format
        iris.io.save(my_cube, "myfile.nc", netcdf_format="NETCDF3_CLASSIC")

    """ 
    # Determine format from filename
    if isinstance(target, str) and saver is None:
        saver = find_saver(target)
    elif isinstance(target, types.FileType):
        saver = find_saver(target.name)
    if saver is None:
        raise ValueError("Cannot save; no saver")
    
    # Single cube?
    if isinstance(source, iris.cube.Cube):
        saver(source, target, **kwargs)
        
    # CubeList or sequence of cubes?
    elif isinstance(source, iris.cube.CubeList) or \
       (isinstance(source, (list,tuple)) and all([type(i)==iris.cube.Cube for i in source])):
        # Make sure the saver accepts an append keyword
        if not "append" in saver.__code__.co_varnames:
            raise ValueError("Cannot append cubes using saver function '%s' in '%s'" % \
                             (saver.__code__.co_name, saver.__code__.co_filename))
        # Force append=True for the tail cubes. Don't modify the incoming kwargs.
        kwargs = kwargs.copy()
        for i, cube in enumerate(source):
            if i != 0:
                kwargs['append'] = True
            saver(cube, target, **kwargs)
    else:
        raise ValueError("Cannot save; non Cube found in source")



