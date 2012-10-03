What's new in Iris 1.0
**********************

:Release: 1.0.0
:Date: dd mmm, 2012

This document explains the new/changed features of Iris in version 1.0.

With the release of Iris 1.0, we have broadly completed the transition
to the CF data model. Future releases will see additional features added
on top of this foundation, and some of the remaining details filled out
(e.g. more CF coordinate systems).


The role of 1.x
===============

The 1.x series of releases is intended to provide a relatively stable,
backwards-compatible platform based on the CF-netCDF data model, upon
which long-lived services can be built.

Iris 1.0 targets the data model implicit in CF-netCDF 1.5. This will be
extended to cover the new features of CF-netCDF 1.6 (e.g. discrete
sampling geometries) and any subsequent versions which maintain
backwards compatibility. Similarly, as the efforts of the CF community
to formalise their data model reach maturity, they will be included
in Iris where significant backwards-compatibility can be maintained.


Iris 1.0 features
=================

A summary of the main features added with version 1.0:

* Hybrid-pressure vertical coordinates, and the ability to load from GRIB.
* Initial support for CF-style coordinate systems.
* Load data from NIMROD files.
* Automatic and manual use of Cynthia Brewer colour palettes.
* Ensures netCDF files are properly closed.
* The ability to bypass merging when loading data.
* Save netCDF files with an unlimited dimension.
* A more explicit set of load functions, which also allow the automatic
  cube merging to be bypassed as a last resort.

Incompatible changes
--------------------
* The "source" and "history" metadata are now represented as Cube
  attributes, where previously they used coordinates.
* Cube.coord_dims() now returns a tuple instead of a list.

Deprecations
------------
* The methods :meth:`iris.coords.Coord.cos()` and
  :meth:`iris.coords.Coord.sin()` have been deprecated.
* The :func:`iris.load_strict()` function has been deprecated. Code
  should now use the :func:`iris.load_cube()` and
  :func:`iris.load_cubes()` functions instead.


CF-netCDF coordinate systems
============================

The coordinate systems in Iris are now defined by the CF-netCDF
`grid mappings <http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/apf.html>`_.
As of Iris 1.0 a subset of the CF-netCDF coordinate systems are
supported, but this will be expanded in subsequent versions. Adding
this code is a relatively simple, incremental process - it would make a
good task to tackle for users interested in getting involved in
contributing to the project.

The coordinate systems available in Iris 1.0 and their corresponding
Iris classes are:

================================================================================================== =========================================
CF name                                                                                            Iris class
================================================================================================== =========================================
`Latitude-longitude <http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/apf.html#idp7779520>`_  :class:`~iris.coord_systems.GeogCS`
`Rotated pole <http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/apf.html#idp7844592>`_        :class:`~iris.coord_systems.RotatedGeogCS`
`Transverse Mercator <http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/apf.html#idp7872672>`_ :class:`~iris.coord_systems.TransverseMercator`
================================================================================================== =========================================

For convenience, Iris also includes the :class:`~iris.coord_systems.OSGB`
class which provides a simple way to create the transverse Mercator
coordinate system used by the British
`Ordnance Survey <http://www.ordnancesurvey.co.uk/>`_.


Hybrid-pressure
===============

With the introduction of the :class:`~iris.aux_factory.HybridPressureFactory`
class, it is now possible to represent data expressed on a
hybrid-pressure vertical coordinate, as defined by the second variant in
`Appendix D <http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/apd.html#idp7406304>`_.
A hybrid-pressure factory is created with references to the coordinates
which provide the components of the hybrid coordinate ("ap" and "b") and
the surface pressure. In return, it provides a virtual "pressure"
coordinate whose values are derived from the given components.

This facility is utilised by the GRIB2 loader to automatically provide
the derived "pressure" coordinate for certain data [#f1]_ from the
`ECMWF <http://www.ecmwf.int/>`_.

.. [#f1] Where the level type is either 105 or 119, and where the
         surface pressure has an ECMWF paramId of
         `152 <http://www.ecmwf.int/publications/manuals/d/gribapi/param/detail/format=grib2/pid=152/>`_.


NetCDF
======

When saving a Cube to a netCDF file, Iris will now define the outermost
dimension as an unlimited/record dimension. In combination with the
:meth:`iris.cube.Cube.transpose` method, this allows any dimension to
take the role of the unlimited/record dimension.

For example, a Cube with the structure::

    <iris 'Cube' of air_potential_temperature (time: 6; model_level_number: 70; grid_latitude: 100; grid_longitude: 100)>

would result in a netCDF file whose CDL definition would include::

    dimensions:
            time = UNLIMITED ; // (6 currently)
            model_level_number = 70 ;
            grid_latitude = 100 ;
            grid_longitude = 100 ;

Also, Iris will now ensure that netCDF files are properly closed when
they are no longer in use. Previously this could cause problems when
dealing with large numbers of netCDF files, or in long running
processes.


Brewer colour palettes
======================

Iris includes a selection of carefully designed colour palettes produced
by Cynthia Brewer. Unless an explicit palette is selected, the plotting 
routines in :mod:`iris.plot` (and hence, :mod:`iris.quickplot` also),
will attempt to choose an appropriate Brewer palette based on the Cube's
standard name.

For example, a Cube of `stratiform_precipitation`
will default to a sequential white-blue palette, but a Cube of
`air_temperature_anomaly` will default to a diverging, red-white-blue
palette.

This behaviour is controlled by the `keyword` and `std_name` tags in
the palette definition files in `iris/etc/palette/...`. Further
contributions to these tag values are very welcome.

The :mod:`iris.palette` module, as used by :mod:`iris.plot`, also
registers the Brewer colour palettes with matplotlib, so they are
explicitly selectable via the :func:`matplotlib.pyplot.set_cmap`
function. For example::

    import iris.palette
    import matplotlib.pyplot as plt
    import numpy as np
    plt.contourf(np.random.randn(10, 10))
    plt.set_cmap('RdBu_11')
    plt.show()

Citations
---------
When the Iris plotting routines detect the selection of a Brewer palette
they also add an appropriate citation to the plot. In other
circumstances, citations can easily be explicitly added using the
:func:`iris.plot.citation` function.

To include a reference in a journal article or report please refer to
`section 5 <http://www.personal.psu.edu/cab38/ColorBrewer/ColorBrewer_updates.html>`_
in the citation guidance provided by Cynthia Brewer.


Metadata attributes
===================

Iris now stores "source" and "history" metadata in Cube attributes.
For example::

    >>> print iris.tests.stock.global_pp()
    air_temperature                     (latitude: 73; longitude: 96)
         ...
         Attributes:
              ...
              source: Data from Met Office Unified Model
         ...

Where previously it would have appeared as::

    air_temperature                     (latitude: 73; longitude: 96)
         ...
         Scalar coordinates:
              ...
              source: Data from Met Office Unified Model
         ...

.. note:: This change breaks backwards compatibility with Iris 0.9. But
    if it is desirable to have the "source" metadata expressed as a
    coordinate then it can be done with the following pattern::

        src = cube.attributes.pop('source')
        src_coord = iris.coords.AuxCoord(src, long_name='source')
        cube.add_aux_coord(src_coord)


New loading functions
=====================

The main functions for loading cubes are now:
  - :func:`iris.load()`
  - :func:`iris.load_cube()`
  - :func:`iris.load_cubes()`

These provide convenient cube loading suitable for both interactive
(:func:`iris.load()`) and scripted (:func:`iris.load_cube()`,
:func:`iris.load_cubes()`) usage.

In addition, :func:`iris.load_raw()` has been provided as a last resort
for situations where the automatic cube merging is not appropriate.
However, if you find you need to use this function we would encourage
you to contact the Iris developers so we can see if a fix can be made
to the cube merge algorithm.

The :func:`iris.load_strict()` function has been deprecated. Code should
now use the :func:`iris.load_cube()` and :func:`iris.load_cubes()`
functions instead.


Other changes
=============
* Cube summaries are now more readable when the scalar coordinates
  contain bounds.
* Iris can now load NIMROD files.
* The ability to bypass merging when loading data.
* The methods `Coord.cos()` and `Coord.sin()` have been deprecated.
