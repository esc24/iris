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
Automatic concatenation of multiple cubes over one or more common and
already existing dimensions.

.. warning::
    Currently, the :func:`concatenate` routine will load the data payload
    of all cubes passed to it.

    This restriction will be relaxed in future revisions.

"""

from collections import namedtuple
from copy import deepcopy
import numpy
import operator

import iris.coords
import iris.cube
from iris.util import guess_coord_axis, array_equal


# Restrict the names imported from this namespace.
__all__ = ['concatenate']


#
# TODO:
#
#   * Deal with scalar coordinate promotion to a new dimension
#     e.g. promote scalar z coordinate in 2D cube (y:m, x:n) to
#     give the similar 3D cube (z:1, y:m, x:n). These two types
#     of cubes are one and the same, and as such should concatenate
#     together.
#
#   * Cope with auxiliary coordinate factories.
#
#   * Don't load the cube data payload.
#


class _CoordAndDims(namedtuple('CoordAndDims',
                               ['coord', 'dims'])):
    """
    Container for a coordinate and the associated data dimension/s
    spanned over a :class:`iris.cube.Cube`.

    Args:

    * coord:
        A :class:`iris.coords.DimCoord` or :class:`iris.coords.AuxCoord`
        coordinate instance.

    * dims:
        A tuple of the data dimesion/s spanned by the coordinate.

    """


class _CoordMetaData(namedtuple('CoordMetaData',
                                ['defn', 'points_dtype',
                                 'bounds_dtype', 'kwargs'])):
    """
    Container for the metadata that defines a dimension or auxiliary
    coordinate.

    Args:

    * defn:
        The :class:`iris.coords.CoordDefn` metadata that represents a
        coordinate.

    * points_dtype:
        The points data :class:`numpy.dtype` of an associated coordinate.

    * bounds_dtype:
        The bounds data :class:`numpy.dtype` of an associated coordinate.

    * kwargs:
        A dictionary of key/value pairs required to define a coordinate.

    """


class _Skeleton(namedtuple('Skeleton',
                           ['signature', 'data'])):
    """
    Basis of a source-cube, containing the associated coordinate metadata,
    coordinates and cube data payload.

    Args:

    * signature:
        The :class:`CoordSignature` of an associated source-cube.

    * data:
        The data payload of an associated :class:`iris.cube.Cube` source-cube.

    """


class _Extent(namedtuple('Extent',
                         ['min', 'max'])):
    """
    Container representing the limits of a one-dimensional extent/range.

    Args:

    * min:
        The minimum value of the extent.

    * max:
        The maximum value of the extent.

    """


class _CoordExtent(namedtuple('CoordExtent',
                              ['points', 'bounds'])):
    """
    Container representing the points and bounds extent of a one dimensional
    coordinate.

    Args:

    * points:
        The :class:`_Extent` of the coordinate point values.

    * bounds:
        A list containing the :class:`_Extent` of the coordinate lower
        bound and the upper bound. Defaults to None if no associated
        bounds exist for the coordinate.

    """


# Direction of dimension coordinate value order.
_CONSTANT = 0
_DECREASING = -1
_INCREASING = 1


def concatenate(cubes):
    """
    Concatenate the provided cubes over common existing dimensions.

    Args:

    * cubes:
        The :class:`iris.cube.CubeList` containing the :class:`iris.cube.Cube`s
        to be concatenated together.

    Returns:
        A :class:`iris.cube.CubeList` of concatenated :class:`iris.cube.Cube`s.

    .. warning::
        This routine will load your data payload!

    """
    proto_cubes_by_name = {}
    axis = None

    # Register each cube with its appropriate proto-cube.
    for cube in cubes:
        # Avoid deferred data/data manager issues, and load the cube data!!
        cube.data

        name = cube.standard_name
        proto_cubes = proto_cubes_by_name.setdefault(name, [])
        registered = False

        # Register cube with an existing proto-cube.
        for proto_cube in proto_cubes:
            registered = proto_cube.register(cube, axis)
            if registered:
                axis = proto_cube.axis
                break

        # Create a new proto-cube for an unregistered cube.
        if not registered:
            proto_cubes.append(ProtoCube(cube, axis))

    # Construct a concatenated cube from each of the proto-cubes.
    concatenated_cubes = iris.cube.CubeList()

    for name in sorted(proto_cubes_by_name):
        for proto_cube in proto_cubes_by_name[name]:
            # Construct the concatenated cube.
            concatenated_cubes.append(proto_cube.concatenate())

    # Perform concatenation until we've reached an equilibrium.
    count = len(concatenated_cubes)
    if count != 1 and count != len(cubes):
        concatenated_cubes = concatenate(concatenated_cubes)

    return concatenated_cubes


class CubeSignature(object):
    """
    Template for identifying a specific type of :class:`iris.cube.Cube` based
    on its metadata and coordinates.

    """
    def __init__(self, cube):
        """
        Represents the cube metadata and associated coordinate metadata that
        allows suitable cubes for concatenation to be identified.

        Args:

        * cube:
            The :class:`iris.cube.Cube` source-cube.

        """
        self.dim_coords = cube.dim_coords
        self.ndim = cube.ndim
        self.dim_metadata = []
        self.aux_coords_and_dims = []

        # Determine whether there are any anonymous cube dimensions.
        covered = set([cube.coord_dims(coord)[0] for coord in self.dim_coords])
        self.anonymous = covered != set(range(self.ndim))

        if not self.anonymous:
            self.defn = cube.metadata
            self.data_type = cube.data.dtype.name
            self.mdi = None

            if isinstance(cube.data, numpy.ma.core.MaskedArray):
                # Only set when we're dealing with a masked payload.
                self.mdi = cube.data.fill_value

            # Collate the dimension coordinate metadata.
            for coord in self.dim_coords:
                defn = coord._as_defn()
                points_dtype = coord.points.dtype
                bounds_dtype = coord.bounds.dtype if coord.bounds is not None \
                    else None

                if coord.points[0] == coord.points[-1]:
                    order = _CONSTANT
                elif coord.points[-1] > coord.points[0]:
                    order = _INCREASING
                else:
                    order = _DECREASING

                kwargs = dict(circular=coord.circular, order=order)
                metadata = _CoordMetaData(defn, points_dtype,
                                          bounds_dtype, kwargs)
                self.dim_metadata.append(metadata)

            # Collate the auxiliary coordinate metadata and scalar coordinates.
            self.scalar_coords = []

            # Coordinate axis ordering dictionary.
            axes = dict(T=0, Z=1, Y=2, X=3)
            # Coordinate sort function - by guessed coordinate axis, then
            # by coordinate definition, then by dimensions, in ascending order.
            key_func = lambda coord: (axes.get(guess_coord_axis(coord),
                                               len(axes) + 1),
                                      coord._as_defn(),
                                      cube.coord_dims(coord))

            self.aux_metadata = []
            for coord in sorted(cube.aux_coords, key=key_func):
                dims = cube.coord_dims(coord)
                if dims:
                    defn = coord._as_defn()
                    points_dtype = coord.points.dtype
                    bounds_dtype = coord.bounds.dtype \
                        if coord.bounds is not None else None
                    kwargs = dict(dims=dims)
                    # Factor the coordinate dimensional mapping into
                    # the metadata criterion.
                    metadata = _CoordMetaData(defn, points_dtype,
                                              bounds_dtype, kwargs)
                    self.aux_metadata.append(metadata)
                    coord_and_dims = _CoordAndDims(coord, tuple(dims))
                    self.aux_coords_and_dims.append(coord_and_dims)
                else:
                    self.scalar_coords.append(coord)

    def __eq__(self, other):
        result = NotImplemented

        if isinstance(other, CubeSignature):
            # Only concatenate with fully described cubes.
            if self.anonymous or other.anonymous:
                result = False
            else:
                result = self.defn == other.defn and \
                    self.data_type == other.data_type and \
                    self.mdi == other.mdi and \
                    self.dim_metadata == other.dim_metadata and \
                    self.aux_metadata == other.aux_metadata and \
                    self.scalar_coords == other.scalar_coords

        return result

    def __ne__(self, other):
        return not self == other


class CoordSignature(object):
    """
    Template for identifying a specific type of :class:`iris.cube.Cube` based
    on its coordinates.

    """
    def __init__(self, cube_signature):
        """
        Represents the coordinate metadata required to identify suitable
        non-overlapping :class:`iris.cube.Cube` source-cubes for concatenation
        over a common single dimension.

        Args:

        * cube_signature:
            The :class:`CubeSignature` that defines the source-cube.

        """
        # The nominated dimension of concatenation.
        self.axis = None
        # Controls whether axis can be negotiated.
        self.axis_lock = False
        self.dim_coords = cube_signature.dim_coords
        self.dim_order = [metadata.kwargs['order']
                          for metadata in cube_signature.dim_metadata]
        self.dim_extents = None
        self._cache = []
        self.aux_coords_and_dims = cube_signature.aux_coords_and_dims

    @property
    def axis_order(self):
        """
        Return the sort order for the nominated dimension of concatenation.

        """
        result = None

        if self.axis is not None:
            result = self.dim_order[self.axis]

        return result

    def calculate_extents(self):
        """
        Calculate the extents over the dimension coordinate points and bounds.

        """
        def _extents(coord, order):
            if order == _CONSTANT or order == _INCREASING:
                points = _Extent(coord.points[0], coord.points[-1])
                if coord.bounds is not None:
                    bounds = [_Extent(coord.bounds[0, 0], coord.bounds[-1, 0]),
                              _Extent(coord.bounds[0, 1], coord.bounds[-1, 1])]
                else:
                    bounds = None
            else:
                # Then the order must be decreasing ...
                points = _Extent(coord.points[-1], coord.points[0])
                if coord.bounds is not None:
                    bounds = [_Extent(coord.bounds[-1, 0], coord.bounds[0, 0]),
                              _Extent(coord.bounds[-1, 1], coord.bounds[0, 1])]
                else:
                    bounds = None

            return _CoordExtent(points, bounds)

        if self.axis is None:
            if not self._cache:
                # Populate the cache with the extents for each
                # dimension coordinate.
                for coord, order in zip(self.dim_coords, self.dim_order):
                    self._cache.append(_extents(coord, order))
        else:
            if self._cache:
                self.dim_extents = self._cache[self.axis]
            else:
                self.dim_extents = _extents(self.dim_coords[self.axis],
                                            self.dim_order[self.axis])

    def has_bounds(self):
        """
        Determine whether the nominated dimension of concatenation
        has a dimension coordinate with bounds.

        Returns:
            Boolean.

        """
        result = None

        if self.dim_extents is not None:
            result = self.dim_extents.bounds is not None

        return result

    def lock_axis(self, axis):
        """
        Attempt to lock the axis down to the given dimension.

        Args:

        * axis:
            The dimension of concatenation.

        """
        if axis is not None and not self.axis_lock:
            if axis < len(self.dim_coords):
                self.axis = axis
                self.axis_lock = True

    def reset_axis(self):
        """Revert the nominated dimension of concatenation."""

        self.axis = None
        self.axis_lock = False

    def _cmp(self, coord, other_coord):
        # Points comparison.
        eq = array_equal(coord.points, other_coord.points)

        # Bounds comparison.
        if eq:
            if coord.bounds is not None and other_coord.bounds is not None:
                result = array_equal(coord.bounds, other_coord.bounds)
            else:
                result = coord.bounds is None and other_coord.bounds is None
        else:
            # Ensure equal bounded status.
            result = (coord.bounds is None) == (other_coord.bounds is None)

        return eq, result

    def __eq__(self, other):
        result = NotImplemented

        if isinstance(other, CoordSignature):
            result = True
            candidate_axes = []

            # Compare dimension coordinates.
            for dim, coord in enumerate(self.dim_coords):
                eq, result = self._cmp(coord, other.dim_coords[dim])
                if not result:
                    break
                if not eq:
                    candidate_axes.append(dim)

            if result:
                if len(candidate_axes) != 1:
                    # Only permit one degree of dimensional freedom.
                    result = False
                else:
                    if self.axis is not None:
                        if self.axis != candidate_axes[0]:
                            result = False
                    else:
                        self.axis = candidate_axes[0]

            if result:
                # Inform the other signature of the nominated dimension
                # of concatenation.
                other.axis = self.axis

        return result

    def __ne__(self, other):
        return not self == other


class ProtoCube(object):
    """
    Framework for concatenating multiple source-cubes over one
    common dimension.

    """
    def __init__(self, cube, axis=None):
        """
        Create a new ProtoCube from the given cube and record the cube
        as a source-cube.

        Args:

        * cube:
            Source :class:`iris.cube.Cube` of the :class:`ProtoCube`.

        Kwargs:

        * axis:
            Seed the dimension of concatenation for the :class:`ProtoCube`
            rather than rely on negotiation with source-cubes.

        """
        # Cache the source-cube of this proto-cube.
        self._cube = cube

        # The cube signature is a combination of cube and coordinate
        # metadata that defines this proto-cube.
        self._cube_signature = CubeSignature(cube)

        # The coordinate signature allows suitable non-overlapping
        # source-cubes to be identified.
        self._coord_signature = CoordSignature(self._cube_signature)

        # Calculate the extents of the proto-cube dimension coordinates.
        self._coord_signature.calculate_extents()

        # Attempt to lock the axis, if appropriate ...
        self._coord_signature.lock_axis(axis)

        # The list of source-cubes relevant to this proto-cube.
        self._skeletons = []
        self._add_cube(cube, self._coord_signature)

    @property
    def axis(self):
        """Return the dimension of concatenation."""

        return self._coord_signature.axis

    def concatenate(self):
        """
        Concatenates all the source-cubes registered with the
        :class:`ProtoCube` over the nominated common dimension.

        Returns:
            The concatenated :class:`iris.cube.Cube`.

        """
        if len(self._skeletons) > 1:
            skeletons = self._skeletons
            order = self._coord_signature.axis_order
            cube_signature = self._cube_signature

            # Sequence the skeleton segments into the correct order
            # pending concatenation.
            key_func = lambda skeleton: skeleton.signature.dim_extents
            skeletons.sort(key=key_func, reverse=order == _DECREASING)

            # Concatenate the new dimension coordinate.
            dim_coords_and_dims = self._build_dim_coordinates()

            # Concatenate the new auxiliary coordinates.
            aux_coords_and_dims = self._build_aux_coordinates()

            # Concatenate the new data payload.
            data = self._build_data()

            # Build the new cube.
            kwargs = dict(zip(iris.cube.CubeMetadata._fields,
                              cube_signature.defn))
            cube = iris.cube.Cube(data,
                                  dim_coords_and_dims=dim_coords_and_dims,
                                  aux_coords_and_dims=aux_coords_and_dims,
                                  **kwargs)
        else:
            # There is nothing else to concatenate with the source-cube
            # of this proto-cube
            cube = self._cube

        return cube

    def register(self, cube, axis=None):
        """
        Determine whether the given source-cube is suitable for concatenation
        with this :class:`ProtoCube`.

        Args:

        * cube:
            The :class:`iris.cube.Cube` source-cube candidate for
            concatenation.

        Kwargs:

        * axis:
            Seed the dimension of concatenation for the :class:`ProtoCube`
            rather than rely on negotiation with source-cubes.

        Returns:
            Boolean.

        """
        # Attempt to lock the axis, if appropriate ...
        self._coord_signature.lock_axis(axis)

        # Check for compatible cube signatures.
        cube_signature = CubeSignature(cube)
        match = self._cube_signature == cube_signature

        # Check for compatible coodinate signatures.
        if match:
            coord_signature = CoordSignature(cube_signature)
            match = self._coord_signature == coord_signature

        # Check for compatible coordinate extents.
        if match:
            self._coord_signature.calculate_extents()
            coord_signature.calculate_extents()
            match = self._sequence(coord_signature.dim_extents)

        if match:
            # Register the cube as a source-cube for this proto-cube.
            self._add_cube(cube, coord_signature)
            # Prevent further axis negotiation.
            self._coord_signature.axis_lock = True
        else:
            if len(self._skeletons) == 1 and \
                    not self._coord_signature.axis_lock:
                # Ensure dimension of concatenation is renegotiated.
                self._coord_signature.reset_axis()

        return match

    def _add_cube(self, cube, coord_signature):
        """
        Add the given source-cube to the list of cubes
        suitable for concatenation with this :class:`ProtoCube`.

        Create and add the source-cube skeleton to the
        :class:`ProtoCube`.

        Args:

        * cube:
            The :class:`iris.cube.Cube` source-cube which
            is compatible for concatenation with this
            :class:`ProtoCube`.

        * coord_signature:
            The :class:`CoordSignature` of the associated
            given source-cube.

        """
        skeleton = _Skeleton(coord_signature, cube.data)
        self._skeletons.append(skeleton)

    def _build_aux_coordinates(self):
        """
        Generate the auxiliary coordinates with associated dimension/s
        mapping for the new concatenated cube.

        Returns:
            A list of auxiliary coordinates and dimension/s tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        axis = self._coord_signature.axis
        cube_signature = self._cube_signature

        aux_coords_and_dims = []

        # Generate all the auxiliary coordinates for the new concatenated cube.
        for i, (coord, dims) in enumerate(cube_signature.aux_coords_and_dims):
            # Check whether the coordinate spans the nominated
            # dimension of concatenation.
            if axis in dims:
                # Concatenate the points together.
                dim = dims.index(axis)
                points = [skeleton.signature.aux_coords_and_dims[i][0].points
                          for skeleton in skeletons]
                points = numpy.concatenate(tuple(points), axis=dim)

                # Concatenate the bounds together.
                bnds = None
                if coord.has_bounds():
                    bnds = [skeleton.signature.aux_coords_and_dims[i][0].bounds
                            for skeleton in skeletons]
                    bnds = numpy.concatenate(tuple(bnds), axis=dim)

                # Generate the associated coordinate metadata.
                defn = cube_signature.aux_metadata[i].defn
                kwargs = dict(zip(iris.coords.CoordDefn._fields, defn))

                # Build the concatenated coordinate.
                if isinstance(cube_signature.aux_coords_and_dims[i][0],
                              iris.coords.AuxCoord):
                    coord = iris.coords.AuxCoord(points, bounds=bnds,
                                                 **kwargs)
                else:
                    # Attempt to create a DimCoord, otherwise default to
                    # an AuxCoord on failure.
                    try:
                        coord = iris.coords.DimCoord(points, bounds=bnds,
                                                     **kwargs)
                    except ValueError:
                        coord = iris.coords.AuxCoord(points, bounds=bnds,
                                                     **kwargs)

            aux_coords_and_dims.append((deepcopy(coord), dims))

        # Generate all the scalar coordinates for the new concatenated cube.
        for coord in cube_signature.scalar_coords:
            aux_coords_and_dims.append((deepcopy(coord), ()))

        return aux_coords_and_dims

    def _build_data(self):
        """
        Generate the data payload for the new concatenated cube.

        Returns:
            The concatenated :class:`iris.cube.Cube` data payload.

        """
        skeletons = self._skeletons
        axis = self._coord_signature.axis
        data = [skeleton.data for skeleton in skeletons]

        if self._cube_signature.mdi is not None:
            # Preserve masked entries.
            data = numpy.ma.concatenate(tuple(data), axis=axis)
        else:
            data = numpy.concatenate(tuple(data), axis=axis)

        return data

    def _build_dim_coordinates(self):
        """
        Generate the dimension coordinates with associated dimension
        mapping for the new concatenated cube.

        Return:
            A list of dimension coordinate and dimension tuple pairs.

        """
        # Setup convenience hooks.
        skeletons = self._skeletons
        axis = self._coord_signature.axis
        defn = self._cube_signature.dim_metadata[axis].defn
        circular = self._cube_signature.dim_metadata[axis].kwargs['circular']

        # Concatenate the points together for the nominated dimension.
        points = [skeleton.signature.dim_coords[axis].points
                  for skeleton in skeletons]
        points = numpy.concatenate(tuple(points))

        # Concatenate the bounds together for the nominated dimension.
        bounds = None
        if self._coord_signature.has_bounds():
            bounds = [skeleton.signature.dim_coords[axis].bounds
                      for skeleton in skeletons]
            bounds = numpy.concatenate(tuple(bounds))

        # Populate the new dimension coordinate with the concatenated
        # points, bounds and associated metadata.
        kwargs = dict(zip(iris.coords.CoordDefn._fields, defn))
        kwargs['circular'] = circular
        dim_coord = iris.coords.DimCoord(points, bounds=bounds, **kwargs)

        # Generate all the dimension coordinates for the new concatenated cube.
        dim_coords_and_dims = []
        for dim, coord in enumerate(self._cube_signature.dim_coords):
            if dim == axis:
                dim_coords_and_dims.append((deepcopy(dim_coord), dim))
            else:
                dim_coords_and_dims.append((deepcopy(coord), dim))

        return dim_coords_and_dims

    def _sequence(self, extent):
        """
        Determine whether the given extent can be sequenced along with
        all the other extents from source-cubes already registered with
        this :class:`ProtoCube` into non-overlapping segments.

        Args:

        * extent:
            The :class:`_CoordExtent` of the candidate source-cube.

        Returns:
            Boolean.

        """
        result = True

        # Add the new extent to the current extents collection.
        dim_extents = [skeleton.signature.dim_extents
                       for skeleton in self._skeletons]
        dim_extents.append(extent)

        # Sort into the appropriate dimension order.
        order = self._coord_signature.axis_order
        dim_extents.sort(reverse=order == _DECREASING)

        # Determine the comparison operator.
        compare = operator.le if order == _DECREASING else operator.ge

        # Ensure that the extents don't overlap.
        if len(dim_extents) > 1:
            for i, extent in enumerate(dim_extents[1:]):
                # Check the points - must be strictly monotonic.
                if compare(dim_extents[i].points.max, extent.points.min):
                    result = False
                    break
                # Check the bounds - must be strictly monotonic.
                if extent.bounds is not None:
                    lower_bound_fail = compare(dim_extents[i].bounds[0].max,
                                               extent.bounds[0].min)
                    upper_bound_fail = compare(dim_extents[i].bounds[1].max,
                                               extent.bounds[1].min)
                    if lower_bound_fail or upper_bound_fail:
                        result = False
                        break

        return result
