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
Provides objects for building up expressions useful for pattern matching.

"""
import collections
import operator

import numpy as np

import iris.exceptions


class Constraint(object):
    """
    Constraints are the mechanism by which cubes can be pattern matched and
    filtered according to specific criteria.

    Once a constraint has been defined, it can be applied to cubes using the
    :meth:`Constraint.extract` method.

    """
    def __init__(self, name=None, cube_func=None, coord_values=None, **kwargs):
        """
        Creates a new instance of a Constraint which can be used for filtering
        cube loading or cube list extraction.

        Args:

        * name:   string or None
            If a string, it is used as the name to match against Cube.name().
        * cube_func:   callable or None
            If a callable, it must accept a Cube as its first and only argument
            and return either True or False.
        * coord_values:   dict or None
            If a dict, it must map coordinate name to the condition on the
            associated coordinate.
        * `**kwargs`:
            The remaining keyword arguments are converted to coordinate
            constraints. The name of the argument gives the name of a
            coordinate, and the value of the argument is the condition to meet
            on that coordinate::

                Constraint(model_level_number=10)

            Coordinate level constraints can be of several types:

            * **string, int or float** - the value of the coordinate to match.
              e.g. ``model_level_number=10``

            * **list of values** - the possible values that the coordinate may
              have to match. e.g. ``model_level_number=[10, 12]``

            * **callable** - a function which accepts a
              :class:`iris.coords.Cell` instance as its first and only argument
              returning True or False if the value of the Cell is desired.
              e.g. ``model_level_number=lambda cell: 5 < cell < 10``

        The :ref:`user guide <loading_iris_cubes>` covers cube much of
        constraining in detail, however an example which uses all of the
        features of this class is given here for completeness::

            Constraint(name='air_potential_temperature',
                       cube_func=lambda cube: cube.units == 'kelvin',
                       coord_values={'latitude':lambda cell: 0 < cell < 90},
                       model_level_number=[10, 12])
                       & Constraint(ensemble_member=2)

        Constraint filtering is performed at the cell level.
        For further details on how cell comparisons are performed see
        :class:`iris.coords.Cell`.

        """
        if not (name is None or isinstance(name, basestring)):
            raise TypeError('name must be None or string, got %r' % name)
        if not (cube_func is None or callable(cube_func)):
            raise TypeError('cube_func must be None or callable, got %r'
                            % cube_func)
        if not (coord_values is None or isinstance(coord_values,
                                                   collections.Mapping)):
            raise TypeError('coord_values must be None or a '
                            'collections.Mapping, got %r' % coord_values)

        coord_values = coord_values or {}
        duplicate_keys = coord_values.viewkeys() & kwargs.viewkeys()
        if duplicate_keys:
            raise ValueError('Duplicate coordinate conditions specified for: '
                             '%s' % list(duplicate_keys))

        self._name = name
        self._cube_func = cube_func

        self._coord_values = coord_values.copy()
        self._coord_values.update(kwargs)

        self._coord_constraints = []
        for coord_name, coord_thing in self._coord_values.items():
            self._coord_constraints.append(_CoordConstraint(coord_name,
                                                            coord_thing))

    def __repr__(self):
        args = []
        if self._name:
            args.append(('name', self._name))
        if self._cube_func:
            args.append(('cube_func', self._cube_func))
        if self._coord_values:
            args.append(('coord_values', self._coord_values))
        return 'Constraint(%s)' % ', '.join('%s=%r' % (k, v) for k, v in args)

    def _coordless_match(self, cube):
        """
        Return whether this constraint matches the given cube when not
        taking coordinates into account.

        """
        match = True
        if self._name:
            match = self._name == cube.name()
        if match and self._cube_func:
            match = self._cube_func(cube)
        return match

    def extract(self, cube):
        """
        Return the subset of the given cube which matches this constraint,
        else return None.

        """
        resultant_CIM = self._CIM_extract(cube)
        slice_tuple = resultant_CIM.as_slice()
        result = None
        if slice_tuple is not None:
            # Slicing the cube is an expensive operation.
            if all([item == slice(None) for item in slice_tuple]):
                # Don't perform a full slice, just return the cube.
                result = cube
            else:
                # Performing the partial slice.
                result = cube[slice_tuple]
        return result

    def _CIM_extract(self, cube):
        # Returns _ColumnIndexManager
        resultant_CIM = _ColumnIndexManager(len(cube.shape))

        if not self._coordless_match(cube):
            resultant_CIM.all_false()
        else:
            for coord_constraint in self._coord_constraints:
                resultant_CIM = resultant_CIM & coord_constraint.extract(cube)

        return resultant_CIM

    def __and__(self, other):
        return ConstraintCombination(self, other, operator.__and__)

    def __rand__(self, other):
        return ConstraintCombination(other, self, operator.__and__)


class ConstraintCombination(Constraint):
    """Represents the binary combination of two Constraint instances."""
    def __init__(self, lhs, rhs, operator):
        """
        A ConstraintCombination instance is created by providing two
        Constraint instances and the appropriate :mod:`operator`.

        """
        try:
            lhs_constraint = as_constraint(lhs)
            rhs_constraint = as_constraint(rhs)
        except TypeError:
            raise TypeError('Can only combine Constraint instances, '
                            'got: %s and %s' % (type(lhs), type(rhs)))
        self.lhs = lhs_constraint
        self.rhs = rhs_constraint
        self.operator = operator

    def _coordless_match(self, cube):
        return self.operator(self.lhs._coordless_match(cube),
                             self.rhs._coordless_match(cube))

    def __repr__(self):
        return 'ConstraintCombination(%r, %r, %r)' % (self.lhs, self.rhs,
                                                      self.operator)

    def _CIM_extract(self, cube):
        return self.operator(self.lhs._CIM_extract(cube),
                             self.rhs._CIM_extract(cube))


class _CoordConstraint(object):
    """Represents the atomic elements which might build up a Constraint."""
    def __init__(self, coord_name, coord_thing):
        """
        Create a coordinate constraint given the coordinate name and a
        thing to compare it with.

        Arguments:

        * coord_name  -  string
            The name of the coordinate to constrain
        * coord_thing
            The object to compare

        """
        self.coord_name = coord_name
        self._coord_thing = coord_thing

    def __repr__(self):
        return '_CoordConstraint(%r, %r)' % (self.coord_name,
                                             self._coord_thing)

    def extract(self, cube):
        """
        Returns the the column based indices of the given cube which
        match the constraint.

        """
        cube_cim = _ColumnIndexManager(len(cube.shape))
        try:
            coord = cube.coord(self.coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            cube_cim.all_false()
            return cube_cim
        dims = cube.coord_dims(coord)
        if len(dims) > 1:
            msg = 'Cannot apply constraints to multidimensional coordinates'
            raise iris.exceptions.CoordinateMultiDimError(msg)

        try_quick = False
        if callable(self._coord_thing):
            call_func = self._coord_thing
        elif (isinstance(self._coord_thing, collections.Iterable) and
              not isinstance(self._coord_thing, basestring)):
            call_func = lambda cell: cell in list(self._coord_thing)
        else:
            call_func = lambda c: c == self._coord_thing
            try_quick = isinstance(coord, iris.coords.DimCoord)

        # Simple, yet dramatic, optimisation for the monotonic case.
        if try_quick:
            try:
                i = coord.nearest_neighbour_index(self._coord_thing)
            except TypeError:
                try_quick = False
        if try_quick:
            r = np.zeros(coord.shape, dtype=np.bool)
            if coord.cell(i) == self._coord_thing:
                r[i] = True
        else:
            r = np.array([call_func(cell) for cell in coord.cells()])
        if dims:
            cube_cim[dims[0]] = r
        elif not all(r):
            cube_cim.all_false()
        return cube_cim


class _ColumnIndexManager(object):
    """
    A class to represent column aligned slices which can be operated on
    using ``&``, ``|`` or ``^``.

    ::

        # 4 Dimensional slices
        import numpy as np
        cim = _ColumnIndexManager(4)
        cim[1] = np.array([3, 4, 5]) > 3
        print cim.as_slice()

    """
    def __init__(self, ndims):
        """
        A _ColumnIndexManager is always created to span the given
        number of dimensions.

        """
        self._column_arrays = [True] * ndims
        self.ndims = ndims

    def __and__(self, other):
        return self._bitwise_operator(other, operator.__and__)

    def __or__(self, other):
        return self._bitwise_operator(other, operator.__or__)

    def __xor__(self, other):
        return self._bitwise_operator(other, operator.__xor__)

    def _bitwise_operator(self, other, operator):
        if not isinstance(other, _ColumnIndexManager):
            return NotImplemented

        if self.ndims != other.ndims:
            raise ValueError('Cannot do %s for %r and %r as they have a '
                             'different number of dimensions.' % operator)
        r = _ColumnIndexManager(self.ndims)
        # iterate over each dimension an combine appropriately
        for i, (lhs, rhs) in enumerate(zip(self, other)):
            r[i] = operator(lhs, rhs)
        return r

    def all_false(self):
        """Turn all slices into False."""
        for i in range(self.ndims):
            self[i] = False

    def __getitem__(self, key):
        return self._column_arrays[key]

    def __setitem__(self, key, value):
        is_vector = isinstance(value, np.ndarray) and value.ndim == 1
        if is_vector or isinstance(value, bool):
            self._column_arrays[key] = value
        else:
            raise TypeError('Expecting value to be a 1 dimensional numpy array'
                            ', or a boolean. Got %s' % (type(value)))

    def as_slice(self):
        """
        Turns a _ColumnIndexManager into a tuple which can be used in an
        indexing operation.

        If no index is possible, None will be returned.
        """
        result = [None] * self.ndims

        for dim, dimension_array in enumerate(self):
            # If dimension_array has not been set, span the entire dimension
            if isinstance(dimension_array, np.ndarray):
                where_true = np.where(dimension_array)[0]
                # If the array had no True values in it, then the dimension
                # is equivalent to False
                if len(where_true) == 0:
                    result = None
                    break

                # If there was exactly one match, the key should be an integer
                if where_true.shape == (1,):
                    result[dim] = where_true[0]
                else:
                    # Finally, we can either provide a slice if possible,
                    # or a tuple of indices which match. In order to determine
                    # if we can provide a slice, calculate the deltas between
                    # the indices and check if they are the same.
                    delta = np.diff(where_true, axis=0)
                    # if the diff is consistent we can create a slice object
                    if all(delta[0] == delta):
                        result[dim] = slice(where_true[0], where_true[-1] + 1,
                                            delta[0])
                    else:
                        # otherwise, key is a tuple
                        result[dim] = tuple(where_true)

            # Handle the case where dimension_array is a boolean
            elif dimension_array:
                result[dim] = slice(None, None)
            else:
                result = None
                break

        if result is None:
            return result
        else:
            return tuple(result)


def list_of_constraints(constraints):
    """
    Turns the given constraints into a list of valid constraints
    using :func:`as_constraint`.

    """
    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]

    return [as_constraint(constraint) for constraint in constraints]


def as_constraint(thing):
    """
    Casts an object into a cube constraint where possible, otherwise
    a TypeError will be raised.

    If the given object is already a valid constraint then the given object
    will be returned, else a TypeError will be raised.

    """
    if isinstance(thing, Constraint):
        return thing
    elif thing is None:
        return Constraint()
    elif isinstance(thing, basestring):
        return Constraint(thing)
    else:
        raise TypeError('%r cannot be cast to a constraint.' % thing)


class AttributeConstraint(Constraint):
    """Provides a simple Cube-attribute based :class:`Constraint`."""
    def __init__(self, **attributes):
        """
        Example usage::

            iris.AttributeConstraint(STASH='m01s16i004')

            iris.AttributeConstraint(
                STASH=lambda stash: stash.endswith('i005'))

        .. note:: Attribute constraint names are case sensitive.

        """
        self._attributes = attributes
        Constraint.__init__(self, cube_func=self._cube_func)

    def _cube_func(self, cube):
        match = True
        for name, value in self._attributes.iteritems():
            if name in cube.attributes:
                cube_attr = cube.attributes.get(name)
                # if we have a callable, then call it with the value,
                # otherwise, assert equality
                if callable(value):
                    if not value(cube_attr):
                        match = False
                        break
                else:
                    if cube_attr != value:
                        match = False
                        break
            else:
                match = False
                break
        return match

    def __repr__(self):
        return 'AttributeConstraint(%r)' % self._attributes


# Useful lookup tables for TimeConstraint and TimePeriodConstraint
months = {1 : (1, 'jan', 'january', 'Jan', 'January'),
          2 : (2, 'feb', 'february', 'Feb', 'February'),
          3 : (3, 'mar', 'march', 'Mar', 'March'),
          4 : (4, 'apr', 'april', 'Apr', 'April'),
          5 : (5, 'may', 'May'),
          6 : (6, 'jun', 'june', 'Jun', 'June'),
          7 : (7, 'jul', 'july', 'Jul', 'July'),
          8 : (8, 'aug', 'august', 'Aug', 'August'),
          9 : (9, 'sep', 'september', 'Sep', 'September'),
          10 : (10, 'oct', 'october', 'Oct', 'October'),
          11 : (11, 'nov', 'november', 'Nov', 'November'),
          12 : (12, 'dec', 'december', 'Dec', 'December')}

seasons = {'djf' : (12, 'dec', 'december',
                    1, 'jan', 'january',
                    2, 'feb', 'february'),
           'mam' : (3, 'mar', 'march',
                    4, 'apr', 'april',
                    5, 'may', 'may'),
           'jja' : (6, 'jun', 'june',
                    7, 'jul', 'july',
                    8, 'aug', 'august'),
           'son' : (9, 'sep', 'september',
                    10, 'oct', 'october',
                    11, 'nov', 'november')}


class TimeConstraint(Constraint):
    """
    Provides a simplified interface for a time-based :class:`Constraint`
    
    Example here... TODO

    """ 
    def __init__(self, year=None, month=None, day=None, hour=None,
                 minute=None, second=None, microsecond=None,
                 season=None, season_year=None, coord_name='time'):
        self._coord_name = coord_name
        PeriodValues = collections.namedtuple(
            'PeriodValues', ['year', 'month', 'day', 'hour',
                             'minute', 'second', 'microsecond',
                             'season', 'season_year'])
        self._period_values = PeriodValues(year, month, day, hour,
                                           minute, second, microsecond,
                                           season, season_year)

        # Create predicates.
        self._predicates = []
        if year is not None:
            self._predicates.append(self._cast_year_thing(year))
        if month is not None:
            self._predicates.append(self._cast_month_thing(month))
        if day is not None:
            self._predicates.append(self._cast_day_thing(day))
        if hour is not None:
            self._predicates.append(self._cast_time_thing('hour', hour))
        if minute is not None:
            self._predicates.append(self._cast_time_thing('minute', minute))
        if second is not None:
            self._predicates.append(self._cast_time_thing('second', second))
        if microsecond is not None:
            self._predicates.append(self._cast_time_thing('microsecond',
                                                          microsecond))
        if season is not None:
            self._predicates.append(self._cast_season_thing(season))
        if season_year is not None:
            self._predicates.append(self._cast_season_year_thing(season_year))

        Constraint.__init__(self, cube_func=self._cube_func)

    def _cube_func(self, cube):
        try:
            _ = cube.coord(self._coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            return False
        else:
            return True

    def _cast_year_thing(self, year_thing):
        """Turn the year thing into a function where appropriate."""
        def _cell_year_eq(cell, year_collection):
            if cell.style() == iris.coords.POINT:
                return cell.point.year in year_collection
            else:
                lower = numpy.min(cell.bound)
                upper = numpy.max(cell.bound) - numpy.max(cell.bound).resolution # Non-inclusive upper bound
                for year in year_collection:
                    if lower.year <= year <= upper.year:
                        return True
                return False

        if callable(year_thing):      # Think about preventing this.
            def _period_attribute_callable_wrapper(cell, period_name, period_thing):
                if cell.style() == iris.coords.POINT:
                    pseudo_cell = iris.coords.Cell(cell.point.__getattribute__(period_name))
                elif cell.style() == iris.coords.BOUND:
                    pseudo_bound = [numpy.min(cell.bound).__getattribute__(period_name), 
                                    (numpy.max(cell.bound) - numpy.max(cell.bound).resolution).__getattribute__(period_name)]
                    pseudo_cell = iris.coords.Cell(None, pseudo_bound)
                else:
                    pseudo_bound = [numpy.min(cell.bound).__getattribute__(period_name), 
                                    (numpy.max(cell.bound) - numpy.max(cell.bound).resolution).__getattribute__(period_name)]
                    pseudo_cell = iris.coords.Cell(cell.point.__getattribute__(period_name), pseudo_bound)
                return period_thing(pseudo_cell)
            result = lambda cell: _period_attribute_callable_wrapper(cell, 'year', year_thing)
        elif isinstance(year_thing, collections.Iterable) and not isinstance(year_thing, basestring):
            result = lambda cell: _cell_year_eq(cell, year_thing)
        else:
            result = lambda cell: _cell_year_eq(cell, [year_thing])
        return result

    def _cast_time_thing(self, period_name, period_thing):
        """Turn the periodic thing into a function where appropriate."""
        def _cell_time_eq(cell, period_name, period_collection):
            if cell.style() == iris.coords.POINT:
                return cell.point.__getattribute__(period_name) in period_collection
            else:
                PeriodLimits = collections.namedtuple('PeriodLimits', ['length', 'min_val', 'max_val'])
                if period_name == 'hour':
                    truncated_bound_extent = (numpy.max(cell.bound).replace(minute=0, second=0, microsecond=0) - 
                                              numpy.min(cell.bound).replace(minute=0, second=0, microsecond=0))
                    period_limits = PeriodLimits(3600, 0, 23)
                elif period_name == 'minute':
                    truncated_bound_extent = (numpy.max(cell.bound).replace(second=0, microsecond=0) - 
                                              numpy.min(cell.bound).replace(second=0, microsecond=0))
                    period_limits = PeriodLimits(60, 0, 59)
                elif period_name == 'second':
                    truncated_bound_extent = (numpy.max(cell.bound).replace(microsecond=0) - 
                                              numpy.min(cell.bound).replace(microsecond=0))
                    period_limits = PeriodLimits(1, 0, 59)
                elif period_name == 'microsecond':
                    truncated_bound_extent = numpy.max(cell.bound) - numpy.min(cell.bound)
                    period_limits = PeriodLimits(0.000001, 0, 999999)
                else:
                    raise ValueError('Expected period_name to be hour, minute, second or microsecond, got %r' % period_name)

                for period in period_collection:
                    # Check whether bound is wider than the modulus of the period_name e.g. for hours is the 
                    # bound wider than 24 hours
                    if truncated_bound_extent.total_seconds() >= ((period_limits.max_val - period_limits.min_val) * 
                                                                  period_limits.length):
                        if period_limits.min_val <= period <= period_limits.max_val:
                            return True
                    else:
                        # Case where the bound spans less than the modulus e.g. for hours this would be 1 day (24 hours)
                        lower_period = numpy.min(cell.bound).__getattribute__(period_name)
                        upper_period = (numpy.max(cell.bound) - numpy.max(cell.bound).resolution).__getattribute__(period_name)
                        if lower_period <= upper_period:
                            if lower_period <= period <= upper_period:
                                return True
                        else:
                            if lower_period <= period <= period_limits.max_val or period_limits.min_val <= period <= upper_period:
                                return True
                return False

        if callable(period_thing):
            raise ValueError('TimeConstraint cannot accept functions as parameters')
        elif isinstance(period_thing, collections.Iterable) and not isinstance(period_thing, basestring):
            result = lambda cell: _cell_time_eq(cell, period_name, period_thing)
        else:
            result = lambda cell: _cell_time_eq(cell, period_name, [period_thing])
        return result

    def _cast_month_thing(self, month_thing):
        def _cell_month_eq(cell, month_collection):
            # Check the collection and raise an exception if their constraint uses an unknown month
            for month in month_collection:
                if month not in itertools.chain.from_iterable(months.itervalues()):
                    raise ValueError('Unknown month %r' % month)
        
            if cell.style() == iris.coords.POINT:
                return bool(set(months[cell.point.month]) & set(month_collection))
            else:
                lower = numpy.min(cell.bound)
                upper = numpy.max(cell.bound) - numpy.max(cell.bound).resolution
                l_year = lower.year
                l_month = lower.month
                u_year = upper.year
                u_month = upper.month
                delta_months = (u_year - l_year) * 12 + u_month - l_month
                if u_year < 0 != l_year < 0:        # special case to handle the missing year 0
                    delta_months -= 12
                
                if delta_months >= 11:
                    for month in month_collection:
                        if month in itertools.chain.from_iterable(months.itervalues()):
                            return True
                else:
                    if l_month <= u_month:
                        for month in xrange(l_month, u_month + 1):
                            if set(months[month]) & set(month_collection):
                                return True
                    else:
                        for month in xrange(l_month, 13):
                            if set(months[month]) & set(month_collection):
                                return True
                        for month in xrange(1, u_month + 1):
                            if set(months[month]) & set(month_collection):
                                return True
                return False
        
        if callable(month_thing):      
            raise ValueError('TimeConstraint cannot accept functions as parameters')
        elif isinstance(month_thing, collections.Iterable) and not isinstance(month_thing, basestring):
            result = lambda cell: _cell_month_eq(cell, month_thing)
        else:
            result = lambda cell: _cell_month_eq(cell, [month_thing])
        return result


    def _cast_day_thing(self, day_thing):
        """Turn the day thing into a function where appropriate."""
        def _cell_day_eq(cell, day_collection):
            if cell.style() == iris.coords.POINT:
                return cell.point.day in day_collection
            else:
                lower = numpy.min(cell.bound)
                upper = numpy.max(cell.bound) - numpy.max(cell.bound).resolution
                # The day number is linked to the calendar e.g. Feb 29th exists in leap years in gregorian, 
                # but exists every year (and every month) in a 360 day calendar
                if isinstance(lower, iris.datetime360.date) and isinstance(upper, iris.datetime360.date):   # 360 day calendar
                    PeriodLimits = collections.namedtuple('PeriodLimits', ['length', 'min_val', 'max_val'])
                    if isinstance(lower, iris.datetime360.datetime) and isinstance(upper, iris.datetime360.datetime):
                        truncated_bound_extent = (upper.replace(hour=0, minute=0, second=0, microsecond=0) - 
                                                  lower.replace(hour=0, minute=0, second=0, microsecond=0))
                        period_limits = PeriodLimits(24*60*60, 1, 30)
                    else:
                        truncated_bound_extent = upper - lower
                        period_limits = PeriodLimits(24*60*60, 1, 30)

                    for day in day_collection:
                        # Check whether bound is wider than the modulus of the period_name, i.e. greater than 29
                        if truncated_bound_extent.total_seconds() >= ((period_limits.max_val - period_limits.min_val) * 
                                                                      period_limits.length):
                            if period_limits.min_val <= day <= period_limits.max_val:
                                return True
                        else:
                            # Case where the bound spans less than the modulus e.g. for days this would be 30 days
                            lower_day = lower.day
                            upper_day = upper.day
                            if lower.day <= upper.day:
                                if lower.day <= day <= upper.day:
                                    return True
                            else:
                                if lower.day <= day <= period_limits.max_val or period_limits.min_val <= day <= upper.day:
                                    return True
                    return False
                elif isinstance(numpy.min(cell.bound), datetime.date) and isinstance(numpy.max(cell.bound), datetime.date):
                    # gregorian calendar
                    lower = numpy.min(cell.bound)
                    upper = numpy.max(cell.bound) - numpy.max(cell.bound).resolution
                    if (upper - lower).days >= 59:  # covers all month days regardless of leap years, months etc.
                        for day in day_collection:
                            if day in xrange(1, 32):   # valid
                                return True
                    elif lower.month == upper.month:
                        for day in day_collection:
                            if day in xrange(lower.day, upper.day + 1):
                                return True
                    else:
                        for day in day_collection:
                            if day in xrange(lower.day, calendar.monthrange(lower.year, lower.month)[1] + 1) or day in xrange(1, upper.day + 1):
                                return True
                            else: # middle month (if present)
                                if lower.year == upper.year:
                                    for month in xrange(lower.month + 1, upper.month): 
                                        if day in xrange(1, calendar.monthrange(lower.year, month)[1] + 1):
                                            return True
                                else:
                                    if upper.month == 2: # need to cover Jan 
                                        if day in xrange(1, 32):
                                            return True
                    return False
                else:
                    raise ValueError('Unsuported datetime objects of type %r, %r' % (type(lower), type(upper)))

        if callable(day_thing):
            raise ValueError('TimeConstraint cannot accept functions as parameters')
        elif isinstance(day_thing, collections.Iterable) and not isinstance(day_thing, basestring):
            result = lambda cell: _cell_day_eq(cell, day_thing)
        else:
            result = lambda cell: _cell_day_eq(cell, [day_thing])
        return result

    def _cast_season_thing(self, season_thing):
        def season_from_month(month):
            if isinstance(month, basestring):
                month = month.lower()

            for season, season_months in seasons.iteritems():
                if month in season_months:
                    return season
            raise ValueError('Unknown month %r' % month)

        def _cell_season_eq(cell, season_collection):
            # Check the collection and raise an exception if the constraint uses an unknown season
            for season in season_collection:
                if season.lower() not in seasons:
                    raise ValueError('Unknown season %r' % season)
            
            # Ignore case
            season_collection_lc = [season.lower() for season in season_collection]

            if cell.style() == iris.coords.POINT:
                return season_from_month(cell.point.month) in season_collection_lc
            else:
                lower = numpy.min(cell.bound)
                upper = numpy.max(cell.bound) - numpy.max(cell.bound).resolution
                l_year = lower.year
                l_month = lower.month
                u_year = upper.year
                u_month = upper.month
                delta_months = (u_year - l_year) * 12 + u_month - l_month
                if u_year < 0 != l_year < 0:        # special case to handle the missing year 0
                    delta_months -= 12
                if delta_months >= 11:
                    for season in season_collection_lc:
                        if season in seasons.iterkeys():
                            return True
                if l_month <= u_month:
                    for month in xrange(l_month, u_month + 1):
                        if season_from_month(month) in season_collection_lc:
                            return True
                else:
                    for month in xrange(l_month, 13):
                        if season_from_month(month) in season_collection_lc:
                            return True
                    for month in xrange(1, u_month + 1):
                        if season_from_month(month) in season_collection_lc:
                            return True
                return False

        if callable(season_thing):
            raise ValueError('TimeConstraint cannot accept functions as parameters')
        elif isinstance(season_thing, collections.Iterable) and not isinstance(season_thing, basestring):
            result = lambda cell: _cell_season_eq(cell, season_thing)
        else:
            result = lambda cell: _cell_season_eq(cell, [season_thing])
        return result

    def _cast_season_year_thing(self, year_thing):
        """Turn the year thing into a function where appropriate."""
        def season_year(dt):
            if dt.month == 12:
                return dt.year + 1
            else:
                return dt.year

        def _cell_season_year_eq(cell, year_collection):
            if cell.style() == iris.coords.POINT:
                return season_year(cell.point) in year_collection
            else:
                lower = numpy.min(cell.bound)
                upper = numpy.max(cell.bound) - numpy.max(cell.bound).resolution
                for year in year_collection:
                    if season_year(lower) <= year <= season_year(upper):
                        return True
                return False

        if callable(year_thing):
            raise ValueError('TimeConstraint cannot accept functions as parameters')
        elif isinstance(year_thing, collections.Iterable) and not isinstance(year_thing, basestring):
            result = lambda cell: _cell_season_year_eq(cell, year_thing)
        else:
            result = lambda cell: _cell_season_year_eq(cell, [year_thing])
        return result


    def _CIM_extract(self, cube):
        cube_cim = _ColumnIndexManager(len(cube.shape))
        try:
            coord = cube.coord(self._coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            cube_cim.all_false()
            return cube_cim

        dims = cube.coord_dims(coord)
        if not dims:
            return cube_cim.all_false()
        elif len(dims) > 1:
            raise iris.exceptions.CoordinateMultiDimError(coord)

        #axis = cube._axes.get(coord.axis_name)
        for predicate in self._predicates:
            cim = _ColumnIndexManager(len(cube.shape))
            #r = numpy.array([predicate(cell) for cell in coord.cells()], ndmin=1)
            #if axis is not None:
            #    cim[axis] = r
            cim[dim] = numpy.array([predicate(cell) for cell in coord.cells()], ndmin=1)
            cube_cim = cube_cim & cim
        
        return cube_cim

    def __repr__(self):
        return 'TimeConstraint(%r, %r)' % (self._coord_name, self._period_values)

# Type factory to produce a named tuple that is used as a base class for 
# _DateTimeTuple, a calendar agnostic container for start or end parameters
_DateTimeTupleParent = collections.namedtuple('_DateTimeTupleParent', ['year', 'month', 'day', 'hour', 
                                                                       'minute', 'second', 'microsecond'])

class _DateTimeTuple(_DateTimeTupleParent):
    """
    A calendar agnostic container used to store year, month, day, hour, minute, second and microsecond
    values passed as parameters to the TimePeriodConstraint constructor. 
    
    This class allows any/all of the named values to be None.
    
    """
    def __new__(cls, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None):
        month_number = None
        month_numbers = {'jan':1 , 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 
                         'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
        if month is not None:
            if month not in itertools.chain.from_iterable(months.itervalues()):
                raise ValueError('Unknown month %r' % month)
            # Convert to number if provide as string e.g. 'March'
            if isinstance(month, basestring):
                month_number = month_numbers[month.lower()[:3]]
            else:
                month_number = month
        
        return _DateTimeTupleParent.__new__(cls, year, month_number, day, hour, minute, second, microsecond)

    def __init__(self, *_args):
        # Require contiguous values (no gaps)
        self._start_index = None
        for i, val in enumerate(self):
            if val is not None:
                self._start_index = i
                break

        self._end_index = None
        if self._start_index is not None:
            for i, val in enumerate(reversed(self)):
                if val is not None:
                    self._end_index = len(self) - i
                    break
            for i in range(self._start_index + 1, self._end_index):
                if self[i] is None:
                    raise ValueError('Specified parameters must not span an unspecified '\
                                     'parameter e.g. if you specify year and day, you must also specify month')
    def __repr__(self):
        return '_DateTimeTuple(year=%r, month=%r, day=%r, hour=%r, minute=%r, second=%r, microsecond=%r)' % self
    
    @property
    def start_index(self):
        return self._start_index
    
    @property
    def end_index(self):
        return self._end_index

    def valid_fields(self):
        """Returns a tuple of the field names (strings) that are specified i.e. not None."""
        if self._start_index is None:
            return []
        if self._end_index is None:
            return self._fields
        return self._fields[self._start_index:self._end_index]

    def is_all_none(self):
        return self.start_index == None
    
    def blended(self, other):
        """
        Returns a new _DateTimeTuple with any None values populated with the values from the object passed in.
        Note that this method should accept another _DateTimeTuple or a datetime object e.g. an iris.datetime360.datetime
     
        Arguments:
    
        * other  -  _DateTimeTuple or datetime object
        Object from which to obtain field values to replace corresponding None values in self.

        """
        self_as_list = list(self)
        for index, field in enumerate(self._fields):
            if self_as_list[index] is None:
                self_as_list[index] = other.__getattribute__(field)
        return _DateTimeTuple(*self_as_list)

    def truncated(self, other):
        """
        Returns a new _DateTimeTuple equal to self except that the necessary field values will be replaced by 
        None such that the startindex of the resulting _DateTimeTuple will equal that of the object passed in. 
                    
        Arguments:
    
        * other  -  _DateTimeTuple
        
        For example:
        
            >>> start = iris._constraints._DateTimeTuple(2002, 2, 15, None, None, None, None)
            >>> other = iris._constraints._DateTimeTuple(None, 12, 13, None, None, None, None)
            >>> start.truncated(other)
            _DateTimeTuple(year=None, month=2, day=15, hour=None, minute=None, second=None, microsecond=None)

        """            

        if other.start_index is None:
            return _DateTimeTuple()
        self_as_list = list(self)
        self_as_list[0:other.start_index] = [None] * other.start_index
        return _DateTimeTuple(*self_as_list)

class TimePeriodConstraint(Constraint):
    """
    Provides a simplified interface for a time-based :class:`Constraint` between two datetimes. For example:
    iris.TimePeriodConstraint(start_month='Dec', start_day=25, end_month='Jan', end_day=2)
    
    """
    
    
    def __init__(self, start_year=None, start_month=None, start_day=None, start_hour=None, start_minute=None, 
                 start_second=None, start_microsecond=None, end_year=None, end_month=None, end_day=None, 
                 end_hour=None, end_minute=None, end_second=None, end_microsecond=None, coord_name='time'):
        self._coord_name = coord_name
        self._start = _DateTimeTuple(start_year, start_month, start_day, 
                                     start_hour, start_minute, start_second,
                                     start_microsecond)
        self._end = _DateTimeTuple(end_year, end_month, end_day, 
                                   end_hour, end_minute, end_second, 
                                   end_microsecond)

        self._predicate = self._generate_predicate(self._start, self._end)
        Constraint.__init__(self, cube_func=self._cube_func)
    
    def _cube_func(self, cube):
        try:
            cube.coord(self._coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            return False
        else:
            return True

    def _generate_predicate(self, start, end):
        def _cell_point_function(point, start, end):
            if start.is_all_none() and end.is_all_none():
                return True
            
            # Handle difficult case where the largest period is only specified at one end 
            # e.g. start_year=2008, start_month='Nov', end_month='March'
            if not start.is_all_none() and not end.is_all_none():
                if start.start_index != end.start_index:
                    if start.start_index < end.start_index:
                        return (_cell_point_function(point, start, _DateTimeTuple()) and 
                                _cell_point_function(point, start.truncated(end), end))
                    else:
                        return (_cell_point_function(point, _DateTimeTuple(), end) and 
                                _cell_point_function(point, start, end.truncated(start)))
                
            long_dt = type(point)(2000,1,1) # Chosen to have all day numbers in 360, gregorian, no_leap and all_leap calendars
            start_dt = type(point)(*start.blended(long_dt))
            end_dt = type(point)(*end.blended(long_dt))
            point_tuple = _DateTimeTuple(point.year, point.month, point.day, point.hour, 
                                                            point.minute, point.second, point.microsecond)
            point_start_dt = type(point)(*point_tuple.truncated(start).blended(long_dt))
            point_end_dt = type(point)(*point_tuple.truncated(end).blended(long_dt))
        
            after_start = start_dt <= point_start_dt
            before_end = point_end_dt < end_dt

            if start.is_all_none():
                return before_end
            elif end.is_all_none():
                return after_start 
            else:
                if start_dt <= end_dt:
                    return after_start and before_end
                else:
                    return  after_start or before_end

        def _cell_bound_function(bound, start, end):
            if start.is_all_none() and end.is_all_none():
                return True
        
            # Handle difficult case where the largest period is only specified at one end 
            # e.g. start_year=2008, start_month='Nov', end_month='March'
            if not start.is_all_none() and not end.is_all_none():
                if start.start_index != end.start_index:
                    if start.start_index < end.start_index:
                        return (_cell_bound_function(bound, start, _DateTimeTuple()) and 
                                _cell_bound_function(bound, start.truncated(end), end))
                    else:
                        return (_cell_bound_function(bound, _DateTimeTuple(), end) and 
                                _cell_bound_function(bound, start, end.truncated(start)))

            lower = numpy.min(bound)
            upper = numpy.max(bound) - numpy.max(bound).resolution
            
            long_dt = type(lower)(2000,1,1) # Chosen to have all day numbers in 360, gregorian, no_leap and all_leap calendars
            start_dt = type(lower)(*start.blended(long_dt))
            end_dt = type(lower)(*end.blended(long_dt))

            upper_after_start = _cell_point_function(upper, start, _DateTimeTuple())
            lower_before_end = _cell_point_function(lower, _DateTimeTuple(), end)

            #upper_after_end = _cell_point_function(upper, end, _DateTimeTuple())
            #lower_before_start = _cell_point_function(lower, _DateTimeTuple(), start)
            
            lower_truncated = type(lower)(*_DateTimeTuple(lower.year, lower.month, lower.day, lower.hour, lower.minute, 
                                                          lower.second, lower.microsecond).truncated(start).blended(long_dt))
            upper_truncated = type(upper)(*_DateTimeTuple(upper.year, upper.month, upper.day, upper.hour, upper.minute,
                                                          upper.second, upper.microsecond).truncated(start).blended(long_dt))
            if lower_truncated <= upper_truncated:
                if start.is_all_none():
                    return lower_before_end
                elif end.is_all_none():
                    return upper_after_start
                elif start_dt <= end_dt:
                    return lower_before_end and upper_after_start
                else:
                    return lower_before_end or upper_after_start
            else:
                raise NotImplementedYetError()      ## Need to fix this - issue is very large bounds e.g Oct 91 to Dec 94 and constraint mar-may

        def _cell_function(cell, start, end):
            if cell.style() == iris.coords.POINT:
                return _cell_point_function(cell.point, start, end)
            else:
                return _cell_bound_function(cell.bound, start, end)

        return lambda cell: _cell_function(cell, start, end) 
        
    def _CIM_extract(self, cube):
        cube_cim = _ColumnIndexManager(len(cube.shape))
        try:
            coord = cube.coord(self._coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            cube_cim.all_false()
            return cube_cim
        
        axis = cube._axes.get(coord.axis_name)
        r = numpy.array([self._predicate(cell) for cell in coord.cells()], ndmin=1)
        if axis is not None:
            cube_cim[axis] = r
        elif not all(r):
            cube_cim.all_false()
        
        return cube_cim

    def __repr__(self):
        return 'TimePeriodConstraint(coord=%r, start=%r, end=%r)' % (self._coord_name, self._start, self._end)

