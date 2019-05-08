#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys


def check_alpha(alpha):
    '''
    Check value of fractional order.

    The package is designed to work on problems where the fractional
    order has a value in in the range [0, 1).

    Args:
        * **alpha** (:py:class:`float`): Order of fractional derivative.

    Raises:
        * System exit for `ZeroDivisionError`.
    '''
    try:
        1/(1-alpha)
    except ZeroDivisionError as err:
        print('Invalid value! The value of alpha cannot'
              'be 1.0. alpha = {}:\n\t{}'.format(alpha, err))
        raise SystemExit


def check_value(value, default_value, varname=None):
    '''
    Check value with respect to default.

    This routine checks that value is defined by either a user-defined
    value or default value.  If still `None`, then something went wrong
    and this raises an error message.

    Args:
        * **value** (user defined): Value being tested
        * **default_value** (method defined): Default value

    Kwargs: name (type) - default
        * **varname** (:py:class:`str`) - `None`: Name of
          variable/value being tested. This provides a more
          descriptive output for debugging.

    If properly defined

    Returns:
        * **value**

    else

    Raises:
        * System exist for no value defined.
    '''
    if value is None:
        value = default_value
    if value is None:
        sys.exit(str('No value defined. '
                     'Provide value for {}'.format(varname)))
    return value


def check_range(lower, upper, value):
    '''
    Check that value falls within [lower, upper] range.

    Args:
        * **lower** (:py:class:`float`): Lower limit
        * **upper** (:py:class:`float`): Upper limit
        * **value** (:py:class:`float`): Value to check with respect to range

    If **value** in range,

    Returns:
        * **value** (:py:class:`float`)

    else

    Raises:
        * System exit for value out of domain.
    '''
    if lower <= value <= upper:
        return value
    else:
        sys.exit(str('Switch time out of domain.'))


def check_input(value, varname=None):
    '''
    Check value is defined.

    This routine checks that value is defined.

    Args:
        * **value** (user defined): Value being tested

    Kwargs: name (type) - default
        * **varname** (:py:class:`str`) - `None`: Name of
          variable/value being tested. This provides a more
          descriptive output for debugging.

    If not properly defined

    Raises:
        * System exist for no value defined.
    '''
    if value is None:
        sys.exit(str('No value defined. '
                     'Provide value for {}.'.format(varname)))


def check_singularity(singularity, upper):
    '''
    Check singularity was defined.

    This routine checks if user provided a singularity location.
    Default behavior is to use the upper limit of integration
    as the singularity location.

    Args:
        * **singularity** (:py:class:`float`): User defined location
        * **upper** (:py:class:`float`): Upper limit

    Returns:
        * **upper** - if `singularity is None`
        * **singularity** - if `singularity not None`
    '''
    if singularity is None:
        return upper
    else:
        return singularity


def check_node_type(n):
    '''
    Check that number of nodes is an integer.

    Args:
        * **n** (user defined): Number of nodes.

    Returns:
        * `int(n)`
    '''
    return int(n)
