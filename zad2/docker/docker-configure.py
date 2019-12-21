#!/usr/bin/env python
import os
from beam_integrals import DEFAULT_MAX_MODE, DEFAULT_DECIMAL_PRECISION
from beam_integrals.beam_types import BaseBeamType
from beam_integrals.characteristic_equation_solvers import best_roots_cache
from beam_integrals.exceptions import BestRootsCacheError


def verify_best_roots_cache(max_mode, decimal_precision):
    try:
        print "Checking best roots cache for max_mode=%d, decimal_precision=%d..." % (max_mode, decimal_precision)
        for beam_type in BaseBeamType.plugins.instances_sorted_by_id:
            best_roots_cache.get(beam_type, max_mode, decimal_precision)

        print 'Cache OK.'
    except BestRootsCacheError, e: # Regen if there's any kind of a problem with the cache
        # The original exception message is too verbose
        short_message = e.message.split(" You'll need to regenerate the cache", 1)[0]
        print "Cache is invalid: %s" % short_message

        best_roots_cache.regenerate(max_mode, decimal_precision)
        print 'Cache regenerated.'

def main():
    verify_best_roots_cache(
        max_mode=int(os.getenv('BEAM_INTEGRALS_MAX_MODE', DEFAULT_MAX_MODE)),
        decimal_precision=int(os.getenv('BEAM_INTEGRALS_DECIMAL_PRECISION', DEFAULT_DECIMAL_PRECISION))
    )

if __name__ == '__main__':
    main()
