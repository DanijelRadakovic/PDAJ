from math import log

from beam_integrals.beam_types import BaseBeamType 
from beam_integrals.exceptions import UnableToGuessScaleFunctionError
from beam_integrals.integrals import BaseIntegral, integrate

from ..app import app


SCALE_FACTOR_HELPER = 2.


@app.task
def compute_integral(integral_id, beam_type_id, m, t, v, n):
    beam_type = BaseBeamType.coerce(beam_type_id)
    integral = BaseIntegral.coerce(integral_id)
    max_mode = app.conf.BEAM_INTEGRALS_MAX_MODE
    decimal_precision = app.conf.BEAM_INTEGRALS_DECIMAL_PRECISION
    normalize_integrals_smaller_than = app.conf.BEAM_INTEGRALS_NORMALIZE_INTEGRALS_SMALLER_THAN

    result, error = integrate(
        integral, beam_type,
        a=1.,
        m=m, t=t, v=v, n=n,
        decimal_precision=decimal_precision,
        error=True
    )

    # Normalize the `result` to zero for small values, as per the section 3.4 of my PhD thesis
    if abs(result) <= normalize_integrals_smaller_than:
        result = 0.
        scale_factor = 0
    else:
        try:
            scale_factor = integral.guess_scale_factor(beam_type, m, t, v, n)
        except UnableToGuessScaleFunctionError:
            scaled_result = integrate(
                integral, beam_type,
                a=SCALE_FACTOR_HELPER,
                m=m, t=t, v=v, n=n,
                decimal_precision=decimal_precision
            )
            scale_factor = round(log(scaled_result / result, SCALE_FACTOR_HELPER))

    cache_key = integral.cache_key(m, t, v, n, max_mode=max_mode)
    data = {
        'integral_float64': float(result),
        'error_float64': float(error),
        'scale_factor': int(scale_factor),
        'integral_str': str(result),
        'error_str': str(error),
    }
    return cache_key, data

@app.task
def combine_computed_integrals_into_a_table(computed_integrals, integral_id):
    return integral_id, dict(computed_integrals)
