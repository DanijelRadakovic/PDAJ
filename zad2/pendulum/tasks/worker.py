import numpy as np
from scipy.integrate import odeint


from ..app import app

# The gravitational acceleration (m.s-2).
g = 9.81


# @app.task
# def compute_integral(integral_id, beam_type_id, m, t, v, n):
#     print 'usao u jebeni compute integral na workeru'
#     beam_type = BaseBeamType.coerce(beam_type_id)
#     integral = BaseIntegral.coerce(integral_id)
#     max_mode = app.conf.BEAM_INTEGRALS_MAX_MODE
#     decimal_precision = app.conf.BEAM_INTEGRALS_DECIMAL_PRECISION
#     normalize_integrals_smaller_than = app.conf.BEAM_INTEGRALS_NORMALIZE_INTEGRALS_SMALLER_THAN
#
#     result, error = integrate(
#         integral, beam_type,
#         a=1.,
#         m=m, t=t, v=v, n=n,
#         decimal_precision=decimal_precision,
#         error=True
#     )
#
#     # Normalize the `result` to zero for small values, as per the section 3.4 of my PhD thesis
#     if abs(result) <= normalize_integrals_smaller_than:
#         result = 0.
#         scale_factor = 0
#     else:
#         try:
#             scale_factor = integral.guess_scale_factor(beam_type, m, t, v, n)
#         except UnableToGuessScaleFunctionError:
#             scaled_result = integrate(
#                 integral, beam_type,
#                 a=SCALE_FACTOR_HELPER,
#                 m=m, t=t, v=v, n=n,
#                 decimal_precision=decimal_precision
#             )
#             scale_factor = round(log(scaled_result / result, SCALE_FACTOR_HELPER))
#
#     cache_key = integral.cache_key(m, t, v, n, max_mode=max_mode)
#     data = {
#         'integral_float64': float(result),
#         'error_float64': float(error),
#         'scale_factor': int(scale_factor),
#         'integral_str': str(result),
#         'error_str': str(error),
#     }
#     return cache_key, data
#
# @app.task
# def combine_computed_integrals_into_a_table(computed_integrals, integral_id):
#     return integral_id, dict(computed_integrals)


def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) -
             (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s ** 2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
             m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)
    return theta1dot, z1dot, theta2dot, z2dot


@app.task
def solve(L1, L2, m1, m2, tmax, dt, y0):
    t = np.arange(0, tmax + dt, dt)

    # Do the numerical integration of the equations of motion
    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))
    theta1, theta2 = y[:, 0], y[:, 2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    return theta1, theta2, x1, y1, x2, y2
