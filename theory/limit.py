import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.stats import chi2

# Calculate TS threshold
delta_chi2_thresh = chi2.isf(1 - 0.68, 1)


def get_lim(delta_chi2_ary, eps_ary):
    """Get one-sided 68% limits for a TS delta_chi2_ary computed over eps_ary"""

    TS_m_xsec = np.zeros(3)
    TS_m_xsec[2] = eps_ary[0]

    for i_m in range(len(delta_chi2_ary)):

        TS_eps_ary = np.nan_to_num(delta_chi2_ary[i_m], nan=1e10)

        # Find value, location and xsec at the max TS (as a function of theta)
        max_loc = np.argmin(TS_eps_ary)
        max_TS = TS_eps_ary[max_loc]

        if max_TS > TS_m_xsec[0]:
            TS_m_xsec[0] = max_TS
            TS_m_xsec[1] = i_m
            TS_m_xsec[2] = eps_ary[max_loc]

        # Calculate limit
        for xi in range(max_loc, len(eps_ary)):
            val = TS_eps_ary[xi]
            if val > delta_chi2_thresh:
                interp = interp1d(np.log10([eps_ary[xi - 1], eps_ary[xi]]), [TS_eps_ary[xi - 1], TS_eps_ary[xi]])
                upper_crossing = 10 ** brentq(lambda eps: interp(eps) - delta_chi2_thresh, np.log10(eps_ary[xi - 1]), np.log10(eps_ary[xi]))
                break

        # Calculate limit
        for xi in range(0, max_loc):
            val = TS_eps_ary[xi]
            if val < delta_chi2_thresh:
                interp = interp1d(np.log10([eps_ary[xi - 1], eps_ary[xi]]), [TS_eps_ary[xi - 1], TS_eps_ary[xi]])
                lower_crossing = 10 ** brentq(lambda eps: interp(eps) - delta_chi2_thresh, np.log10(eps_ary[xi - 1]), np.log10(eps_ary[xi]))
                break

    return upper_crossing - lower_crossing


def get_interval(delta_chi2_ary, eps_ary, p=0.68):
    """Get one-sided 68% limits for a TS delta_chi2_ary computed over eps_ary"""

    # Calculate TS threshold
    delta_chi2_thresh = chi2.isf(1 - p, 1)

    TS_m_xsec = np.zeros(3)
    TS_m_xsec[2] = eps_ary[0]

    for i_m in range(len(delta_chi2_ary)):

        TS_eps_ary = np.nan_to_num(delta_chi2_ary[i_m], nan=1e10)

        # Find value, location and xsec at the max TS (as a function of theta)
        max_loc = np.argmin(TS_eps_ary)
        max_TS = TS_eps_ary[max_loc]

        if max_TS > TS_m_xsec[0]:
            TS_m_xsec[0] = max_TS
            TS_m_xsec[1] = i_m
            TS_m_xsec[2] = eps_ary[max_loc]

        # Calculate limit
        for xi in range(max_loc, len(eps_ary)):
            val = TS_eps_ary[xi]
            if val > delta_chi2_thresh:
                interp = interp1d(np.log10([eps_ary[xi - 1], eps_ary[xi]]), [TS_eps_ary[xi - 1], TS_eps_ary[xi]])
                upper_crossing = 10 ** brentq(lambda eps: interp(eps) - delta_chi2_thresh, np.log10(eps_ary[xi - 1]), np.log10(eps_ary[xi]))
                break

        # Calculate limit
        for xi in range(0, max_loc):
            val = TS_eps_ary[xi]
            if val < delta_chi2_thresh:
                interp = interp1d(np.log10([eps_ary[xi - 1], eps_ary[xi]]), [TS_eps_ary[xi - 1], TS_eps_ary[xi]])
                lower_crossing = 10 ** brentq(lambda eps: interp(eps) - delta_chi2_thresh, np.log10(eps_ary[xi - 1]), np.log10(eps_ary[xi]))
                break

    return lower_crossing, upper_crossing