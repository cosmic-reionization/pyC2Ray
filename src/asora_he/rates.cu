#include "rates.cuh"

#include <iostream>

// ========================================================================
// Define macros. Could be passed as parameters but are kept as
// compile-time constants for now
// ========================================================================
#define TAU_PHOTO_LIMIT 1.0e-7  // Limit to consider a cell "optically thin/thick"
#define S_STAR_REF \
    1e48  // Reference ionizing flux (strength of source is given in this unit)

// ========================================================================
// Compute photoionization rate from in/out column density by looking up
// values of the integral ∫L_v*e^(-τ_v)/hv in precalculated tables. These
// tables are assumed to have been copied to device memory in advance using
// photo_table_to_device()
// ========================================================================
__device__ double photoion_rates_gpu(
    const double &strength, const double &tau_in, const double &tau_out, const int &nf,
    const double &Vfact, const double *photo_thin_table,
    const double *photo_thick_table, const double &minlogtau, const double &dlogtau,
    const int &NumTau, const int &NumFreq
) {
    // MB (23.09.24): Rather then re-calculating the tau_in and tau_out as in the
    // HI only raytracing. Here, we pass these two variables (TODO: this could be
    // implemented also in the HI only).
    double prefact = strength / Vfact;

    // PH (08.10.23) I'm confused about the way the rates are calculated
    // differently for thin/thick cells. The following is taken verbatim from
    // radiation_photoionrates.F90 lines 276 - 303 but without true
    // understanding... Names are slightly different to simpify notation
    double phi_photo_in =
        prefact *
        photo_lookuptable(photo_thick_table, nf, tau_in, minlogtau, dlogtau, NumTau);

    // printf("%i %e\n", nf, photo_thick_table[0+nf*NumTau]);
    if (abs(tau_out - tau_in) > TAU_PHOTO_LIMIT) {
        double phi_photo_out =
            prefact * photo_lookuptable(
                          photo_thick_table, nf, tau_out, minlogtau, dlogtau, NumTau
                      );
        return phi_photo_in - phi_photo_out;
    } else {
        return prefact * (tau_out - tau_in) *
               photo_lookuptable(
                   photo_thin_table, nf, tau_out, minlogtau, dlogtau, NumTau
               );
    }
    // MB (07.10.2024): for debuggin comment out the if else condition above and
    // un-comment the following line:
    // return phi_photo_in;
}

__device__ double photoheat_rates_gpu(
    const double &strength, const double &tau_in, const double &tau_out, const int &nf,
    const double &Vfact, const double *photo_thin_table,
    const double *photo_thick_table, const double &minlogtau, const double &dlogtau,
    const int &NumTau, const int &NumFreq
) {
    // MB (07.10.24): this is a copy of the photoion_rates_gpu method, but we pass
    // the photo-heating tables instead
    double prefact = strength / Vfact;

    double phi_photo_in =
        prefact *
        photo_lookuptable(photo_thick_table, nf, tau_in, minlogtau, dlogtau, NumTau);

    // printf("%i %e\n", nf, photo_thick_table[0+nf*NumTau]);
    if (abs(tau_out - tau_in) > TAU_PHOTO_LIMIT) {
        double phi_photo_out =
            prefact * photo_lookuptable(
                          photo_thick_table, nf, tau_out, minlogtau, dlogtau, NumTau
                      );
        return phi_photo_in - phi_photo_out;
    } else {
        return prefact * (tau_out - tau_in) *
               photo_lookuptable(
                   photo_thin_table, nf, tau_out, minlogtau, dlogtau, NumTau
               );
    }
    // MB (07.10.2024): for debuggin comment out the if else condition above and
    // un-comment the following line:
    // return phi_photo_in;
}

// ========================================================================
// Grey-opacity test case photoionization rate, computed from analytical
// expression rather than using tables. To use this version, compile with the
// -DGREY_NOTABLES flag
// ========================================================================
__device__ double photoion_rates_test_gpu(
    const double &strength, const double &coldens_in, const double &coldens_out,
    const double &Vfact, const double &sig
) {
    // Compute optical depth and ionization rate depending on whether the cell is
    // optically thick or thin
    double tau_in = coldens_in * sig;
    double tau_out = coldens_out * sig;

    // If cell is optically thick
    if (fabs(tau_out - tau_in) > TAU_PHOTO_LIMIT)
        // return strength * INV4PI / (Vfact * nHI) * (exp(-tau_in) -
        // exp(-tau_out));
        return (strength * S_STAR_REF / (Vfact)) * (exp(-tau_in) - exp(-tau_out));
    // If cell is optically thin
    else
        // return strength * INV4PI * sig * (tau_out - tau_in) / (Vfact) *
        // exp(-tau_in);
        return (strength * S_STAR_REF / (Vfact)) * (tau_out - tau_in) * exp(-tau_in);
}

// ========================================================================
// Utility function to look up the integral value corresponding to an optical
// depth τ by doing linear interpolation.
// ========================================================================
__device__ double photo_lookuptable(
    const double *table, const int &nf, const double &tau, const double &minlogtau,
    const double &dlogtau, const int &NumTau
) {
    // MB (04.10.24): Upset with the old method I decided to change it and write
    // my own.
    int i0, i1;
    double real_i, w1, w0;

    // Find table index and do linear interpolation
    // Recall that tau(0) = 0 and tau(1:NumTau) ~ logspace(minlogtau,maxlogtau)
    // (so in reality the table has size NumTau+1)
    if (log10(tau) < minlogtau) {
        real_i = 0.0;
        // Find table index and weight for linear interpolation
        i0 = 0;
        i1 = 1;
    } else if (log10(tau) > float(NumTau) * dlogtau + minlogtau) {
        real_i = float(NumTau);
        // Find table index and weight for linear interpolation
        i0 = NumTau;
        i1 = NumTau + 1;
    } else {
        // real_i = float(NumTau)*(log10(tau) - minlogtau)/(float(NumTau-1)*dlogtau)
        // + 1.0;
        real_i =
            float(NumTau) * (log10(tau) - minlogtau) / (float(NumTau) * dlogtau) + 1.0;
        // Find table index and weight for linear interpolation
        i0 = int(floor(real_i));
        i1 = int(ceil(real_i));
    }
    w1 = real_i - float(i0);
    w0 = float(i1) - real_i;

    // MB (02.10.204): Look for the table value. In the Helium update the tables
    // are a 2D array with shape (N_tau, N_freq) that is table.T.ravel() before
    // being passed to C++ routine.
    double tab = table[int(i0 + nf * (NumTau + 1))] * w0 +
                 table[int(i1 + nf * (NumTau + 1))] * w1;
    // printf("%d %.3f %d %d %.3f %.3f %3f %.1e %.1e %e\n", nf, real_i, i0, i1,
    // w0, w1, tau, table[int(i0+nf*(NumTau+1))], table[int(i1+nf*(NumTau+1))],
    // tab); printf("%d %d %f %.1e %.1e\n", nf, i0, tau,
    // table[int(i0+nf*(NumTau+1))], table[int(i1+nf*(NumTau+1))]);
    return tab;
}
/*
double logtau;
double real_i, residual;
int i0, i1;
// Find table index and do linear interpolation
// Recall that tau(0) = 0 and tau(1:NumTau) ~ logspace(minlogtau,maxlogtau) (so
in reality the table has size NumTau+1) logtau = log10(max(1.0e-20,tau)); real_i
= min(float(NumTau),max(0.0,1.0+(logtau-minlogtau)/dlogtau)); i0 = int(real_i);
i1 = min(NumTau, i0+1);
residual = real_i - double(i0);
printf("%i %i %i %e %e\n", i0, nf, int(i0+nf*NumTau), tau,
table[int(i0+nf*NumTau)]); return table[int(i0+nf*NumTau)] +
residual*(table[int(i1+nf*NumTau)] - table[int(i0+nf*NumTau)]);
*/
