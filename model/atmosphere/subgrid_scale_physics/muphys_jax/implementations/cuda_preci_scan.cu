#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void cuda_precip_scan(
    int nlev, int ncells, int nspecies,
    const float* __restrict__ prefactor,
    const float* __restrict__ exponent,
    const float* __restrict__ offset,
    const float* __restrict__ zeta,
    const float* __restrict__ vc,
    const float* __restrict__ q,
    const float* __restrict__ rho,
    const bool* __restrict__ mask,
    float* __restrict__ q_out,
    float* __restrict__ flx_out
) {
    int cell = blockIdx.x;
    int species = blockIdx.y;
    if (cell >= ncells || species >= nspecies) return;

    float q_prev = 0.0f, flx_prev = 0.0f, rhox_prev = 0.0f;
    bool activated_prev = false;

    for (int k = 0; k < nlev; ++k) {
        int idx = species * nlev * ncells + k * ncells + cell;

        float pf = prefactor[idx];
        float ex = exponent[idx];
        float off = offset[idx];
        float z = zeta[idx];
        float v = vc[idx];
        float qk = q[idx];
        float r = rho[idx];
        bool m = mask[idx];

        bool activated = activated_prev || m;
        float rho_x = qk * r;
        float flx_eff = (rho_x / z) + 2.0f * flx_prev;
        float fall_speed = pf * powf(rho_x + off, ex);
        float flx_partial = fminf(rho_x * v * fall_speed, flx_eff);
        float vt_active = v * pf * powf(rhox_prev + off, ex);
        float vt = activated_prev ? vt_active : 0.0f;
        float q_activated = (z * (flx_eff - flx_partial)) / ((1.0f + z * vt) * r);
        float flx_activated = (q_activated * r * vt + flx_partial) * 0.5f;

        float qk_out = activated ? q_activated : qk;
        float flxk_out = activated ? flx_activated : 0.0f;

        q_out[idx] = qk_out;
        flx_out[idx] = flxk_out;

        q_prev = qk_out;
        flx_prev = flxk_out;
        rhox_prev = qk_out * r;
        activated_prev = activated;
    }
}