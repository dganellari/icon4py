
#include <hip/hip_runtime.h>
#include <dace/dace.h>

                                                                                  ////__DACE:0
struct theta_shared_probe_native_state_t {                                        ////__DACE:0
    dace::cuda::Context *gpu_context;                                             ////__DACE:0
};                                                                                ////__DACE:0
                                                                                  ////__DACE:0


DACE_EXPORTED int __dace_init_cuda(theta_shared_probe_native_state_t *__state, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __grf_tend_vn_K_stride, int __gt_conn_C2E2CO_neighbor_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __ikoffset_E2C_stride, int __ikoffset_K_stride, int __next_vn_K_stride, int __normal_wind_iau_increment_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, int __zdiff_gradp_E2C_stride, int __zdiff_gradp_K_stride, double dtime);
DACE_EXPORTED int __dace_exit_cuda(theta_shared_probe_native_state_t *__state);
DACE_EXPORTED int __dace_gpu_last_error(theta_shared_probe_native_state_t *__state);
DACE_EXPORTED void __dace_gpu_drain_error(theta_shared_probe_native_state_t *__state);
DACE_EXPORTED bool __dace_gpu_set_stream(theta_shared_probe_native_state_t *__state, int streamid, gpuStream_t stream);
DACE_EXPORTED void __dace_gpu_set_all_streams(theta_shared_probe_native_state_t *__state, gpuStream_t stream);

DACE_DFI void reduce_0_0_306(double* __restrict__ _in, double&  _out) {           ////__DACE:0:0:306
    ////__DACE:52
    {                                                                                 ////__DACE:52
        ////__DACE:52
        {                                                                                 ////__DACE:52:0:0    ////__DACE:52
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {                                       ////__DACE:52:0:0    ////__DACE:52
                {                                                                         ////__DACE:52:0:1    ////__DACE:52
                    double __out;                                                                     ////__DACE:52:0:1    ////__DACE:52:0:1    ////__DACE:52
                    ////__DACE:52:0:1                                                     ////__DACE:52:0:1    ////__DACE:52
                    ///////////////////                                                               ////__DACE:52:0:1    ////__DACE:52:0:1    ////__DACE:52
                    // Tasklet code (reduce_init)                                                     ////__DACE:52:0:1    ////__DACE:52:0:1    ////__DACE:52
                    __out = 0;                                                                        ////__DACE:52:0:1    ////__DACE:52:0:1    ////__DACE:52
                    ///////////////////                                                               ////__DACE:52:0:1    ////__DACE:52:0:1    ////__DACE:52
                    ////__DACE:52:0:1                                                     ////__DACE:52:0:1    ////__DACE:52
                    _out = __out;                                                                     ////__DACE:52:0:1    ////__DACE:52:0:1    ////__DACE:52
                }                                                                         ////__DACE:52:0:1    ////__DACE:52
            }                                                                             ////__DACE:52:0:2    ////__DACE:52
        }                                                                                 ////__DACE:52:0:2    ////__DACE:52
        ////__DACE:52
    }                                                                                 ////__DACE:52
    {                                                                                 ////__DACE:52
        ////__DACE:52
        {                                                                                 ////__DACE:52:1:0    ////__DACE:52
            for (auto _i0 = 0; _i0 < 4; _i0 += 1) {                                       ////__DACE:52:1:0    ////__DACE:52
                {                                                                         ////__DACE:52:1:2    ////__DACE:52
                    double __inp = _in[_i0];                                                          ////__DACE:52:1:3,2    ////__DACE:52:1:2    ////__DACE:52
                    double __out;                                                                     ////__DACE:52:1:2    ////__DACE:52:1:2    ////__DACE:52
                    ////__DACE:52:1:2                                                     ////__DACE:52:1:2    ////__DACE:52
                    ///////////////////                                                               ////__DACE:52:1:2    ////__DACE:52:1:2    ////__DACE:52
                    // Tasklet code (identity)                                                        ////__DACE:52:1:2    ////__DACE:52:1:2    ////__DACE:52
                    __out = __inp;                                                                    ////__DACE:52:1:2    ////__DACE:52:1:2    ////__DACE:52
                    ///////////////////                                                               ////__DACE:52:1:2    ////__DACE:52:1:2    ////__DACE:52
                    ////__DACE:52:1:2                                                     ////__DACE:52:1:2    ////__DACE:52
                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(&_out, __out);          ////__DACE:52:1:2    ////__DACE:52:1:2    ////__DACE:52
                }                                                                         ////__DACE:52:1:2    ////__DACE:52
            }                                                                             ////__DACE:52:1:1    ////__DACE:52
        }                                                                                 ////__DACE:52:1:1    ////__DACE:52
        ////__DACE:52
    }                                                                                 ////__DACE:52
}                                                                                 ////__DACE:0:0:306
////__DACE:0:0:306
DACE_DFI void if_stmt_0_0_0_194(const bool&  __arg1___, const bool&  __arg2, const bool&  __cond, bool&  __output) {    ////__DACE:0:0:194
    ////__DACE:29
    if (__cond) {                                                                     ////__DACE:29
        {                                                                             ////__DACE:29
            ////__DACE:29
            {                                                                                 ////__DACE:31:0:2    ////__DACE:29
                bool _cpy_in = __arg1___;                                                         ////__DACE:31:0:1,2    ////__DACE:31:0:2    ////__DACE:29
                bool _cpy_out;                                                                    ////__DACE:31:0:2    ////__DACE:31:0:2    ////__DACE:29
                ////__DACE:31:0:2                                                             ////__DACE:31:0:2    ////__DACE:29
                ///////////////////                                                               ////__DACE:31:0:2    ////__DACE:31:0:2    ////__DACE:29
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:31:0:2    ////__DACE:31:0:2    ////__DACE:29
                _cpy_out = _cpy_in;                                                               ////__DACE:31:0:2    ////__DACE:31:0:2    ////__DACE:29
                ///////////////////                                                               ////__DACE:31:0:2    ////__DACE:31:0:2    ////__DACE:29
                ////__DACE:31:0:2                                                             ////__DACE:31:0:2    ////__DACE:29
                __output = _cpy_out;                                                              ////__DACE:31:0:2    ////__DACE:31:0:2    ////__DACE:29
            }                                                                                 ////__DACE:31:0:2    ////__DACE:29
            ////__DACE:29
        }                                                                             ////__DACE:29
    } else {                                                                          ////__DACE:29
        {                                                                             ////__DACE:29
            ////__DACE:29
            {                                                                                 ////__DACE:32:0:2    ////__DACE:29
                bool _cpy_in = __arg2;                                                            ////__DACE:32:0:1,2    ////__DACE:32:0:2    ////__DACE:29
                bool _cpy_out;                                                                    ////__DACE:32:0:2    ////__DACE:32:0:2    ////__DACE:29
                ////__DACE:32:0:2                                                             ////__DACE:32:0:2    ////__DACE:29
                ///////////////////                                                               ////__DACE:32:0:2    ////__DACE:32:0:2    ////__DACE:29
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:32:0:2    ////__DACE:32:0:2    ////__DACE:29
                _cpy_out = _cpy_in;                                                               ////__DACE:32:0:2    ////__DACE:32:0:2    ////__DACE:29
                ///////////////////                                                               ////__DACE:32:0:2    ////__DACE:32:0:2    ////__DACE:29
                ////__DACE:32:0:2                                                             ////__DACE:32:0:2    ////__DACE:29
                __output = _cpy_out;                                                              ////__DACE:32:0:2    ////__DACE:32:0:2    ////__DACE:29
            }                                                                                 ////__DACE:32:0:2    ////__DACE:29
            ////__DACE:29
        }                                                                             ////__DACE:29
    }                                                                                 ////__DACE:29
}                                                                                 ////__DACE:0:0:194
////__DACE:0:0:194
DACE_DFI void if_stmt_1_0_0_199(const double&  __arg1___, const double&  __arg1____from_cb_fusion_3, const double&  __arg2, const double&  __arg2_from_cb_fusion_3, const bool&  __cond, double&  __output, double&  __output_from_cb_fusion_3) {    ////__DACE:0:0:199
    ////__DACE:33
    if (__cond) {                                                                     ////__DACE:33
        {                                                                             ////__DACE:33
            ////__DACE:33
            {                                                                                 ////__DACE:35:0:4    ////__DACE:33
                double _cpy_in = __arg1___;                                                       ////__DACE:35:0:1,4    ////__DACE:35:0:4    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                ////__DACE:35:0:4                                                             ////__DACE:35:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                ////__DACE:35:0:4                                                             ////__DACE:35:0:4    ////__DACE:33
                __output = _cpy_out;                                                              ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
            }                                                                                 ////__DACE:35:0:4    ////__DACE:33
            {                                                                                 ////__DACE:35:0:5    ////__DACE:33
                double _cpy_in = __arg1____from_cb_fusion_3;                                      ////__DACE:35:0:3,5    ////__DACE:35:0:5    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:35:0:5    ////__DACE:35:0:5    ////__DACE:33
                ////__DACE:35:0:5                                                             ////__DACE:35:0:5    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:5    ////__DACE:35:0:5    ////__DACE:33
                // Tasklet code (copy___arg1____from_cb_fusion_3_to___output_from_cb_fusion_3)    ////__DACE:35:0:5    ////__DACE:35:0:5    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:35:0:5    ////__DACE:35:0:5    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:5    ////__DACE:35:0:5    ////__DACE:33
                ////__DACE:35:0:5                                                             ////__DACE:35:0:5    ////__DACE:33
                __output_from_cb_fusion_3 = _cpy_out;                                             ////__DACE:35:0:5    ////__DACE:35:0:5    ////__DACE:33
            }                                                                                 ////__DACE:35:0:5    ////__DACE:33
            ////__DACE:33
        }                                                                             ////__DACE:33
    } else {                                                                          ////__DACE:33
        {                                                                             ////__DACE:33
            ////__DACE:33
            {                                                                                 ////__DACE:36:0:4    ////__DACE:33
                double _cpy_in = __arg2;                                                          ////__DACE:36:0:1,4    ////__DACE:36:0:4    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                ////__DACE:36:0:4                                                             ////__DACE:36:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                ////__DACE:36:0:4                                                             ////__DACE:36:0:4    ////__DACE:33
                __output = _cpy_out;                                                              ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
            }                                                                                 ////__DACE:36:0:4    ////__DACE:33
            {                                                                                 ////__DACE:36:0:5    ////__DACE:33
                double _cpy_in = __arg2_from_cb_fusion_3;                                         ////__DACE:36:0:3,5    ////__DACE:36:0:5    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:36:0:5    ////__DACE:36:0:5    ////__DACE:33
                ////__DACE:36:0:5                                                             ////__DACE:36:0:5    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:5    ////__DACE:36:0:5    ////__DACE:33
                // Tasklet code (copy___arg2_from_cb_fusion_3_to___output_from_cb_fusion_3)       ////__DACE:36:0:5    ////__DACE:36:0:5    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:36:0:5    ////__DACE:36:0:5    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:5    ////__DACE:36:0:5    ////__DACE:33
                ////__DACE:36:0:5                                                             ////__DACE:36:0:5    ////__DACE:33
                __output_from_cb_fusion_3 = _cpy_out;                                             ////__DACE:36:0:5    ////__DACE:36:0:5    ////__DACE:33
            }                                                                                 ////__DACE:36:0:5    ////__DACE:33
            ////__DACE:33
        }                                                                             ////__DACE:33
    }                                                                                 ////__DACE:33
}                                                                                 ////__DACE:0:0:199
////__DACE:0:0:199
DACE_DFI void if_stmt_4_0_0_210(const bool&  __cond, const double&  __map_fusion_gtir_tmp_21_1_1, const double&  __map_fusion_gtir_tmp_33_1_1, const double&  gtir_tmp_34_1, const double&  gtir_tmp_38_1, const double&  gtir_tmp_44_1, const double&  gtir_tmp_48_1, const double&  gtir_tmp_56_1, const double&  gtir_tmp_60_1, const double&  gtir_tmp_66_1, const double&  gtir_tmp_70_1, double&  __output, double&  __output_from_cb_fusion_2) {    ////__DACE:0:0:210
    ////__DACE:37
    if (__cond) {                                                                     ////__DACE:37
        {                                                                             ////__DACE:37
            double __arg1____;                                                                ////__DACE:39:0:1    ////__DACE:37
            double __arg1____from_cb_fusion_2;                                                ////__DACE:39:0:3    ////__DACE:37
            double __map_fusion_gtir_tmp_63_1;                                                ////__DACE:39:0:5    ////__DACE:37
            double __map_fusion_gtir_tmp_59_1;                                                ////__DACE:39:0:7    ////__DACE:37
            double __map_fusion_gtir_tmp_41_1;                                                ////__DACE:39:0:10    ////__DACE:37
            double __map_fusion_gtir_tmp_37_1;                                                ////__DACE:39:0:12    ////__DACE:37
            ////__DACE:37
            {                                                                                 ////__DACE:39:0:6    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_60_1;                                               ////__DACE:39:0:14,6    ////__DACE:39:0:6    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1;                                ////__DACE:39:0:15,6    ////__DACE:39:0:6    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                ////__DACE:39:0:6                                                             ////__DACE:39:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                // Tasklet code (tlet_21_multiplies_1)                                            ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                ////__DACE:39:0:6                                                             ////__DACE:39:0:6    ////__DACE:37
                __map_fusion_gtir_tmp_63_1 = __tlet_result;                                       ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
            }                                                                                 ////__DACE:39:0:6    ////__DACE:37
            {                                                                                 ////__DACE:39:0:8    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_56_1;                                               ////__DACE:39:0:16,8    ////__DACE:39:0:8    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1;                                ////__DACE:39:0:17,8    ////__DACE:39:0:8    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                ////__DACE:39:0:8                                                             ////__DACE:39:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                // Tasklet code (tlet_20_multiplies_1)                                            ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                ////__DACE:39:0:8                                                             ////__DACE:39:0:8    ////__DACE:37
                __map_fusion_gtir_tmp_59_1 = __tlet_result;                                       ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
            }                                                                                 ////__DACE:39:0:8    ////__DACE:37
            {                                                                                 ////__DACE:39:0:4    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_63_1;                                  ////__DACE:39:0:5,4    ////__DACE:39:0:4    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_59_1;                                  ////__DACE:39:0:7,4    ////__DACE:39:0:4    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                ////__DACE:39:0:4                                                             ////__DACE:39:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                // Tasklet code (tlet_22_plus_1)                                                  ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                ////__DACE:39:0:4                                                             ////__DACE:39:0:4    ////__DACE:37
                __arg1____ = __tlet_result;                                                       ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
            }                                                                                 ////__DACE:39:0:4    ////__DACE:37
            {                                                                                 ////__DACE:39:0:20    ////__DACE:37
                double _cpy_in = __arg1____;                                                      ////__DACE:39:0:1,20    ////__DACE:39:0:20    ////__DACE:37
                double _cpy_out;                                                                  ////__DACE:39:0:20    ////__DACE:39:0:20    ////__DACE:37
                ////__DACE:39:0:20                                                            ////__DACE:39:0:20    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:20    ////__DACE:39:0:20    ////__DACE:37
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:39:0:20    ////__DACE:39:0:20    ////__DACE:37
                _cpy_out = _cpy_in;                                                               ////__DACE:39:0:20    ////__DACE:39:0:20    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:20    ////__DACE:39:0:20    ////__DACE:37
                ////__DACE:39:0:20                                                            ////__DACE:39:0:20    ////__DACE:37
                __output = _cpy_out;                                                              ////__DACE:39:0:20    ////__DACE:39:0:20    ////__DACE:37
            }                                                                                 ////__DACE:39:0:20    ////__DACE:37
            {                                                                                 ////__DACE:39:0:11    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1;                                ////__DACE:39:0:15,11    ////__DACE:39:0:11    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_38_1;                                               ////__DACE:39:0:18,11    ////__DACE:39:0:11    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:11    ////__DACE:39:0:11    ////__DACE:37
                ////__DACE:39:0:11                                                            ////__DACE:39:0:11    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:11    ////__DACE:39:0:11    ////__DACE:37
                // Tasklet code (tlet_15_multiplies_1)                                            ////__DACE:39:0:11    ////__DACE:39:0:11    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:39:0:11    ////__DACE:39:0:11    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:11    ////__DACE:39:0:11    ////__DACE:37
                ////__DACE:39:0:11                                                            ////__DACE:39:0:11    ////__DACE:37
                __map_fusion_gtir_tmp_41_1 = __tlet_result;                                       ////__DACE:39:0:11    ////__DACE:39:0:11    ////__DACE:37
            }                                                                                 ////__DACE:39:0:11    ////__DACE:37
            {                                                                                 ////__DACE:39:0:13    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1;                                ////__DACE:39:0:17,13    ////__DACE:39:0:13    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_34_1;                                               ////__DACE:39:0:19,13    ////__DACE:39:0:13    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:13    ////__DACE:39:0:13    ////__DACE:37
                ////__DACE:39:0:13                                                            ////__DACE:39:0:13    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:13    ////__DACE:39:0:13    ////__DACE:37
                // Tasklet code (tlet_14_multiplies_1)                                            ////__DACE:39:0:13    ////__DACE:39:0:13    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:39:0:13    ////__DACE:39:0:13    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:13    ////__DACE:39:0:13    ////__DACE:37
                ////__DACE:39:0:13                                                            ////__DACE:39:0:13    ////__DACE:37
                __map_fusion_gtir_tmp_37_1 = __tlet_result;                                       ////__DACE:39:0:13    ////__DACE:39:0:13    ////__DACE:37
            }                                                                                 ////__DACE:39:0:13    ////__DACE:37
            {                                                                                 ////__DACE:39:0:9    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_41_1;                                  ////__DACE:39:0:10,9    ////__DACE:39:0:9    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_37_1;                                  ////__DACE:39:0:12,9    ////__DACE:39:0:9    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:9    ////__DACE:39:0:9    ////__DACE:37
                ////__DACE:39:0:9                                                             ////__DACE:39:0:9    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:9    ////__DACE:39:0:9    ////__DACE:37
                // Tasklet code (tlet_16_plus_1)                                                  ////__DACE:39:0:9    ////__DACE:39:0:9    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:39:0:9    ////__DACE:39:0:9    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:9    ////__DACE:39:0:9    ////__DACE:37
                ////__DACE:39:0:9                                                             ////__DACE:39:0:9    ////__DACE:37
                __arg1____from_cb_fusion_2 = __tlet_result;                                       ////__DACE:39:0:9    ////__DACE:39:0:9    ////__DACE:37
            }                                                                                 ////__DACE:39:0:9    ////__DACE:37
            {                                                                                 ////__DACE:39:0:21    ////__DACE:37
                double _cpy_in = __arg1____from_cb_fusion_2;                                      ////__DACE:39:0:3,21    ////__DACE:39:0:21    ////__DACE:37
                double _cpy_out;                                                                  ////__DACE:39:0:21    ////__DACE:39:0:21    ////__DACE:37
                ////__DACE:39:0:21                                                            ////__DACE:39:0:21    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:21    ////__DACE:39:0:21    ////__DACE:37
                // Tasklet code (copy___arg1____from_cb_fusion_2_to___output_from_cb_fusion_2)    ////__DACE:39:0:21    ////__DACE:39:0:21    ////__DACE:37
                _cpy_out = _cpy_in;                                                               ////__DACE:39:0:21    ////__DACE:39:0:21    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:21    ////__DACE:39:0:21    ////__DACE:37
                ////__DACE:39:0:21                                                            ////__DACE:39:0:21    ////__DACE:37
                __output_from_cb_fusion_2 = _cpy_out;                                             ////__DACE:39:0:21    ////__DACE:39:0:21    ////__DACE:37
            }                                                                                 ////__DACE:39:0:21    ////__DACE:37
            ////__DACE:37
        }                                                                             ////__DACE:37
    } else {                                                                          ////__DACE:37
        {                                                                             ////__DACE:37
            double __arg2_;                                                                   ////__DACE:40:0:1    ////__DACE:37
            double __arg2_from_cb_fusion_2;                                                   ////__DACE:40:0:3    ////__DACE:37
            double __map_fusion_gtir_tmp_73_1;                                                ////__DACE:40:0:5    ////__DACE:37
            double __map_fusion_gtir_tmp_69_1;                                                ////__DACE:40:0:7    ////__DACE:37
            double __map_fusion_gtir_tmp_51_1;                                                ////__DACE:40:0:10    ////__DACE:37
            double __map_fusion_gtir_tmp_47_1;                                                ////__DACE:40:0:12    ////__DACE:37
            ////__DACE:37
            {                                                                                 ////__DACE:40:0:6    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_70_1;                                               ////__DACE:40:0:14,6    ////__DACE:40:0:6    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1;                                ////__DACE:40:0:15,6    ////__DACE:40:0:6    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                ////__DACE:40:0:6                                                             ////__DACE:40:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                // Tasklet code (tlet_24_multiplies_1)                                            ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                ////__DACE:40:0:6                                                             ////__DACE:40:0:6    ////__DACE:37
                __map_fusion_gtir_tmp_73_1 = __tlet_result;                                       ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
            }                                                                                 ////__DACE:40:0:6    ////__DACE:37
            {                                                                                 ////__DACE:40:0:8    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_66_1;                                               ////__DACE:40:0:16,8    ////__DACE:40:0:8    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1;                                ////__DACE:40:0:17,8    ////__DACE:40:0:8    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                ////__DACE:40:0:8                                                             ////__DACE:40:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                // Tasklet code (tlet_23_multiplies_1)                                            ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                ////__DACE:40:0:8                                                             ////__DACE:40:0:8    ////__DACE:37
                __map_fusion_gtir_tmp_69_1 = __tlet_result;                                       ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
            }                                                                                 ////__DACE:40:0:8    ////__DACE:37
            {                                                                                 ////__DACE:40:0:4    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_73_1;                                  ////__DACE:40:0:5,4    ////__DACE:40:0:4    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_69_1;                                  ////__DACE:40:0:7,4    ////__DACE:40:0:4    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                ////__DACE:40:0:4                                                             ////__DACE:40:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                // Tasklet code (tlet_25_plus_1)                                                  ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                ////__DACE:40:0:4                                                             ////__DACE:40:0:4    ////__DACE:37
                __arg2_ = __tlet_result;                                                          ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
            }                                                                                 ////__DACE:40:0:4    ////__DACE:37
            {                                                                                 ////__DACE:40:0:20    ////__DACE:37
                double _cpy_in = __arg2_;                                                         ////__DACE:40:0:1,20    ////__DACE:40:0:20    ////__DACE:37
                double _cpy_out;                                                                  ////__DACE:40:0:20    ////__DACE:40:0:20    ////__DACE:37
                ////__DACE:40:0:20                                                            ////__DACE:40:0:20    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:20    ////__DACE:40:0:20    ////__DACE:37
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:40:0:20    ////__DACE:40:0:20    ////__DACE:37
                _cpy_out = _cpy_in;                                                               ////__DACE:40:0:20    ////__DACE:40:0:20    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:20    ////__DACE:40:0:20    ////__DACE:37
                ////__DACE:40:0:20                                                            ////__DACE:40:0:20    ////__DACE:37
                __output = _cpy_out;                                                              ////__DACE:40:0:20    ////__DACE:40:0:20    ////__DACE:37
            }                                                                                 ////__DACE:40:0:20    ////__DACE:37
            {                                                                                 ////__DACE:40:0:11    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_48_1;                                               ////__DACE:40:0:18,11    ////__DACE:40:0:11    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1;                                ////__DACE:40:0:15,11    ////__DACE:40:0:11    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:11    ////__DACE:40:0:11    ////__DACE:37
                ////__DACE:40:0:11                                                            ////__DACE:40:0:11    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:11    ////__DACE:40:0:11    ////__DACE:37
                // Tasklet code (tlet_18_multiplies_1)                                            ////__DACE:40:0:11    ////__DACE:40:0:11    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:40:0:11    ////__DACE:40:0:11    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:11    ////__DACE:40:0:11    ////__DACE:37
                ////__DACE:40:0:11                                                            ////__DACE:40:0:11    ////__DACE:37
                __map_fusion_gtir_tmp_51_1 = __tlet_result;                                       ////__DACE:40:0:11    ////__DACE:40:0:11    ////__DACE:37
            }                                                                                 ////__DACE:40:0:11    ////__DACE:37
            {                                                                                 ////__DACE:40:0:13    ////__DACE:37
                double __tlet_arg1 = gtir_tmp_44_1;                                               ////__DACE:40:0:19,13    ////__DACE:40:0:13    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1;                                ////__DACE:40:0:17,13    ////__DACE:40:0:13    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:13    ////__DACE:40:0:13    ////__DACE:37
                ////__DACE:40:0:13                                                            ////__DACE:40:0:13    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:13    ////__DACE:40:0:13    ////__DACE:37
                // Tasklet code (tlet_17_multiplies_1)                                            ////__DACE:40:0:13    ////__DACE:40:0:13    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:40:0:13    ////__DACE:40:0:13    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:13    ////__DACE:40:0:13    ////__DACE:37
                ////__DACE:40:0:13                                                            ////__DACE:40:0:13    ////__DACE:37
                __map_fusion_gtir_tmp_47_1 = __tlet_result;                                       ////__DACE:40:0:13    ////__DACE:40:0:13    ////__DACE:37
            }                                                                                 ////__DACE:40:0:13    ////__DACE:37
            {                                                                                 ////__DACE:40:0:9    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_51_1;                                  ////__DACE:40:0:10,9    ////__DACE:40:0:9    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_47_1;                                  ////__DACE:40:0:12,9    ////__DACE:40:0:9    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:9    ////__DACE:40:0:9    ////__DACE:37
                ////__DACE:40:0:9                                                             ////__DACE:40:0:9    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:9    ////__DACE:40:0:9    ////__DACE:37
                // Tasklet code (tlet_19_plus_1)                                                  ////__DACE:40:0:9    ////__DACE:40:0:9    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:40:0:9    ////__DACE:40:0:9    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:9    ////__DACE:40:0:9    ////__DACE:37
                ////__DACE:40:0:9                                                             ////__DACE:40:0:9    ////__DACE:37
                __arg2_from_cb_fusion_2 = __tlet_result;                                          ////__DACE:40:0:9    ////__DACE:40:0:9    ////__DACE:37
            }                                                                                 ////__DACE:40:0:9    ////__DACE:37
            {                                                                                 ////__DACE:40:0:21    ////__DACE:37
                double _cpy_in = __arg2_from_cb_fusion_2;                                         ////__DACE:40:0:3,21    ////__DACE:40:0:21    ////__DACE:37
                double _cpy_out;                                                                  ////__DACE:40:0:21    ////__DACE:40:0:21    ////__DACE:37
                ////__DACE:40:0:21                                                            ////__DACE:40:0:21    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:21    ////__DACE:40:0:21    ////__DACE:37
                // Tasklet code (copy___arg2_from_cb_fusion_2_to___output_from_cb_fusion_2)       ////__DACE:40:0:21    ////__DACE:40:0:21    ////__DACE:37
                _cpy_out = _cpy_in;                                                               ////__DACE:40:0:21    ////__DACE:40:0:21    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:21    ////__DACE:40:0:21    ////__DACE:37
                ////__DACE:40:0:21                                                            ////__DACE:40:0:21    ////__DACE:37
                __output_from_cb_fusion_2 = _cpy_out;                                             ////__DACE:40:0:21    ////__DACE:40:0:21    ////__DACE:37
            }                                                                                 ////__DACE:40:0:21    ////__DACE:37
            ////__DACE:37
        }                                                                             ////__DACE:37
    }                                                                                 ////__DACE:37
}                                                                                 ////__DACE:0:0:210
////__DACE:0:0:210
DACE_DFI void if_stmt_7_0_0_226(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double&  gtir_tmp_54_1, const double&  gtir_tmp_76_1, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double* __restrict__ perturbed_rho_at_cells_on_model_levels, const double* __restrict__ reference_rho_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_rho_at_cells_on_model_levels_K_stride_0, int __reference_rho_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:226
    ////__DACE:45
    if (__cond) {                                                                     ////__DACE:45
        {                                                                             ////__DACE:45
            double __arg1___;                                                                 ////__DACE:47:0:1    ////__DACE:45
            double __map_fusion_gtir_tmp_191_1;                                               ////__DACE:47:0:3    ////__DACE:45
            double __map_fusion_gtir_tmp_189_1;                                               ////__DACE:47:0:5    ////__DACE:45
            double __map_fusion_gtir_tmp_187_1;                                               ////__DACE:47:0:7    ////__DACE:45
            double __map_fusion_gtir_tmp_185_1;                                               ////__DACE:47:0:9    ////__DACE:45
            double __map_fusion_gtir_tmp_183_1;                                               ////__DACE:47:0:11    ////__DACE:45
            double __map_fusion_gtir_tmp_181_1;                                               ////__DACE:47:0:13    ////__DACE:45
            double __map_fusion_gtir_tmp_179_1;                                               ////__DACE:47:0:15    ////__DACE:45
            ////__DACE:45
            {                                                                                 ////__DACE:47:0:6    ////__DACE:45
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:47:0:18,6    ////__DACE:47:0:6    ////__DACE:45
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:47:0:19,6    ////__DACE:47:0:6    ////__DACE:45
                double __tlet_val;                                                                ////__DACE:47:0:6    ////__DACE:47:0:6    ////__DACE:45
                ////__DACE:47:0:6                                                             ////__DACE:47:0:6    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:6    ////__DACE:47:0:6    ////__DACE:45
                // Tasklet code (tlet_75_deref_1)                                                 ////__DACE:47:0:6    ////__DACE:47:0:6    ////__DACE:45
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:47:0:6    ////__DACE:47:0:6    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:6    ////__DACE:47:0:6    ////__DACE:45
                ////__DACE:47:0:6                                                             ////__DACE:47:0:6    ////__DACE:45
                __map_fusion_gtir_tmp_189_1 = __tlet_val;                                         ////__DACE:47:0:6    ////__DACE:47:0:6    ////__DACE:45
            }                                                                                 ////__DACE:47:0:6    ////__DACE:45
            {                                                                                 ////__DACE:47:0:4    ////__DACE:45
                double __tlet_arg0 = gtir_tmp_76_1;                                               ////__DACE:47:0:17,4    ////__DACE:47:0:4    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_189_1;                                 ////__DACE:47:0:5,4    ////__DACE:47:0:4    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:47:0:4    ////__DACE:47:0:4    ////__DACE:45
                ////__DACE:47:0:4                                                             ////__DACE:47:0:4    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:4    ////__DACE:47:0:4    ////__DACE:45
                // Tasklet code (tlet_76_multiplies_1)                                            ////__DACE:47:0:4    ////__DACE:47:0:4    ////__DACE:45
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:47:0:4    ////__DACE:47:0:4    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:4    ////__DACE:47:0:4    ////__DACE:45
                ////__DACE:47:0:4                                                             ////__DACE:47:0:4    ////__DACE:45
                __map_fusion_gtir_tmp_191_1 = __tlet_result;                                      ////__DACE:47:0:4    ////__DACE:47:0:4    ////__DACE:45
            }                                                                                 ////__DACE:47:0:4    ////__DACE:45
            {                                                                                 ////__DACE:47:0:12    ////__DACE:45
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:47:0:21,12    ////__DACE:47:0:12    ////__DACE:45
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:47:0:19,12    ////__DACE:47:0:12    ////__DACE:45
                double __tlet_val;                                                                ////__DACE:47:0:12    ////__DACE:47:0:12    ////__DACE:45
                ////__DACE:47:0:12                                                            ////__DACE:47:0:12    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:12    ////__DACE:47:0:12    ////__DACE:45
                // Tasklet code (tlet_72_deref_1)                                                 ////__DACE:47:0:12    ////__DACE:47:0:12    ////__DACE:45
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:47:0:12    ////__DACE:47:0:12    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:12    ////__DACE:47:0:12    ////__DACE:45
                ////__DACE:47:0:12                                                            ////__DACE:47:0:12    ////__DACE:45
                __map_fusion_gtir_tmp_183_1 = __tlet_val;                                         ////__DACE:47:0:12    ////__DACE:47:0:12    ////__DACE:45
            }                                                                                 ////__DACE:47:0:12    ////__DACE:45
            {                                                                                 ////__DACE:47:0:10    ////__DACE:45
                double __tlet_arg0 = gtir_tmp_54_1;                                               ////__DACE:47:0:20,10    ////__DACE:47:0:10    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_183_1;                                 ////__DACE:47:0:11,10    ////__DACE:47:0:10    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:47:0:10    ////__DACE:47:0:10    ////__DACE:45
                ////__DACE:47:0:10                                                            ////__DACE:47:0:10    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:10    ////__DACE:47:0:10    ////__DACE:45
                // Tasklet code (tlet_73_multiplies_1)                                            ////__DACE:47:0:10    ////__DACE:47:0:10    ////__DACE:45
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:47:0:10    ////__DACE:47:0:10    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:10    ////__DACE:47:0:10    ////__DACE:45
                ////__DACE:47:0:10                                                            ////__DACE:47:0:10    ////__DACE:45
                __map_fusion_gtir_tmp_185_1 = __tlet_result;                                      ////__DACE:47:0:10    ////__DACE:47:0:10    ////__DACE:45
            }                                                                                 ////__DACE:47:0:10    ////__DACE:45
            {                                                                                 ////__DACE:47:0:16    ////__DACE:45
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:47:0:23,16    ////__DACE:47:0:16    ////__DACE:45
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:47:0:19,16    ////__DACE:47:0:16    ////__DACE:45
                double __tlet_val;                                                                ////__DACE:47:0:16    ////__DACE:47:0:16    ////__DACE:45
                ////__DACE:47:0:16                                                            ////__DACE:47:0:16    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:16    ////__DACE:47:0:16    ////__DACE:45
                // Tasklet code (tlet_70_deref_1)                                                 ////__DACE:47:0:16    ////__DACE:47:0:16    ////__DACE:45
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:47:0:16    ////__DACE:47:0:16    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:16    ////__DACE:47:0:16    ////__DACE:45
                ////__DACE:47:0:16                                                            ////__DACE:47:0:16    ////__DACE:45
                __map_fusion_gtir_tmp_179_1 = __tlet_val;                                         ////__DACE:47:0:16    ////__DACE:47:0:16    ////__DACE:45
            }                                                                                 ////__DACE:47:0:16    ////__DACE:45
            {                                                                                 ////__DACE:47:0:14    ////__DACE:45
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:47:0:22,14    ////__DACE:47:0:14    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_179_1;                                 ////__DACE:47:0:15,14    ////__DACE:47:0:14    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:47:0:14    ////__DACE:47:0:14    ////__DACE:45
                ////__DACE:47:0:14                                                            ////__DACE:47:0:14    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:14    ////__DACE:47:0:14    ////__DACE:45
                // Tasklet code (tlet_71_plus_1)                                                  ////__DACE:47:0:14    ////__DACE:47:0:14    ////__DACE:45
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:47:0:14    ////__DACE:47:0:14    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:14    ////__DACE:47:0:14    ////__DACE:45
                ////__DACE:47:0:14                                                            ////__DACE:47:0:14    ////__DACE:45
                __map_fusion_gtir_tmp_181_1 = __tlet_result;                                      ////__DACE:47:0:14    ////__DACE:47:0:14    ////__DACE:45
            }                                                                                 ////__DACE:47:0:14    ////__DACE:45
            {                                                                                 ////__DACE:47:0:8    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_185_1;                                 ////__DACE:47:0:9,8    ////__DACE:47:0:8    ////__DACE:45
                double __tlet_arg0 = __map_fusion_gtir_tmp_181_1;                                 ////__DACE:47:0:13,8    ////__DACE:47:0:8    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:47:0:8    ////__DACE:47:0:8    ////__DACE:45
                ////__DACE:47:0:8                                                             ////__DACE:47:0:8    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:8    ////__DACE:47:0:8    ////__DACE:45
                // Tasklet code (tlet_74_plus_1)                                                  ////__DACE:47:0:8    ////__DACE:47:0:8    ////__DACE:45
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:47:0:8    ////__DACE:47:0:8    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:8    ////__DACE:47:0:8    ////__DACE:45
                ////__DACE:47:0:8                                                             ////__DACE:47:0:8    ////__DACE:45
                __map_fusion_gtir_tmp_187_1 = __tlet_result;                                      ////__DACE:47:0:8    ////__DACE:47:0:8    ////__DACE:45
            }                                                                                 ////__DACE:47:0:8    ////__DACE:45
            {                                                                                 ////__DACE:47:0:2    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_191_1;                                 ////__DACE:47:0:3,2    ////__DACE:47:0:2    ////__DACE:45
                double __tlet_arg0 = __map_fusion_gtir_tmp_187_1;                                 ////__DACE:47:0:7,2    ////__DACE:47:0:2    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:47:0:2    ////__DACE:47:0:2    ////__DACE:45
                ////__DACE:47:0:2                                                             ////__DACE:47:0:2    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:2    ////__DACE:47:0:2    ////__DACE:45
                // Tasklet code (tlet_77_plus_1)                                                  ////__DACE:47:0:2    ////__DACE:47:0:2    ////__DACE:45
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:47:0:2    ////__DACE:47:0:2    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:2    ////__DACE:47:0:2    ////__DACE:45
                ////__DACE:47:0:2                                                             ////__DACE:47:0:2    ////__DACE:45
                __arg1___ = __tlet_result;                                                        ////__DACE:47:0:2    ////__DACE:47:0:2    ////__DACE:45
            }                                                                                 ////__DACE:47:0:2    ////__DACE:45
            {                                                                                 ////__DACE:47:0:24    ////__DACE:45
                double _cpy_in = __arg1___;                                                       ////__DACE:47:0:1,24    ////__DACE:47:0:24    ////__DACE:45
                double _cpy_out;                                                                  ////__DACE:47:0:24    ////__DACE:47:0:24    ////__DACE:45
                ////__DACE:47:0:24                                                            ////__DACE:47:0:24    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:24    ////__DACE:47:0:24    ////__DACE:45
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:47:0:24    ////__DACE:47:0:24    ////__DACE:45
                _cpy_out = _cpy_in;                                                               ////__DACE:47:0:24    ////__DACE:47:0:24    ////__DACE:45
                ///////////////////                                                               ////__DACE:47:0:24    ////__DACE:47:0:24    ////__DACE:45
                ////__DACE:47:0:24                                                            ////__DACE:47:0:24    ////__DACE:45
                __output = _cpy_out;                                                              ////__DACE:47:0:24    ////__DACE:47:0:24    ////__DACE:45
            }                                                                                 ////__DACE:47:0:24    ////__DACE:45
            ////__DACE:45
        }                                                                             ////__DACE:45
    } else {                                                                          ////__DACE:45
        {                                                                             ////__DACE:45
            double __arg2;                                                                    ////__DACE:48:0:1    ////__DACE:45
            double __map_fusion_gtir_tmp_207_1;                                               ////__DACE:48:0:3    ////__DACE:45
            double __map_fusion_gtir_tmp_205_1;                                               ////__DACE:48:0:5    ////__DACE:45
            double __map_fusion_gtir_tmp_203_1;                                               ////__DACE:48:0:7    ////__DACE:45
            double __map_fusion_gtir_tmp_201_1;                                               ////__DACE:48:0:9    ////__DACE:45
            double __map_fusion_gtir_tmp_199_1;                                               ////__DACE:48:0:11    ////__DACE:45
            double __map_fusion_gtir_tmp_197_1;                                               ////__DACE:48:0:13    ////__DACE:45
            double __map_fusion_gtir_tmp_195_1;                                               ////__DACE:48:0:15    ////__DACE:45
            ////__DACE:45
            {                                                                                 ////__DACE:48:0:6    ////__DACE:45
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:48:0:18,6    ////__DACE:48:0:6    ////__DACE:45
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:48:0:19,6    ////__DACE:48:0:6    ////__DACE:45
                double __tlet_val;                                                                ////__DACE:48:0:6    ////__DACE:48:0:6    ////__DACE:45
                ////__DACE:48:0:6                                                             ////__DACE:48:0:6    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:6    ////__DACE:48:0:6    ////__DACE:45
                // Tasklet code (tlet_83_deref_1)                                                 ////__DACE:48:0:6    ////__DACE:48:0:6    ////__DACE:45
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:48:0:6    ////__DACE:48:0:6    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:6    ////__DACE:48:0:6    ////__DACE:45
                ////__DACE:48:0:6                                                             ////__DACE:48:0:6    ////__DACE:45
                __map_fusion_gtir_tmp_205_1 = __tlet_val;                                         ////__DACE:48:0:6    ////__DACE:48:0:6    ////__DACE:45
            }                                                                                 ////__DACE:48:0:6    ////__DACE:45
            {                                                                                 ////__DACE:48:0:4    ////__DACE:45
                double __tlet_arg0 = gtir_tmp_76_1;                                               ////__DACE:48:0:17,4    ////__DACE:48:0:4    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_205_1;                                 ////__DACE:48:0:5,4    ////__DACE:48:0:4    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:48:0:4    ////__DACE:48:0:4    ////__DACE:45
                ////__DACE:48:0:4                                                             ////__DACE:48:0:4    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:4    ////__DACE:48:0:4    ////__DACE:45
                // Tasklet code (tlet_84_multiplies_1)                                            ////__DACE:48:0:4    ////__DACE:48:0:4    ////__DACE:45
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:48:0:4    ////__DACE:48:0:4    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:4    ////__DACE:48:0:4    ////__DACE:45
                ////__DACE:48:0:4                                                             ////__DACE:48:0:4    ////__DACE:45
                __map_fusion_gtir_tmp_207_1 = __tlet_result;                                      ////__DACE:48:0:4    ////__DACE:48:0:4    ////__DACE:45
            }                                                                                 ////__DACE:48:0:4    ////__DACE:45
            {                                                                                 ////__DACE:48:0:12    ////__DACE:45
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:48:0:21,12    ////__DACE:48:0:12    ////__DACE:45
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:48:0:19,12    ////__DACE:48:0:12    ////__DACE:45
                double __tlet_val;                                                                ////__DACE:48:0:12    ////__DACE:48:0:12    ////__DACE:45
                ////__DACE:48:0:12                                                            ////__DACE:48:0:12    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:12    ////__DACE:48:0:12    ////__DACE:45
                // Tasklet code (tlet_80_deref_1)                                                 ////__DACE:48:0:12    ////__DACE:48:0:12    ////__DACE:45
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:48:0:12    ////__DACE:48:0:12    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:12    ////__DACE:48:0:12    ////__DACE:45
                ////__DACE:48:0:12                                                            ////__DACE:48:0:12    ////__DACE:45
                __map_fusion_gtir_tmp_199_1 = __tlet_val;                                         ////__DACE:48:0:12    ////__DACE:48:0:12    ////__DACE:45
            }                                                                                 ////__DACE:48:0:12    ////__DACE:45
            {                                                                                 ////__DACE:48:0:10    ////__DACE:45
                double __tlet_arg0 = gtir_tmp_54_1;                                               ////__DACE:48:0:20,10    ////__DACE:48:0:10    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_199_1;                                 ////__DACE:48:0:11,10    ////__DACE:48:0:10    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:48:0:10    ////__DACE:48:0:10    ////__DACE:45
                ////__DACE:48:0:10                                                            ////__DACE:48:0:10    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:10    ////__DACE:48:0:10    ////__DACE:45
                // Tasklet code (tlet_81_multiplies_1)                                            ////__DACE:48:0:10    ////__DACE:48:0:10    ////__DACE:45
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:48:0:10    ////__DACE:48:0:10    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:10    ////__DACE:48:0:10    ////__DACE:45
                ////__DACE:48:0:10                                                            ////__DACE:48:0:10    ////__DACE:45
                __map_fusion_gtir_tmp_201_1 = __tlet_result;                                      ////__DACE:48:0:10    ////__DACE:48:0:10    ////__DACE:45
            }                                                                                 ////__DACE:48:0:10    ////__DACE:45
            {                                                                                 ////__DACE:48:0:16    ////__DACE:45
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:48:0:23,16    ////__DACE:48:0:16    ////__DACE:45
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:48:0:19,16    ////__DACE:48:0:16    ////__DACE:45
                double __tlet_val;                                                                ////__DACE:48:0:16    ////__DACE:48:0:16    ////__DACE:45
                ////__DACE:48:0:16                                                            ////__DACE:48:0:16    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:16    ////__DACE:48:0:16    ////__DACE:45
                // Tasklet code (tlet_78_deref_1)                                                 ////__DACE:48:0:16    ////__DACE:48:0:16    ////__DACE:45
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:48:0:16    ////__DACE:48:0:16    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:16    ////__DACE:48:0:16    ////__DACE:45
                ////__DACE:48:0:16                                                            ////__DACE:48:0:16    ////__DACE:45
                __map_fusion_gtir_tmp_195_1 = __tlet_val;                                         ////__DACE:48:0:16    ////__DACE:48:0:16    ////__DACE:45
            }                                                                                 ////__DACE:48:0:16    ////__DACE:45
            {                                                                                 ////__DACE:48:0:14    ////__DACE:45
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:48:0:22,14    ////__DACE:48:0:14    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_195_1;                                 ////__DACE:48:0:15,14    ////__DACE:48:0:14    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:48:0:14    ////__DACE:48:0:14    ////__DACE:45
                ////__DACE:48:0:14                                                            ////__DACE:48:0:14    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:14    ////__DACE:48:0:14    ////__DACE:45
                // Tasklet code (tlet_79_plus_1)                                                  ////__DACE:48:0:14    ////__DACE:48:0:14    ////__DACE:45
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:48:0:14    ////__DACE:48:0:14    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:14    ////__DACE:48:0:14    ////__DACE:45
                ////__DACE:48:0:14                                                            ////__DACE:48:0:14    ////__DACE:45
                __map_fusion_gtir_tmp_197_1 = __tlet_result;                                      ////__DACE:48:0:14    ////__DACE:48:0:14    ////__DACE:45
            }                                                                                 ////__DACE:48:0:14    ////__DACE:45
            {                                                                                 ////__DACE:48:0:8    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_201_1;                                 ////__DACE:48:0:9,8    ////__DACE:48:0:8    ////__DACE:45
                double __tlet_arg0 = __map_fusion_gtir_tmp_197_1;                                 ////__DACE:48:0:13,8    ////__DACE:48:0:8    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:48:0:8    ////__DACE:48:0:8    ////__DACE:45
                ////__DACE:48:0:8                                                             ////__DACE:48:0:8    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:8    ////__DACE:48:0:8    ////__DACE:45
                // Tasklet code (tlet_82_plus_1)                                                  ////__DACE:48:0:8    ////__DACE:48:0:8    ////__DACE:45
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:48:0:8    ////__DACE:48:0:8    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:8    ////__DACE:48:0:8    ////__DACE:45
                ////__DACE:48:0:8                                                             ////__DACE:48:0:8    ////__DACE:45
                __map_fusion_gtir_tmp_203_1 = __tlet_result;                                      ////__DACE:48:0:8    ////__DACE:48:0:8    ////__DACE:45
            }                                                                                 ////__DACE:48:0:8    ////__DACE:45
            {                                                                                 ////__DACE:48:0:2    ////__DACE:45
                double __tlet_arg1 = __map_fusion_gtir_tmp_207_1;                                 ////__DACE:48:0:3,2    ////__DACE:48:0:2    ////__DACE:45
                double __tlet_arg0 = __map_fusion_gtir_tmp_203_1;                                 ////__DACE:48:0:7,2    ////__DACE:48:0:2    ////__DACE:45
                double __tlet_result;                                                             ////__DACE:48:0:2    ////__DACE:48:0:2    ////__DACE:45
                ////__DACE:48:0:2                                                             ////__DACE:48:0:2    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:2    ////__DACE:48:0:2    ////__DACE:45
                // Tasklet code (tlet_85_plus_1)                                                  ////__DACE:48:0:2    ////__DACE:48:0:2    ////__DACE:45
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:48:0:2    ////__DACE:48:0:2    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:2    ////__DACE:48:0:2    ////__DACE:45
                ////__DACE:48:0:2                                                             ////__DACE:48:0:2    ////__DACE:45
                __arg2 = __tlet_result;                                                           ////__DACE:48:0:2    ////__DACE:48:0:2    ////__DACE:45
            }                                                                                 ////__DACE:48:0:2    ////__DACE:45
            {                                                                                 ////__DACE:48:0:24    ////__DACE:45
                double _cpy_in = __arg2;                                                          ////__DACE:48:0:1,24    ////__DACE:48:0:24    ////__DACE:45
                double _cpy_out;                                                                  ////__DACE:48:0:24    ////__DACE:48:0:24    ////__DACE:45
                ////__DACE:48:0:24                                                            ////__DACE:48:0:24    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:24    ////__DACE:48:0:24    ////__DACE:45
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:48:0:24    ////__DACE:48:0:24    ////__DACE:45
                _cpy_out = _cpy_in;                                                               ////__DACE:48:0:24    ////__DACE:48:0:24    ////__DACE:45
                ///////////////////                                                               ////__DACE:48:0:24    ////__DACE:48:0:24    ////__DACE:45
                ////__DACE:48:0:24                                                            ////__DACE:48:0:24    ////__DACE:45
                __output = _cpy_out;                                                              ////__DACE:48:0:24    ////__DACE:48:0:24    ////__DACE:45
            }                                                                                 ////__DACE:48:0:24    ////__DACE:45
            ////__DACE:45
        }                                                                             ////__DACE:45
    }                                                                                 ////__DACE:45
}                                                                                 ////__DACE:0:0:226
////__DACE:0:0:226
DACE_DFI void if_stmt_5_0_0_222(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double&  gtir_tmp_54_1, const double&  gtir_tmp_76_1, const double * __restrict__ gtir_tmp_95, const double* __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double* __restrict__ reference_theta_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_theta_v_at_cells_on_model_levels_K_stride_0, int __reference_theta_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:222
    ////__DACE:41
    if (__cond) {                                                                     ////__DACE:41
        {                                                                             ////__DACE:41
            double __arg1____;                                                                ////__DACE:43:0:1    ////__DACE:41
            double __map_fusion_gtir_tmp_118_1;                                               ////__DACE:43:0:3    ////__DACE:41
            double __map_fusion_gtir_tmp_116_1;                                               ////__DACE:43:0:5    ////__DACE:41
            double __map_fusion_gtir_tmp_114_1;                                               ////__DACE:43:0:7    ////__DACE:41
            double __map_fusion_gtir_tmp_112_1;                                               ////__DACE:43:0:9    ////__DACE:41
            double __map_fusion_gtir_tmp_110_1;                                               ////__DACE:43:0:11    ////__DACE:41
            double __map_fusion_gtir_tmp_108_1;                                               ////__DACE:43:0:13    ////__DACE:41
            double __map_fusion_gtir_tmp_106_1;                                               ////__DACE:43:0:15    ////__DACE:41
            ////__DACE:41
            {                                                                                 ////__DACE:43:0:6    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:43:0:18,6    ////__DACE:43:0:6    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:43:0:19,6    ////__DACE:43:0:6    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                ////__DACE:43:0:6                                                             ////__DACE:43:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                // Tasklet code (tlet_41_deref_1)                                                 ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                ////__DACE:43:0:6                                                             ////__DACE:43:0:6    ////__DACE:41
                __map_fusion_gtir_tmp_116_1 = __tlet_val;                                         ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
            }                                                                                 ////__DACE:43:0:6    ////__DACE:41
            {                                                                                 ////__DACE:43:0:4    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_76_1;                                               ////__DACE:43:0:17,4    ////__DACE:43:0:4    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_116_1;                                 ////__DACE:43:0:5,4    ////__DACE:43:0:4    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                ////__DACE:43:0:4                                                             ////__DACE:43:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                // Tasklet code (tlet_42_multiplies_1)                                            ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                ////__DACE:43:0:4                                                             ////__DACE:43:0:4    ////__DACE:41
                __map_fusion_gtir_tmp_118_1 = __tlet_result;                                      ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
            }                                                                                 ////__DACE:43:0:4    ////__DACE:41
            {                                                                                 ////__DACE:43:0:12    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:43:0:21,12    ////__DACE:43:0:12    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:43:0:19,12    ////__DACE:43:0:12    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                ////__DACE:43:0:12                                                            ////__DACE:43:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                // Tasklet code (tlet_38_deref_1)                                                 ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                ////__DACE:43:0:12                                                            ////__DACE:43:0:12    ////__DACE:41
                __map_fusion_gtir_tmp_110_1 = __tlet_val;                                         ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
            }                                                                                 ////__DACE:43:0:12    ////__DACE:41
            {                                                                                 ////__DACE:43:0:10    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_54_1;                                               ////__DACE:43:0:20,10    ////__DACE:43:0:10    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_110_1;                                 ////__DACE:43:0:11,10    ////__DACE:43:0:10    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                ////__DACE:43:0:10                                                            ////__DACE:43:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                // Tasklet code (tlet_39_multiplies_1)                                            ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                ////__DACE:43:0:10                                                            ////__DACE:43:0:10    ////__DACE:41
                __map_fusion_gtir_tmp_112_1 = __tlet_result;                                      ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
            }                                                                                 ////__DACE:43:0:10    ////__DACE:41
            {                                                                                 ////__DACE:43:0:16    ////__DACE:41
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:43:0:23,16    ////__DACE:43:0:16    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:43:0:19,16    ////__DACE:43:0:16    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                ////__DACE:43:0:16                                                            ////__DACE:43:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                // Tasklet code (tlet_36_deref_1)                                                 ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                ////__DACE:43:0:16                                                            ////__DACE:43:0:16    ////__DACE:41
                __map_fusion_gtir_tmp_106_1 = __tlet_val;                                         ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
            }                                                                                 ////__DACE:43:0:16    ////__DACE:41
            {                                                                                 ////__DACE:43:0:14    ////__DACE:41
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:43:0:22,14    ////__DACE:43:0:14    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_106_1;                                 ////__DACE:43:0:15,14    ////__DACE:43:0:14    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                ////__DACE:43:0:14                                                            ////__DACE:43:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                // Tasklet code (tlet_37_plus_1)                                                  ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                ////__DACE:43:0:14                                                            ////__DACE:43:0:14    ////__DACE:41
                __map_fusion_gtir_tmp_108_1 = __tlet_result;                                      ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
            }                                                                                 ////__DACE:43:0:14    ////__DACE:41
            {                                                                                 ////__DACE:43:0:8    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_112_1;                                 ////__DACE:43:0:9,8    ////__DACE:43:0:8    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_108_1;                                 ////__DACE:43:0:13,8    ////__DACE:43:0:8    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                ////__DACE:43:0:8                                                             ////__DACE:43:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                // Tasklet code (tlet_40_plus_1)                                                  ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                ////__DACE:43:0:8                                                             ////__DACE:43:0:8    ////__DACE:41
                __map_fusion_gtir_tmp_114_1 = __tlet_result;                                      ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
            }                                                                                 ////__DACE:43:0:8    ////__DACE:41
            {                                                                                 ////__DACE:43:0:2    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_118_1;                                 ////__DACE:43:0:3,2    ////__DACE:43:0:2    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_114_1;                                 ////__DACE:43:0:7,2    ////__DACE:43:0:2    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                ////__DACE:43:0:2                                                             ////__DACE:43:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                // Tasklet code (tlet_43_plus_1)                                                  ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                ////__DACE:43:0:2                                                             ////__DACE:43:0:2    ////__DACE:41
                __arg1____ = __tlet_result;                                                       ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
            }                                                                                 ////__DACE:43:0:2    ////__DACE:41
            {                                                                                 ////__DACE:43:0:24    ////__DACE:41
                double _cpy_in = __arg1____;                                                      ////__DACE:43:0:1,24    ////__DACE:43:0:24    ////__DACE:41
                double _cpy_out;                                                                  ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                ////__DACE:43:0:24                                                            ////__DACE:43:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                _cpy_out = _cpy_in;                                                               ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                ////__DACE:43:0:24                                                            ////__DACE:43:0:24    ////__DACE:41
                __output = _cpy_out;                                                              ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
            }                                                                                 ////__DACE:43:0:24    ////__DACE:41
            ////__DACE:41
        }                                                                             ////__DACE:41
    } else {                                                                          ////__DACE:41
        {                                                                             ////__DACE:41
            double __arg2_;                                                                   ////__DACE:44:0:1    ////__DACE:41
            double __map_fusion_gtir_tmp_134_1;                                               ////__DACE:44:0:3    ////__DACE:41
            double __map_fusion_gtir_tmp_132_1;                                               ////__DACE:44:0:5    ////__DACE:41
            double __map_fusion_gtir_tmp_130_1;                                               ////__DACE:44:0:7    ////__DACE:41
            double __map_fusion_gtir_tmp_128_1;                                               ////__DACE:44:0:9    ////__DACE:41
            double __map_fusion_gtir_tmp_126_1;                                               ////__DACE:44:0:11    ////__DACE:41
            double __map_fusion_gtir_tmp_124_1;                                               ////__DACE:44:0:13    ////__DACE:41
            double __map_fusion_gtir_tmp_122_1;                                               ////__DACE:44:0:15    ////__DACE:41
            ////__DACE:41
            {                                                                                 ////__DACE:44:0:6    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:44:0:18,6    ////__DACE:44:0:6    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:44:0:19,6    ////__DACE:44:0:6    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                ////__DACE:44:0:6                                                             ////__DACE:44:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                // Tasklet code (tlet_49_deref_1)                                                 ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                ////__DACE:44:0:6                                                             ////__DACE:44:0:6    ////__DACE:41
                __map_fusion_gtir_tmp_132_1 = __tlet_val;                                         ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
            }                                                                                 ////__DACE:44:0:6    ////__DACE:41
            {                                                                                 ////__DACE:44:0:4    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_76_1;                                               ////__DACE:44:0:17,4    ////__DACE:44:0:4    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_132_1;                                 ////__DACE:44:0:5,4    ////__DACE:44:0:4    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                ////__DACE:44:0:4                                                             ////__DACE:44:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                // Tasklet code (tlet_50_multiplies_1)                                            ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                ////__DACE:44:0:4                                                             ////__DACE:44:0:4    ////__DACE:41
                __map_fusion_gtir_tmp_134_1 = __tlet_result;                                      ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
            }                                                                                 ////__DACE:44:0:4    ////__DACE:41
            {                                                                                 ////__DACE:44:0:12    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:44:0:21,12    ////__DACE:44:0:12    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:44:0:19,12    ////__DACE:44:0:12    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                ////__DACE:44:0:12                                                            ////__DACE:44:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                // Tasklet code (tlet_46_deref_1)                                                 ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                ////__DACE:44:0:12                                                            ////__DACE:44:0:12    ////__DACE:41
                __map_fusion_gtir_tmp_126_1 = __tlet_val;                                         ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
            }                                                                                 ////__DACE:44:0:12    ////__DACE:41
            {                                                                                 ////__DACE:44:0:10    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_54_1;                                               ////__DACE:44:0:20,10    ////__DACE:44:0:10    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_126_1;                                 ////__DACE:44:0:11,10    ////__DACE:44:0:10    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                ////__DACE:44:0:10                                                            ////__DACE:44:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                // Tasklet code (tlet_47_multiplies_1)                                            ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                ////__DACE:44:0:10                                                            ////__DACE:44:0:10    ////__DACE:41
                __map_fusion_gtir_tmp_128_1 = __tlet_result;                                      ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
            }                                                                                 ////__DACE:44:0:10    ////__DACE:41
            {                                                                                 ////__DACE:44:0:16    ////__DACE:41
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:44:0:23,16    ////__DACE:44:0:16    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:44:0:19,16    ////__DACE:44:0:16    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                ////__DACE:44:0:16                                                            ////__DACE:44:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                // Tasklet code (tlet_44_deref_1)                                                 ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                ////__DACE:44:0:16                                                            ////__DACE:44:0:16    ////__DACE:41
                __map_fusion_gtir_tmp_122_1 = __tlet_val;                                         ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
            }                                                                                 ////__DACE:44:0:16    ////__DACE:41
            {                                                                                 ////__DACE:44:0:14    ////__DACE:41
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:44:0:22,14    ////__DACE:44:0:14    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_122_1;                                 ////__DACE:44:0:15,14    ////__DACE:44:0:14    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                ////__DACE:44:0:14                                                            ////__DACE:44:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                // Tasklet code (tlet_45_plus_1)                                                  ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                ////__DACE:44:0:14                                                            ////__DACE:44:0:14    ////__DACE:41
                __map_fusion_gtir_tmp_124_1 = __tlet_result;                                      ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
            }                                                                                 ////__DACE:44:0:14    ////__DACE:41
            {                                                                                 ////__DACE:44:0:8    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_128_1;                                 ////__DACE:44:0:9,8    ////__DACE:44:0:8    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_124_1;                                 ////__DACE:44:0:13,8    ////__DACE:44:0:8    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                ////__DACE:44:0:8                                                             ////__DACE:44:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                // Tasklet code (tlet_48_plus_1)                                                  ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                ////__DACE:44:0:8                                                             ////__DACE:44:0:8    ////__DACE:41
                __map_fusion_gtir_tmp_130_1 = __tlet_result;                                      ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
            }                                                                                 ////__DACE:44:0:8    ////__DACE:41
            {                                                                                 ////__DACE:44:0:2    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_134_1;                                 ////__DACE:44:0:3,2    ////__DACE:44:0:2    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_130_1;                                 ////__DACE:44:0:7,2    ////__DACE:44:0:2    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                ////__DACE:44:0:2                                                             ////__DACE:44:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                // Tasklet code (tlet_51_plus_1)                                                  ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                ////__DACE:44:0:2                                                             ////__DACE:44:0:2    ////__DACE:41
                __arg2_ = __tlet_result;                                                          ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
            }                                                                                 ////__DACE:44:0:2    ////__DACE:41
            {                                                                                 ////__DACE:44:0:24    ////__DACE:41
                double _cpy_in = __arg2_;                                                         ////__DACE:44:0:1,24    ////__DACE:44:0:24    ////__DACE:41
                double _cpy_out;                                                                  ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                ////__DACE:44:0:24                                                            ////__DACE:44:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                _cpy_out = _cpy_in;                                                               ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                ////__DACE:44:0:24                                                            ////__DACE:44:0:24    ////__DACE:41
                __output = _cpy_out;                                                              ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
            }                                                                                 ////__DACE:44:0:24    ////__DACE:41
            ////__DACE:41
        }                                                                             ////__DACE:41
    }                                                                                 ////__DACE:41
}                                                                                 ////__DACE:0:0:222
////__DACE:0:0:222
DACE_DFI void if_stmt_1_0_0_155(const double&  __arg1___, const double&  __arg1____from_cb_fusion_0, const double&  __arg2, const double&  __arg2_from_cb_fusion_0, const bool&  __cond, double&  __output, double&  __output_from_cb_fusion_0) {    ////__DACE:0:0:155
    ////__DACE:13
    if (__cond) {                                                                     ////__DACE:13
        {                                                                             ////__DACE:13
            ////__DACE:13
            {                                                                                 ////__DACE:15:0:4    ////__DACE:13
                double _cpy_in = __arg1___;                                                       ////__DACE:15:0:1,4    ////__DACE:15:0:4    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                ////__DACE:15:0:4                                                             ////__DACE:15:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                ////__DACE:15:0:4                                                             ////__DACE:15:0:4    ////__DACE:13
                __output = _cpy_out;                                                              ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
            }                                                                                 ////__DACE:15:0:4    ////__DACE:13
            {                                                                                 ////__DACE:15:0:5    ////__DACE:13
                double _cpy_in = __arg1____from_cb_fusion_0;                                      ////__DACE:15:0:3,5    ////__DACE:15:0:5    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:15:0:5    ////__DACE:15:0:5    ////__DACE:13
                ////__DACE:15:0:5                                                             ////__DACE:15:0:5    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:5    ////__DACE:15:0:5    ////__DACE:13
                // Tasklet code (copy___arg1____from_cb_fusion_0_to___output_from_cb_fusion_0)    ////__DACE:15:0:5    ////__DACE:15:0:5    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:15:0:5    ////__DACE:15:0:5    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:5    ////__DACE:15:0:5    ////__DACE:13
                ////__DACE:15:0:5                                                             ////__DACE:15:0:5    ////__DACE:13
                __output_from_cb_fusion_0 = _cpy_out;                                             ////__DACE:15:0:5    ////__DACE:15:0:5    ////__DACE:13
            }                                                                                 ////__DACE:15:0:5    ////__DACE:13
            ////__DACE:13
        }                                                                             ////__DACE:13
    } else {                                                                          ////__DACE:13
        {                                                                             ////__DACE:13
            ////__DACE:13
            {                                                                                 ////__DACE:16:0:4    ////__DACE:13
                double _cpy_in = __arg2;                                                          ////__DACE:16:0:1,4    ////__DACE:16:0:4    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                ////__DACE:16:0:4                                                             ////__DACE:16:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                ////__DACE:16:0:4                                                             ////__DACE:16:0:4    ////__DACE:13
                __output = _cpy_out;                                                              ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
            }                                                                                 ////__DACE:16:0:4    ////__DACE:13
            {                                                                                 ////__DACE:16:0:5    ////__DACE:13
                double _cpy_in = __arg2_from_cb_fusion_0;                                         ////__DACE:16:0:3,5    ////__DACE:16:0:5    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:16:0:5    ////__DACE:16:0:5    ////__DACE:13
                ////__DACE:16:0:5                                                             ////__DACE:16:0:5    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:5    ////__DACE:16:0:5    ////__DACE:13
                // Tasklet code (copy___arg2_from_cb_fusion_0_to___output_from_cb_fusion_0)       ////__DACE:16:0:5    ////__DACE:16:0:5    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:16:0:5    ////__DACE:16:0:5    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:5    ////__DACE:16:0:5    ////__DACE:13
                ////__DACE:16:0:5                                                             ////__DACE:16:0:5    ////__DACE:13
                __output_from_cb_fusion_0 = _cpy_out;                                             ////__DACE:16:0:5    ////__DACE:16:0:5    ////__DACE:13
            }                                                                                 ////__DACE:16:0:5    ////__DACE:13
            ////__DACE:13
        }                                                                             ////__DACE:13
    }                                                                                 ////__DACE:13
}                                                                                 ////__DACE:0:0:155
////__DACE:0:0:155
DACE_DFI void if_stmt_4_0_0_166(const bool&  __cond, const double&  __map_fusion_gtir_tmp_21_0_0, const double&  __map_fusion_gtir_tmp_33_0_0, const double&  gtir_tmp_34_0, const double&  gtir_tmp_38_0, const double&  gtir_tmp_44_0, const double&  gtir_tmp_48_0, const double&  gtir_tmp_56_0, const double&  gtir_tmp_60_0, const double&  gtir_tmp_66_0, const double&  gtir_tmp_70_0, double&  __output, double&  __output_from_cb_fusion_1) {    ////__DACE:0:0:166
    ////__DACE:17
    if (__cond) {                                                                     ////__DACE:17
        {                                                                             ////__DACE:17
            double __arg1____;                                                                ////__DACE:19:0:1    ////__DACE:17
            double __arg1____from_cb_fusion_1;                                                ////__DACE:19:0:3    ////__DACE:17
            double __map_fusion_gtir_tmp_63_0;                                                ////__DACE:19:0:5    ////__DACE:17
            double __map_fusion_gtir_tmp_59_0;                                                ////__DACE:19:0:7    ////__DACE:17
            double __map_fusion_gtir_tmp_41_0;                                                ////__DACE:19:0:10    ////__DACE:17
            double __map_fusion_gtir_tmp_37_0;                                                ////__DACE:19:0:12    ////__DACE:17
            ////__DACE:17
            {                                                                                 ////__DACE:19:0:6    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_60_0;                                               ////__DACE:19:0:14,6    ////__DACE:19:0:6    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:19:0:15,6    ////__DACE:19:0:6    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                ////__DACE:19:0:6                                                             ////__DACE:19:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                // Tasklet code (tlet_21_multiplies_0)                                            ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                ////__DACE:19:0:6                                                             ////__DACE:19:0:6    ////__DACE:17
                __map_fusion_gtir_tmp_63_0 = __tlet_result;                                       ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
            }                                                                                 ////__DACE:19:0:6    ////__DACE:17
            {                                                                                 ////__DACE:19:0:8    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_56_0;                                               ////__DACE:19:0:16,8    ////__DACE:19:0:8    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:19:0:17,8    ////__DACE:19:0:8    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                ////__DACE:19:0:8                                                             ////__DACE:19:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                // Tasklet code (tlet_20_multiplies_0)                                            ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                ////__DACE:19:0:8                                                             ////__DACE:19:0:8    ////__DACE:17
                __map_fusion_gtir_tmp_59_0 = __tlet_result;                                       ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
            }                                                                                 ////__DACE:19:0:8    ////__DACE:17
            {                                                                                 ////__DACE:19:0:4    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_63_0;                                  ////__DACE:19:0:5,4    ////__DACE:19:0:4    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_59_0;                                  ////__DACE:19:0:7,4    ////__DACE:19:0:4    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                ////__DACE:19:0:4                                                             ////__DACE:19:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                // Tasklet code (tlet_22_plus_0)                                                  ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                ////__DACE:19:0:4                                                             ////__DACE:19:0:4    ////__DACE:17
                __arg1____ = __tlet_result;                                                       ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
            }                                                                                 ////__DACE:19:0:4    ////__DACE:17
            {                                                                                 ////__DACE:19:0:20    ////__DACE:17
                double _cpy_in = __arg1____;                                                      ////__DACE:19:0:1,20    ////__DACE:19:0:20    ////__DACE:17
                double _cpy_out;                                                                  ////__DACE:19:0:20    ////__DACE:19:0:20    ////__DACE:17
                ////__DACE:19:0:20                                                            ////__DACE:19:0:20    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:20    ////__DACE:19:0:20    ////__DACE:17
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:19:0:20    ////__DACE:19:0:20    ////__DACE:17
                _cpy_out = _cpy_in;                                                               ////__DACE:19:0:20    ////__DACE:19:0:20    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:20    ////__DACE:19:0:20    ////__DACE:17
                ////__DACE:19:0:20                                                            ////__DACE:19:0:20    ////__DACE:17
                __output = _cpy_out;                                                              ////__DACE:19:0:20    ////__DACE:19:0:20    ////__DACE:17
            }                                                                                 ////__DACE:19:0:20    ////__DACE:17
            {                                                                                 ////__DACE:19:0:11    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:19:0:15,11    ////__DACE:19:0:11    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_38_0;                                               ////__DACE:19:0:18,11    ////__DACE:19:0:11    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:11    ////__DACE:19:0:11    ////__DACE:17
                ////__DACE:19:0:11                                                            ////__DACE:19:0:11    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:11    ////__DACE:19:0:11    ////__DACE:17
                // Tasklet code (tlet_15_multiplies_0)                                            ////__DACE:19:0:11    ////__DACE:19:0:11    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:19:0:11    ////__DACE:19:0:11    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:11    ////__DACE:19:0:11    ////__DACE:17
                ////__DACE:19:0:11                                                            ////__DACE:19:0:11    ////__DACE:17
                __map_fusion_gtir_tmp_41_0 = __tlet_result;                                       ////__DACE:19:0:11    ////__DACE:19:0:11    ////__DACE:17
            }                                                                                 ////__DACE:19:0:11    ////__DACE:17
            {                                                                                 ////__DACE:19:0:13    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:19:0:17,13    ////__DACE:19:0:13    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_34_0;                                               ////__DACE:19:0:19,13    ////__DACE:19:0:13    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:13    ////__DACE:19:0:13    ////__DACE:17
                ////__DACE:19:0:13                                                            ////__DACE:19:0:13    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:13    ////__DACE:19:0:13    ////__DACE:17
                // Tasklet code (tlet_14_multiplies_0)                                            ////__DACE:19:0:13    ////__DACE:19:0:13    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:19:0:13    ////__DACE:19:0:13    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:13    ////__DACE:19:0:13    ////__DACE:17
                ////__DACE:19:0:13                                                            ////__DACE:19:0:13    ////__DACE:17
                __map_fusion_gtir_tmp_37_0 = __tlet_result;                                       ////__DACE:19:0:13    ////__DACE:19:0:13    ////__DACE:17
            }                                                                                 ////__DACE:19:0:13    ////__DACE:17
            {                                                                                 ////__DACE:19:0:9    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_41_0;                                  ////__DACE:19:0:10,9    ////__DACE:19:0:9    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_37_0;                                  ////__DACE:19:0:12,9    ////__DACE:19:0:9    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:9    ////__DACE:19:0:9    ////__DACE:17
                ////__DACE:19:0:9                                                             ////__DACE:19:0:9    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:9    ////__DACE:19:0:9    ////__DACE:17
                // Tasklet code (tlet_16_plus_0)                                                  ////__DACE:19:0:9    ////__DACE:19:0:9    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:19:0:9    ////__DACE:19:0:9    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:9    ////__DACE:19:0:9    ////__DACE:17
                ////__DACE:19:0:9                                                             ////__DACE:19:0:9    ////__DACE:17
                __arg1____from_cb_fusion_1 = __tlet_result;                                       ////__DACE:19:0:9    ////__DACE:19:0:9    ////__DACE:17
            }                                                                                 ////__DACE:19:0:9    ////__DACE:17
            {                                                                                 ////__DACE:19:0:21    ////__DACE:17
                double _cpy_in = __arg1____from_cb_fusion_1;                                      ////__DACE:19:0:3,21    ////__DACE:19:0:21    ////__DACE:17
                double _cpy_out;                                                                  ////__DACE:19:0:21    ////__DACE:19:0:21    ////__DACE:17
                ////__DACE:19:0:21                                                            ////__DACE:19:0:21    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:21    ////__DACE:19:0:21    ////__DACE:17
                // Tasklet code (copy___arg1____from_cb_fusion_1_to___output_from_cb_fusion_1)    ////__DACE:19:0:21    ////__DACE:19:0:21    ////__DACE:17
                _cpy_out = _cpy_in;                                                               ////__DACE:19:0:21    ////__DACE:19:0:21    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:21    ////__DACE:19:0:21    ////__DACE:17
                ////__DACE:19:0:21                                                            ////__DACE:19:0:21    ////__DACE:17
                __output_from_cb_fusion_1 = _cpy_out;                                             ////__DACE:19:0:21    ////__DACE:19:0:21    ////__DACE:17
            }                                                                                 ////__DACE:19:0:21    ////__DACE:17
            ////__DACE:17
        }                                                                             ////__DACE:17
    } else {                                                                          ////__DACE:17
        {                                                                             ////__DACE:17
            double __arg2_;                                                                   ////__DACE:20:0:1    ////__DACE:17
            double __arg2_from_cb_fusion_1;                                                   ////__DACE:20:0:3    ////__DACE:17
            double __map_fusion_gtir_tmp_73_0;                                                ////__DACE:20:0:5    ////__DACE:17
            double __map_fusion_gtir_tmp_69_0;                                                ////__DACE:20:0:7    ////__DACE:17
            double __map_fusion_gtir_tmp_51_0;                                                ////__DACE:20:0:10    ////__DACE:17
            double __map_fusion_gtir_tmp_47_0;                                                ////__DACE:20:0:12    ////__DACE:17
            ////__DACE:17
            {                                                                                 ////__DACE:20:0:6    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_70_0;                                               ////__DACE:20:0:14,6    ////__DACE:20:0:6    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:20:0:15,6    ////__DACE:20:0:6    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                ////__DACE:20:0:6                                                             ////__DACE:20:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                // Tasklet code (tlet_24_multiplies_0)                                            ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                ////__DACE:20:0:6                                                             ////__DACE:20:0:6    ////__DACE:17
                __map_fusion_gtir_tmp_73_0 = __tlet_result;                                       ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
            }                                                                                 ////__DACE:20:0:6    ////__DACE:17
            {                                                                                 ////__DACE:20:0:8    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_66_0;                                               ////__DACE:20:0:16,8    ////__DACE:20:0:8    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:20:0:17,8    ////__DACE:20:0:8    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                ////__DACE:20:0:8                                                             ////__DACE:20:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                // Tasklet code (tlet_23_multiplies_0)                                            ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                ////__DACE:20:0:8                                                             ////__DACE:20:0:8    ////__DACE:17
                __map_fusion_gtir_tmp_69_0 = __tlet_result;                                       ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
            }                                                                                 ////__DACE:20:0:8    ////__DACE:17
            {                                                                                 ////__DACE:20:0:4    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_73_0;                                  ////__DACE:20:0:5,4    ////__DACE:20:0:4    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_69_0;                                  ////__DACE:20:0:7,4    ////__DACE:20:0:4    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                ////__DACE:20:0:4                                                             ////__DACE:20:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                // Tasklet code (tlet_25_plus_0)                                                  ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                ////__DACE:20:0:4                                                             ////__DACE:20:0:4    ////__DACE:17
                __arg2_ = __tlet_result;                                                          ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
            }                                                                                 ////__DACE:20:0:4    ////__DACE:17
            {                                                                                 ////__DACE:20:0:20    ////__DACE:17
                double _cpy_in = __arg2_;                                                         ////__DACE:20:0:1,20    ////__DACE:20:0:20    ////__DACE:17
                double _cpy_out;                                                                  ////__DACE:20:0:20    ////__DACE:20:0:20    ////__DACE:17
                ////__DACE:20:0:20                                                            ////__DACE:20:0:20    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:20    ////__DACE:20:0:20    ////__DACE:17
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:20:0:20    ////__DACE:20:0:20    ////__DACE:17
                _cpy_out = _cpy_in;                                                               ////__DACE:20:0:20    ////__DACE:20:0:20    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:20    ////__DACE:20:0:20    ////__DACE:17
                ////__DACE:20:0:20                                                            ////__DACE:20:0:20    ////__DACE:17
                __output = _cpy_out;                                                              ////__DACE:20:0:20    ////__DACE:20:0:20    ////__DACE:17
            }                                                                                 ////__DACE:20:0:20    ////__DACE:17
            {                                                                                 ////__DACE:20:0:11    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_48_0;                                               ////__DACE:20:0:18,11    ////__DACE:20:0:11    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:20:0:15,11    ////__DACE:20:0:11    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:11    ////__DACE:20:0:11    ////__DACE:17
                ////__DACE:20:0:11                                                            ////__DACE:20:0:11    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:11    ////__DACE:20:0:11    ////__DACE:17
                // Tasklet code (tlet_18_multiplies_0)                                            ////__DACE:20:0:11    ////__DACE:20:0:11    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:20:0:11    ////__DACE:20:0:11    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:11    ////__DACE:20:0:11    ////__DACE:17
                ////__DACE:20:0:11                                                            ////__DACE:20:0:11    ////__DACE:17
                __map_fusion_gtir_tmp_51_0 = __tlet_result;                                       ////__DACE:20:0:11    ////__DACE:20:0:11    ////__DACE:17
            }                                                                                 ////__DACE:20:0:11    ////__DACE:17
            {                                                                                 ////__DACE:20:0:13    ////__DACE:17
                double __tlet_arg1 = gtir_tmp_44_0;                                               ////__DACE:20:0:19,13    ////__DACE:20:0:13    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:20:0:17,13    ////__DACE:20:0:13    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:13    ////__DACE:20:0:13    ////__DACE:17
                ////__DACE:20:0:13                                                            ////__DACE:20:0:13    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:13    ////__DACE:20:0:13    ////__DACE:17
                // Tasklet code (tlet_17_multiplies_0)                                            ////__DACE:20:0:13    ////__DACE:20:0:13    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:20:0:13    ////__DACE:20:0:13    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:13    ////__DACE:20:0:13    ////__DACE:17
                ////__DACE:20:0:13                                                            ////__DACE:20:0:13    ////__DACE:17
                __map_fusion_gtir_tmp_47_0 = __tlet_result;                                       ////__DACE:20:0:13    ////__DACE:20:0:13    ////__DACE:17
            }                                                                                 ////__DACE:20:0:13    ////__DACE:17
            {                                                                                 ////__DACE:20:0:9    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_51_0;                                  ////__DACE:20:0:10,9    ////__DACE:20:0:9    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_47_0;                                  ////__DACE:20:0:12,9    ////__DACE:20:0:9    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:9    ////__DACE:20:0:9    ////__DACE:17
                ////__DACE:20:0:9                                                             ////__DACE:20:0:9    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:9    ////__DACE:20:0:9    ////__DACE:17
                // Tasklet code (tlet_19_plus_0)                                                  ////__DACE:20:0:9    ////__DACE:20:0:9    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:20:0:9    ////__DACE:20:0:9    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:9    ////__DACE:20:0:9    ////__DACE:17
                ////__DACE:20:0:9                                                             ////__DACE:20:0:9    ////__DACE:17
                __arg2_from_cb_fusion_1 = __tlet_result;                                          ////__DACE:20:0:9    ////__DACE:20:0:9    ////__DACE:17
            }                                                                                 ////__DACE:20:0:9    ////__DACE:17
            {                                                                                 ////__DACE:20:0:21    ////__DACE:17
                double _cpy_in = __arg2_from_cb_fusion_1;                                         ////__DACE:20:0:3,21    ////__DACE:20:0:21    ////__DACE:17
                double _cpy_out;                                                                  ////__DACE:20:0:21    ////__DACE:20:0:21    ////__DACE:17
                ////__DACE:20:0:21                                                            ////__DACE:20:0:21    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:21    ////__DACE:20:0:21    ////__DACE:17
                // Tasklet code (copy___arg2_from_cb_fusion_1_to___output_from_cb_fusion_1)       ////__DACE:20:0:21    ////__DACE:20:0:21    ////__DACE:17
                _cpy_out = _cpy_in;                                                               ////__DACE:20:0:21    ////__DACE:20:0:21    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:21    ////__DACE:20:0:21    ////__DACE:17
                ////__DACE:20:0:21                                                            ////__DACE:20:0:21    ////__DACE:17
                __output_from_cb_fusion_1 = _cpy_out;                                             ////__DACE:20:0:21    ////__DACE:20:0:21    ////__DACE:17
            }                                                                                 ////__DACE:20:0:21    ////__DACE:17
            ////__DACE:17
        }                                                                             ////__DACE:17
    }                                                                                 ////__DACE:17
}                                                                                 ////__DACE:0:0:166
////__DACE:0:0:166
DACE_DFI void if_stmt_7_0_0_182(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double&  gtir_tmp_54_0, const double&  gtir_tmp_76_0, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double* __restrict__ perturbed_rho_at_cells_on_model_levels, const double* __restrict__ reference_rho_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_rho_at_cells_on_model_levels_K_stride_0, int __reference_rho_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:182
    ////__DACE:25
    if (__cond) {                                                                     ////__DACE:25
        {                                                                             ////__DACE:25
            double __arg1___;                                                                 ////__DACE:27:0:1    ////__DACE:25
            double __map_fusion_gtir_tmp_191_0;                                               ////__DACE:27:0:3    ////__DACE:25
            double __map_fusion_gtir_tmp_189_0;                                               ////__DACE:27:0:5    ////__DACE:25
            double __map_fusion_gtir_tmp_187_0;                                               ////__DACE:27:0:7    ////__DACE:25
            double __map_fusion_gtir_tmp_185_0;                                               ////__DACE:27:0:9    ////__DACE:25
            double __map_fusion_gtir_tmp_183_0;                                               ////__DACE:27:0:11    ////__DACE:25
            double __map_fusion_gtir_tmp_181_0;                                               ////__DACE:27:0:13    ////__DACE:25
            double __map_fusion_gtir_tmp_179_0;                                               ////__DACE:27:0:15    ////__DACE:25
            ////__DACE:25
            {                                                                                 ////__DACE:27:0:6    ////__DACE:25
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:27:0:18,6    ////__DACE:27:0:6    ////__DACE:25
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:27:0:19,6    ////__DACE:27:0:6    ////__DACE:25
                double __tlet_val;                                                                ////__DACE:27:0:6    ////__DACE:27:0:6    ////__DACE:25
                ////__DACE:27:0:6                                                             ////__DACE:27:0:6    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:6    ////__DACE:27:0:6    ////__DACE:25
                // Tasklet code (tlet_75_deref_0)                                                 ////__DACE:27:0:6    ////__DACE:27:0:6    ////__DACE:25
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:27:0:6    ////__DACE:27:0:6    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:6    ////__DACE:27:0:6    ////__DACE:25
                ////__DACE:27:0:6                                                             ////__DACE:27:0:6    ////__DACE:25
                __map_fusion_gtir_tmp_189_0 = __tlet_val;                                         ////__DACE:27:0:6    ////__DACE:27:0:6    ////__DACE:25
            }                                                                                 ////__DACE:27:0:6    ////__DACE:25
            {                                                                                 ////__DACE:27:0:4    ////__DACE:25
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:27:0:17,4    ////__DACE:27:0:4    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_189_0;                                 ////__DACE:27:0:5,4    ////__DACE:27:0:4    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:27:0:4    ////__DACE:27:0:4    ////__DACE:25
                ////__DACE:27:0:4                                                             ////__DACE:27:0:4    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:4    ////__DACE:27:0:4    ////__DACE:25
                // Tasklet code (tlet_76_multiplies_0)                                            ////__DACE:27:0:4    ////__DACE:27:0:4    ////__DACE:25
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:27:0:4    ////__DACE:27:0:4    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:4    ////__DACE:27:0:4    ////__DACE:25
                ////__DACE:27:0:4                                                             ////__DACE:27:0:4    ////__DACE:25
                __map_fusion_gtir_tmp_191_0 = __tlet_result;                                      ////__DACE:27:0:4    ////__DACE:27:0:4    ////__DACE:25
            }                                                                                 ////__DACE:27:0:4    ////__DACE:25
            {                                                                                 ////__DACE:27:0:12    ////__DACE:25
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:27:0:21,12    ////__DACE:27:0:12    ////__DACE:25
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:27:0:19,12    ////__DACE:27:0:12    ////__DACE:25
                double __tlet_val;                                                                ////__DACE:27:0:12    ////__DACE:27:0:12    ////__DACE:25
                ////__DACE:27:0:12                                                            ////__DACE:27:0:12    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:12    ////__DACE:27:0:12    ////__DACE:25
                // Tasklet code (tlet_72_deref_0)                                                 ////__DACE:27:0:12    ////__DACE:27:0:12    ////__DACE:25
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:27:0:12    ////__DACE:27:0:12    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:12    ////__DACE:27:0:12    ////__DACE:25
                ////__DACE:27:0:12                                                            ////__DACE:27:0:12    ////__DACE:25
                __map_fusion_gtir_tmp_183_0 = __tlet_val;                                         ////__DACE:27:0:12    ////__DACE:27:0:12    ////__DACE:25
            }                                                                                 ////__DACE:27:0:12    ////__DACE:25
            {                                                                                 ////__DACE:27:0:10    ////__DACE:25
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:27:0:20,10    ////__DACE:27:0:10    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_183_0;                                 ////__DACE:27:0:11,10    ////__DACE:27:0:10    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:27:0:10    ////__DACE:27:0:10    ////__DACE:25
                ////__DACE:27:0:10                                                            ////__DACE:27:0:10    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:10    ////__DACE:27:0:10    ////__DACE:25
                // Tasklet code (tlet_73_multiplies_0)                                            ////__DACE:27:0:10    ////__DACE:27:0:10    ////__DACE:25
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:27:0:10    ////__DACE:27:0:10    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:10    ////__DACE:27:0:10    ////__DACE:25
                ////__DACE:27:0:10                                                            ////__DACE:27:0:10    ////__DACE:25
                __map_fusion_gtir_tmp_185_0 = __tlet_result;                                      ////__DACE:27:0:10    ////__DACE:27:0:10    ////__DACE:25
            }                                                                                 ////__DACE:27:0:10    ////__DACE:25
            {                                                                                 ////__DACE:27:0:16    ////__DACE:25
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:27:0:23,16    ////__DACE:27:0:16    ////__DACE:25
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:27:0:19,16    ////__DACE:27:0:16    ////__DACE:25
                double __tlet_val;                                                                ////__DACE:27:0:16    ////__DACE:27:0:16    ////__DACE:25
                ////__DACE:27:0:16                                                            ////__DACE:27:0:16    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:16    ////__DACE:27:0:16    ////__DACE:25
                // Tasklet code (tlet_70_deref_0)                                                 ////__DACE:27:0:16    ////__DACE:27:0:16    ////__DACE:25
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:27:0:16    ////__DACE:27:0:16    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:16    ////__DACE:27:0:16    ////__DACE:25
                ////__DACE:27:0:16                                                            ////__DACE:27:0:16    ////__DACE:25
                __map_fusion_gtir_tmp_179_0 = __tlet_val;                                         ////__DACE:27:0:16    ////__DACE:27:0:16    ////__DACE:25
            }                                                                                 ////__DACE:27:0:16    ////__DACE:25
            {                                                                                 ////__DACE:27:0:14    ////__DACE:25
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:27:0:22,14    ////__DACE:27:0:14    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_179_0;                                 ////__DACE:27:0:15,14    ////__DACE:27:0:14    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:27:0:14    ////__DACE:27:0:14    ////__DACE:25
                ////__DACE:27:0:14                                                            ////__DACE:27:0:14    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:14    ////__DACE:27:0:14    ////__DACE:25
                // Tasklet code (tlet_71_plus_0)                                                  ////__DACE:27:0:14    ////__DACE:27:0:14    ////__DACE:25
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:27:0:14    ////__DACE:27:0:14    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:14    ////__DACE:27:0:14    ////__DACE:25
                ////__DACE:27:0:14                                                            ////__DACE:27:0:14    ////__DACE:25
                __map_fusion_gtir_tmp_181_0 = __tlet_result;                                      ////__DACE:27:0:14    ////__DACE:27:0:14    ////__DACE:25
            }                                                                                 ////__DACE:27:0:14    ////__DACE:25
            {                                                                                 ////__DACE:27:0:8    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_185_0;                                 ////__DACE:27:0:9,8    ////__DACE:27:0:8    ////__DACE:25
                double __tlet_arg0 = __map_fusion_gtir_tmp_181_0;                                 ////__DACE:27:0:13,8    ////__DACE:27:0:8    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:27:0:8    ////__DACE:27:0:8    ////__DACE:25
                ////__DACE:27:0:8                                                             ////__DACE:27:0:8    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:8    ////__DACE:27:0:8    ////__DACE:25
                // Tasklet code (tlet_74_plus_0)                                                  ////__DACE:27:0:8    ////__DACE:27:0:8    ////__DACE:25
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:27:0:8    ////__DACE:27:0:8    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:8    ////__DACE:27:0:8    ////__DACE:25
                ////__DACE:27:0:8                                                             ////__DACE:27:0:8    ////__DACE:25
                __map_fusion_gtir_tmp_187_0 = __tlet_result;                                      ////__DACE:27:0:8    ////__DACE:27:0:8    ////__DACE:25
            }                                                                                 ////__DACE:27:0:8    ////__DACE:25
            {                                                                                 ////__DACE:27:0:2    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_191_0;                                 ////__DACE:27:0:3,2    ////__DACE:27:0:2    ////__DACE:25
                double __tlet_arg0 = __map_fusion_gtir_tmp_187_0;                                 ////__DACE:27:0:7,2    ////__DACE:27:0:2    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                ////__DACE:27:0:2                                                             ////__DACE:27:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                // Tasklet code (tlet_77_plus_0)                                                  ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                ////__DACE:27:0:2                                                             ////__DACE:27:0:2    ////__DACE:25
                __arg1___ = __tlet_result;                                                        ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
            }                                                                                 ////__DACE:27:0:2    ////__DACE:25
            {                                                                                 ////__DACE:27:0:24    ////__DACE:25
                double _cpy_in = __arg1___;                                                       ////__DACE:27:0:1,24    ////__DACE:27:0:24    ////__DACE:25
                double _cpy_out;                                                                  ////__DACE:27:0:24    ////__DACE:27:0:24    ////__DACE:25
                ////__DACE:27:0:24                                                            ////__DACE:27:0:24    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:24    ////__DACE:27:0:24    ////__DACE:25
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:27:0:24    ////__DACE:27:0:24    ////__DACE:25
                _cpy_out = _cpy_in;                                                               ////__DACE:27:0:24    ////__DACE:27:0:24    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:24    ////__DACE:27:0:24    ////__DACE:25
                ////__DACE:27:0:24                                                            ////__DACE:27:0:24    ////__DACE:25
                __output = _cpy_out;                                                              ////__DACE:27:0:24    ////__DACE:27:0:24    ////__DACE:25
            }                                                                                 ////__DACE:27:0:24    ////__DACE:25
            ////__DACE:25
        }                                                                             ////__DACE:25
    } else {                                                                          ////__DACE:25
        {                                                                             ////__DACE:25
            double __arg2;                                                                    ////__DACE:28:0:1    ////__DACE:25
            double __map_fusion_gtir_tmp_207_0;                                               ////__DACE:28:0:3    ////__DACE:25
            double __map_fusion_gtir_tmp_205_0;                                               ////__DACE:28:0:5    ////__DACE:25
            double __map_fusion_gtir_tmp_203_0;                                               ////__DACE:28:0:7    ////__DACE:25
            double __map_fusion_gtir_tmp_201_0;                                               ////__DACE:28:0:9    ////__DACE:25
            double __map_fusion_gtir_tmp_199_0;                                               ////__DACE:28:0:11    ////__DACE:25
            double __map_fusion_gtir_tmp_197_0;                                               ////__DACE:28:0:13    ////__DACE:25
            double __map_fusion_gtir_tmp_195_0;                                               ////__DACE:28:0:15    ////__DACE:25
            ////__DACE:25
            {                                                                                 ////__DACE:28:0:6    ////__DACE:25
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:28:0:18,6    ////__DACE:28:0:6    ////__DACE:25
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:28:0:19,6    ////__DACE:28:0:6    ////__DACE:25
                double __tlet_val;                                                                ////__DACE:28:0:6    ////__DACE:28:0:6    ////__DACE:25
                ////__DACE:28:0:6                                                             ////__DACE:28:0:6    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:6    ////__DACE:28:0:6    ////__DACE:25
                // Tasklet code (tlet_83_deref_0)                                                 ////__DACE:28:0:6    ////__DACE:28:0:6    ////__DACE:25
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:28:0:6    ////__DACE:28:0:6    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:6    ////__DACE:28:0:6    ////__DACE:25
                ////__DACE:28:0:6                                                             ////__DACE:28:0:6    ////__DACE:25
                __map_fusion_gtir_tmp_205_0 = __tlet_val;                                         ////__DACE:28:0:6    ////__DACE:28:0:6    ////__DACE:25
            }                                                                                 ////__DACE:28:0:6    ////__DACE:25
            {                                                                                 ////__DACE:28:0:4    ////__DACE:25
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:28:0:17,4    ////__DACE:28:0:4    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_205_0;                                 ////__DACE:28:0:5,4    ////__DACE:28:0:4    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:28:0:4    ////__DACE:28:0:4    ////__DACE:25
                ////__DACE:28:0:4                                                             ////__DACE:28:0:4    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:4    ////__DACE:28:0:4    ////__DACE:25
                // Tasklet code (tlet_84_multiplies_0)                                            ////__DACE:28:0:4    ////__DACE:28:0:4    ////__DACE:25
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:28:0:4    ////__DACE:28:0:4    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:4    ////__DACE:28:0:4    ////__DACE:25
                ////__DACE:28:0:4                                                             ////__DACE:28:0:4    ////__DACE:25
                __map_fusion_gtir_tmp_207_0 = __tlet_result;                                      ////__DACE:28:0:4    ////__DACE:28:0:4    ////__DACE:25
            }                                                                                 ////__DACE:28:0:4    ////__DACE:25
            {                                                                                 ////__DACE:28:0:12    ////__DACE:25
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:28:0:21,12    ////__DACE:28:0:12    ////__DACE:25
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:28:0:19,12    ////__DACE:28:0:12    ////__DACE:25
                double __tlet_val;                                                                ////__DACE:28:0:12    ////__DACE:28:0:12    ////__DACE:25
                ////__DACE:28:0:12                                                            ////__DACE:28:0:12    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:12    ////__DACE:28:0:12    ////__DACE:25
                // Tasklet code (tlet_80_deref_0)                                                 ////__DACE:28:0:12    ////__DACE:28:0:12    ////__DACE:25
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:28:0:12    ////__DACE:28:0:12    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:12    ////__DACE:28:0:12    ////__DACE:25
                ////__DACE:28:0:12                                                            ////__DACE:28:0:12    ////__DACE:25
                __map_fusion_gtir_tmp_199_0 = __tlet_val;                                         ////__DACE:28:0:12    ////__DACE:28:0:12    ////__DACE:25
            }                                                                                 ////__DACE:28:0:12    ////__DACE:25
            {                                                                                 ////__DACE:28:0:10    ////__DACE:25
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:28:0:20,10    ////__DACE:28:0:10    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_199_0;                                 ////__DACE:28:0:11,10    ////__DACE:28:0:10    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:28:0:10    ////__DACE:28:0:10    ////__DACE:25
                ////__DACE:28:0:10                                                            ////__DACE:28:0:10    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:10    ////__DACE:28:0:10    ////__DACE:25
                // Tasklet code (tlet_81_multiplies_0)                                            ////__DACE:28:0:10    ////__DACE:28:0:10    ////__DACE:25
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:28:0:10    ////__DACE:28:0:10    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:10    ////__DACE:28:0:10    ////__DACE:25
                ////__DACE:28:0:10                                                            ////__DACE:28:0:10    ////__DACE:25
                __map_fusion_gtir_tmp_201_0 = __tlet_result;                                      ////__DACE:28:0:10    ////__DACE:28:0:10    ////__DACE:25
            }                                                                                 ////__DACE:28:0:10    ////__DACE:25
            {                                                                                 ////__DACE:28:0:16    ////__DACE:25
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:28:0:23,16    ////__DACE:28:0:16    ////__DACE:25
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:28:0:19,16    ////__DACE:28:0:16    ////__DACE:25
                double __tlet_val;                                                                ////__DACE:28:0:16    ////__DACE:28:0:16    ////__DACE:25
                ////__DACE:28:0:16                                                            ////__DACE:28:0:16    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:16    ////__DACE:28:0:16    ////__DACE:25
                // Tasklet code (tlet_78_deref_0)                                                 ////__DACE:28:0:16    ////__DACE:28:0:16    ////__DACE:25
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:28:0:16    ////__DACE:28:0:16    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:16    ////__DACE:28:0:16    ////__DACE:25
                ////__DACE:28:0:16                                                            ////__DACE:28:0:16    ////__DACE:25
                __map_fusion_gtir_tmp_195_0 = __tlet_val;                                         ////__DACE:28:0:16    ////__DACE:28:0:16    ////__DACE:25
            }                                                                                 ////__DACE:28:0:16    ////__DACE:25
            {                                                                                 ////__DACE:28:0:14    ////__DACE:25
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:28:0:22,14    ////__DACE:28:0:14    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_195_0;                                 ////__DACE:28:0:15,14    ////__DACE:28:0:14    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:28:0:14    ////__DACE:28:0:14    ////__DACE:25
                ////__DACE:28:0:14                                                            ////__DACE:28:0:14    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:14    ////__DACE:28:0:14    ////__DACE:25
                // Tasklet code (tlet_79_plus_0)                                                  ////__DACE:28:0:14    ////__DACE:28:0:14    ////__DACE:25
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:28:0:14    ////__DACE:28:0:14    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:14    ////__DACE:28:0:14    ////__DACE:25
                ////__DACE:28:0:14                                                            ////__DACE:28:0:14    ////__DACE:25
                __map_fusion_gtir_tmp_197_0 = __tlet_result;                                      ////__DACE:28:0:14    ////__DACE:28:0:14    ////__DACE:25
            }                                                                                 ////__DACE:28:0:14    ////__DACE:25
            {                                                                                 ////__DACE:28:0:8    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_201_0;                                 ////__DACE:28:0:9,8    ////__DACE:28:0:8    ////__DACE:25
                double __tlet_arg0 = __map_fusion_gtir_tmp_197_0;                                 ////__DACE:28:0:13,8    ////__DACE:28:0:8    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:28:0:8    ////__DACE:28:0:8    ////__DACE:25
                ////__DACE:28:0:8                                                             ////__DACE:28:0:8    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:8    ////__DACE:28:0:8    ////__DACE:25
                // Tasklet code (tlet_82_plus_0)                                                  ////__DACE:28:0:8    ////__DACE:28:0:8    ////__DACE:25
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:28:0:8    ////__DACE:28:0:8    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:8    ////__DACE:28:0:8    ////__DACE:25
                ////__DACE:28:0:8                                                             ////__DACE:28:0:8    ////__DACE:25
                __map_fusion_gtir_tmp_203_0 = __tlet_result;                                      ////__DACE:28:0:8    ////__DACE:28:0:8    ////__DACE:25
            }                                                                                 ////__DACE:28:0:8    ////__DACE:25
            {                                                                                 ////__DACE:28:0:2    ////__DACE:25
                double __tlet_arg1 = __map_fusion_gtir_tmp_207_0;                                 ////__DACE:28:0:3,2    ////__DACE:28:0:2    ////__DACE:25
                double __tlet_arg0 = __map_fusion_gtir_tmp_203_0;                                 ////__DACE:28:0:7,2    ////__DACE:28:0:2    ////__DACE:25
                double __tlet_result;                                                             ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                ////__DACE:28:0:2                                                             ////__DACE:28:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                // Tasklet code (tlet_85_plus_0)                                                  ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                ////__DACE:28:0:2                                                             ////__DACE:28:0:2    ////__DACE:25
                __arg2 = __tlet_result;                                                           ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
            }                                                                                 ////__DACE:28:0:2    ////__DACE:25
            {                                                                                 ////__DACE:28:0:24    ////__DACE:25
                double _cpy_in = __arg2;                                                          ////__DACE:28:0:1,24    ////__DACE:28:0:24    ////__DACE:25
                double _cpy_out;                                                                  ////__DACE:28:0:24    ////__DACE:28:0:24    ////__DACE:25
                ////__DACE:28:0:24                                                            ////__DACE:28:0:24    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:24    ////__DACE:28:0:24    ////__DACE:25
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:28:0:24    ////__DACE:28:0:24    ////__DACE:25
                _cpy_out = _cpy_in;                                                               ////__DACE:28:0:24    ////__DACE:28:0:24    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:24    ////__DACE:28:0:24    ////__DACE:25
                ////__DACE:28:0:24                                                            ////__DACE:28:0:24    ////__DACE:25
                __output = _cpy_out;                                                              ////__DACE:28:0:24    ////__DACE:28:0:24    ////__DACE:25
            }                                                                                 ////__DACE:28:0:24    ////__DACE:25
            ////__DACE:25
        }                                                                             ////__DACE:25
    }                                                                                 ////__DACE:25
}                                                                                 ////__DACE:0:0:182
////__DACE:0:0:182
DACE_DFI void if_stmt_5_0_0_178(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double&  gtir_tmp_54_0, const double&  gtir_tmp_76_0, const double * __restrict__ gtir_tmp_95, const double* __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double* __restrict__ reference_theta_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_theta_v_at_cells_on_model_levels_K_stride_0, int __reference_theta_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:178
    ////__DACE:21
    if (__cond) {                                                                     ////__DACE:21
        {                                                                             ////__DACE:21
            double __arg1____;                                                                ////__DACE:23:0:1    ////__DACE:21
            double __map_fusion_gtir_tmp_118_0;                                               ////__DACE:23:0:3    ////__DACE:21
            double __map_fusion_gtir_tmp_116_0;                                               ////__DACE:23:0:5    ////__DACE:21
            double __map_fusion_gtir_tmp_114_0;                                               ////__DACE:23:0:7    ////__DACE:21
            double __map_fusion_gtir_tmp_112_0;                                               ////__DACE:23:0:9    ////__DACE:21
            double __map_fusion_gtir_tmp_110_0;                                               ////__DACE:23:0:11    ////__DACE:21
            double __map_fusion_gtir_tmp_108_0;                                               ////__DACE:23:0:13    ////__DACE:21
            double __map_fusion_gtir_tmp_106_0;                                               ////__DACE:23:0:15    ////__DACE:21
            ////__DACE:21
            {                                                                                 ////__DACE:23:0:6    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:23:0:18,6    ////__DACE:23:0:6    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:23:0:19,6    ////__DACE:23:0:6    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                ////__DACE:23:0:6                                                             ////__DACE:23:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                // Tasklet code (tlet_41_deref_0)                                                 ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                ////__DACE:23:0:6                                                             ////__DACE:23:0:6    ////__DACE:21
                __map_fusion_gtir_tmp_116_0 = __tlet_val;                                         ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
            }                                                                                 ////__DACE:23:0:6    ////__DACE:21
            {                                                                                 ////__DACE:23:0:4    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:23:0:17,4    ////__DACE:23:0:4    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_116_0;                                 ////__DACE:23:0:5,4    ////__DACE:23:0:4    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                ////__DACE:23:0:4                                                             ////__DACE:23:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                // Tasklet code (tlet_42_multiplies_0)                                            ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                ////__DACE:23:0:4                                                             ////__DACE:23:0:4    ////__DACE:21
                __map_fusion_gtir_tmp_118_0 = __tlet_result;                                      ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
            }                                                                                 ////__DACE:23:0:4    ////__DACE:21
            {                                                                                 ////__DACE:23:0:12    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:23:0:21,12    ////__DACE:23:0:12    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:23:0:19,12    ////__DACE:23:0:12    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                ////__DACE:23:0:12                                                            ////__DACE:23:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                // Tasklet code (tlet_38_deref_0)                                                 ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                ////__DACE:23:0:12                                                            ////__DACE:23:0:12    ////__DACE:21
                __map_fusion_gtir_tmp_110_0 = __tlet_val;                                         ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
            }                                                                                 ////__DACE:23:0:12    ////__DACE:21
            {                                                                                 ////__DACE:23:0:10    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:23:0:20,10    ////__DACE:23:0:10    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_110_0;                                 ////__DACE:23:0:11,10    ////__DACE:23:0:10    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                ////__DACE:23:0:10                                                            ////__DACE:23:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                // Tasklet code (tlet_39_multiplies_0)                                            ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                ////__DACE:23:0:10                                                            ////__DACE:23:0:10    ////__DACE:21
                __map_fusion_gtir_tmp_112_0 = __tlet_result;                                      ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
            }                                                                                 ////__DACE:23:0:10    ////__DACE:21
            {                                                                                 ////__DACE:23:0:16    ////__DACE:21
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:23:0:23,16    ////__DACE:23:0:16    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:23:0:19,16    ////__DACE:23:0:16    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                ////__DACE:23:0:16                                                            ////__DACE:23:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                // Tasklet code (tlet_36_deref_0)                                                 ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                ////__DACE:23:0:16                                                            ////__DACE:23:0:16    ////__DACE:21
                __map_fusion_gtir_tmp_106_0 = __tlet_val;                                         ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
            }                                                                                 ////__DACE:23:0:16    ////__DACE:21
            {                                                                                 ////__DACE:23:0:14    ////__DACE:21
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:23:0:22,14    ////__DACE:23:0:14    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_106_0;                                 ////__DACE:23:0:15,14    ////__DACE:23:0:14    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                ////__DACE:23:0:14                                                            ////__DACE:23:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                // Tasklet code (tlet_37_plus_0)                                                  ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                ////__DACE:23:0:14                                                            ////__DACE:23:0:14    ////__DACE:21
                __map_fusion_gtir_tmp_108_0 = __tlet_result;                                      ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
            }                                                                                 ////__DACE:23:0:14    ////__DACE:21
            {                                                                                 ////__DACE:23:0:8    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_112_0;                                 ////__DACE:23:0:9,8    ////__DACE:23:0:8    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_108_0;                                 ////__DACE:23:0:13,8    ////__DACE:23:0:8    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                ////__DACE:23:0:8                                                             ////__DACE:23:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                // Tasklet code (tlet_40_plus_0)                                                  ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                ////__DACE:23:0:8                                                             ////__DACE:23:0:8    ////__DACE:21
                __map_fusion_gtir_tmp_114_0 = __tlet_result;                                      ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
            }                                                                                 ////__DACE:23:0:8    ////__DACE:21
            {                                                                                 ////__DACE:23:0:2    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_118_0;                                 ////__DACE:23:0:3,2    ////__DACE:23:0:2    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_114_0;                                 ////__DACE:23:0:7,2    ////__DACE:23:0:2    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                ////__DACE:23:0:2                                                             ////__DACE:23:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                // Tasklet code (tlet_43_plus_0)                                                  ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                ////__DACE:23:0:2                                                             ////__DACE:23:0:2    ////__DACE:21
                __arg1____ = __tlet_result;                                                       ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
            }                                                                                 ////__DACE:23:0:2    ////__DACE:21
            {                                                                                 ////__DACE:23:0:24    ////__DACE:21
                double _cpy_in = __arg1____;                                                      ////__DACE:23:0:1,24    ////__DACE:23:0:24    ////__DACE:21
                double _cpy_out;                                                                  ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                ////__DACE:23:0:24                                                            ////__DACE:23:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                _cpy_out = _cpy_in;                                                               ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                ////__DACE:23:0:24                                                            ////__DACE:23:0:24    ////__DACE:21
                __output = _cpy_out;                                                              ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
            }                                                                                 ////__DACE:23:0:24    ////__DACE:21
            ////__DACE:21
        }                                                                             ////__DACE:21
    } else {                                                                          ////__DACE:21
        {                                                                             ////__DACE:21
            double __arg2_;                                                                   ////__DACE:24:0:1    ////__DACE:21
            double __map_fusion_gtir_tmp_134_0;                                               ////__DACE:24:0:3    ////__DACE:21
            double __map_fusion_gtir_tmp_132_0;                                               ////__DACE:24:0:5    ////__DACE:21
            double __map_fusion_gtir_tmp_130_0;                                               ////__DACE:24:0:7    ////__DACE:21
            double __map_fusion_gtir_tmp_128_0;                                               ////__DACE:24:0:9    ////__DACE:21
            double __map_fusion_gtir_tmp_126_0;                                               ////__DACE:24:0:11    ////__DACE:21
            double __map_fusion_gtir_tmp_124_0;                                               ////__DACE:24:0:13    ////__DACE:21
            double __map_fusion_gtir_tmp_122_0;                                               ////__DACE:24:0:15    ////__DACE:21
            ////__DACE:21
            {                                                                                 ////__DACE:24:0:6    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:24:0:18,6    ////__DACE:24:0:6    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:24:0:19,6    ////__DACE:24:0:6    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                ////__DACE:24:0:6                                                             ////__DACE:24:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                // Tasklet code (tlet_49_deref_0)                                                 ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                ////__DACE:24:0:6                                                             ////__DACE:24:0:6    ////__DACE:21
                __map_fusion_gtir_tmp_132_0 = __tlet_val;                                         ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
            }                                                                                 ////__DACE:24:0:6    ////__DACE:21
            {                                                                                 ////__DACE:24:0:4    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:24:0:17,4    ////__DACE:24:0:4    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_132_0;                                 ////__DACE:24:0:5,4    ////__DACE:24:0:4    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                ////__DACE:24:0:4                                                             ////__DACE:24:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                // Tasklet code (tlet_50_multiplies_0)                                            ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                ////__DACE:24:0:4                                                             ////__DACE:24:0:4    ////__DACE:21
                __map_fusion_gtir_tmp_134_0 = __tlet_result;                                      ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
            }                                                                                 ////__DACE:24:0:4    ////__DACE:21
            {                                                                                 ////__DACE:24:0:12    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:24:0:21,12    ////__DACE:24:0:12    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:24:0:19,12    ////__DACE:24:0:12    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                ////__DACE:24:0:12                                                            ////__DACE:24:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                // Tasklet code (tlet_46_deref_0)                                                 ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                ////__DACE:24:0:12                                                            ////__DACE:24:0:12    ////__DACE:21
                __map_fusion_gtir_tmp_126_0 = __tlet_val;                                         ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
            }                                                                                 ////__DACE:24:0:12    ////__DACE:21
            {                                                                                 ////__DACE:24:0:10    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:24:0:20,10    ////__DACE:24:0:10    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_126_0;                                 ////__DACE:24:0:11,10    ////__DACE:24:0:10    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                ////__DACE:24:0:10                                                            ////__DACE:24:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                // Tasklet code (tlet_47_multiplies_0)                                            ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                ////__DACE:24:0:10                                                            ////__DACE:24:0:10    ////__DACE:21
                __map_fusion_gtir_tmp_128_0 = __tlet_result;                                      ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
            }                                                                                 ////__DACE:24:0:10    ////__DACE:21
            {                                                                                 ////__DACE:24:0:16    ////__DACE:21
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:24:0:23,16    ////__DACE:24:0:16    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:24:0:19,16    ////__DACE:24:0:16    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                ////__DACE:24:0:16                                                            ////__DACE:24:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                // Tasklet code (tlet_44_deref_0)                                                 ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                ////__DACE:24:0:16                                                            ////__DACE:24:0:16    ////__DACE:21
                __map_fusion_gtir_tmp_122_0 = __tlet_val;                                         ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
            }                                                                                 ////__DACE:24:0:16    ////__DACE:21
            {                                                                                 ////__DACE:24:0:14    ////__DACE:21
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:24:0:22,14    ////__DACE:24:0:14    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_122_0;                                 ////__DACE:24:0:15,14    ////__DACE:24:0:14    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                ////__DACE:24:0:14                                                            ////__DACE:24:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                // Tasklet code (tlet_45_plus_0)                                                  ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                ////__DACE:24:0:14                                                            ////__DACE:24:0:14    ////__DACE:21
                __map_fusion_gtir_tmp_124_0 = __tlet_result;                                      ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
            }                                                                                 ////__DACE:24:0:14    ////__DACE:21
            {                                                                                 ////__DACE:24:0:8    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_128_0;                                 ////__DACE:24:0:9,8    ////__DACE:24:0:8    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_124_0;                                 ////__DACE:24:0:13,8    ////__DACE:24:0:8    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                ////__DACE:24:0:8                                                             ////__DACE:24:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                // Tasklet code (tlet_48_plus_0)                                                  ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                ////__DACE:24:0:8                                                             ////__DACE:24:0:8    ////__DACE:21
                __map_fusion_gtir_tmp_130_0 = __tlet_result;                                      ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
            }                                                                                 ////__DACE:24:0:8    ////__DACE:21
            {                                                                                 ////__DACE:24:0:2    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_134_0;                                 ////__DACE:24:0:3,2    ////__DACE:24:0:2    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_130_0;                                 ////__DACE:24:0:7,2    ////__DACE:24:0:2    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                ////__DACE:24:0:2                                                             ////__DACE:24:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                // Tasklet code (tlet_51_plus_0)                                                  ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                ////__DACE:24:0:2                                                             ////__DACE:24:0:2    ////__DACE:21
                __arg2_ = __tlet_result;                                                          ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
            }                                                                                 ////__DACE:24:0:2    ////__DACE:21
            {                                                                                 ////__DACE:24:0:24    ////__DACE:21
                double _cpy_in = __arg2_;                                                         ////__DACE:24:0:1,24    ////__DACE:24:0:24    ////__DACE:21
                double _cpy_out;                                                                  ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                ////__DACE:24:0:24                                                            ////__DACE:24:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                _cpy_out = _cpy_in;                                                               ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                ////__DACE:24:0:24                                                            ////__DACE:24:0:24    ////__DACE:21
                __output = _cpy_out;                                                              ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
            }                                                                                 ////__DACE:24:0:24    ////__DACE:21
            ////__DACE:21
        }                                                                             ////__DACE:21
    }                                                                                 ////__DACE:21
}                                                                                 ////__DACE:0:0:178
////__DACE:0:0:178
DACE_DFI void if_stmt_6_0_0_116(const double&  __arg2_, const bool&  __cond, const double* __restrict__ hydrostatic_correction_on_lowest_level, const double* __restrict__ pg_exdist, double&  __output, int __pg_exdist_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:116
    ////__DACE:1
    if (__cond) {                                                                     ////__DACE:1
        {                                                                             ////__DACE:1
            double __arg1__;                                                                  ////__DACE:3:0:1    ////__DACE:1
            double __map_fusion_gtir_tmp_170_0;                                               ////__DACE:3:0:3    ////__DACE:1
            ////__DACE:1
            {                                                                                 ////__DACE:3:0:4    ////__DACE:1
                double __tlet_arg0 = hydrostatic_correction_on_lowest_level[i_Edge_gtx_horizontal];    ////__DACE:3:0:6,4    ////__DACE:3:0:4    ////__DACE:1
                double __tlet_arg1 = pg_exdist[((__pg_exdist_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:3:0:7,4    ////__DACE:3:0:4    ////__DACE:1
                double __tlet_result;                                                             ////__DACE:3:0:4    ////__DACE:3:0:4    ////__DACE:1
                ////__DACE:3:0:4                                                              ////__DACE:3:0:4    ////__DACE:1
                ///////////////////                                                               ////__DACE:3:0:4    ////__DACE:3:0:4    ////__DACE:1
                // Tasklet code (tlet_66_multiplies_0)                                            ////__DACE:3:0:4    ////__DACE:3:0:4    ////__DACE:1
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:3:0:4    ////__DACE:3:0:4    ////__DACE:1
                ///////////////////                                                               ////__DACE:3:0:4    ////__DACE:3:0:4    ////__DACE:1
                ////__DACE:3:0:4                                                              ////__DACE:3:0:4    ////__DACE:1
                __map_fusion_gtir_tmp_170_0 = __tlet_result;                                      ////__DACE:3:0:4    ////__DACE:3:0:4    ////__DACE:1
            }                                                                                 ////__DACE:3:0:4    ////__DACE:1
            {                                                                                 ////__DACE:3:0:2    ////__DACE:1
                double __tlet_arg0 = __arg2_;                                                     ////__DACE:3:0:5,2    ////__DACE:3:0:2    ////__DACE:1
                double __tlet_arg1 = __map_fusion_gtir_tmp_170_0;                                 ////__DACE:3:0:3,2    ////__DACE:3:0:2    ////__DACE:1
                double __tlet_result;                                                             ////__DACE:3:0:2    ////__DACE:3:0:2    ////__DACE:1
                ////__DACE:3:0:2                                                              ////__DACE:3:0:2    ////__DACE:1
                ///////////////////                                                               ////__DACE:3:0:2    ////__DACE:3:0:2    ////__DACE:1
                // Tasklet code (tlet_67_plus_0)                                                  ////__DACE:3:0:2    ////__DACE:3:0:2    ////__DACE:1
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:3:0:2    ////__DACE:3:0:2    ////__DACE:1
                ///////////////////                                                               ////__DACE:3:0:2    ////__DACE:3:0:2    ////__DACE:1
                ////__DACE:3:0:2                                                              ////__DACE:3:0:2    ////__DACE:1
                __arg1__ = __tlet_result;                                                         ////__DACE:3:0:2    ////__DACE:3:0:2    ////__DACE:1
            }                                                                                 ////__DACE:3:0:2    ////__DACE:1
            {                                                                                 ////__DACE:3:0:8    ////__DACE:1
                double _cpy_in = __arg1__;                                                        ////__DACE:3:0:1,8    ////__DACE:3:0:8    ////__DACE:1
                double _cpy_out;                                                                  ////__DACE:3:0:8    ////__DACE:3:0:8    ////__DACE:1
                ////__DACE:3:0:8                                                              ////__DACE:3:0:8    ////__DACE:1
                ///////////////////                                                               ////__DACE:3:0:8    ////__DACE:3:0:8    ////__DACE:1
                // Tasklet code (copy___arg1___to___output)                                       ////__DACE:3:0:8    ////__DACE:3:0:8    ////__DACE:1
                _cpy_out = _cpy_in;                                                               ////__DACE:3:0:8    ////__DACE:3:0:8    ////__DACE:1
                ///////////////////                                                               ////__DACE:3:0:8    ////__DACE:3:0:8    ////__DACE:1
                ////__DACE:3:0:8                                                              ////__DACE:3:0:8    ////__DACE:1
                __output = _cpy_out;                                                              ////__DACE:3:0:8    ////__DACE:3:0:8    ////__DACE:1
            }                                                                                 ////__DACE:3:0:8    ////__DACE:1
            ////__DACE:1
        }                                                                             ////__DACE:1
    } else {                                                                          ////__DACE:1
        {                                                                             ////__DACE:1
            ////__DACE:1
            {                                                                                 ////__DACE:4:0:2    ////__DACE:1
                double _cpy_in = __arg2_;                                                         ////__DACE:4:0:1,2    ////__DACE:4:0:2    ////__DACE:1
                double _cpy_out;                                                                  ////__DACE:4:0:2    ////__DACE:4:0:2    ////__DACE:1
                ////__DACE:4:0:2                                                              ////__DACE:4:0:2    ////__DACE:1
                ///////////////////                                                               ////__DACE:4:0:2    ////__DACE:4:0:2    ////__DACE:1
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:4:0:2    ////__DACE:4:0:2    ////__DACE:1
                _cpy_out = _cpy_in;                                                               ////__DACE:4:0:2    ////__DACE:4:0:2    ////__DACE:1
                ///////////////////                                                               ////__DACE:4:0:2    ////__DACE:4:0:2    ////__DACE:1
                ////__DACE:4:0:2                                                              ////__DACE:4:0:2    ////__DACE:1
                __output = _cpy_out;                                                              ////__DACE:4:0:2    ////__DACE:4:0:2    ////__DACE:1
            }                                                                                 ////__DACE:4:0:2    ////__DACE:1
            ////__DACE:1
        }                                                                             ////__DACE:1
    }                                                                                 ////__DACE:1
}                                                                                 ////__DACE:0:0:116
////__DACE:0:0:116
DACE_DFI void reduce_0_0_307(double* __restrict__ _in, double&  _out) {           ////__DACE:0:0:307
    ////__DACE:53
    {                                                                                 ////__DACE:53
        ////__DACE:53
        {                                                                                 ////__DACE:53:0:0    ////__DACE:53
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {                                       ////__DACE:53:0:0    ////__DACE:53
                {                                                                         ////__DACE:53:0:1    ////__DACE:53
                    double __out;                                                                     ////__DACE:53:0:1    ////__DACE:53:0:1    ////__DACE:53
                    ////__DACE:53:0:1                                                     ////__DACE:53:0:1    ////__DACE:53
                    ///////////////////                                                               ////__DACE:53:0:1    ////__DACE:53:0:1    ////__DACE:53
                    // Tasklet code (reduce_init)                                                     ////__DACE:53:0:1    ////__DACE:53:0:1    ////__DACE:53
                    __out = 0;                                                                        ////__DACE:53:0:1    ////__DACE:53:0:1    ////__DACE:53
                    ///////////////////                                                               ////__DACE:53:0:1    ////__DACE:53:0:1    ////__DACE:53
                    ////__DACE:53:0:1                                                     ////__DACE:53:0:1    ////__DACE:53
                    _out = __out;                                                                     ////__DACE:53:0:1    ////__DACE:53:0:1    ////__DACE:53
                }                                                                         ////__DACE:53:0:1    ////__DACE:53
            }                                                                             ////__DACE:53:0:2    ////__DACE:53
        }                                                                                 ////__DACE:53:0:2    ////__DACE:53
        ////__DACE:53
    }                                                                                 ////__DACE:53
    {                                                                                 ////__DACE:53
        ////__DACE:53
        {                                                                                 ////__DACE:53:1:0    ////__DACE:53
            for (auto _i0 = 0; _i0 < 2; _i0 += 1) {                                       ////__DACE:53:1:0    ////__DACE:53
                {                                                                         ////__DACE:53:1:2    ////__DACE:53
                    double __inp = _in[_i0];                                                          ////__DACE:53:1:3,2    ////__DACE:53:1:2    ////__DACE:53
                    double __out;                                                                     ////__DACE:53:1:2    ////__DACE:53:1:2    ////__DACE:53
                    ////__DACE:53:1:2                                                     ////__DACE:53:1:2    ////__DACE:53
                    ///////////////////                                                               ////__DACE:53:1:2    ////__DACE:53:1:2    ////__DACE:53
                    // Tasklet code (identity)                                                        ////__DACE:53:1:2    ////__DACE:53:1:2    ////__DACE:53
                    __out = __inp;                                                                    ////__DACE:53:1:2    ////__DACE:53:1:2    ////__DACE:53
                    ///////////////////                                                               ////__DACE:53:1:2    ////__DACE:53:1:2    ////__DACE:53
                    ////__DACE:53:1:2                                                     ////__DACE:53:1:2    ////__DACE:53
                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(&_out, __out);          ////__DACE:53:1:2    ////__DACE:53:1:2    ////__DACE:53
                }                                                                         ////__DACE:53:1:2    ////__DACE:53
            }                                                                             ////__DACE:53:1:1    ////__DACE:53
        }                                                                                 ////__DACE:53:1:1    ////__DACE:53
        ////__DACE:53
    }                                                                                 ////__DACE:53
}                                                                                 ////__DACE:0:0:307
////__DACE:0:0:307
DACE_DFI void if_stmt_6_0_0_132(const double&  __arg2_, const bool&  __cond, const double* __restrict__ hydrostatic_correction_on_lowest_level, const double* __restrict__ pg_exdist, double&  __output, int __pg_exdist_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:132
    ////__DACE:5
    if (__cond) {                                                                     ////__DACE:5
        {                                                                             ////__DACE:5
            double __arg1__;                                                                  ////__DACE:7:0:1    ////__DACE:5
            double __map_fusion_gtir_tmp_170_1;                                               ////__DACE:7:0:3    ////__DACE:5
            ////__DACE:5
            {                                                                                 ////__DACE:7:0:4    ////__DACE:5
                double __tlet_arg0 = hydrostatic_correction_on_lowest_level[i_Edge_gtx_horizontal];    ////__DACE:7:0:6,4    ////__DACE:7:0:4    ////__DACE:5
                double __tlet_arg1 = pg_exdist[((__pg_exdist_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:7:0:7,4    ////__DACE:7:0:4    ////__DACE:5
                double __tlet_result;                                                             ////__DACE:7:0:4    ////__DACE:7:0:4    ////__DACE:5
                ////__DACE:7:0:4                                                              ////__DACE:7:0:4    ////__DACE:5
                ///////////////////                                                               ////__DACE:7:0:4    ////__DACE:7:0:4    ////__DACE:5
                // Tasklet code (tlet_66_multiplies_1)                                            ////__DACE:7:0:4    ////__DACE:7:0:4    ////__DACE:5
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:7:0:4    ////__DACE:7:0:4    ////__DACE:5
                ///////////////////                                                               ////__DACE:7:0:4    ////__DACE:7:0:4    ////__DACE:5
                ////__DACE:7:0:4                                                              ////__DACE:7:0:4    ////__DACE:5
                __map_fusion_gtir_tmp_170_1 = __tlet_result;                                      ////__DACE:7:0:4    ////__DACE:7:0:4    ////__DACE:5
            }                                                                                 ////__DACE:7:0:4    ////__DACE:5
            {                                                                                 ////__DACE:7:0:2    ////__DACE:5
                double __tlet_arg0 = __arg2_;                                                     ////__DACE:7:0:5,2    ////__DACE:7:0:2    ////__DACE:5
                double __tlet_arg1 = __map_fusion_gtir_tmp_170_1;                                 ////__DACE:7:0:3,2    ////__DACE:7:0:2    ////__DACE:5
                double __tlet_result;                                                             ////__DACE:7:0:2    ////__DACE:7:0:2    ////__DACE:5
                ////__DACE:7:0:2                                                              ////__DACE:7:0:2    ////__DACE:5
                ///////////////////                                                               ////__DACE:7:0:2    ////__DACE:7:0:2    ////__DACE:5
                // Tasklet code (tlet_67_plus_1)                                                  ////__DACE:7:0:2    ////__DACE:7:0:2    ////__DACE:5
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:7:0:2    ////__DACE:7:0:2    ////__DACE:5
                ///////////////////                                                               ////__DACE:7:0:2    ////__DACE:7:0:2    ////__DACE:5
                ////__DACE:7:0:2                                                              ////__DACE:7:0:2    ////__DACE:5
                __arg1__ = __tlet_result;                                                         ////__DACE:7:0:2    ////__DACE:7:0:2    ////__DACE:5
            }                                                                                 ////__DACE:7:0:2    ////__DACE:5
            {                                                                                 ////__DACE:7:0:8    ////__DACE:5
                double _cpy_in = __arg1__;                                                        ////__DACE:7:0:1,8    ////__DACE:7:0:8    ////__DACE:5
                double _cpy_out;                                                                  ////__DACE:7:0:8    ////__DACE:7:0:8    ////__DACE:5
                ////__DACE:7:0:8                                                              ////__DACE:7:0:8    ////__DACE:5
                ///////////////////                                                               ////__DACE:7:0:8    ////__DACE:7:0:8    ////__DACE:5
                // Tasklet code (copy___arg1___to___output)                                       ////__DACE:7:0:8    ////__DACE:7:0:8    ////__DACE:5
                _cpy_out = _cpy_in;                                                               ////__DACE:7:0:8    ////__DACE:7:0:8    ////__DACE:5
                ///////////////////                                                               ////__DACE:7:0:8    ////__DACE:7:0:8    ////__DACE:5
                ////__DACE:7:0:8                                                              ////__DACE:7:0:8    ////__DACE:5
                __output = _cpy_out;                                                              ////__DACE:7:0:8    ////__DACE:7:0:8    ////__DACE:5
            }                                                                                 ////__DACE:7:0:8    ////__DACE:5
            ////__DACE:5
        }                                                                             ////__DACE:5
    } else {                                                                          ////__DACE:5
        {                                                                             ////__DACE:5
            ////__DACE:5
            {                                                                                 ////__DACE:8:0:2    ////__DACE:5
                double _cpy_in = __arg2_;                                                         ////__DACE:8:0:1,2    ////__DACE:8:0:2    ////__DACE:5
                double _cpy_out;                                                                  ////__DACE:8:0:2    ////__DACE:8:0:2    ////__DACE:5
                ////__DACE:8:0:2                                                              ////__DACE:8:0:2    ////__DACE:5
                ///////////////////                                                               ////__DACE:8:0:2    ////__DACE:8:0:2    ////__DACE:5
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:8:0:2    ////__DACE:8:0:2    ////__DACE:5
                _cpy_out = _cpy_in;                                                               ////__DACE:8:0:2    ////__DACE:8:0:2    ////__DACE:5
                ///////////////////                                                               ////__DACE:8:0:2    ////__DACE:8:0:2    ////__DACE:5
                ////__DACE:8:0:2                                                              ////__DACE:8:0:2    ////__DACE:5
                __output = _cpy_out;                                                              ////__DACE:8:0:2    ////__DACE:8:0:2    ////__DACE:5
            }                                                                                 ////__DACE:8:0:2    ////__DACE:5
            ////__DACE:5
        }                                                                             ////__DACE:5
    }                                                                                 ////__DACE:5
}                                                                                 ////__DACE:0:0:132
////__DACE:0:0:132


int __dace_init_cuda(theta_shared_probe_native_state_t *__state, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __grf_tend_vn_K_stride, int __gt_conn_C2E2CO_neighbor_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __ikoffset_E2C_stride, int __ikoffset_K_stride, int __next_vn_K_stride, int __normal_wind_iau_increment_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, int __zdiff_gradp_E2C_stride, int __zdiff_gradp_K_stride, double dtime) {
    int count;

    // Check that we are able to run hip code
    if (hipGetDeviceCount(&count) != hipSuccess)
    {
        printf("ERROR: GPU drivers are not configured or hip-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No hip-capable devices found\n");
        return 2;
    }

    // One GPU per process, selected here and never changed, so the memory pool, every kernel and
    // every library handle share it. Which physical GPU is the process's business: the visible-
    // devices variable renumbers what it exposes, so a rank's own GPU is device 0. An ordinal
    // fixed at codegen time cannot do that -- every rank shares one build.
    const int __dace_device = 0;
    if (hipSetDevice(__dace_device) != hipSuccess)
    {
        printf("ERROR: could not select hip device 0 out of %d visible\n", count);
        return 4;
    }

    __dace_gpu_drain_error(__state);

    // Initialize hip before we run the application
    float *dev_X;
    DACE_GPU_CHECK(hipMalloc((void **) &dev_X, 1));
    DACE_GPU_CHECK(hipFree(dev_X));

    __state->gpu_context = new dace::cuda::Context(1, 1);

    // After the context exists: DACE_GPU_CHECK records into it.
    
    hipMemPool_t mempool;
    DACE_GPU_CHECK(hipDeviceGetDefaultMemPool(&mempool, __dace_device));
    uint64_t threshold = UINT64_MAX;
    DACE_GPU_CHECK(hipMemPoolSetAttribute(mempool, hipMemPoolAttrReleaseThreshold, &threshold));


    // Create hip streams and events
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(hipStreamCreateWithFlags(&__state->gpu_context->internal_streams[i], hipStreamNonBlocking));
        __state->gpu_context->streams[i] = __state->gpu_context->internal_streams[i]; // Allow for externals to modify streams
    }
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(hipEventCreateWithFlags(&__state->gpu_context->events[i], hipEventDisableTiming));
    }

    

    return 0;
}

int __dace_exit_cuda(theta_shared_probe_native_state_t *__state) {
    

    // Synchronize and check for CUDA errors
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    if (__err == 0)
        __err = static_cast<int>(hipDeviceSynchronize());

    // Destroy hip streams and events
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(hipStreamDestroy(__state->gpu_context->internal_streams[i]));
    }
    for(int i = 0; i < 1; ++i) {
        DACE_GPU_CHECK(hipEventDestroy(__state->gpu_context->events[i]));
    }

    delete __state->gpu_context;
    return __err;
}

// Discard a pending error left by another GPU user in this process, so the next checked call does
// not report it as its own. Sticky errors survive this and are reported normally.
// Must not touch __state->gpu_context: init calls this before the context exists.
void __dace_gpu_drain_error(theta_shared_probe_native_state_t *__state) {
    (void)__state;
    gpuError_t __pre_existing = hipGetLastError();
    if (__pre_existing != (gpuError_t)0) {
        printf("WARNING: a GPU error was already pending on entry to a DaCe program and has been "
               "discarded: %s (%d). It was not caused by this SDFG.\n",
               gpuGetErrorString(__pre_existing), __pre_existing);
    }
}

// Returns what the generated code recorded, not the runtime's shared slot, and clears it.
int __dace_gpu_last_error(theta_shared_probe_native_state_t *__state) {
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    __state->gpu_context->lasterror = (gpuError_t)0;
    return __err;
}

bool __dace_gpu_set_stream(theta_shared_probe_native_state_t *__state, int streamid, gpuStream_t stream)
{
    if (streamid < 0 || streamid >= 1)
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}

void __dace_gpu_set_all_streams(theta_shared_probe_native_state_t *__state, gpuStream_t stream)
{
    for (int i = 0; i < 1; ++i)
        __state->gpu_context->streams[i] = stream;
}

__global__ void  __launch_bounds__(256) map_37_fieldop_0_0_354(const double * __restrict__ geofac_grg_x, const double * __restrict__ geofac_grg_y, const int * __restrict__ gt_conn_C2E2CO, double * __restrict__ gtir_tmp_101, double * __restrict__ gtir_tmp_83, double * __restrict__ gtir_tmp_89, double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __gt_conn_C2E2CO_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride) {    ////__DACE:0:0:354
    {                                                                                 ////__DACE:0:0:354
        {                                                                             ////__DACE:0:0:354
            int b_i_Cell_gtx_horizontal = ((256 * blockIdx.x) + 2406);                ////__DACE:0:0:354
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:354
            {                                                                         ////__DACE:0:0:20
                {                                                                     ////__DACE:0:0:20
                    {                                                                 ////__DACE:0:0:20
                        int i_Cell_gtx_horizontal = (threadIdx.x + b_i_Cell_gtx_horizontal);    ////__DACE:0:0:20
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:20
                        if (i_Cell_gtx_horizontal >= b_i_Cell_gtx_horizontal && i_Cell_gtx_horizontal < (Min(44527, (b_i_Cell_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:20
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(29, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:20
                                {                                                     ////__DACE:0:0:245
                                    #pragma unroll 4                                  ////__DACE:0:0:245
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:245
                                        double gtir_tmp_100;                          ////__DACE:0:0:13
                                        double gtir_tmp_94;                           ////__DACE:0:0:25
                                        double gtir_tmp_88;                           ////__DACE:0:0:34
                                        double gtir_tmp_82;                           ////__DACE:0:0:43
                                        double __map_fusion_gtir_tmp_99[4]  DACE_ALIGN(64);    ////__DACE:0:0:73
                                        double __map_fusion_gtir_tmp_97[4]  DACE_ALIGN(64);    ////__DACE:0:0:74
                                        double __map_fusion_gtir_tmp_93[4]  DACE_ALIGN(64);    ////__DACE:0:0:75
                                        double __map_fusion_gtir_tmp_91[4]  DACE_ALIGN(64);    ////__DACE:0:0:76
                                        double __map_fusion_gtir_tmp_87[4]  DACE_ALIGN(64);    ////__DACE:0:0:77
                                        double __map_fusion_gtir_tmp_85[4]  DACE_ALIGN(64);    ////__DACE:0:0:78
                                        double __map_fusion_gtir_tmp_81[4]  DACE_ALIGN(64);    ////__DACE:0:0:79
                                        double __map_fusion_gtir_tmp_79[4]  DACE_ALIGN(64);    ////__DACE:0:0:80
                                        {                                             ////__DACE:0:0:49
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:49
                                                double __gtx_double_write_remover_inner_inner_distribution_node_6;    ////__DACE:0:0:91
                                                {                                     ////__DACE:0:0:48
                                                    int __tlet_index = gt_conn_C2E2CO[((__gt_conn_C2E2CO_neighbor_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:21,48    ////__DACE:0:0:48
                                                    const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[(__perturbed_rho_at_cells_on_model_levels_K_stride * i_K_gtx_vertical)];    ////__DACE:0:0:41,48    ////__DACE:0:0:48
                                                    double __tlet_val;                                                                ////__DACE:0:0:48    ////__DACE:0:0:48
                                                    ////__DACE:0:0:48                 ////__DACE:0:0:48
                                                    ///////////////////                                                               ////__DACE:0:0:48    ////__DACE:0:0:48
                                                    // Tasklet code (tlet_26_C2E2CO_neighbors)                                        ////__DACE:0:0:48    ////__DACE:0:0:48
                                                    __tlet_val = __tlet_field[__tlet_index];                                          ////__DACE:0:0:48    ////__DACE:0:0:48
                                                    ///////////////////                                                               ////__DACE:0:0:48    ////__DACE:0:0:48
                                                    ////__DACE:0:0:48                 ////__DACE:0:0:48
                                                    __gtx_double_write_remover_inner_inner_distribution_node_6 = __tlet_val;          ////__DACE:0:0:48    ////__DACE:0:0:48
                                                }                                     ////__DACE:0:0:48
                                                {                                     ////__DACE:0:0:315
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_6;      ////__DACE:0:0:91,315    ////__DACE:0:0:315
                                                    double _cpy_out;                                                                  ////__DACE:0:0:315    ////__DACE:0:0:315
                                                    ////__DACE:0:0:315                ////__DACE:0:0:315
                                                    ///////////////////                                                               ////__DACE:0:0:315    ////__DACE:0:0:315
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_6_to___map_fusion_gtir_tmp_79)    ////__DACE:0:0:315    ////__DACE:0:0:315
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:315    ////__DACE:0:0:315
                                                    ///////////////////                                                               ////__DACE:0:0:315    ////__DACE:0:0:315
                                                    ////__DACE:0:0:315                ////__DACE:0:0:315
                                                    __map_fusion_gtir_tmp_79[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:315    ////__DACE:0:0:315
                                                }                                     ////__DACE:0:0:315
                                            }                                         ////__DACE:0:0:47
                                        }                                             ////__DACE:0:0:47
                                        {                                             ////__DACE:0:0:46
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:46
                                                double __gtx_double_write_remover_inner_inner_distribution_node_5;    ////__DACE:0:0:90
                                                {                                     ////__DACE:0:0:45
                                                    double __tlet_arg0 = geofac_grg_x[((__geofac_grg_x_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:32,45    ////__DACE:0:0:45
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_79[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:80,45    ////__DACE:0:0:45
                                                    double __tlet_out;                                                                ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    ////__DACE:0:0:45                 ////__DACE:0:0:45
                                                    ///////////////////                                                               ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    // Tasklet code (tlet_27_map)                                                     ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    ///////////////////                                                               ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    ////__DACE:0:0:45                 ////__DACE:0:0:45
                                                    __gtx_double_write_remover_inner_inner_distribution_node_5 = __tlet_out;          ////__DACE:0:0:45    ////__DACE:0:0:45
                                                }                                     ////__DACE:0:0:45
                                                {                                     ////__DACE:0:0:314
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_5;      ////__DACE:0:0:90,314    ////__DACE:0:0:314
                                                    double _cpy_out;                                                                  ////__DACE:0:0:314    ////__DACE:0:0:314
                                                    ////__DACE:0:0:314                ////__DACE:0:0:314
                                                    ///////////////////                                                               ////__DACE:0:0:314    ////__DACE:0:0:314
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_5_to___map_fusion_gtir_tmp_81)    ////__DACE:0:0:314    ////__DACE:0:0:314
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:314    ////__DACE:0:0:314
                                                    ///////////////////                                                               ////__DACE:0:0:314    ////__DACE:0:0:314
                                                    ////__DACE:0:0:314                ////__DACE:0:0:314
                                                    __map_fusion_gtir_tmp_81[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:314    ////__DACE:0:0:314
                                                }                                     ////__DACE:0:0:314
                                            }                                         ////__DACE:0:0:44
                                        }                                             ////__DACE:0:0:44
                                        reduce_0_0_306(&__map_fusion_gtir_tmp_81[0], gtir_tmp_82);    ////__DACE:0:0:306
                                        {                                             ////__DACE:0:0:343
                                            double _cpy_in = gtir_tmp_82;                                                     ////__DACE:0:0:43,343    ////__DACE:0:0:343
                                            double _cpy_out;                                                                  ////__DACE:0:0:343    ////__DACE:0:0:343
                                            ////__DACE:0:0:343                        ////__DACE:0:0:343
                                            ///////////////////                                                               ////__DACE:0:0:343    ////__DACE:0:0:343
                                            // Tasklet code (copy_gtir_tmp_82_to_gtir_tmp_83)                                 ////__DACE:0:0:343    ////__DACE:0:0:343
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:343    ////__DACE:0:0:343
                                            ///////////////////                                                               ////__DACE:0:0:343    ////__DACE:0:0:343
                                            ////__DACE:0:0:343                        ////__DACE:0:0:343
                                            gtir_tmp_83[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:343    ////__DACE:0:0:343
                                        }                                             ////__DACE:0:0:343
                                        {                                             ////__DACE:0:0:40
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:40
                                                double __gtx_double_write_remover_inner_inner_distribution_node_4;    ////__DACE:0:0:89
                                                {                                     ////__DACE:0:0:39
                                                    int __tlet_index = gt_conn_C2E2CO[((__gt_conn_C2E2CO_neighbor_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:21,39    ////__DACE:0:0:39
                                                    const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[(__perturbed_rho_at_cells_on_model_levels_K_stride * i_K_gtx_vertical)];    ////__DACE:0:0:41,39    ////__DACE:0:0:39
                                                    double __tlet_val;                                                                ////__DACE:0:0:39    ////__DACE:0:0:39
                                                    ////__DACE:0:0:39                 ////__DACE:0:0:39
                                                    ///////////////////                                                               ////__DACE:0:0:39    ////__DACE:0:0:39
                                                    // Tasklet code (tlet_28_C2E2CO_neighbors)                                        ////__DACE:0:0:39    ////__DACE:0:0:39
                                                    __tlet_val = __tlet_field[__tlet_index];                                          ////__DACE:0:0:39    ////__DACE:0:0:39
                                                    ///////////////////                                                               ////__DACE:0:0:39    ////__DACE:0:0:39
                                                    ////__DACE:0:0:39                 ////__DACE:0:0:39
                                                    __gtx_double_write_remover_inner_inner_distribution_node_4 = __tlet_val;          ////__DACE:0:0:39    ////__DACE:0:0:39
                                                }                                     ////__DACE:0:0:39
                                                {                                     ////__DACE:0:0:313
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_4;      ////__DACE:0:0:89,313    ////__DACE:0:0:313
                                                    double _cpy_out;                                                                  ////__DACE:0:0:313    ////__DACE:0:0:313
                                                    ////__DACE:0:0:313                ////__DACE:0:0:313
                                                    ///////////////////                                                               ////__DACE:0:0:313    ////__DACE:0:0:313
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_4_to___map_fusion_gtir_tmp_85)    ////__DACE:0:0:313    ////__DACE:0:0:313
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:313    ////__DACE:0:0:313
                                                    ///////////////////                                                               ////__DACE:0:0:313    ////__DACE:0:0:313
                                                    ////__DACE:0:0:313                ////__DACE:0:0:313
                                                    __map_fusion_gtir_tmp_85[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:313    ////__DACE:0:0:313
                                                }                                     ////__DACE:0:0:313
                                            }                                         ////__DACE:0:0:38
                                        }                                             ////__DACE:0:0:38
                                        {                                             ////__DACE:0:0:37
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:37
                                                double __gtx_double_write_remover_inner_inner_distribution_node_3;    ////__DACE:0:0:88
                                                {                                     ////__DACE:0:0:36
                                                    double __tlet_arg0 = geofac_grg_y[((__geofac_grg_y_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:23,36    ////__DACE:0:0:36
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_85[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:78,36    ////__DACE:0:0:36
                                                    double __tlet_out;                                                                ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    ////__DACE:0:0:36                 ////__DACE:0:0:36
                                                    ///////////////////                                                               ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    // Tasklet code (tlet_29_map)                                                     ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    ///////////////////                                                               ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    ////__DACE:0:0:36                 ////__DACE:0:0:36
                                                    __gtx_double_write_remover_inner_inner_distribution_node_3 = __tlet_out;          ////__DACE:0:0:36    ////__DACE:0:0:36
                                                }                                     ////__DACE:0:0:36
                                                {                                     ////__DACE:0:0:312
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_3;      ////__DACE:0:0:88,312    ////__DACE:0:0:312
                                                    double _cpy_out;                                                                  ////__DACE:0:0:312    ////__DACE:0:0:312
                                                    ////__DACE:0:0:312                ////__DACE:0:0:312
                                                    ///////////////////                                                               ////__DACE:0:0:312    ////__DACE:0:0:312
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_3_to___map_fusion_gtir_tmp_87)    ////__DACE:0:0:312    ////__DACE:0:0:312
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:312    ////__DACE:0:0:312
                                                    ///////////////////                                                               ////__DACE:0:0:312    ////__DACE:0:0:312
                                                    ////__DACE:0:0:312                ////__DACE:0:0:312
                                                    __map_fusion_gtir_tmp_87[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:312    ////__DACE:0:0:312
                                                }                                     ////__DACE:0:0:312
                                            }                                         ////__DACE:0:0:35
                                        }                                             ////__DACE:0:0:35
                                        reduce_0_0_306(&__map_fusion_gtir_tmp_87[0], gtir_tmp_88);    ////__DACE:0:0:305
                                        {                                             ////__DACE:0:0:344
                                            double _cpy_in = gtir_tmp_88;                                                     ////__DACE:0:0:34,344    ////__DACE:0:0:344
                                            double _cpy_out;                                                                  ////__DACE:0:0:344    ////__DACE:0:0:344
                                            ////__DACE:0:0:344                        ////__DACE:0:0:344
                                            ///////////////////                                                               ////__DACE:0:0:344    ////__DACE:0:0:344
                                            // Tasklet code (copy_gtir_tmp_88_to_gtir_tmp_89)                                 ////__DACE:0:0:344    ////__DACE:0:0:344
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:344    ////__DACE:0:0:344
                                            ///////////////////                                                               ////__DACE:0:0:344    ////__DACE:0:0:344
                                            ////__DACE:0:0:344                        ////__DACE:0:0:344
                                            gtir_tmp_89[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:344    ////__DACE:0:0:344
                                        }                                             ////__DACE:0:0:344
                                        {                                             ////__DACE:0:0:31
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:31
                                                double __gtx_double_write_remover_inner_inner_distribution_node_2;    ////__DACE:0:0:87
                                                {                                     ////__DACE:0:0:30
                                                    int __tlet_index = gt_conn_C2E2CO[((__gt_conn_C2E2CO_neighbor_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:21,30    ////__DACE:0:0:30
                                                    const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[(__perturbed_theta_v_at_cells_on_model_levels_K_stride * i_K_gtx_vertical)];    ////__DACE:0:0:22,30    ////__DACE:0:0:30
                                                    double __tlet_val;                                                                ////__DACE:0:0:30    ////__DACE:0:0:30
                                                    ////__DACE:0:0:30                 ////__DACE:0:0:30
                                                    ///////////////////                                                               ////__DACE:0:0:30    ////__DACE:0:0:30
                                                    // Tasklet code (tlet_30_C2E2CO_neighbors)                                        ////__DACE:0:0:30    ////__DACE:0:0:30
                                                    __tlet_val = __tlet_field[__tlet_index];                                          ////__DACE:0:0:30    ////__DACE:0:0:30
                                                    ///////////////////                                                               ////__DACE:0:0:30    ////__DACE:0:0:30
                                                    ////__DACE:0:0:30                 ////__DACE:0:0:30
                                                    __gtx_double_write_remover_inner_inner_distribution_node_2 = __tlet_val;          ////__DACE:0:0:30    ////__DACE:0:0:30
                                                }                                     ////__DACE:0:0:30
                                                {                                     ////__DACE:0:0:311
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_2;      ////__DACE:0:0:87,311    ////__DACE:0:0:311
                                                    double _cpy_out;                                                                  ////__DACE:0:0:311    ////__DACE:0:0:311
                                                    ////__DACE:0:0:311                ////__DACE:0:0:311
                                                    ///////////////////                                                               ////__DACE:0:0:311    ////__DACE:0:0:311
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_2_to___map_fusion_gtir_tmp_91)    ////__DACE:0:0:311    ////__DACE:0:0:311
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:311    ////__DACE:0:0:311
                                                    ///////////////////                                                               ////__DACE:0:0:311    ////__DACE:0:0:311
                                                    ////__DACE:0:0:311                ////__DACE:0:0:311
                                                    __map_fusion_gtir_tmp_91[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:311    ////__DACE:0:0:311
                                                }                                     ////__DACE:0:0:311
                                            }                                         ////__DACE:0:0:29
                                        }                                             ////__DACE:0:0:29
                                        {                                             ////__DACE:0:0:28
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:28
                                                double __gtx_double_write_remover_inner_inner_distribution_node_1;    ////__DACE:0:0:86
                                                {                                     ////__DACE:0:0:27
                                                    double __tlet_arg0 = geofac_grg_x[((__geofac_grg_x_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:32,27    ////__DACE:0:0:27
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_91[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:76,27    ////__DACE:0:0:27
                                                    double __tlet_out;                                                                ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    ////__DACE:0:0:27                 ////__DACE:0:0:27
                                                    ///////////////////                                                               ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    // Tasklet code (tlet_31_map)                                                     ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    ///////////////////                                                               ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    ////__DACE:0:0:27                 ////__DACE:0:0:27
                                                    __gtx_double_write_remover_inner_inner_distribution_node_1 = __tlet_out;          ////__DACE:0:0:27    ////__DACE:0:0:27
                                                }                                     ////__DACE:0:0:27
                                                {                                     ////__DACE:0:0:310
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_1;      ////__DACE:0:0:86,310    ////__DACE:0:0:310
                                                    double _cpy_out;                                                                  ////__DACE:0:0:310    ////__DACE:0:0:310
                                                    ////__DACE:0:0:310                ////__DACE:0:0:310
                                                    ///////////////////                                                               ////__DACE:0:0:310    ////__DACE:0:0:310
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_1_to___map_fusion_gtir_tmp_93)    ////__DACE:0:0:310    ////__DACE:0:0:310
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:310    ////__DACE:0:0:310
                                                    ///////////////////                                                               ////__DACE:0:0:310    ////__DACE:0:0:310
                                                    ////__DACE:0:0:310                ////__DACE:0:0:310
                                                    __map_fusion_gtir_tmp_93[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:310    ////__DACE:0:0:310
                                                }                                     ////__DACE:0:0:310
                                            }                                         ////__DACE:0:0:26
                                        }                                             ////__DACE:0:0:26
                                        reduce_0_0_306(&__map_fusion_gtir_tmp_93[0], gtir_tmp_94);    ////__DACE:0:0:304
                                        {                                             ////__DACE:0:0:345
                                            double _cpy_in = gtir_tmp_94;                                                     ////__DACE:0:0:25,345    ////__DACE:0:0:345
                                            double _cpy_out;                                                                  ////__DACE:0:0:345    ////__DACE:0:0:345
                                            ////__DACE:0:0:345                        ////__DACE:0:0:345
                                            ///////////////////                                                               ////__DACE:0:0:345    ////__DACE:0:0:345
                                            // Tasklet code (copy_gtir_tmp_94_to_gtir_tmp_95)                                 ////__DACE:0:0:345    ////__DACE:0:0:345
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:345    ////__DACE:0:0:345
                                            ///////////////////                                                               ////__DACE:0:0:345    ////__DACE:0:0:345
                                            ////__DACE:0:0:345                        ////__DACE:0:0:345
                                            gtir_tmp_95[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:345    ////__DACE:0:0:345
                                        }                                             ////__DACE:0:0:345
                                        {                                             ////__DACE:0:0:19
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:19
                                                double __gtx_double_write_remover_inner_inner_distribution_node_0;    ////__DACE:0:0:85
                                                {                                     ////__DACE:0:0:18
                                                    int __tlet_index = gt_conn_C2E2CO[((__gt_conn_C2E2CO_neighbor_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:21,18    ////__DACE:0:0:18
                                                    const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[(__perturbed_theta_v_at_cells_on_model_levels_K_stride * i_K_gtx_vertical)];    ////__DACE:0:0:22,18    ////__DACE:0:0:18
                                                    double __tlet_val;                                                                ////__DACE:0:0:18    ////__DACE:0:0:18
                                                    ////__DACE:0:0:18                 ////__DACE:0:0:18
                                                    ///////////////////                                                               ////__DACE:0:0:18    ////__DACE:0:0:18
                                                    // Tasklet code (tlet_32_C2E2CO_neighbors)                                        ////__DACE:0:0:18    ////__DACE:0:0:18
                                                    __tlet_val = __tlet_field[__tlet_index];                                          ////__DACE:0:0:18    ////__DACE:0:0:18
                                                    ///////////////////                                                               ////__DACE:0:0:18    ////__DACE:0:0:18
                                                    ////__DACE:0:0:18                 ////__DACE:0:0:18
                                                    __gtx_double_write_remover_inner_inner_distribution_node_0 = __tlet_val;          ////__DACE:0:0:18    ////__DACE:0:0:18
                                                }                                     ////__DACE:0:0:18
                                                {                                     ////__DACE:0:0:309
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_0;      ////__DACE:0:0:85,309    ////__DACE:0:0:309
                                                    double _cpy_out;                                                                  ////__DACE:0:0:309    ////__DACE:0:0:309
                                                    ////__DACE:0:0:309                ////__DACE:0:0:309
                                                    ///////////////////                                                               ////__DACE:0:0:309    ////__DACE:0:0:309
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_0_to___map_fusion_gtir_tmp_97)    ////__DACE:0:0:309    ////__DACE:0:0:309
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:309    ////__DACE:0:0:309
                                                    ///////////////////                                                               ////__DACE:0:0:309    ////__DACE:0:0:309
                                                    ////__DACE:0:0:309                ////__DACE:0:0:309
                                                    __map_fusion_gtir_tmp_97[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:309    ////__DACE:0:0:309
                                                }                                     ////__DACE:0:0:309
                                            }                                         ////__DACE:0:0:17
                                        }                                             ////__DACE:0:0:17
                                        {                                             ////__DACE:0:0:16
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:16
                                                double __gtx_double_write_remover_inner_inner_distribution_node;    ////__DACE:0:0:84
                                                {                                     ////__DACE:0:0:15
                                                    double __tlet_arg0 = geofac_grg_y[((__geofac_grg_y_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:23,15    ////__DACE:0:0:15
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_97[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:74,15    ////__DACE:0:0:15
                                                    double __tlet_out;                                                                ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    ////__DACE:0:0:15                 ////__DACE:0:0:15
                                                    ///////////////////                                                               ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    // Tasklet code (tlet_33_map)                                                     ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    ///////////////////                                                               ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    ////__DACE:0:0:15                 ////__DACE:0:0:15
                                                    __gtx_double_write_remover_inner_inner_distribution_node = __tlet_out;            ////__DACE:0:0:15    ////__DACE:0:0:15
                                                }                                     ////__DACE:0:0:15
                                                {                                     ////__DACE:0:0:308
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node;        ////__DACE:0:0:84,308    ////__DACE:0:0:308
                                                    double _cpy_out;                                                                  ////__DACE:0:0:308    ////__DACE:0:0:308
                                                    ////__DACE:0:0:308                ////__DACE:0:0:308
                                                    ///////////////////                                                               ////__DACE:0:0:308    ////__DACE:0:0:308
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_to___map_fusion_gtir_tmp_99)    ////__DACE:0:0:308    ////__DACE:0:0:308
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:308    ////__DACE:0:0:308
                                                    ///////////////////                                                               ////__DACE:0:0:308    ////__DACE:0:0:308
                                                    ////__DACE:0:0:308                ////__DACE:0:0:308
                                                    __map_fusion_gtir_tmp_99[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:308    ////__DACE:0:0:308
                                                }                                     ////__DACE:0:0:308
                                            }                                         ////__DACE:0:0:14
                                        }                                             ////__DACE:0:0:14
                                        reduce_0_0_306(&__map_fusion_gtir_tmp_99[0], gtir_tmp_100);    ////__DACE:0:0:303
                                        {                                             ////__DACE:0:0:342
                                            double _cpy_in = gtir_tmp_100;                                                    ////__DACE:0:0:13,342    ////__DACE:0:0:342
                                            double _cpy_out;                                                                  ////__DACE:0:0:342    ////__DACE:0:0:342
                                            ////__DACE:0:0:342                        ////__DACE:0:0:342
                                            ///////////////////                                                               ////__DACE:0:0:342    ////__DACE:0:0:342
                                            // Tasklet code (copy_gtir_tmp_100_to_gtir_tmp_101)                               ////__DACE:0:0:342    ////__DACE:0:0:342
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:342    ////__DACE:0:0:342
                                            ///////////////////                                                               ////__DACE:0:0:342    ////__DACE:0:0:342
                                            ////__DACE:0:0:342                        ////__DACE:0:0:342
                                            gtir_tmp_101[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:342    ////__DACE:0:0:342
                                        }                                             ////__DACE:0:0:342
                                    }                                                 ////__DACE:0:0:246
                                }                                                     ////__DACE:0:0:246
                            }                                                         ////__DACE:0:0:12
                        }                                                             ////__DACE:0:0:12
                    }                                                                 ////__DACE:0:0:12
                }                                                                     ////__DACE:0:0:12
            }                                                                         ////__DACE:0:0:12
        }                                                                             ////__DACE:0:0:355
    }                                                                                 ////__DACE:0:0:355
}                                                                                 ////__DACE:0:0:355

                                                                                  ////__DACE:0:0:354
DACE_EXPORTED void __dace_runkernel_map_37_fieldop_0_0_354(theta_shared_probe_native_state_t *__state, const double * __restrict__ geofac_grg_x, const double * __restrict__ geofac_grg_y, const int * __restrict__ gt_conn_C2E2CO, double * __restrict__ gtir_tmp_101, double * __restrict__ gtir_tmp_83, double * __restrict__ gtir_tmp_89, double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __gt_conn_C2E2CO_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride);    ////__DACE:0:0:354
void __dace_runkernel_map_37_fieldop_0_0_354(theta_shared_probe_native_state_t *__state, const double * __restrict__ geofac_grg_x, const double * __restrict__ geofac_grg_y, const int * __restrict__ gt_conn_C2E2CO, double * __restrict__ gtir_tmp_101, double * __restrict__ gtir_tmp_83, double * __restrict__ gtir_tmp_89, double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __gt_conn_C2E2CO_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride)    ////__DACE:0:0:354
{                                                                                 ////__DACE:0:0:354
                                                                                  ////__DACE:0:0:354
    void  *map_37_fieldop_0_0_354_args[] = { (void *)&geofac_grg_x, (void *)&geofac_grg_y, (void *)&gt_conn_C2E2CO, (void *)&gtir_tmp_101, (void *)&gtir_tmp_83, (void *)&gtir_tmp_89, (void *)&gtir_tmp_95, (void *)&perturbed_rho_at_cells_on_model_levels, (void *)&perturbed_theta_v_at_cells_on_model_levels, (void *)&__geofac_grg_x_C2E2CO_stride, (void *)&__geofac_grg_y_C2E2CO_stride, (void *)&__gt_conn_C2E2CO_neighbor_stride, (void *)&__perturbed_rho_at_cells_on_model_levels_K_stride, (void *)&__perturbed_theta_v_at_cells_on_model_levels_K_stride };    ////__DACE:0:0:354
    gpuError_t __err = hipLaunchKernel((void*)map_37_fieldop_0_0_354, dim3(165, 30, 1), dim3(256, 1, 1), map_37_fieldop_0_0_354_args, 0, nullptr);    ////__DACE:0:0:354
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_37_fieldop_0_0_354", 165, 30, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_100_fieldop_1_0_0_362(const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:362
    {                                                                                 ////__DACE:0:0:362
        {                                                                             ////__DACE:0:0:362
            int b_i_Edge_gtx_horizontal = ((256 * blockIdx.x) + 7701);                ////__DACE:0:0:362
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:362
            {                                                                         ////__DACE:0:0:220
                {                                                                     ////__DACE:0:0:220
                    {                                                                 ////__DACE:0:0:220
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:220
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:220
                        double gtir_tmp_14_1;                                         ////__DACE:0:0:200
                        double gtir_tmp_12_1;                                         ////__DACE:0:0:201
                        double gtir_tmp_26_1;                                         ////__DACE:0:0:206
                        double gtir_tmp_24_1;                                         ////__DACE:0:0:207
                        double gtir_tmp_70_1;                                         ////__DACE:0:0:211
                        double gtir_tmp_66_1;                                         ////__DACE:0:0:212
                        double gtir_tmp_60_1;                                         ////__DACE:0:0:213
                        double gtir_tmp_56_1;                                         ////__DACE:0:0:214
                        double gtir_tmp_48_1;                                         ////__DACE:0:0:216
                        double gtir_tmp_44_1;                                         ////__DACE:0:0:217
                        double gtir_tmp_38_1;                                         ////__DACE:0:0:218
                        double gtir_tmp_34_1;                                         ////__DACE:0:0:219
                        bool gtir_tmp_7_0;                                            ////__DACE:0:0:258
                        bool gtir_tmp_6_0;                                            ////__DACE:0:0:262
                        double gtir_tmp_3_1;                                          ////__DACE:0:0:268
                        double __p_dthalf_1;                                          ////__DACE:0:0:272
                        double lambda_4___p_dthalf_0;                                 ////__DACE:0:0:274
                        double gtir_tmp_102_0;                                        ////__DACE:0:0:278
                        double gtir_tmp_175_0;                                        ////__DACE:0:0:288
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(67095, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:220
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(29, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:220
                                {                                                     ////__DACE:0:0:257
                                    bool __tlet_out;                                                                  ////__DACE:0:0:257    ////__DACE:0:0:257
                                    ////__DACE:0:0:257                                ////__DACE:0:0:257
                                    ///////////////////                                                               ////__DACE:0:0:257    ////__DACE:0:0:257
                                    // Tasklet code (tlet_5_get_value__clone_0)                                       ////__DACE:0:0:257    ////__DACE:0:0:257
                                    __tlet_out = false;                                                               ////__DACE:0:0:257    ////__DACE:0:0:257
                                    ///////////////////                                                               ////__DACE:0:0:257    ////__DACE:0:0:257
                                    ////__DACE:0:0:257                                ////__DACE:0:0:257
                                    gtir_tmp_7_0 = __tlet_out;                                                        ////__DACE:0:0:257    ////__DACE:0:0:257
                                }                                                     ////__DACE:0:0:257
                                {                                                     ////__DACE:0:0:261
                                    bool __tlet_out;                                                                  ////__DACE:0:0:261    ////__DACE:0:0:261
                                    ////__DACE:0:0:261                                ////__DACE:0:0:261
                                    ///////////////////                                                               ////__DACE:0:0:261    ////__DACE:0:0:261
                                    // Tasklet code (tlet_4_get_value__clone_0)                                       ////__DACE:0:0:261    ////__DACE:0:0:261
                                    __tlet_out = true;                                                                ////__DACE:0:0:261    ////__DACE:0:0:261
                                    ///////////////////                                                               ////__DACE:0:0:261    ////__DACE:0:0:261
                                    ////__DACE:0:0:261                                ////__DACE:0:0:261
                                    gtir_tmp_6_0 = __tlet_out;                                                        ////__DACE:0:0:261    ////__DACE:0:0:261
                                }                                                     ////__DACE:0:0:261
                                {                                                     ////__DACE:0:0:267
                                    double __tlet_out;                                                                ////__DACE:0:0:267    ////__DACE:0:0:267
                                    ////__DACE:0:0:267                                ////__DACE:0:0:267
                                    ///////////////////                                                               ////__DACE:0:0:267    ////__DACE:0:0:267
                                    // Tasklet code (tlet_2_get_value__clone_1)                                       ////__DACE:0:0:267    ////__DACE:0:0:267
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:267    ////__DACE:0:0:267
                                    ///////////////////                                                               ////__DACE:0:0:267    ////__DACE:0:0:267
                                    ////__DACE:0:0:267                                ////__DACE:0:0:267
                                    gtir_tmp_3_1 = __tlet_out;                                                        ////__DACE:0:0:267    ////__DACE:0:0:267
                                }                                                     ////__DACE:0:0:267
                                {                                                     ////__DACE:0:0:271
                                    double __tlet_out;                                                                ////__DACE:0:0:271    ////__DACE:0:0:271
                                    ////__DACE:0:0:271                                ////__DACE:0:0:271
                                    ///////////////////                                                               ////__DACE:0:0:271    ////__DACE:0:0:271
                                    // Tasklet code (tlet_6_get_value__clone_1)                                       ////__DACE:0:0:271    ////__DACE:0:0:271
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:271    ////__DACE:0:0:271
                                    ///////////////////                                                               ////__DACE:0:0:271    ////__DACE:0:0:271
                                    ////__DACE:0:0:271                                ////__DACE:0:0:271
                                    __p_dthalf_1 = __tlet_out;                                                        ////__DACE:0:0:271    ////__DACE:0:0:271
                                }                                                     ////__DACE:0:0:271
                                {                                                     ////__DACE:0:0:273
                                    double __tlet_out;                                                                ////__DACE:0:0:273    ////__DACE:0:0:273
                                    ////__DACE:0:0:273                                ////__DACE:0:0:273
                                    ///////////////////                                                               ////__DACE:0:0:273    ////__DACE:0:0:273
                                    // Tasklet code (tlet_10_get_value__clone_0)                                      ////__DACE:0:0:273    ////__DACE:0:0:273
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:273    ////__DACE:0:0:273
                                    ///////////////////                                                               ////__DACE:0:0:273    ////__DACE:0:0:273
                                    ////__DACE:0:0:273                                ////__DACE:0:0:273
                                    lambda_4___p_dthalf_0 = __tlet_out;                                               ////__DACE:0:0:273    ////__DACE:0:0:273
                                }                                                     ////__DACE:0:0:273
                                {                                                     ////__DACE:0:0:277
                                    double __tlet_out;                                                                ////__DACE:0:0:277    ////__DACE:0:0:277
                                    ////__DACE:0:0:277                                ////__DACE:0:0:277
                                    ///////////////////                                                               ////__DACE:0:0:277    ////__DACE:0:0:277
                                    // Tasklet code (tlet_34_get_value__clone_0)                                      ////__DACE:0:0:277    ////__DACE:0:0:277
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:277    ////__DACE:0:0:277
                                    ///////////////////                                                               ////__DACE:0:0:277    ////__DACE:0:0:277
                                    ////__DACE:0:0:277                                ////__DACE:0:0:277
                                    gtir_tmp_102_0 = __tlet_out;                                                      ////__DACE:0:0:277    ////__DACE:0:0:277
                                }                                                     ////__DACE:0:0:277
                                {                                                     ////__DACE:0:0:287
                                    double __tlet_out;                                                                ////__DACE:0:0:287    ////__DACE:0:0:287
                                    ////__DACE:0:0:287                                ////__DACE:0:0:287
                                    ///////////////////                                                               ////__DACE:0:0:287    ////__DACE:0:0:287
                                    // Tasklet code (tlet_68_get_value__clone_0)                                      ////__DACE:0:0:287    ////__DACE:0:0:287
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:287    ////__DACE:0:0:287
                                    ///////////////////                                                               ////__DACE:0:0:287    ////__DACE:0:0:287
                                    ////__DACE:0:0:287                                ////__DACE:0:0:287
                                    gtir_tmp_175_0 = __tlet_out;                                                      ////__DACE:0:0:287    ////__DACE:0:0:287
                                }                                                     ////__DACE:0:0:287
                                {                                                     ////__DACE:0:0:330
                                    double _cpy_in = dual_normal_cell_x[i_Edge_gtx_horizontal];                       ////__DACE:0:0:9,330    ////__DACE:0:0:330
                                    double _cpy_out;                                                                  ////__DACE:0:0:330    ////__DACE:0:0:330
                                    ////__DACE:0:0:330                                ////__DACE:0:0:330
                                    ///////////////////                                                               ////__DACE:0:0:330    ////__DACE:0:0:330
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_38_1)                        ////__DACE:0:0:330    ////__DACE:0:0:330
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:330    ////__DACE:0:0:330
                                    ///////////////////                                                               ////__DACE:0:0:330    ////__DACE:0:0:330
                                    ////__DACE:0:0:330                                ////__DACE:0:0:330
                                    gtir_tmp_38_1 = _cpy_out;                                                         ////__DACE:0:0:330    ////__DACE:0:0:330
                                }                                                     ////__DACE:0:0:330
                                {                                                     ////__DACE:0:0:331
                                    double _cpy_in = dual_normal_cell_x[(__dual_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:9,331    ////__DACE:0:0:331
                                    double _cpy_out;                                                                  ////__DACE:0:0:331    ////__DACE:0:0:331
                                    ////__DACE:0:0:331                                ////__DACE:0:0:331
                                    ///////////////////                                                               ////__DACE:0:0:331    ////__DACE:0:0:331
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_48_1)                        ////__DACE:0:0:331    ////__DACE:0:0:331
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:331    ////__DACE:0:0:331
                                    ///////////////////                                                               ////__DACE:0:0:331    ////__DACE:0:0:331
                                    ////__DACE:0:0:331                                ////__DACE:0:0:331
                                    gtir_tmp_48_1 = _cpy_out;                                                         ////__DACE:0:0:331    ////__DACE:0:0:331
                                }                                                     ////__DACE:0:0:331
                                {                                                     ////__DACE:0:0:332
                                    double _cpy_in = dual_normal_cell_y[i_Edge_gtx_horizontal];                       ////__DACE:0:0:7,332    ////__DACE:0:0:332
                                    double _cpy_out;                                                                  ////__DACE:0:0:332    ////__DACE:0:0:332
                                    ////__DACE:0:0:332                                ////__DACE:0:0:332
                                    ///////////////////                                                               ////__DACE:0:0:332    ////__DACE:0:0:332
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_60_1)                        ////__DACE:0:0:332    ////__DACE:0:0:332
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:332    ////__DACE:0:0:332
                                    ///////////////////                                                               ////__DACE:0:0:332    ////__DACE:0:0:332
                                    ////__DACE:0:0:332                                ////__DACE:0:0:332
                                    gtir_tmp_60_1 = _cpy_out;                                                         ////__DACE:0:0:332    ////__DACE:0:0:332
                                }                                                     ////__DACE:0:0:332
                                {                                                     ////__DACE:0:0:333
                                    double _cpy_in = dual_normal_cell_y[(__dual_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:7,333    ////__DACE:0:0:333
                                    double _cpy_out;                                                                  ////__DACE:0:0:333    ////__DACE:0:0:333
                                    ////__DACE:0:0:333                                ////__DACE:0:0:333
                                    ///////////////////                                                               ////__DACE:0:0:333    ////__DACE:0:0:333
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_70_1)                        ////__DACE:0:0:333    ////__DACE:0:0:333
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:333    ////__DACE:0:0:333
                                    ///////////////////                                                               ////__DACE:0:0:333    ////__DACE:0:0:333
                                    ////__DACE:0:0:333                                ////__DACE:0:0:333
                                    gtir_tmp_70_1 = _cpy_out;                                                         ////__DACE:0:0:333    ////__DACE:0:0:333
                                }                                                     ////__DACE:0:0:333
                                {                                                     ////__DACE:0:0:334
                                    double _cpy_in = pos_on_tplane_e_x[i_Edge_gtx_horizontal];                        ////__DACE:0:0:4,334    ////__DACE:0:0:334
                                    double _cpy_out;                                                                  ////__DACE:0:0:334    ////__DACE:0:0:334
                                    ////__DACE:0:0:334                                ////__DACE:0:0:334
                                    ///////////////////                                                               ////__DACE:0:0:334    ////__DACE:0:0:334
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_12_1)                         ////__DACE:0:0:334    ////__DACE:0:0:334
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:334    ////__DACE:0:0:334
                                    ///////////////////                                                               ////__DACE:0:0:334    ////__DACE:0:0:334
                                    ////__DACE:0:0:334                                ////__DACE:0:0:334
                                    gtir_tmp_12_1 = _cpy_out;                                                         ////__DACE:0:0:334    ////__DACE:0:0:334
                                }                                                     ////__DACE:0:0:334
                                {                                                     ////__DACE:0:0:335
                                    double _cpy_in = pos_on_tplane_e_x[(__pos_on_tplane_e_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:4,335    ////__DACE:0:0:335
                                    double _cpy_out;                                                                  ////__DACE:0:0:335    ////__DACE:0:0:335
                                    ////__DACE:0:0:335                                ////__DACE:0:0:335
                                    ///////////////////                                                               ////__DACE:0:0:335    ////__DACE:0:0:335
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_14_1)                         ////__DACE:0:0:335    ////__DACE:0:0:335
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:335    ////__DACE:0:0:335
                                    ///////////////////                                                               ////__DACE:0:0:335    ////__DACE:0:0:335
                                    ////__DACE:0:0:335                                ////__DACE:0:0:335
                                    gtir_tmp_14_1 = _cpy_out;                                                         ////__DACE:0:0:335    ////__DACE:0:0:335
                                }                                                     ////__DACE:0:0:335
                                {                                                     ////__DACE:0:0:336
                                    double _cpy_in = pos_on_tplane_e_y[i_Edge_gtx_horizontal];                        ////__DACE:0:0:5,336    ////__DACE:0:0:336
                                    double _cpy_out;                                                                  ////__DACE:0:0:336    ////__DACE:0:0:336
                                    ////__DACE:0:0:336                                ////__DACE:0:0:336
                                    ///////////////////                                                               ////__DACE:0:0:336    ////__DACE:0:0:336
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_24_1)                         ////__DACE:0:0:336    ////__DACE:0:0:336
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:336    ////__DACE:0:0:336
                                    ///////////////////                                                               ////__DACE:0:0:336    ////__DACE:0:0:336
                                    ////__DACE:0:0:336                                ////__DACE:0:0:336
                                    gtir_tmp_24_1 = _cpy_out;                                                         ////__DACE:0:0:336    ////__DACE:0:0:336
                                }                                                     ////__DACE:0:0:336
                                {                                                     ////__DACE:0:0:337
                                    double _cpy_in = pos_on_tplane_e_y[(__pos_on_tplane_e_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:5,337    ////__DACE:0:0:337
                                    double _cpy_out;                                                                  ////__DACE:0:0:337    ////__DACE:0:0:337
                                    ////__DACE:0:0:337                                ////__DACE:0:0:337
                                    ///////////////////                                                               ////__DACE:0:0:337    ////__DACE:0:0:337
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_26_1)                         ////__DACE:0:0:337    ////__DACE:0:0:337
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:337    ////__DACE:0:0:337
                                    ///////////////////                                                               ////__DACE:0:0:337    ////__DACE:0:0:337
                                    ////__DACE:0:0:337                                ////__DACE:0:0:337
                                    gtir_tmp_26_1 = _cpy_out;                                                         ////__DACE:0:0:337    ////__DACE:0:0:337
                                }                                                     ////__DACE:0:0:337
                                {                                                     ////__DACE:0:0:338
                                    double _cpy_in = primal_normal_cell_x[i_Edge_gtx_horizontal];                     ////__DACE:0:0:10,338    ////__DACE:0:0:338
                                    double _cpy_out;                                                                  ////__DACE:0:0:338    ////__DACE:0:0:338
                                    ////__DACE:0:0:338                                ////__DACE:0:0:338
                                    ///////////////////                                                               ////__DACE:0:0:338    ////__DACE:0:0:338
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_34_1)                      ////__DACE:0:0:338    ////__DACE:0:0:338
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:338    ////__DACE:0:0:338
                                    ///////////////////                                                               ////__DACE:0:0:338    ////__DACE:0:0:338
                                    ////__DACE:0:0:338                                ////__DACE:0:0:338
                                    gtir_tmp_34_1 = _cpy_out;                                                         ////__DACE:0:0:338    ////__DACE:0:0:338
                                }                                                     ////__DACE:0:0:338
                                {                                                     ////__DACE:0:0:339
                                    double _cpy_in = primal_normal_cell_x[(__primal_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:10,339    ////__DACE:0:0:339
                                    double _cpy_out;                                                                  ////__DACE:0:0:339    ////__DACE:0:0:339
                                    ////__DACE:0:0:339                                ////__DACE:0:0:339
                                    ///////////////////                                                               ////__DACE:0:0:339    ////__DACE:0:0:339
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_44_1)                      ////__DACE:0:0:339    ////__DACE:0:0:339
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:339    ////__DACE:0:0:339
                                    ///////////////////                                                               ////__DACE:0:0:339    ////__DACE:0:0:339
                                    ////__DACE:0:0:339                                ////__DACE:0:0:339
                                    gtir_tmp_44_1 = _cpy_out;                                                         ////__DACE:0:0:339    ////__DACE:0:0:339
                                }                                                     ////__DACE:0:0:339
                                {                                                     ////__DACE:0:0:340
                                    double _cpy_in = primal_normal_cell_y[i_Edge_gtx_horizontal];                     ////__DACE:0:0:8,340    ////__DACE:0:0:340
                                    double _cpy_out;                                                                  ////__DACE:0:0:340    ////__DACE:0:0:340
                                    ////__DACE:0:0:340                                ////__DACE:0:0:340
                                    ///////////////////                                                               ////__DACE:0:0:340    ////__DACE:0:0:340
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_56_1)                      ////__DACE:0:0:340    ////__DACE:0:0:340
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:340    ////__DACE:0:0:340
                                    ///////////////////                                                               ////__DACE:0:0:340    ////__DACE:0:0:340
                                    ////__DACE:0:0:340                                ////__DACE:0:0:340
                                    gtir_tmp_56_1 = _cpy_out;                                                         ////__DACE:0:0:340    ////__DACE:0:0:340
                                }                                                     ////__DACE:0:0:340
                                {                                                     ////__DACE:0:0:341
                                    double _cpy_in = primal_normal_cell_y[(__primal_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:8,341    ////__DACE:0:0:341
                                    double _cpy_out;                                                                  ////__DACE:0:0:341    ////__DACE:0:0:341
                                    ////__DACE:0:0:341                                ////__DACE:0:0:341
                                    ///////////////////                                                               ////__DACE:0:0:341    ////__DACE:0:0:341
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_66_1)                      ////__DACE:0:0:341    ////__DACE:0:0:341
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:341    ////__DACE:0:0:341
                                    ///////////////////                                                               ////__DACE:0:0:341    ////__DACE:0:0:341
                                    ////__DACE:0:0:341                                ////__DACE:0:0:341
                                    gtir_tmp_66_1 = _cpy_out;                                                         ////__DACE:0:0:341    ////__DACE:0:0:341
                                }                                                     ////__DACE:0:0:341
                                {                                                     ////__DACE:0:0:253
                                    #pragma unroll 4                                  ////__DACE:0:0:253
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:253
                                        bool gtir_tmp_8_1;                            ////__DACE:0:0:193
                                        double gtir_tmp_16_1;                         ////__DACE:0:0:198
                                        double gtir_tmp_28_1;                         ////__DACE:0:0:205
                                        double gtir_tmp_76_1;                         ////__DACE:0:0:209
                                        double gtir_tmp_54_1;                         ////__DACE:0:0:215
                                        double gtir_tmp_137_1;                        ////__DACE:0:0:221
                                        double gtir_tmp_210_1;                        ////__DACE:0:0:225
                                        bool __map_fusion_gtir_tmp_5_1;               ////__DACE:0:0:228
                                        double __map_fusion_gtir_tmp_21_1_1;          ////__DACE:0:0:229
                                        double __map_fusion_gtir_tmp_19_1;            ////__DACE:0:0:230
                                        double __map_fusion_gtir_tmp_11_1;            ////__DACE:0:0:231
                                        double __map_fusion_gtir_tmp_33_1_1;          ////__DACE:0:0:232
                                        double __map_fusion_gtir_tmp_31_1;            ////__DACE:0:0:233
                                        double __map_fusion_gtir_tmp_23_1;            ////__DACE:0:0:234
                                        bool __map_fusion_gtir_tmp_104_1;             ////__DACE:0:0:235
                                        bool __map_fusion_gtir_tmp_177_1;             ////__DACE:0:0:236
                                        {                                             ////__DACE:0:0:202
                                            double __tlet_arg1 = __p_dthalf_1;                                                ////__DACE:0:0:272,202    ////__DACE:0:0:202
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,202    ////__DACE:0:0:202
                                            double __tlet_result;                                                             ////__DACE:0:0:202    ////__DACE:0:0:202
                                            ////__DACE:0:0:202                        ////__DACE:0:0:202
                                            ///////////////////                                                               ////__DACE:0:0:202    ////__DACE:0:0:202
                                            // Tasklet code (tlet_7_multiplies_1)                                             ////__DACE:0:0:202    ////__DACE:0:0:202
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:202    ////__DACE:0:0:202
                                            ///////////////////                                                               ////__DACE:0:0:202    ////__DACE:0:0:202
                                            ////__DACE:0:0:202                        ////__DACE:0:0:202
                                            __map_fusion_gtir_tmp_11_1 = __tlet_result;                                       ////__DACE:0:0:202    ////__DACE:0:0:202
                                        }                                             ////__DACE:0:0:202
                                        {                                             ////__DACE:0:0:208
                                            double __tlet_arg1 = lambda_4___p_dthalf_0;                                       ////__DACE:0:0:274,208    ////__DACE:0:0:208
                                            double __tlet_arg0 = tangential_wind[((__tangential_wind_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:6,208    ////__DACE:0:0:208
                                            double __tlet_result;                                                             ////__DACE:0:0:208    ////__DACE:0:0:208
                                            ////__DACE:0:0:208                        ////__DACE:0:0:208
                                            ///////////////////                                                               ////__DACE:0:0:208    ////__DACE:0:0:208
                                            // Tasklet code (tlet_11_multiplies_1)                                            ////__DACE:0:0:208    ////__DACE:0:0:208
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:208    ////__DACE:0:0:208
                                            ///////////////////                                                               ////__DACE:0:0:208    ////__DACE:0:0:208
                                            ////__DACE:0:0:208                        ////__DACE:0:0:208
                                            __map_fusion_gtir_tmp_23_1 = __tlet_result;                                       ////__DACE:0:0:208    ////__DACE:0:0:208
                                        }                                             ////__DACE:0:0:208
                                        {                                             ////__DACE:0:0:227
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,227    ////__DACE:0:0:227
                                            double __tlet_arg1 = gtir_tmp_175_0;                                              ////__DACE:0:0:288,227    ////__DACE:0:0:227
                                            bool __tlet_result;                                                               ////__DACE:0:0:227    ////__DACE:0:0:227
                                            ////__DACE:0:0:227                        ////__DACE:0:0:227
                                            ///////////////////                                                               ////__DACE:0:0:227    ////__DACE:0:0:227
                                            // Tasklet code (tlet_69_greater_equal_1)                                         ////__DACE:0:0:227    ////__DACE:0:0:227
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:227    ////__DACE:0:0:227
                                            ///////////////////                                                               ////__DACE:0:0:227    ////__DACE:0:0:227
                                            ////__DACE:0:0:227                        ////__DACE:0:0:227
                                            __map_fusion_gtir_tmp_177_1 = __tlet_result;                                      ////__DACE:0:0:227    ////__DACE:0:0:227
                                        }                                             ////__DACE:0:0:227
                                        {                                             ////__DACE:0:0:195
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,195    ////__DACE:0:0:195
                                            double __tlet_arg1 = gtir_tmp_3_1;                                                ////__DACE:0:0:268,195    ////__DACE:0:0:195
                                            bool __tlet_result;                                                               ////__DACE:0:0:195    ////__DACE:0:0:195
                                            ////__DACE:0:0:195                        ////__DACE:0:0:195
                                            ///////////////////                                                               ////__DACE:0:0:195    ////__DACE:0:0:195
                                            // Tasklet code (tlet_3_greater_equal_1)                                          ////__DACE:0:0:195    ////__DACE:0:0:195
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:195    ////__DACE:0:0:195
                                            ///////////////////                                                               ////__DACE:0:0:195    ////__DACE:0:0:195
                                            ////__DACE:0:0:195                        ////__DACE:0:0:195
                                            __map_fusion_gtir_tmp_5_1 = __tlet_result;                                        ////__DACE:0:0:195    ////__DACE:0:0:195
                                        }                                             ////__DACE:0:0:195
                                        if_stmt_0_0_0_194(gtir_tmp_6_0, gtir_tmp_7_0, __map_fusion_gtir_tmp_5_1, gtir_tmp_8_1);    ////__DACE:0:0:194
                                        if_stmt_1_0_0_199(gtir_tmp_12_1, gtir_tmp_24_1, gtir_tmp_14_1, gtir_tmp_26_1, gtir_tmp_8_1, gtir_tmp_16_1, gtir_tmp_28_1);    ////__DACE:0:0:199
                                        {                                             ////__DACE:0:0:197
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_11_1;                                  ////__DACE:0:0:231,197    ////__DACE:0:0:197
                                            double __tlet_arg1 = gtir_tmp_16_1;                                               ////__DACE:0:0:198,197    ////__DACE:0:0:197
                                            double __tlet_result;                                                             ////__DACE:0:0:197    ////__DACE:0:0:197
                                            ////__DACE:0:0:197                        ////__DACE:0:0:197
                                            ///////////////////                                                               ////__DACE:0:0:197    ////__DACE:0:0:197
                                            // Tasklet code (tlet_8_plus_1)                                                   ////__DACE:0:0:197    ////__DACE:0:0:197
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:197    ////__DACE:0:0:197
                                            ///////////////////                                                               ////__DACE:0:0:197    ////__DACE:0:0:197
                                            ////__DACE:0:0:197                        ////__DACE:0:0:197
                                            __map_fusion_gtir_tmp_19_1 = __tlet_result;                                       ////__DACE:0:0:197    ////__DACE:0:0:197
                                        }                                             ////__DACE:0:0:197
                                        {                                             ////__DACE:0:0:196
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_19_1;                                  ////__DACE:0:0:230,196    ////__DACE:0:0:196
                                            double __tlet_result;                                                             ////__DACE:0:0:196    ////__DACE:0:0:196
                                            ////__DACE:0:0:196                        ////__DACE:0:0:196
                                            ///////////////////                                                               ////__DACE:0:0:196    ////__DACE:0:0:196
                                            // Tasklet code (tlet_9_neg_1)                                                    ////__DACE:0:0:196    ////__DACE:0:0:196
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:196    ////__DACE:0:0:196
                                            ///////////////////                                                               ////__DACE:0:0:196    ////__DACE:0:0:196
                                            ////__DACE:0:0:196                        ////__DACE:0:0:196
                                            __map_fusion_gtir_tmp_21_1_1 = __tlet_result;                                     ////__DACE:0:0:196    ////__DACE:0:0:196
                                        }                                             ////__DACE:0:0:196
                                        {                                             ////__DACE:0:0:204
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_23_1;                                  ////__DACE:0:0:234,204    ////__DACE:0:0:204
                                            double __tlet_arg1 = gtir_tmp_28_1;                                               ////__DACE:0:0:205,204    ////__DACE:0:0:204
                                            double __tlet_result;                                                             ////__DACE:0:0:204    ////__DACE:0:0:204
                                            ////__DACE:0:0:204                        ////__DACE:0:0:204
                                            ///////////////////                                                               ////__DACE:0:0:204    ////__DACE:0:0:204
                                            // Tasklet code (tlet_12_plus_1)                                                  ////__DACE:0:0:204    ////__DACE:0:0:204
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:204    ////__DACE:0:0:204
                                            ///////////////////                                                               ////__DACE:0:0:204    ////__DACE:0:0:204
                                            ////__DACE:0:0:204                        ////__DACE:0:0:204
                                            __map_fusion_gtir_tmp_31_1 = __tlet_result;                                       ////__DACE:0:0:204    ////__DACE:0:0:204
                                        }                                             ////__DACE:0:0:204
                                        {                                             ////__DACE:0:0:203
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_31_1;                                  ////__DACE:0:0:233,203    ////__DACE:0:0:203
                                            double __tlet_result;                                                             ////__DACE:0:0:203    ////__DACE:0:0:203
                                            ////__DACE:0:0:203                        ////__DACE:0:0:203
                                            ///////////////////                                                               ////__DACE:0:0:203    ////__DACE:0:0:203
                                            // Tasklet code (tlet_13_neg_1)                                                   ////__DACE:0:0:203    ////__DACE:0:0:203
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:203    ////__DACE:0:0:203
                                            ///////////////////                                                               ////__DACE:0:0:203    ////__DACE:0:0:203
                                            ////__DACE:0:0:203                        ////__DACE:0:0:203
                                            __map_fusion_gtir_tmp_33_1_1 = __tlet_result;                                     ////__DACE:0:0:203    ////__DACE:0:0:203
                                        }                                             ////__DACE:0:0:203
                                        if_stmt_4_0_0_210(gtir_tmp_8_1, __map_fusion_gtir_tmp_21_1_1, __map_fusion_gtir_tmp_33_1_1, gtir_tmp_34_1, gtir_tmp_38_1, gtir_tmp_44_1, gtir_tmp_48_1, gtir_tmp_56_1, gtir_tmp_60_1, gtir_tmp_66_1, gtir_tmp_70_1, gtir_tmp_76_1, gtir_tmp_54_1);    ////__DACE:0:0:210
                                        if_stmt_7_0_0_226(__map_fusion_gtir_tmp_177_1, &gt_conn_E2C[0], gtir_tmp_54_1, gtir_tmp_76_1, &gtir_tmp_83[0], &gtir_tmp_89[0], &perturbed_rho_at_cells_on_model_levels[0], &reference_rho_at_edges_on_model_levels[0], gtir_tmp_210_1, __gt_conn_E2C_neighbor_stride, __perturbed_rho_at_cells_on_model_levels_K_stride, __reference_rho_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:226
                                        {                                             ////__DACE:0:0:351
                                            double _cpy_in = gtir_tmp_210_1;                                                  ////__DACE:0:0:225,351    ////__DACE:0:0:351
                                            double _cpy_out;                                                                  ////__DACE:0:0:351    ////__DACE:0:0:351
                                            ////__DACE:0:0:351                        ////__DACE:0:0:351
                                            ///////////////////                                                               ////__DACE:0:0:351    ////__DACE:0:0:351
                                            // Tasklet code (copy_gtir_tmp_210_1_to_rho_at_edges_on_model_levels)             ////__DACE:0:0:351    ////__DACE:0:0:351
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:351    ////__DACE:0:0:351
                                            ///////////////////                                                               ////__DACE:0:0:351    ////__DACE:0:0:351
                                            ////__DACE:0:0:351                        ////__DACE:0:0:351
                                            rho_at_edges_on_model_levels[((__rho_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:351    ////__DACE:0:0:351
                                        }                                             ////__DACE:0:0:351
                                        {                                             ////__DACE:0:0:223
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,223    ////__DACE:0:0:223
                                            double __tlet_arg1 = gtir_tmp_102_0;                                              ////__DACE:0:0:278,223    ////__DACE:0:0:223
                                            bool __tlet_result;                                                               ////__DACE:0:0:223    ////__DACE:0:0:223
                                            ////__DACE:0:0:223                        ////__DACE:0:0:223
                                            ///////////////////                                                               ////__DACE:0:0:223    ////__DACE:0:0:223
                                            // Tasklet code (tlet_35_greater_equal_1)                                         ////__DACE:0:0:223    ////__DACE:0:0:223
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:223    ////__DACE:0:0:223
                                            ///////////////////                                                               ////__DACE:0:0:223    ////__DACE:0:0:223
                                            ////__DACE:0:0:223                        ////__DACE:0:0:223
                                            __map_fusion_gtir_tmp_104_1 = __tlet_result;                                      ////__DACE:0:0:223    ////__DACE:0:0:223
                                        }                                             ////__DACE:0:0:223
                                        if_stmt_5_0_0_222(__map_fusion_gtir_tmp_104_1, &gt_conn_E2C[0], &gtir_tmp_101[0], gtir_tmp_54_1, gtir_tmp_76_1, &gtir_tmp_95[0], &perturbed_theta_v_at_cells_on_model_levels[0], &reference_theta_at_edges_on_model_levels[0], gtir_tmp_137_1, __gt_conn_E2C_neighbor_stride, __perturbed_theta_v_at_cells_on_model_levels_K_stride, __reference_theta_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:222
                                        {                                             ////__DACE:0:0:350
                                            double _cpy_in = gtir_tmp_137_1;                                                  ////__DACE:0:0:221,350    ////__DACE:0:0:350
                                            double _cpy_out;                                                                  ////__DACE:0:0:350    ////__DACE:0:0:350
                                            ////__DACE:0:0:350                        ////__DACE:0:0:350
                                            ///////////////////                                                               ////__DACE:0:0:350    ////__DACE:0:0:350
                                            // Tasklet code (copy_gtir_tmp_137_1_to_theta_v_at_edges_on_model_levels)         ////__DACE:0:0:350    ////__DACE:0:0:350
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:350    ////__DACE:0:0:350
                                            ///////////////////                                                               ////__DACE:0:0:350    ////__DACE:0:0:350
                                            ////__DACE:0:0:350                        ////__DACE:0:0:350
                                            theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:350    ////__DACE:0:0:350
                                        }                                             ////__DACE:0:0:350
                                    }                                                 ////__DACE:0:0:254
                                }                                                     ////__DACE:0:0:254
                            }                                                         ////__DACE:0:0:224
                        }                                                             ////__DACE:0:0:224
                    }                                                                 ////__DACE:0:0:224
                }                                                                     ////__DACE:0:0:224
            }                                                                         ////__DACE:0:0:224
        }                                                                             ////__DACE:0:0:363
    }                                                                                 ////__DACE:0:0:363
}                                                                                 ////__DACE:0:0:363

                                                                                  ////__DACE:0:0:362
DACE_EXPORTED void __dace_runkernel_map_100_fieldop_1_0_0_362(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:362
void __dace_runkernel_map_100_fieldop_1_0_0_362(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:362
{                                                                                 ////__DACE:0:0:362
                                                                                  ////__DACE:0:0:362
    void  *map_100_fieldop_1_0_0_362_args[] = { (void *)&current_vn, (void *)&dual_normal_cell_x, (void *)&dual_normal_cell_y, (void *)&gt_conn_E2C, (void *)&gtir_tmp_101, (void *)&gtir_tmp_83, (void *)&gtir_tmp_89, (void *)&gtir_tmp_95, (void *)&perturbed_rho_at_cells_on_model_levels, (void *)&perturbed_theta_v_at_cells_on_model_levels, (void *)&pos_on_tplane_e_x, (void *)&pos_on_tplane_e_y, (void *)&primal_normal_cell_x, (void *)&primal_normal_cell_y, (void *)&reference_rho_at_edges_on_model_levels, (void *)&reference_theta_at_edges_on_model_levels, (void *)&rho_at_edges_on_model_levels, (void *)&tangential_wind, (void *)&theta_v_at_edges_on_model_levels, (void *)&__current_vn_K_stride, (void *)&__dual_normal_cell_x_E2C_stride, (void *)&__dual_normal_cell_y_E2C_stride, (void *)&__gt_conn_E2C_neighbor_stride, (void *)&__perturbed_rho_at_cells_on_model_levels_K_stride, (void *)&__perturbed_theta_v_at_cells_on_model_levels_K_stride, (void *)&__pos_on_tplane_e_x_E2C_stride, (void *)&__pos_on_tplane_e_y_E2C_stride, (void *)&__primal_normal_cell_x_E2C_stride, (void *)&__primal_normal_cell_y_E2C_stride, (void *)&__reference_rho_at_edges_on_model_levels_K_stride, (void *)&__reference_theta_at_edges_on_model_levels_K_stride, (void *)&__rho_at_edges_on_model_levels_K_stride, (void *)&__tangential_wind_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:362
    gpuError_t __err = hipLaunchKernel((void*)map_100_fieldop_1_0_0_362, dim3(233, 30, 1), dim3(256, 1, 1), map_100_fieldop_1_0_0_362_args, 0, nullptr);    ////__DACE:0:0:362
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_100_fieldop_1_0_0_362", 233, 30, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_0_fieldop_0_0_352(const double * __restrict__ current_vn, const double * __restrict__ grf_tend_vn, double * __restrict__ next_vn, double * __restrict__ rho_at_edges_on_model_levels, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __grf_tend_vn_K_stride, int __next_vn_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:352
    {                                                                                 ////__DACE:0:0:352
        {                                                                             ////__DACE:0:0:352
            int b_i_Edge_gtx_horizontal = (256 * blockIdx.x);                         ////__DACE:0:0:352
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:352
            {                                                                         ////__DACE:0:0:2
                {                                                                     ////__DACE:0:0:2
                    {                                                                 ////__DACE:0:0:2
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:2
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:2
                        double gtir_tmp_0_0;                                          ////__DACE:0:0:256
                        double gtir_tmp_212_0;                                        ////__DACE:0:0:286
                        double __dtime_1;                                             ////__DACE:0:0:302
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(5386, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:2
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(29, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:2
                                {                                                     ////__DACE:0:0:255
                                    double __tlet_out;                                                                ////__DACE:0:0:255    ////__DACE:0:0:255
                                    ////__DACE:0:0:255                                ////__DACE:0:0:255
                                    ///////////////////                                                               ////__DACE:0:0:255    ////__DACE:0:0:255
                                    // Tasklet code (tlet_0_get_value__clone_0)                                       ////__DACE:0:0:255    ////__DACE:0:0:255
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:255    ////__DACE:0:0:255
                                    ///////////////////                                                               ////__DACE:0:0:255    ////__DACE:0:0:255
                                    ////__DACE:0:0:255                                ////__DACE:0:0:255
                                    gtir_tmp_0_0 = __tlet_out;                                                        ////__DACE:0:0:255    ////__DACE:0:0:255
                                }                                                     ////__DACE:0:0:255
                                {                                                     ////__DACE:0:0:285
                                    double __tlet_out;                                                                ////__DACE:0:0:285    ////__DACE:0:0:285
                                    ////__DACE:0:0:285                                ////__DACE:0:0:285
                                    ///////////////////                                                               ////__DACE:0:0:285    ////__DACE:0:0:285
                                    // Tasklet code (tlet_86_get_value__clone_0)                                      ////__DACE:0:0:285    ////__DACE:0:0:285
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:285    ////__DACE:0:0:285
                                    ///////////////////                                                               ////__DACE:0:0:285    ////__DACE:0:0:285
                                    ////__DACE:0:0:285                                ////__DACE:0:0:285
                                    gtir_tmp_212_0 = __tlet_out;                                                      ////__DACE:0:0:285    ////__DACE:0:0:285
                                }                                                     ////__DACE:0:0:285
                                {                                                     ////__DACE:0:0:301
                                    double __tlet_out;                                                                ////__DACE:0:0:301    ////__DACE:0:0:301
                                    ////__DACE:0:0:301                                ////__DACE:0:0:301
                                    ///////////////////                                                               ////__DACE:0:0:301    ////__DACE:0:0:301
                                    // Tasklet code (tlet_88_get_value__clone_1)                                      ////__DACE:0:0:301    ////__DACE:0:0:301
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:301    ////__DACE:0:0:301
                                    ///////////////////                                                               ////__DACE:0:0:301    ////__DACE:0:0:301
                                    ////__DACE:0:0:301                                ////__DACE:0:0:301
                                    __dtime_1 = __tlet_out;                                                           ////__DACE:0:0:301    ////__DACE:0:0:301
                                }                                                     ////__DACE:0:0:301
                                {                                                     ////__DACE:0:0:243
                                    #pragma unroll 4                                  ////__DACE:0:0:243
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:243
                                        double __map_fusion_gtir_tmp_220_0;           ////__DACE:0:0:239
                                        {                                             ////__DACE:0:0:238
                                            double __tlet_arg0 = __dtime_1;                                                   ////__DACE:0:0:302,238    ////__DACE:0:0:238
                                            double __tlet_arg1 = grf_tend_vn[((__grf_tend_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:68,238    ////__DACE:0:0:238
                                            double __tlet_result;                                                             ////__DACE:0:0:238    ////__DACE:0:0:238
                                            ////__DACE:0:0:238                        ////__DACE:0:0:238
                                            ///////////////////                                                               ////__DACE:0:0:238    ////__DACE:0:0:238
                                            // Tasklet code (tlet_89_multiplies_0)                                            ////__DACE:0:0:238    ////__DACE:0:0:238
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:238    ////__DACE:0:0:238
                                            ///////////////////                                                               ////__DACE:0:0:238    ////__DACE:0:0:238
                                            ////__DACE:0:0:238                        ////__DACE:0:0:238
                                            __map_fusion_gtir_tmp_220_0 = __tlet_result;                                      ////__DACE:0:0:238    ////__DACE:0:0:238
                                        }                                             ////__DACE:0:0:238
                                        {                                             ////__DACE:0:0:237
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_220_0;                                 ////__DACE:0:0:239,237    ////__DACE:0:0:237
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,237    ////__DACE:0:0:237
                                            double __tlet_result;                                                             ////__DACE:0:0:237    ////__DACE:0:0:237
                                            ////__DACE:0:0:237                        ////__DACE:0:0:237
                                            ///////////////////                                                               ////__DACE:0:0:237    ////__DACE:0:0:237
                                            // Tasklet code (tlet_90_plus_0)                                                  ////__DACE:0:0:237    ////__DACE:0:0:237
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:237    ////__DACE:0:0:237
                                            ///////////////////                                                               ////__DACE:0:0:237    ////__DACE:0:0:237
                                            ////__DACE:0:0:237                        ////__DACE:0:0:237
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:237    ////__DACE:0:0:237
                                        }                                             ////__DACE:0:0:237
                                        {                                             ////__DACE:0:0:1
                                            double __tlet_inp = gtir_tmp_0_0;                                                 ////__DACE:0:0:256,1    ////__DACE:0:0:1
                                            double __tlet_out;                                                                ////__DACE:0:0:1    ////__DACE:0:0:1
                                            ////__DACE:0:0:1                          ////__DACE:0:0:1
                                            ///////////////////                                                               ////__DACE:0:0:1    ////__DACE:0:0:1
                                            // Tasklet code (tlet_1_copy)                                                     ////__DACE:0:0:1    ////__DACE:0:0:1
                                            __tlet_out = __tlet_inp;                                                          ////__DACE:0:0:1    ////__DACE:0:0:1
                                            ///////////////////                                                               ////__DACE:0:0:1    ////__DACE:0:0:1
                                            ////__DACE:0:0:1                          ////__DACE:0:0:1
                                            theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_out;    ////__DACE:0:0:1    ////__DACE:0:0:1
                                        }                                             ////__DACE:0:0:1
                                        {                                             ////__DACE:0:0:64
                                            double __tlet_inp = gtir_tmp_212_0;                                               ////__DACE:0:0:286,64    ////__DACE:0:0:64
                                            double __tlet_out;                                                                ////__DACE:0:0:64    ////__DACE:0:0:64
                                            ////__DACE:0:0:64                         ////__DACE:0:0:64
                                            ///////////////////                                                               ////__DACE:0:0:64    ////__DACE:0:0:64
                                            // Tasklet code (tlet_87_copy)                                                    ////__DACE:0:0:64    ////__DACE:0:0:64
                                            __tlet_out = __tlet_inp;                                                          ////__DACE:0:0:64    ////__DACE:0:0:64
                                            ///////////////////                                                               ////__DACE:0:0:64    ////__DACE:0:0:64
                                            ////__DACE:0:0:64                         ////__DACE:0:0:64
                                            rho_at_edges_on_model_levels[((__rho_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_out;    ////__DACE:0:0:64    ////__DACE:0:0:64
                                        }                                             ////__DACE:0:0:64
                                    }                                                 ////__DACE:0:0:244
                                }                                                     ////__DACE:0:0:244
                            }                                                         ////__DACE:0:0:0
                        }                                                             ////__DACE:0:0:0
                    }                                                                 ////__DACE:0:0:0
                }                                                                     ////__DACE:0:0:0
            }                                                                         ////__DACE:0:0:0
        }                                                                             ////__DACE:0:0:353
    }                                                                                 ////__DACE:0:0:353
}                                                                                 ////__DACE:0:0:353

                                                                                  ////__DACE:0:0:352
DACE_EXPORTED void __dace_runkernel_map_0_fieldop_0_0_352(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ grf_tend_vn, double * __restrict__ next_vn, double * __restrict__ rho_at_edges_on_model_levels, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __grf_tend_vn_K_stride, int __next_vn_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:352
void __dace_runkernel_map_0_fieldop_0_0_352(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ grf_tend_vn, double * __restrict__ next_vn, double * __restrict__ rho_at_edges_on_model_levels, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __grf_tend_vn_K_stride, int __next_vn_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:352
{                                                                                 ////__DACE:0:0:352
                                                                                  ////__DACE:0:0:352
    void  *map_0_fieldop_0_0_352_args[] = { (void *)&current_vn, (void *)&grf_tend_vn, (void *)&next_vn, (void *)&rho_at_edges_on_model_levels, (void *)&theta_v_at_edges_on_model_levels, (void *)&__current_vn_K_stride, (void *)&__grf_tend_vn_K_stride, (void *)&__next_vn_K_stride, (void *)&__rho_at_edges_on_model_levels_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:352
    gpuError_t __err = hipLaunchKernel((void*)map_0_fieldop_0_0_352, dim3(22, 30, 1), dim3(256, 1, 1), map_0_fieldop_0_0_352_args, 0, nullptr);    ////__DACE:0:0:352
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_0_fieldop_0_0_352", 22, 30, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_100_fieldop_0_0_0_360(const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const double * __restrict__ grf_tend_vn, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ next_vn, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __grf_tend_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __next_vn_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:360
    {                                                                                 ////__DACE:0:0:360
        {                                                                             ////__DACE:0:0:360
            int b_i_Edge_gtx_horizontal = ((256 * blockIdx.x) + 5387);                ////__DACE:0:0:360
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:360
            {                                                                         ////__DACE:0:0:176
                {                                                                     ////__DACE:0:0:176
                    {                                                                 ////__DACE:0:0:176
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:176
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:176
                        double gtir_tmp_14_0;                                         ////__DACE:0:0:156
                        double gtir_tmp_12_0;                                         ////__DACE:0:0:157
                        double gtir_tmp_26_0;                                         ////__DACE:0:0:162
                        double gtir_tmp_24_0;                                         ////__DACE:0:0:163
                        double gtir_tmp_70_0;                                         ////__DACE:0:0:167
                        double gtir_tmp_66_0;                                         ////__DACE:0:0:168
                        double gtir_tmp_60_0;                                         ////__DACE:0:0:169
                        double gtir_tmp_56_0;                                         ////__DACE:0:0:170
                        double gtir_tmp_48_0;                                         ////__DACE:0:0:172
                        double gtir_tmp_44_0;                                         ////__DACE:0:0:173
                        double gtir_tmp_38_0;                                         ////__DACE:0:0:174
                        double gtir_tmp_34_0;                                         ////__DACE:0:0:175
                        bool gtir_tmp_7_1;                                            ////__DACE:0:0:260
                        bool gtir_tmp_6_1;                                            ////__DACE:0:0:264
                        double gtir_tmp_3_0;                                          ////__DACE:0:0:266
                        double __p_dthalf_0;                                          ////__DACE:0:0:270
                        double lambda_4___p_dthalf_1;                                 ////__DACE:0:0:276
                        double gtir_tmp_102_1;                                        ////__DACE:0:0:280
                        double gtir_tmp_175_1;                                        ////__DACE:0:0:290
                        double __dtime_0;                                             ////__DACE:0:0:300
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(7700, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:176
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(29, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:176
                                {                                                     ////__DACE:0:0:259
                                    bool __tlet_out;                                                                  ////__DACE:0:0:259    ////__DACE:0:0:259
                                    ////__DACE:0:0:259                                ////__DACE:0:0:259
                                    ///////////////////                                                               ////__DACE:0:0:259    ////__DACE:0:0:259
                                    // Tasklet code (tlet_5_get_value__clone_1)                                       ////__DACE:0:0:259    ////__DACE:0:0:259
                                    __tlet_out = false;                                                               ////__DACE:0:0:259    ////__DACE:0:0:259
                                    ///////////////////                                                               ////__DACE:0:0:259    ////__DACE:0:0:259
                                    ////__DACE:0:0:259                                ////__DACE:0:0:259
                                    gtir_tmp_7_1 = __tlet_out;                                                        ////__DACE:0:0:259    ////__DACE:0:0:259
                                }                                                     ////__DACE:0:0:259
                                {                                                     ////__DACE:0:0:263
                                    bool __tlet_out;                                                                  ////__DACE:0:0:263    ////__DACE:0:0:263
                                    ////__DACE:0:0:263                                ////__DACE:0:0:263
                                    ///////////////////                                                               ////__DACE:0:0:263    ////__DACE:0:0:263
                                    // Tasklet code (tlet_4_get_value__clone_1)                                       ////__DACE:0:0:263    ////__DACE:0:0:263
                                    __tlet_out = true;                                                                ////__DACE:0:0:263    ////__DACE:0:0:263
                                    ///////////////////                                                               ////__DACE:0:0:263    ////__DACE:0:0:263
                                    ////__DACE:0:0:263                                ////__DACE:0:0:263
                                    gtir_tmp_6_1 = __tlet_out;                                                        ////__DACE:0:0:263    ////__DACE:0:0:263
                                }                                                     ////__DACE:0:0:263
                                {                                                     ////__DACE:0:0:265
                                    double __tlet_out;                                                                ////__DACE:0:0:265    ////__DACE:0:0:265
                                    ////__DACE:0:0:265                                ////__DACE:0:0:265
                                    ///////////////////                                                               ////__DACE:0:0:265    ////__DACE:0:0:265
                                    // Tasklet code (tlet_2_get_value__clone_0)                                       ////__DACE:0:0:265    ////__DACE:0:0:265
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:265    ////__DACE:0:0:265
                                    ///////////////////                                                               ////__DACE:0:0:265    ////__DACE:0:0:265
                                    ////__DACE:0:0:265                                ////__DACE:0:0:265
                                    gtir_tmp_3_0 = __tlet_out;                                                        ////__DACE:0:0:265    ////__DACE:0:0:265
                                }                                                     ////__DACE:0:0:265
                                {                                                     ////__DACE:0:0:269
                                    double __tlet_out;                                                                ////__DACE:0:0:269    ////__DACE:0:0:269
                                    ////__DACE:0:0:269                                ////__DACE:0:0:269
                                    ///////////////////                                                               ////__DACE:0:0:269    ////__DACE:0:0:269
                                    // Tasklet code (tlet_6_get_value__clone_0)                                       ////__DACE:0:0:269    ////__DACE:0:0:269
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:269    ////__DACE:0:0:269
                                    ///////////////////                                                               ////__DACE:0:0:269    ////__DACE:0:0:269
                                    ////__DACE:0:0:269                                ////__DACE:0:0:269
                                    __p_dthalf_0 = __tlet_out;                                                        ////__DACE:0:0:269    ////__DACE:0:0:269
                                }                                                     ////__DACE:0:0:269
                                {                                                     ////__DACE:0:0:275
                                    double __tlet_out;                                                                ////__DACE:0:0:275    ////__DACE:0:0:275
                                    ////__DACE:0:0:275                                ////__DACE:0:0:275
                                    ///////////////////                                                               ////__DACE:0:0:275    ////__DACE:0:0:275
                                    // Tasklet code (tlet_10_get_value__clone_1)                                      ////__DACE:0:0:275    ////__DACE:0:0:275
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:275    ////__DACE:0:0:275
                                    ///////////////////                                                               ////__DACE:0:0:275    ////__DACE:0:0:275
                                    ////__DACE:0:0:275                                ////__DACE:0:0:275
                                    lambda_4___p_dthalf_1 = __tlet_out;                                               ////__DACE:0:0:275    ////__DACE:0:0:275
                                }                                                     ////__DACE:0:0:275
                                {                                                     ////__DACE:0:0:279
                                    double __tlet_out;                                                                ////__DACE:0:0:279    ////__DACE:0:0:279
                                    ////__DACE:0:0:279                                ////__DACE:0:0:279
                                    ///////////////////                                                               ////__DACE:0:0:279    ////__DACE:0:0:279
                                    // Tasklet code (tlet_34_get_value__clone_1)                                      ////__DACE:0:0:279    ////__DACE:0:0:279
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:279    ////__DACE:0:0:279
                                    ///////////////////                                                               ////__DACE:0:0:279    ////__DACE:0:0:279
                                    ////__DACE:0:0:279                                ////__DACE:0:0:279
                                    gtir_tmp_102_1 = __tlet_out;                                                      ////__DACE:0:0:279    ////__DACE:0:0:279
                                }                                                     ////__DACE:0:0:279
                                {                                                     ////__DACE:0:0:289
                                    double __tlet_out;                                                                ////__DACE:0:0:289    ////__DACE:0:0:289
                                    ////__DACE:0:0:289                                ////__DACE:0:0:289
                                    ///////////////////                                                               ////__DACE:0:0:289    ////__DACE:0:0:289
                                    // Tasklet code (tlet_68_get_value__clone_1)                                      ////__DACE:0:0:289    ////__DACE:0:0:289
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:289    ////__DACE:0:0:289
                                    ///////////////////                                                               ////__DACE:0:0:289    ////__DACE:0:0:289
                                    ////__DACE:0:0:289                                ////__DACE:0:0:289
                                    gtir_tmp_175_1 = __tlet_out;                                                      ////__DACE:0:0:289    ////__DACE:0:0:289
                                }                                                     ////__DACE:0:0:289
                                {                                                     ////__DACE:0:0:299
                                    double __tlet_out;                                                                ////__DACE:0:0:299    ////__DACE:0:0:299
                                    ////__DACE:0:0:299                                ////__DACE:0:0:299
                                    ///////////////////                                                               ////__DACE:0:0:299    ////__DACE:0:0:299
                                    // Tasklet code (tlet_88_get_value__clone_0)                                      ////__DACE:0:0:299    ////__DACE:0:0:299
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:299    ////__DACE:0:0:299
                                    ///////////////////                                                               ////__DACE:0:0:299    ////__DACE:0:0:299
                                    ////__DACE:0:0:299                                ////__DACE:0:0:299
                                    __dtime_0 = __tlet_out;                                                           ////__DACE:0:0:299    ////__DACE:0:0:299
                                }                                                     ////__DACE:0:0:299
                                {                                                     ////__DACE:0:0:318
                                    double _cpy_in = dual_normal_cell_x[i_Edge_gtx_horizontal];                       ////__DACE:0:0:9,318    ////__DACE:0:0:318
                                    double _cpy_out;                                                                  ////__DACE:0:0:318    ////__DACE:0:0:318
                                    ////__DACE:0:0:318                                ////__DACE:0:0:318
                                    ///////////////////                                                               ////__DACE:0:0:318    ////__DACE:0:0:318
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_38_0)                        ////__DACE:0:0:318    ////__DACE:0:0:318
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:318    ////__DACE:0:0:318
                                    ///////////////////                                                               ////__DACE:0:0:318    ////__DACE:0:0:318
                                    ////__DACE:0:0:318                                ////__DACE:0:0:318
                                    gtir_tmp_38_0 = _cpy_out;                                                         ////__DACE:0:0:318    ////__DACE:0:0:318
                                }                                                     ////__DACE:0:0:318
                                {                                                     ////__DACE:0:0:319
                                    double _cpy_in = dual_normal_cell_x[(__dual_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:9,319    ////__DACE:0:0:319
                                    double _cpy_out;                                                                  ////__DACE:0:0:319    ////__DACE:0:0:319
                                    ////__DACE:0:0:319                                ////__DACE:0:0:319
                                    ///////////////////                                                               ////__DACE:0:0:319    ////__DACE:0:0:319
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_48_0)                        ////__DACE:0:0:319    ////__DACE:0:0:319
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:319    ////__DACE:0:0:319
                                    ///////////////////                                                               ////__DACE:0:0:319    ////__DACE:0:0:319
                                    ////__DACE:0:0:319                                ////__DACE:0:0:319
                                    gtir_tmp_48_0 = _cpy_out;                                                         ////__DACE:0:0:319    ////__DACE:0:0:319
                                }                                                     ////__DACE:0:0:319
                                {                                                     ////__DACE:0:0:320
                                    double _cpy_in = dual_normal_cell_y[i_Edge_gtx_horizontal];                       ////__DACE:0:0:7,320    ////__DACE:0:0:320
                                    double _cpy_out;                                                                  ////__DACE:0:0:320    ////__DACE:0:0:320
                                    ////__DACE:0:0:320                                ////__DACE:0:0:320
                                    ///////////////////                                                               ////__DACE:0:0:320    ////__DACE:0:0:320
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_60_0)                        ////__DACE:0:0:320    ////__DACE:0:0:320
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:320    ////__DACE:0:0:320
                                    ///////////////////                                                               ////__DACE:0:0:320    ////__DACE:0:0:320
                                    ////__DACE:0:0:320                                ////__DACE:0:0:320
                                    gtir_tmp_60_0 = _cpy_out;                                                         ////__DACE:0:0:320    ////__DACE:0:0:320
                                }                                                     ////__DACE:0:0:320
                                {                                                     ////__DACE:0:0:321
                                    double _cpy_in = dual_normal_cell_y[(__dual_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:7,321    ////__DACE:0:0:321
                                    double _cpy_out;                                                                  ////__DACE:0:0:321    ////__DACE:0:0:321
                                    ////__DACE:0:0:321                                ////__DACE:0:0:321
                                    ///////////////////                                                               ////__DACE:0:0:321    ////__DACE:0:0:321
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_70_0)                        ////__DACE:0:0:321    ////__DACE:0:0:321
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:321    ////__DACE:0:0:321
                                    ///////////////////                                                               ////__DACE:0:0:321    ////__DACE:0:0:321
                                    ////__DACE:0:0:321                                ////__DACE:0:0:321
                                    gtir_tmp_70_0 = _cpy_out;                                                         ////__DACE:0:0:321    ////__DACE:0:0:321
                                }                                                     ////__DACE:0:0:321
                                {                                                     ////__DACE:0:0:322
                                    double _cpy_in = pos_on_tplane_e_x[i_Edge_gtx_horizontal];                        ////__DACE:0:0:4,322    ////__DACE:0:0:322
                                    double _cpy_out;                                                                  ////__DACE:0:0:322    ////__DACE:0:0:322
                                    ////__DACE:0:0:322                                ////__DACE:0:0:322
                                    ///////////////////                                                               ////__DACE:0:0:322    ////__DACE:0:0:322
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_12_0)                         ////__DACE:0:0:322    ////__DACE:0:0:322
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:322    ////__DACE:0:0:322
                                    ///////////////////                                                               ////__DACE:0:0:322    ////__DACE:0:0:322
                                    ////__DACE:0:0:322                                ////__DACE:0:0:322
                                    gtir_tmp_12_0 = _cpy_out;                                                         ////__DACE:0:0:322    ////__DACE:0:0:322
                                }                                                     ////__DACE:0:0:322
                                {                                                     ////__DACE:0:0:323
                                    double _cpy_in = pos_on_tplane_e_x[(__pos_on_tplane_e_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:4,323    ////__DACE:0:0:323
                                    double _cpy_out;                                                                  ////__DACE:0:0:323    ////__DACE:0:0:323
                                    ////__DACE:0:0:323                                ////__DACE:0:0:323
                                    ///////////////////                                                               ////__DACE:0:0:323    ////__DACE:0:0:323
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_14_0)                         ////__DACE:0:0:323    ////__DACE:0:0:323
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:323    ////__DACE:0:0:323
                                    ///////////////////                                                               ////__DACE:0:0:323    ////__DACE:0:0:323
                                    ////__DACE:0:0:323                                ////__DACE:0:0:323
                                    gtir_tmp_14_0 = _cpy_out;                                                         ////__DACE:0:0:323    ////__DACE:0:0:323
                                }                                                     ////__DACE:0:0:323
                                {                                                     ////__DACE:0:0:324
                                    double _cpy_in = pos_on_tplane_e_y[i_Edge_gtx_horizontal];                        ////__DACE:0:0:5,324    ////__DACE:0:0:324
                                    double _cpy_out;                                                                  ////__DACE:0:0:324    ////__DACE:0:0:324
                                    ////__DACE:0:0:324                                ////__DACE:0:0:324
                                    ///////////////////                                                               ////__DACE:0:0:324    ////__DACE:0:0:324
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_24_0)                         ////__DACE:0:0:324    ////__DACE:0:0:324
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:324    ////__DACE:0:0:324
                                    ///////////////////                                                               ////__DACE:0:0:324    ////__DACE:0:0:324
                                    ////__DACE:0:0:324                                ////__DACE:0:0:324
                                    gtir_tmp_24_0 = _cpy_out;                                                         ////__DACE:0:0:324    ////__DACE:0:0:324
                                }                                                     ////__DACE:0:0:324
                                {                                                     ////__DACE:0:0:325
                                    double _cpy_in = pos_on_tplane_e_y[(__pos_on_tplane_e_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:5,325    ////__DACE:0:0:325
                                    double _cpy_out;                                                                  ////__DACE:0:0:325    ////__DACE:0:0:325
                                    ////__DACE:0:0:325                                ////__DACE:0:0:325
                                    ///////////////////                                                               ////__DACE:0:0:325    ////__DACE:0:0:325
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_26_0)                         ////__DACE:0:0:325    ////__DACE:0:0:325
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:325    ////__DACE:0:0:325
                                    ///////////////////                                                               ////__DACE:0:0:325    ////__DACE:0:0:325
                                    ////__DACE:0:0:325                                ////__DACE:0:0:325
                                    gtir_tmp_26_0 = _cpy_out;                                                         ////__DACE:0:0:325    ////__DACE:0:0:325
                                }                                                     ////__DACE:0:0:325
                                {                                                     ////__DACE:0:0:326
                                    double _cpy_in = primal_normal_cell_x[i_Edge_gtx_horizontal];                     ////__DACE:0:0:10,326    ////__DACE:0:0:326
                                    double _cpy_out;                                                                  ////__DACE:0:0:326    ////__DACE:0:0:326
                                    ////__DACE:0:0:326                                ////__DACE:0:0:326
                                    ///////////////////                                                               ////__DACE:0:0:326    ////__DACE:0:0:326
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_34_0)                      ////__DACE:0:0:326    ////__DACE:0:0:326
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:326    ////__DACE:0:0:326
                                    ///////////////////                                                               ////__DACE:0:0:326    ////__DACE:0:0:326
                                    ////__DACE:0:0:326                                ////__DACE:0:0:326
                                    gtir_tmp_34_0 = _cpy_out;                                                         ////__DACE:0:0:326    ////__DACE:0:0:326
                                }                                                     ////__DACE:0:0:326
                                {                                                     ////__DACE:0:0:327
                                    double _cpy_in = primal_normal_cell_x[(__primal_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:10,327    ////__DACE:0:0:327
                                    double _cpy_out;                                                                  ////__DACE:0:0:327    ////__DACE:0:0:327
                                    ////__DACE:0:0:327                                ////__DACE:0:0:327
                                    ///////////////////                                                               ////__DACE:0:0:327    ////__DACE:0:0:327
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_44_0)                      ////__DACE:0:0:327    ////__DACE:0:0:327
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:327    ////__DACE:0:0:327
                                    ///////////////////                                                               ////__DACE:0:0:327    ////__DACE:0:0:327
                                    ////__DACE:0:0:327                                ////__DACE:0:0:327
                                    gtir_tmp_44_0 = _cpy_out;                                                         ////__DACE:0:0:327    ////__DACE:0:0:327
                                }                                                     ////__DACE:0:0:327
                                {                                                     ////__DACE:0:0:328
                                    double _cpy_in = primal_normal_cell_y[i_Edge_gtx_horizontal];                     ////__DACE:0:0:8,328    ////__DACE:0:0:328
                                    double _cpy_out;                                                                  ////__DACE:0:0:328    ////__DACE:0:0:328
                                    ////__DACE:0:0:328                                ////__DACE:0:0:328
                                    ///////////////////                                                               ////__DACE:0:0:328    ////__DACE:0:0:328
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_56_0)                      ////__DACE:0:0:328    ////__DACE:0:0:328
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:328    ////__DACE:0:0:328
                                    ///////////////////                                                               ////__DACE:0:0:328    ////__DACE:0:0:328
                                    ////__DACE:0:0:328                                ////__DACE:0:0:328
                                    gtir_tmp_56_0 = _cpy_out;                                                         ////__DACE:0:0:328    ////__DACE:0:0:328
                                }                                                     ////__DACE:0:0:328
                                {                                                     ////__DACE:0:0:329
                                    double _cpy_in = primal_normal_cell_y[(__primal_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:8,329    ////__DACE:0:0:329
                                    double _cpy_out;                                                                  ////__DACE:0:0:329    ////__DACE:0:0:329
                                    ////__DACE:0:0:329                                ////__DACE:0:0:329
                                    ///////////////////                                                               ////__DACE:0:0:329    ////__DACE:0:0:329
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_66_0)                      ////__DACE:0:0:329    ////__DACE:0:0:329
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:329    ////__DACE:0:0:329
                                    ///////////////////                                                               ////__DACE:0:0:329    ////__DACE:0:0:329
                                    ////__DACE:0:0:329                                ////__DACE:0:0:329
                                    gtir_tmp_66_0 = _cpy_out;                                                         ////__DACE:0:0:329    ////__DACE:0:0:329
                                }                                                     ////__DACE:0:0:329
                                {                                                     ////__DACE:0:0:251
                                    #pragma unroll 4                                  ////__DACE:0:0:251
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:251
                                        bool gtir_tmp_8_0;                            ////__DACE:0:0:149
                                        double gtir_tmp_16_0;                         ////__DACE:0:0:154
                                        double gtir_tmp_28_0;                         ////__DACE:0:0:161
                                        double gtir_tmp_76_0;                         ////__DACE:0:0:165
                                        double gtir_tmp_54_0;                         ////__DACE:0:0:171
                                        double gtir_tmp_137_0;                        ////__DACE:0:0:177
                                        double gtir_tmp_210_0;                        ////__DACE:0:0:181
                                        bool __map_fusion_gtir_tmp_5_0;               ////__DACE:0:0:184
                                        double __map_fusion_gtir_tmp_21_0_0;          ////__DACE:0:0:185
                                        double __map_fusion_gtir_tmp_19_0;            ////__DACE:0:0:186
                                        double __map_fusion_gtir_tmp_11_0;            ////__DACE:0:0:187
                                        double __map_fusion_gtir_tmp_33_0_0;          ////__DACE:0:0:188
                                        double __map_fusion_gtir_tmp_31_0;            ////__DACE:0:0:189
                                        double __map_fusion_gtir_tmp_23_0;            ////__DACE:0:0:190
                                        bool __map_fusion_gtir_tmp_104_0;             ////__DACE:0:0:191
                                        bool __map_fusion_gtir_tmp_177_0;             ////__DACE:0:0:192
                                        double __map_fusion_gtir_tmp_220_1;           ////__DACE:0:0:242
                                        {                                             ////__DACE:0:0:158
                                            double __tlet_arg1 = __p_dthalf_0;                                                ////__DACE:0:0:270,158    ////__DACE:0:0:158
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,158    ////__DACE:0:0:158
                                            double __tlet_result;                                                             ////__DACE:0:0:158    ////__DACE:0:0:158
                                            ////__DACE:0:0:158                        ////__DACE:0:0:158
                                            ///////////////////                                                               ////__DACE:0:0:158    ////__DACE:0:0:158
                                            // Tasklet code (tlet_7_multiplies_0)                                             ////__DACE:0:0:158    ////__DACE:0:0:158
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:158    ////__DACE:0:0:158
                                            ///////////////////                                                               ////__DACE:0:0:158    ////__DACE:0:0:158
                                            ////__DACE:0:0:158                        ////__DACE:0:0:158
                                            __map_fusion_gtir_tmp_11_0 = __tlet_result;                                       ////__DACE:0:0:158    ////__DACE:0:0:158
                                        }                                             ////__DACE:0:0:158
                                        {                                             ////__DACE:0:0:241
                                            double __tlet_arg0 = __dtime_0;                                                   ////__DACE:0:0:300,241    ////__DACE:0:0:241
                                            double __tlet_arg1 = grf_tend_vn[((__grf_tend_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:68,241    ////__DACE:0:0:241
                                            double __tlet_result;                                                             ////__DACE:0:0:241    ////__DACE:0:0:241
                                            ////__DACE:0:0:241                        ////__DACE:0:0:241
                                            ///////////////////                                                               ////__DACE:0:0:241    ////__DACE:0:0:241
                                            // Tasklet code (tlet_89_multiplies_1)                                            ////__DACE:0:0:241    ////__DACE:0:0:241
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:241    ////__DACE:0:0:241
                                            ///////////////////                                                               ////__DACE:0:0:241    ////__DACE:0:0:241
                                            ////__DACE:0:0:241                        ////__DACE:0:0:241
                                            __map_fusion_gtir_tmp_220_1 = __tlet_result;                                      ////__DACE:0:0:241    ////__DACE:0:0:241
                                        }                                             ////__DACE:0:0:241
                                        {                                             ////__DACE:0:0:240
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_220_1;                                 ////__DACE:0:0:242,240    ////__DACE:0:0:240
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,240    ////__DACE:0:0:240
                                            double __tlet_result;                                                             ////__DACE:0:0:240    ////__DACE:0:0:240
                                            ////__DACE:0:0:240                        ////__DACE:0:0:240
                                            ///////////////////                                                               ////__DACE:0:0:240    ////__DACE:0:0:240
                                            // Tasklet code (tlet_90_plus_1)                                                  ////__DACE:0:0:240    ////__DACE:0:0:240
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:240    ////__DACE:0:0:240
                                            ///////////////////                                                               ////__DACE:0:0:240    ////__DACE:0:0:240
                                            ////__DACE:0:0:240                        ////__DACE:0:0:240
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:240    ////__DACE:0:0:240
                                        }                                             ////__DACE:0:0:240
                                        {                                             ////__DACE:0:0:164
                                            double __tlet_arg1 = lambda_4___p_dthalf_1;                                       ////__DACE:0:0:276,164    ////__DACE:0:0:164
                                            double __tlet_arg0 = tangential_wind[((__tangential_wind_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:6,164    ////__DACE:0:0:164
                                            double __tlet_result;                                                             ////__DACE:0:0:164    ////__DACE:0:0:164
                                            ////__DACE:0:0:164                        ////__DACE:0:0:164
                                            ///////////////////                                                               ////__DACE:0:0:164    ////__DACE:0:0:164
                                            // Tasklet code (tlet_11_multiplies_0)                                            ////__DACE:0:0:164    ////__DACE:0:0:164
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:164    ////__DACE:0:0:164
                                            ///////////////////                                                               ////__DACE:0:0:164    ////__DACE:0:0:164
                                            ////__DACE:0:0:164                        ////__DACE:0:0:164
                                            __map_fusion_gtir_tmp_23_0 = __tlet_result;                                       ////__DACE:0:0:164    ////__DACE:0:0:164
                                        }                                             ////__DACE:0:0:164
                                        {                                             ////__DACE:0:0:183
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,183    ////__DACE:0:0:183
                                            double __tlet_arg1 = gtir_tmp_175_1;                                              ////__DACE:0:0:290,183    ////__DACE:0:0:183
                                            bool __tlet_result;                                                               ////__DACE:0:0:183    ////__DACE:0:0:183
                                            ////__DACE:0:0:183                        ////__DACE:0:0:183
                                            ///////////////////                                                               ////__DACE:0:0:183    ////__DACE:0:0:183
                                            // Tasklet code (tlet_69_greater_equal_0)                                         ////__DACE:0:0:183    ////__DACE:0:0:183
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:183    ////__DACE:0:0:183
                                            ///////////////////                                                               ////__DACE:0:0:183    ////__DACE:0:0:183
                                            ////__DACE:0:0:183                        ////__DACE:0:0:183
                                            __map_fusion_gtir_tmp_177_0 = __tlet_result;                                      ////__DACE:0:0:183    ////__DACE:0:0:183
                                        }                                             ////__DACE:0:0:183
                                        {                                             ////__DACE:0:0:151
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,151    ////__DACE:0:0:151
                                            double __tlet_arg1 = gtir_tmp_3_0;                                                ////__DACE:0:0:266,151    ////__DACE:0:0:151
                                            bool __tlet_result;                                                               ////__DACE:0:0:151    ////__DACE:0:0:151
                                            ////__DACE:0:0:151                        ////__DACE:0:0:151
                                            ///////////////////                                                               ////__DACE:0:0:151    ////__DACE:0:0:151
                                            // Tasklet code (tlet_3_greater_equal_0)                                          ////__DACE:0:0:151    ////__DACE:0:0:151
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:151    ////__DACE:0:0:151
                                            ///////////////////                                                               ////__DACE:0:0:151    ////__DACE:0:0:151
                                            ////__DACE:0:0:151                        ////__DACE:0:0:151
                                            __map_fusion_gtir_tmp_5_0 = __tlet_result;                                        ////__DACE:0:0:151    ////__DACE:0:0:151
                                        }                                             ////__DACE:0:0:151
                                        if_stmt_0_0_0_194(gtir_tmp_6_1, gtir_tmp_7_1, __map_fusion_gtir_tmp_5_0, gtir_tmp_8_0);    ////__DACE:0:0:150
                                        if_stmt_1_0_0_155(gtir_tmp_12_0, gtir_tmp_24_0, gtir_tmp_14_0, gtir_tmp_26_0, gtir_tmp_8_0, gtir_tmp_16_0, gtir_tmp_28_0);    ////__DACE:0:0:155
                                        {                                             ////__DACE:0:0:153
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_11_0;                                  ////__DACE:0:0:187,153    ////__DACE:0:0:153
                                            double __tlet_arg1 = gtir_tmp_16_0;                                               ////__DACE:0:0:154,153    ////__DACE:0:0:153
                                            double __tlet_result;                                                             ////__DACE:0:0:153    ////__DACE:0:0:153
                                            ////__DACE:0:0:153                        ////__DACE:0:0:153
                                            ///////////////////                                                               ////__DACE:0:0:153    ////__DACE:0:0:153
                                            // Tasklet code (tlet_8_plus_0)                                                   ////__DACE:0:0:153    ////__DACE:0:0:153
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:153    ////__DACE:0:0:153
                                            ///////////////////                                                               ////__DACE:0:0:153    ////__DACE:0:0:153
                                            ////__DACE:0:0:153                        ////__DACE:0:0:153
                                            __map_fusion_gtir_tmp_19_0 = __tlet_result;                                       ////__DACE:0:0:153    ////__DACE:0:0:153
                                        }                                             ////__DACE:0:0:153
                                        {                                             ////__DACE:0:0:152
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_19_0;                                  ////__DACE:0:0:186,152    ////__DACE:0:0:152
                                            double __tlet_result;                                                             ////__DACE:0:0:152    ////__DACE:0:0:152
                                            ////__DACE:0:0:152                        ////__DACE:0:0:152
                                            ///////////////////                                                               ////__DACE:0:0:152    ////__DACE:0:0:152
                                            // Tasklet code (tlet_9_neg_0)                                                    ////__DACE:0:0:152    ////__DACE:0:0:152
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:152    ////__DACE:0:0:152
                                            ///////////////////                                                               ////__DACE:0:0:152    ////__DACE:0:0:152
                                            ////__DACE:0:0:152                        ////__DACE:0:0:152
                                            __map_fusion_gtir_tmp_21_0_0 = __tlet_result;                                     ////__DACE:0:0:152    ////__DACE:0:0:152
                                        }                                             ////__DACE:0:0:152
                                        {                                             ////__DACE:0:0:160
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_23_0;                                  ////__DACE:0:0:190,160    ////__DACE:0:0:160
                                            double __tlet_arg1 = gtir_tmp_28_0;                                               ////__DACE:0:0:161,160    ////__DACE:0:0:160
                                            double __tlet_result;                                                             ////__DACE:0:0:160    ////__DACE:0:0:160
                                            ////__DACE:0:0:160                        ////__DACE:0:0:160
                                            ///////////////////                                                               ////__DACE:0:0:160    ////__DACE:0:0:160
                                            // Tasklet code (tlet_12_plus_0)                                                  ////__DACE:0:0:160    ////__DACE:0:0:160
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:160    ////__DACE:0:0:160
                                            ///////////////////                                                               ////__DACE:0:0:160    ////__DACE:0:0:160
                                            ////__DACE:0:0:160                        ////__DACE:0:0:160
                                            __map_fusion_gtir_tmp_31_0 = __tlet_result;                                       ////__DACE:0:0:160    ////__DACE:0:0:160
                                        }                                             ////__DACE:0:0:160
                                        {                                             ////__DACE:0:0:159
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_31_0;                                  ////__DACE:0:0:189,159    ////__DACE:0:0:159
                                            double __tlet_result;                                                             ////__DACE:0:0:159    ////__DACE:0:0:159
                                            ////__DACE:0:0:159                        ////__DACE:0:0:159
                                            ///////////////////                                                               ////__DACE:0:0:159    ////__DACE:0:0:159
                                            // Tasklet code (tlet_13_neg_0)                                                   ////__DACE:0:0:159    ////__DACE:0:0:159
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:159    ////__DACE:0:0:159
                                            ///////////////////                                                               ////__DACE:0:0:159    ////__DACE:0:0:159
                                            ////__DACE:0:0:159                        ////__DACE:0:0:159
                                            __map_fusion_gtir_tmp_33_0_0 = __tlet_result;                                     ////__DACE:0:0:159    ////__DACE:0:0:159
                                        }                                             ////__DACE:0:0:159
                                        if_stmt_4_0_0_166(gtir_tmp_8_0, __map_fusion_gtir_tmp_21_0_0, __map_fusion_gtir_tmp_33_0_0, gtir_tmp_34_0, gtir_tmp_38_0, gtir_tmp_44_0, gtir_tmp_48_0, gtir_tmp_56_0, gtir_tmp_60_0, gtir_tmp_66_0, gtir_tmp_70_0, gtir_tmp_76_0, gtir_tmp_54_0);    ////__DACE:0:0:166
                                        if_stmt_7_0_0_182(__map_fusion_gtir_tmp_177_0, &gt_conn_E2C[0], gtir_tmp_54_0, gtir_tmp_76_0, &gtir_tmp_83[0], &gtir_tmp_89[0], &perturbed_rho_at_cells_on_model_levels[0], &reference_rho_at_edges_on_model_levels[0], gtir_tmp_210_0, __gt_conn_E2C_neighbor_stride, __perturbed_rho_at_cells_on_model_levels_K_stride, __reference_rho_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:182
                                        {                                             ////__DACE:0:0:349
                                            double _cpy_in = gtir_tmp_210_0;                                                  ////__DACE:0:0:181,349    ////__DACE:0:0:349
                                            double _cpy_out;                                                                  ////__DACE:0:0:349    ////__DACE:0:0:349
                                            ////__DACE:0:0:349                        ////__DACE:0:0:349
                                            ///////////////////                                                               ////__DACE:0:0:349    ////__DACE:0:0:349
                                            // Tasklet code (copy_gtir_tmp_210_0_to_rho_at_edges_on_model_levels)             ////__DACE:0:0:349    ////__DACE:0:0:349
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:349    ////__DACE:0:0:349
                                            ///////////////////                                                               ////__DACE:0:0:349    ////__DACE:0:0:349
                                            ////__DACE:0:0:349                        ////__DACE:0:0:349
                                            rho_at_edges_on_model_levels[((__rho_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:349    ////__DACE:0:0:349
                                        }                                             ////__DACE:0:0:349
                                        {                                             ////__DACE:0:0:179
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,179    ////__DACE:0:0:179
                                            double __tlet_arg1 = gtir_tmp_102_1;                                              ////__DACE:0:0:280,179    ////__DACE:0:0:179
                                            bool __tlet_result;                                                               ////__DACE:0:0:179    ////__DACE:0:0:179
                                            ////__DACE:0:0:179                        ////__DACE:0:0:179
                                            ///////////////////                                                               ////__DACE:0:0:179    ////__DACE:0:0:179
                                            // Tasklet code (tlet_35_greater_equal_0)                                         ////__DACE:0:0:179    ////__DACE:0:0:179
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:179    ////__DACE:0:0:179
                                            ///////////////////                                                               ////__DACE:0:0:179    ////__DACE:0:0:179
                                            ////__DACE:0:0:179                        ////__DACE:0:0:179
                                            __map_fusion_gtir_tmp_104_0 = __tlet_result;                                      ////__DACE:0:0:179    ////__DACE:0:0:179
                                        }                                             ////__DACE:0:0:179
                                        if_stmt_5_0_0_178(__map_fusion_gtir_tmp_104_0, &gt_conn_E2C[0], &gtir_tmp_101[0], gtir_tmp_54_0, gtir_tmp_76_0, &gtir_tmp_95[0], &perturbed_theta_v_at_cells_on_model_levels[0], &reference_theta_at_edges_on_model_levels[0], gtir_tmp_137_0, __gt_conn_E2C_neighbor_stride, __perturbed_theta_v_at_cells_on_model_levels_K_stride, __reference_theta_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:178
                                        {                                             ////__DACE:0:0:348
                                            double _cpy_in = gtir_tmp_137_0;                                                  ////__DACE:0:0:177,348    ////__DACE:0:0:348
                                            double _cpy_out;                                                                  ////__DACE:0:0:348    ////__DACE:0:0:348
                                            ////__DACE:0:0:348                        ////__DACE:0:0:348
                                            ///////////////////                                                               ////__DACE:0:0:348    ////__DACE:0:0:348
                                            // Tasklet code (copy_gtir_tmp_137_0_to_theta_v_at_edges_on_model_levels)         ////__DACE:0:0:348    ////__DACE:0:0:348
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:348    ////__DACE:0:0:348
                                            ///////////////////                                                               ////__DACE:0:0:348    ////__DACE:0:0:348
                                            ////__DACE:0:0:348                        ////__DACE:0:0:348
                                            theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:348    ////__DACE:0:0:348
                                        }                                             ////__DACE:0:0:348
                                    }                                                 ////__DACE:0:0:252
                                }                                                     ////__DACE:0:0:252
                            }                                                         ////__DACE:0:0:180
                        }                                                             ////__DACE:0:0:180
                    }                                                                 ////__DACE:0:0:180
                }                                                                     ////__DACE:0:0:180
            }                                                                         ////__DACE:0:0:180
        }                                                                             ////__DACE:0:0:361
    }                                                                                 ////__DACE:0:0:361
}                                                                                 ////__DACE:0:0:361

                                                                                  ////__DACE:0:0:360
DACE_EXPORTED void __dace_runkernel_map_100_fieldop_0_0_0_360(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const double * __restrict__ grf_tend_vn, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ next_vn, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __grf_tend_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __next_vn_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:360
void __dace_runkernel_map_100_fieldop_0_0_0_360(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const double * __restrict__ grf_tend_vn, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ next_vn, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __grf_tend_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __next_vn_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:360
{                                                                                 ////__DACE:0:0:360
                                                                                  ////__DACE:0:0:360
    void  *map_100_fieldop_0_0_0_360_args[] = { (void *)&current_vn, (void *)&dual_normal_cell_x, (void *)&dual_normal_cell_y, (void *)&grf_tend_vn, (void *)&gt_conn_E2C, (void *)&gtir_tmp_101, (void *)&gtir_tmp_83, (void *)&gtir_tmp_89, (void *)&gtir_tmp_95, (void *)&next_vn, (void *)&perturbed_rho_at_cells_on_model_levels, (void *)&perturbed_theta_v_at_cells_on_model_levels, (void *)&pos_on_tplane_e_x, (void *)&pos_on_tplane_e_y, (void *)&primal_normal_cell_x, (void *)&primal_normal_cell_y, (void *)&reference_rho_at_edges_on_model_levels, (void *)&reference_theta_at_edges_on_model_levels, (void *)&rho_at_edges_on_model_levels, (void *)&tangential_wind, (void *)&theta_v_at_edges_on_model_levels, (void *)&__current_vn_K_stride, (void *)&__dual_normal_cell_x_E2C_stride, (void *)&__dual_normal_cell_y_E2C_stride, (void *)&__grf_tend_vn_K_stride, (void *)&__gt_conn_E2C_neighbor_stride, (void *)&__next_vn_K_stride, (void *)&__perturbed_rho_at_cells_on_model_levels_K_stride, (void *)&__perturbed_theta_v_at_cells_on_model_levels_K_stride, (void *)&__pos_on_tplane_e_x_E2C_stride, (void *)&__pos_on_tplane_e_y_E2C_stride, (void *)&__primal_normal_cell_x_E2C_stride, (void *)&__primal_normal_cell_y_E2C_stride, (void *)&__reference_rho_at_edges_on_model_levels_K_stride, (void *)&__reference_theta_at_edges_on_model_levels_K_stride, (void *)&__rho_at_edges_on_model_levels_K_stride, (void *)&__tangential_wind_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:360
    gpuError_t __err = hipLaunchKernel((void*)map_100_fieldop_0_0_0_360, dim3(10, 30, 1), dim3(256, 1, 1), map_100_fieldop_0_0_0_360_args, 0, nullptr);    ////__DACE:0:0:360
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_100_fieldop_0_0_0_360", 10, 30, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_105_fieldop_0_0_0_356(const double * __restrict__ current_vn, const int * __restrict__ gt_conn_E2C, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ pg_exdist, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, const double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __pg_exdist_K_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:356
    {                                                                                 ////__DACE:0:0:356
        {                                                                             ////__DACE:0:0:356
            int b_i_Edge_gtx_horizontal = ((256 * blockIdx.x) + 7701);                ////__DACE:0:0:356
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:356
            {                                                                         ////__DACE:0:0:61
                {                                                                     ////__DACE:0:0:61
                    {                                                                 ////__DACE:0:0:61
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:61
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:61
                        double gtir_tmp_166_0;                                        ////__DACE:0:0:282
                        double gtir_tmp_223_0;                                        ////__DACE:0:0:292
                        double __dtime_0_0;                                           ////__DACE:0:0:296
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(67095, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:61
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(6, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:61
                                {                                                     ////__DACE:0:0:281
                                    double __tlet_out;                                                                ////__DACE:0:0:281    ////__DACE:0:0:281
                                    ////__DACE:0:0:281                                ////__DACE:0:0:281
                                    ///////////////////                                                               ////__DACE:0:0:281    ////__DACE:0:0:281
                                    // Tasklet code (tlet_64_get_value__clone_0)                                      ////__DACE:0:0:281    ////__DACE:0:0:281
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:281    ////__DACE:0:0:281
                                    ///////////////////                                                               ////__DACE:0:0:281    ////__DACE:0:0:281
                                    ////__DACE:0:0:281                                ////__DACE:0:0:281
                                    gtir_tmp_166_0 = __tlet_out;                                                      ////__DACE:0:0:281    ////__DACE:0:0:281
                                }                                                     ////__DACE:0:0:281
                                {                                                     ////__DACE:0:0:291
                                    double __tlet_out;                                                                ////__DACE:0:0:291    ////__DACE:0:0:291
                                    ////__DACE:0:0:291                                ////__DACE:0:0:291
                                    ///////////////////                                                               ////__DACE:0:0:291    ////__DACE:0:0:291
                                    // Tasklet code (tlet_92_get_value__clone_0)                                      ////__DACE:0:0:291    ////__DACE:0:0:291
                                    __tlet_out = 1004.64;                                                             ////__DACE:0:0:291    ////__DACE:0:0:291
                                    ///////////////////                                                               ////__DACE:0:0:291    ////__DACE:0:0:291
                                    ////__DACE:0:0:291                                ////__DACE:0:0:291
                                    gtir_tmp_223_0 = __tlet_out;                                                      ////__DACE:0:0:291    ////__DACE:0:0:291
                                }                                                     ////__DACE:0:0:291
                                {                                                     ////__DACE:0:0:295
                                    double __tlet_out;                                                                ////__DACE:0:0:295    ////__DACE:0:0:295
                                    ////__DACE:0:0:295                                ////__DACE:0:0:295
                                    ///////////////////                                                               ////__DACE:0:0:295    ////__DACE:0:0:295
                                    // Tasklet code (tlet_91_get_value__clone_0)                                      ////__DACE:0:0:295    ////__DACE:0:0:295
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:295    ////__DACE:0:0:295
                                    ///////////////////                                                               ////__DACE:0:0:295    ////__DACE:0:0:295
                                    ////__DACE:0:0:295                                ////__DACE:0:0:295
                                    __dtime_0_0 = __tlet_out;                                                         ////__DACE:0:0:295    ////__DACE:0:0:295
                                }                                                     ////__DACE:0:0:295
                                {                                                     ////__DACE:0:0:247
                                    #pragma unroll 4                                  ////__DACE:0:0:247
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(27, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:247
                                        double __map_fusion_gtir_tmp_145;             ////__DACE:0:0:81
                                        double __map_fusion_gtir_tmp_143;             ////__DACE:0:0:82
                                        double __map_fusion_gtir_tmp_141;             ////__DACE:0:0:83
                                        double gtir_tmp_173_0;                        ////__DACE:0:0:115
                                        bool __map_fusion_gtir_tmp_168_0;             ////__DACE:0:0:125
                                        double __map_fusion_gtir_tmp_235_0;           ////__DACE:0:0:126
                                        double __map_fusion_gtir_tmp_233_0;           ////__DACE:0:0:127
                                        double __map_fusion_gtir_tmp_231_0;           ////__DACE:0:0:128
                                        double __map_fusion_gtir_tmp_229_0;           ////__DACE:0:0:129
                                        double __map_fusion_gtir_tmp_227_0;           ////__DACE:0:0:130
                                        double __map_fusion_gtir_tmp_139_split_0;     ////__DACE:0:0:147
                                        {                                             ////__DACE:0:0:59
                                            int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:0:0:50,59    ////__DACE:0:0:59
                                            const double* __tlet_field = &temporal_extrapolation_of_perturbed_exner[0];       ////__DACE:0:0:55,59    ////__DACE:0:0:59
                                            double __tlet_val;                                                                ////__DACE:0:0:59    ////__DACE:0:0:59
                                            ////__DACE:0:0:59                         ////__DACE:0:0:59
                                            ///////////////////                                                               ////__DACE:0:0:59    ////__DACE:0:0:59
                                            // Tasklet code (tlet_53_deref)                                                   ////__DACE:0:0:59    ////__DACE:0:0:59
                                            __tlet_val = __tlet_field[((__temporal_extrapolation_of_perturbed_exner_K_stride * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:0:0:59    ////__DACE:0:0:59
                                            ///////////////////                                                               ////__DACE:0:0:59    ////__DACE:0:0:59
                                            ////__DACE:0:0:59                         ////__DACE:0:0:59
                                            __map_fusion_gtir_tmp_143 = __tlet_val;                                           ////__DACE:0:0:59    ////__DACE:0:0:59
                                        }                                             ////__DACE:0:0:59
                                        {                                             ////__DACE:0:0:60
                                            int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:50,60    ////__DACE:0:0:60
                                            const double* __tlet_field = &temporal_extrapolation_of_perturbed_exner[0];       ////__DACE:0:0:55,60    ////__DACE:0:0:60
                                            double __tlet_val;                                                                ////__DACE:0:0:60    ////__DACE:0:0:60
                                            ////__DACE:0:0:60                         ////__DACE:0:0:60
                                            ///////////////////                                                               ////__DACE:0:0:60    ////__DACE:0:0:60
                                            // Tasklet code (tlet_52_deref)                                                   ////__DACE:0:0:60    ////__DACE:0:0:60
                                            __tlet_val = __tlet_field[((__temporal_extrapolation_of_perturbed_exner_K_stride * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:0:0:60    ////__DACE:0:0:60
                                            ///////////////////                                                               ////__DACE:0:0:60    ////__DACE:0:0:60
                                            ////__DACE:0:0:60                         ////__DACE:0:0:60
                                            __map_fusion_gtir_tmp_141 = __tlet_val;                                           ////__DACE:0:0:60    ////__DACE:0:0:60
                                        }                                             ////__DACE:0:0:60
                                        {                                             ////__DACE:0:0:58
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_141;                                   ////__DACE:0:0:83,58    ////__DACE:0:0:58
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_143;                                   ////__DACE:0:0:82,58    ////__DACE:0:0:58
                                            double __tlet_result;                                                             ////__DACE:0:0:58    ////__DACE:0:0:58
                                            ////__DACE:0:0:58                         ////__DACE:0:0:58
                                            ///////////////////                                                               ////__DACE:0:0:58    ////__DACE:0:0:58
                                            // Tasklet code (tlet_54_minus)                                                   ////__DACE:0:0:58    ////__DACE:0:0:58
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:58    ////__DACE:0:0:58
                                            ///////////////////                                                               ////__DACE:0:0:58    ////__DACE:0:0:58
                                            ////__DACE:0:0:58                         ////__DACE:0:0:58
                                            __map_fusion_gtir_tmp_145 = __tlet_result;                                        ////__DACE:0:0:58    ////__DACE:0:0:58
                                        }                                             ////__DACE:0:0:58
                                        {                                             ////__DACE:0:0:57
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_145;                                   ////__DACE:0:0:81,57    ////__DACE:0:0:57
                                            double __tlet_arg0 = inv_dual_edge_length[i_Edge_gtx_horizontal];                 ////__DACE:0:0:56,57    ////__DACE:0:0:57
                                            double __tlet_result;                                                             ////__DACE:0:0:57    ////__DACE:0:0:57
                                            ////__DACE:0:0:57                         ////__DACE:0:0:57
                                            ///////////////////                                                               ////__DACE:0:0:57    ////__DACE:0:0:57
                                            // Tasklet code (tlet_55_multiplies)                                              ////__DACE:0:0:57    ////__DACE:0:0:57
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:57    ////__DACE:0:0:57
                                            ///////////////////                                                               ////__DACE:0:0:57    ////__DACE:0:0:57
                                            ////__DACE:0:0:57                         ////__DACE:0:0:57
                                            __map_fusion_gtir_tmp_139_split_0 = __tlet_result;                                ////__DACE:0:0:57    ////__DACE:0:0:57
                                        }                                             ////__DACE:0:0:57
                                        {                                             ////__DACE:0:0:117
                                            double __tlet_arg1 = gtir_tmp_166_0;                                              ////__DACE:0:0:282,117    ////__DACE:0:0:117
                                            double __tlet_arg0 = pg_exdist[((__pg_exdist_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:62,117    ////__DACE:0:0:117
                                            bool __tlet_result;                                                               ////__DACE:0:0:117    ////__DACE:0:0:117
                                            ////__DACE:0:0:117                        ////__DACE:0:0:117
                                            ///////////////////                                                               ////__DACE:0:0:117    ////__DACE:0:0:117
                                            // Tasklet code (tlet_65_not_eq_0)                                                ////__DACE:0:0:117    ////__DACE:0:0:117
                                            __tlet_result = (__tlet_arg0 != __tlet_arg1);                                     ////__DACE:0:0:117    ////__DACE:0:0:117
                                            ///////////////////                                                               ////__DACE:0:0:117    ////__DACE:0:0:117
                                            ////__DACE:0:0:117                        ////__DACE:0:0:117
                                            __map_fusion_gtir_tmp_168_0 = __tlet_result;                                      ////__DACE:0:0:117    ////__DACE:0:0:117
                                        }                                             ////__DACE:0:0:117
                                        if_stmt_6_0_0_116(__map_fusion_gtir_tmp_139_split_0, __map_fusion_gtir_tmp_168_0, &hydrostatic_correction_on_lowest_level[0], &pg_exdist[0], gtir_tmp_173_0, __pg_exdist_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:116
                                        {                                             ////__DACE:0:0:346
                                            double _cpy_in = gtir_tmp_173_0;                                                  ////__DACE:0:0:115,346    ////__DACE:0:0:346
                                            double _cpy_out;                                                                  ////__DACE:0:0:346    ////__DACE:0:0:346
                                            ////__DACE:0:0:346                        ////__DACE:0:0:346
                                            ///////////////////                                                               ////__DACE:0:0:346    ////__DACE:0:0:346
                                            // Tasklet code (copy_gtir_tmp_173_0_to_horizontal_pressure_gradient)             ////__DACE:0:0:346    ////__DACE:0:0:346
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:346    ////__DACE:0:0:346
                                            ///////////////////                                                               ////__DACE:0:0:346    ////__DACE:0:0:346
                                            ////__DACE:0:0:346                        ////__DACE:0:0:346
                                            horizontal_pressure_gradient[((__horizontal_pressure_gradient_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:346    ////__DACE:0:0:346
                                        }                                             ////__DACE:0:0:346
                                        {                                             ////__DACE:0:0:124
                                            double __tlet_arg0 = gtir_tmp_223_0;                                              ////__DACE:0:0:292,124    ////__DACE:0:0:124
                                            double __tlet_arg1 = theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:70,124    ////__DACE:0:0:124
                                            double __tlet_result;                                                             ////__DACE:0:0:124    ////__DACE:0:0:124
                                            ////__DACE:0:0:124                        ////__DACE:0:0:124
                                            ///////////////////                                                               ////__DACE:0:0:124    ////__DACE:0:0:124
                                            // Tasklet code (tlet_93_multiplies_0)                                            ////__DACE:0:0:124    ////__DACE:0:0:124
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:124    ////__DACE:0:0:124
                                            ///////////////////                                                               ////__DACE:0:0:124    ////__DACE:0:0:124
                                            ////__DACE:0:0:124                        ////__DACE:0:0:124
                                            __map_fusion_gtir_tmp_227_0 = __tlet_result;                                      ////__DACE:0:0:124    ////__DACE:0:0:124
                                        }                                             ////__DACE:0:0:124
                                        {                                             ////__DACE:0:0:123
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_227_0;                                 ////__DACE:0:0:130,123    ////__DACE:0:0:123
                                            double __tlet_arg1 = gtir_tmp_173_0;                                              ////__DACE:0:0:115,123    ////__DACE:0:0:123
                                            double __tlet_result;                                                             ////__DACE:0:0:123    ////__DACE:0:0:123
                                            ////__DACE:0:0:123                        ////__DACE:0:0:123
                                            ///////////////////                                                               ////__DACE:0:0:123    ////__DACE:0:0:123
                                            // Tasklet code (tlet_94_multiplies_0)                                            ////__DACE:0:0:123    ////__DACE:0:0:123
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:123    ////__DACE:0:0:123
                                            ///////////////////                                                               ////__DACE:0:0:123    ////__DACE:0:0:123
                                            ////__DACE:0:0:123                        ////__DACE:0:0:123
                                            __map_fusion_gtir_tmp_229_0 = __tlet_result;                                      ////__DACE:0:0:123    ////__DACE:0:0:123
                                        }                                             ////__DACE:0:0:123
                                        {                                             ////__DACE:0:0:122
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_229_0;                                 ////__DACE:0:0:129,122    ////__DACE:0:0:122
                                            double __tlet_arg0 = predictor_normal_wind_advective_tendency[((__predictor_normal_wind_advective_tendency_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:67,122    ////__DACE:0:0:122
                                            double __tlet_result;                                                             ////__DACE:0:0:122    ////__DACE:0:0:122
                                            ////__DACE:0:0:122                        ////__DACE:0:0:122
                                            ///////////////////                                                               ////__DACE:0:0:122    ////__DACE:0:0:122
                                            // Tasklet code (tlet_95_minus_0)                                                 ////__DACE:0:0:122    ////__DACE:0:0:122
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:122    ////__DACE:0:0:122
                                            ///////////////////                                                               ////__DACE:0:0:122    ////__DACE:0:0:122
                                            ////__DACE:0:0:122                        ////__DACE:0:0:122
                                            __map_fusion_gtir_tmp_231_0 = __tlet_result;                                      ////__DACE:0:0:122    ////__DACE:0:0:122
                                        }                                             ////__DACE:0:0:122
                                        {                                             ////__DACE:0:0:121
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_231_0;                                 ////__DACE:0:0:128,121    ////__DACE:0:0:121
                                            double __tlet_arg1 = normal_wind_tendency_due_to_slow_physics_process[((__normal_wind_tendency_due_to_slow_physics_process_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:66,121    ////__DACE:0:0:121
                                            double __tlet_result;                                                             ////__DACE:0:0:121    ////__DACE:0:0:121
                                            ////__DACE:0:0:121                        ////__DACE:0:0:121
                                            ///////////////////                                                               ////__DACE:0:0:121    ////__DACE:0:0:121
                                            // Tasklet code (tlet_96_plus_0)                                                  ////__DACE:0:0:121    ////__DACE:0:0:121
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:121    ////__DACE:0:0:121
                                            ///////////////////                                                               ////__DACE:0:0:121    ////__DACE:0:0:121
                                            ////__DACE:0:0:121                        ////__DACE:0:0:121
                                            __map_fusion_gtir_tmp_233_0 = __tlet_result;                                      ////__DACE:0:0:121    ////__DACE:0:0:121
                                        }                                             ////__DACE:0:0:121
                                        {                                             ////__DACE:0:0:120
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_233_0;                                 ////__DACE:0:0:127,120    ////__DACE:0:0:120
                                            double __tlet_arg0 = __dtime_0_0;                                                 ////__DACE:0:0:296,120    ////__DACE:0:0:120
                                            double __tlet_result;                                                             ////__DACE:0:0:120    ////__DACE:0:0:120
                                            ////__DACE:0:0:120                        ////__DACE:0:0:120
                                            ///////////////////                                                               ////__DACE:0:0:120    ////__DACE:0:0:120
                                            // Tasklet code (tlet_97_multiplies_0)                                            ////__DACE:0:0:120    ////__DACE:0:0:120
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:120    ////__DACE:0:0:120
                                            ///////////////////                                                               ////__DACE:0:0:120    ////__DACE:0:0:120
                                            ////__DACE:0:0:120                        ////__DACE:0:0:120
                                            __map_fusion_gtir_tmp_235_0 = __tlet_result;                                      ////__DACE:0:0:120    ////__DACE:0:0:120
                                        }                                             ////__DACE:0:0:120
                                        {                                             ////__DACE:0:0:119
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_235_0;                                 ////__DACE:0:0:126,119    ////__DACE:0:0:119
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,119    ////__DACE:0:0:119
                                            double __tlet_result;                                                             ////__DACE:0:0:119    ////__DACE:0:0:119
                                            ////__DACE:0:0:119                        ////__DACE:0:0:119
                                            ///////////////////                                                               ////__DACE:0:0:119    ////__DACE:0:0:119
                                            // Tasklet code (tlet_98_plus_0)                                                  ////__DACE:0:0:119    ////__DACE:0:0:119
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:119    ////__DACE:0:0:119
                                            ///////////////////                                                               ////__DACE:0:0:119    ////__DACE:0:0:119
                                            ////__DACE:0:0:119                        ////__DACE:0:0:119
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:119    ////__DACE:0:0:119
                                        }                                             ////__DACE:0:0:119
                                    }                                                 ////__DACE:0:0:248
                                }                                                     ////__DACE:0:0:248
                            }                                                         ////__DACE:0:0:118
                        }                                                             ////__DACE:0:0:118
                    }                                                                 ////__DACE:0:0:118
                }                                                                     ////__DACE:0:0:118
            }                                                                         ////__DACE:0:0:118
        }                                                                             ////__DACE:0:0:357
    }                                                                                 ////__DACE:0:0:357
}                                                                                 ////__DACE:0:0:357

                                                                                  ////__DACE:0:0:356
DACE_EXPORTED void __dace_runkernel_map_105_fieldop_0_0_0_356(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const int * __restrict__ gt_conn_E2C, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ pg_exdist, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, const double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __pg_exdist_K_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:356
void __dace_runkernel_map_105_fieldop_0_0_0_356(theta_shared_probe_native_state_t *__state, const double * __restrict__ current_vn, const int * __restrict__ gt_conn_E2C, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ pg_exdist, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, const double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __pg_exdist_K_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:356
{                                                                                 ////__DACE:0:0:356
                                                                                  ////__DACE:0:0:356
    void  *map_105_fieldop_0_0_0_356_args[] = { (void *)&current_vn, (void *)&gt_conn_E2C, (void *)&horizontal_pressure_gradient, (void *)&hydrostatic_correction_on_lowest_level, (void *)&inv_dual_edge_length, (void *)&next_vn, (void *)&normal_wind_tendency_due_to_slow_physics_process, (void *)&pg_exdist, (void *)&predictor_normal_wind_advective_tendency, (void *)&temporal_extrapolation_of_perturbed_exner, (void *)&theta_v_at_edges_on_model_levels, (void *)&__current_vn_K_stride, (void *)&__gt_conn_E2C_neighbor_stride, (void *)&__horizontal_pressure_gradient_K_stride, (void *)&__next_vn_K_stride, (void *)&__normal_wind_tendency_due_to_slow_physics_process_K_stride, (void *)&__pg_exdist_K_stride, (void *)&__predictor_normal_wind_advective_tendency_K_stride, (void *)&__temporal_extrapolation_of_perturbed_exner_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:356
    gpuError_t __err = hipLaunchKernel((void*)map_105_fieldop_0_0_0_356, dim3(233, 7, 1), dim3(256, 1, 1), map_105_fieldop_0_0_0_356_args, 0, nullptr);    ////__DACE:0:0:356
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_105_fieldop_0_0_0_356", 233, 7, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_105_fieldop_1_0_0_358(const double * __restrict__ c_lin_e, const double * __restrict__ current_vn, const double * __restrict__ ddxn_z_full, const double * __restrict__ ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, const int * __restrict__ gt_conn_E2C, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ pg_exdist, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, const double * __restrict__ theta_v_at_edges_on_model_levels, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __pg_exdist_K_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:358
    {                                                                                 ////__DACE:0:0:358
        {                                                                             ////__DACE:0:0:358
            int b_i_Edge_gtx_horizontal = ((256 * blockIdx.x) + 7701);                ////__DACE:0:0:358
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:358
            {                                                                         ////__DACE:0:0:105
                {                                                                     ////__DACE:0:0:105
                    {                                                                 ////__DACE:0:0:105
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:105
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:105
                        double gtir_tmp_166_1;                                        ////__DACE:0:0:284
                        double gtir_tmp_223_1;                                        ////__DACE:0:0:294
                        double __dtime_0_1;                                           ////__DACE:0:0:298
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(67095, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:105
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(23, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:105
                                {                                                     ////__DACE:0:0:283
                                    double __tlet_out;                                                                ////__DACE:0:0:283    ////__DACE:0:0:283
                                    ////__DACE:0:0:283                                ////__DACE:0:0:283
                                    ///////////////////                                                               ////__DACE:0:0:283    ////__DACE:0:0:283
                                    // Tasklet code (tlet_64_get_value__clone_1)                                      ////__DACE:0:0:283    ////__DACE:0:0:283
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:283    ////__DACE:0:0:283
                                    ///////////////////                                                               ////__DACE:0:0:283    ////__DACE:0:0:283
                                    ////__DACE:0:0:283                                ////__DACE:0:0:283
                                    gtir_tmp_166_1 = __tlet_out;                                                      ////__DACE:0:0:283    ////__DACE:0:0:283
                                }                                                     ////__DACE:0:0:283
                                {                                                     ////__DACE:0:0:293
                                    double __tlet_out;                                                                ////__DACE:0:0:293    ////__DACE:0:0:293
                                    ////__DACE:0:0:293                                ////__DACE:0:0:293
                                    ///////////////////                                                               ////__DACE:0:0:293    ////__DACE:0:0:293
                                    // Tasklet code (tlet_92_get_value__clone_1)                                      ////__DACE:0:0:293    ////__DACE:0:0:293
                                    __tlet_out = 1004.64;                                                             ////__DACE:0:0:293    ////__DACE:0:0:293
                                    ///////////////////                                                               ////__DACE:0:0:293    ////__DACE:0:0:293
                                    ////__DACE:0:0:293                                ////__DACE:0:0:293
                                    gtir_tmp_223_1 = __tlet_out;                                                      ////__DACE:0:0:293    ////__DACE:0:0:293
                                }                                                     ////__DACE:0:0:293
                                {                                                     ////__DACE:0:0:297
                                    double __tlet_out;                                                                ////__DACE:0:0:297    ////__DACE:0:0:297
                                    ////__DACE:0:0:297                                ////__DACE:0:0:297
                                    ///////////////////                                                               ////__DACE:0:0:297    ////__DACE:0:0:297
                                    // Tasklet code (tlet_91_get_value__clone_1)                                      ////__DACE:0:0:297    ////__DACE:0:0:297
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:297    ////__DACE:0:0:297
                                    ///////////////////                                                               ////__DACE:0:0:297    ////__DACE:0:0:297
                                    ////__DACE:0:0:297                                ////__DACE:0:0:297
                                    __dtime_0_1 = __tlet_out;                                                         ////__DACE:0:0:297    ////__DACE:0:0:297
                                }                                                     ////__DACE:0:0:297
                                {                                                     ////__DACE:0:0:249
                                    #pragma unroll 4                                  ////__DACE:0:0:249
                                    for (auto i_K_gtx_vertical = ((4 * __gtx_coarse_i_K_gtx_vertical) + 27); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 31)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:249
                                        double gtir_tmp_160_0;                        ////__DACE:0:0:94
                                        double __map_fusion_gtir_tmp_163_0;           ////__DACE:0:0:106
                                        double __map_fusion_gtir_tmp_159_0[2]  DACE_ALIGN(64);    ////__DACE:0:0:107
                                        double __map_fusion_gtir_tmp_157_0[2]  DACE_ALIGN(64);    ////__DACE:0:0:108
                                        double __map_fusion_gtir_tmp_155_0;           ////__DACE:0:0:109
                                        double __map_fusion_gtir_tmp_153_0;           ////__DACE:0:0:110
                                        double __map_fusion_gtir_tmp_151_0;           ////__DACE:0:0:111
                                        double __map_fusion_gtir_tmp_149_0;           ////__DACE:0:0:112
                                        double gtir_tmp_173_1;                        ////__DACE:0:0:131
                                        bool __map_fusion_gtir_tmp_168_1;             ////__DACE:0:0:141
                                        double __map_fusion_gtir_tmp_235_1;           ////__DACE:0:0:142
                                        double __map_fusion_gtir_tmp_233_1;           ////__DACE:0:0:143
                                        double __map_fusion_gtir_tmp_231_1;           ////__DACE:0:0:144
                                        double __map_fusion_gtir_tmp_229_1;           ////__DACE:0:0:145
                                        double __map_fusion_gtir_tmp_227_1;           ////__DACE:0:0:146
                                        double __map_fusion_gtir_tmp_139_split_1;     ////__DACE:0:0:148
                                        {                                             ////__DACE:0:0:100
                                            for (auto i_E2C_gtx_localdim = 0; i_E2C_gtx_localdim < 2; i_E2C_gtx_localdim += 1) {    ////__DACE:0:0:100
                                                double __gtx_double_write_remover_inner_inner_distribution_node_8_0;    ////__DACE:0:0:114
                                                {                                     ////__DACE:0:0:99
                                                    const double* __tlet_field = &ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels[(__ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride * i_K_gtx_vertical)];    ////__DACE:0:0:53,99    ////__DACE:0:0:99
                                                    int __tlet_index = gt_conn_E2C[((__gt_conn_E2C_neighbor_stride * i_E2C_gtx_localdim) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:50,99    ////__DACE:0:0:99
                                                    double __tlet_val;                                                                ////__DACE:0:0:99    ////__DACE:0:0:99
                                                    ////__DACE:0:0:99                 ////__DACE:0:0:99
                                                    ///////////////////                                                               ////__DACE:0:0:99    ////__DACE:0:0:99
                                                    // Tasklet code (tlet_60_E2C_neighbors_0)                                         ////__DACE:0:0:99    ////__DACE:0:0:99
                                                    __tlet_val = __tlet_field[__tlet_index];                                          ////__DACE:0:0:99    ////__DACE:0:0:99
                                                    ///////////////////                                                               ////__DACE:0:0:99    ////__DACE:0:0:99
                                                    ////__DACE:0:0:99                 ////__DACE:0:0:99
                                                    __gtx_double_write_remover_inner_inner_distribution_node_8_0 = __tlet_val;        ////__DACE:0:0:99    ////__DACE:0:0:99
                                                }                                     ////__DACE:0:0:99
                                                {                                     ////__DACE:0:0:317
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_8_0;    ////__DACE:0:0:114,317    ////__DACE:0:0:317
                                                    double _cpy_out;                                                                  ////__DACE:0:0:317    ////__DACE:0:0:317
                                                    ////__DACE:0:0:317                ////__DACE:0:0:317
                                                    ///////////////////                                                               ////__DACE:0:0:317    ////__DACE:0:0:317
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_8_0_to___map_fusion_gtir_tmp_157_0)    ////__DACE:0:0:317    ////__DACE:0:0:317
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:317    ////__DACE:0:0:317
                                                    ///////////////////                                                               ////__DACE:0:0:317    ////__DACE:0:0:317
                                                    ////__DACE:0:0:317                ////__DACE:0:0:317
                                                    __map_fusion_gtir_tmp_157_0[i_E2C_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:317    ////__DACE:0:0:317
                                                }                                     ////__DACE:0:0:317
                                            }                                         ////__DACE:0:0:98
                                        }                                             ////__DACE:0:0:98
                                        {                                             ////__DACE:0:0:97
                                            for (auto i_E2C_gtx_localdim = 0; i_E2C_gtx_localdim < 2; i_E2C_gtx_localdim += 1) {    ////__DACE:0:0:97
                                                double __gtx_double_write_remover_inner_inner_distribution_node_7_0;    ////__DACE:0:0:113
                                                {                                     ////__DACE:0:0:96
                                                    double __tlet_arg1 = c_lin_e[((__c_lin_e_E2C_stride * i_E2C_gtx_localdim) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:52,96    ////__DACE:0:0:96
                                                    double __tlet_arg0 = __map_fusion_gtir_tmp_157_0[i_E2C_gtx_localdim];             ////__DACE:0:0:108,96    ////__DACE:0:0:96
                                                    double __tlet_out;                                                                ////__DACE:0:0:96    ////__DACE:0:0:96
                                                    ////__DACE:0:0:96                 ////__DACE:0:0:96
                                                    ///////////////////                                                               ////__DACE:0:0:96    ////__DACE:0:0:96
                                                    // Tasklet code (tlet_61_map_0)                                                   ////__DACE:0:0:96    ////__DACE:0:0:96
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:96    ////__DACE:0:0:96
                                                    ///////////////////                                                               ////__DACE:0:0:96    ////__DACE:0:0:96
                                                    ////__DACE:0:0:96                 ////__DACE:0:0:96
                                                    __gtx_double_write_remover_inner_inner_distribution_node_7_0 = __tlet_out;        ////__DACE:0:0:96    ////__DACE:0:0:96
                                                }                                     ////__DACE:0:0:96
                                                {                                     ////__DACE:0:0:316
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_7_0;    ////__DACE:0:0:113,316    ////__DACE:0:0:316
                                                    double _cpy_out;                                                                  ////__DACE:0:0:316    ////__DACE:0:0:316
                                                    ////__DACE:0:0:316                ////__DACE:0:0:316
                                                    ///////////////////                                                               ////__DACE:0:0:316    ////__DACE:0:0:316
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_7_0_to___map_fusion_gtir_tmp_159_0)    ////__DACE:0:0:316    ////__DACE:0:0:316
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:316    ////__DACE:0:0:316
                                                    ///////////////////                                                               ////__DACE:0:0:316    ////__DACE:0:0:316
                                                    ////__DACE:0:0:316                ////__DACE:0:0:316
                                                    __map_fusion_gtir_tmp_159_0[i_E2C_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:316    ////__DACE:0:0:316
                                                }                                     ////__DACE:0:0:316
                                            }                                         ////__DACE:0:0:95
                                        }                                             ////__DACE:0:0:95
                                        reduce_0_0_307(&__map_fusion_gtir_tmp_159_0[0], gtir_tmp_160_0);    ////__DACE:0:0:307
                                        {                                             ////__DACE:0:0:93
                                            double __tlet_arg0 = ddxn_z_full[((__ddxn_z_full_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:54,93    ////__DACE:0:0:93
                                            double __tlet_arg1 = gtir_tmp_160_0;                                              ////__DACE:0:0:94,93    ////__DACE:0:0:93
                                            double __tlet_result;                                                             ////__DACE:0:0:93    ////__DACE:0:0:93
                                            ////__DACE:0:0:93                         ////__DACE:0:0:93
                                            ///////////////////                                                               ////__DACE:0:0:93    ////__DACE:0:0:93
                                            // Tasklet code (tlet_62_multiplies_0)                                            ////__DACE:0:0:93    ////__DACE:0:0:93
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:93    ////__DACE:0:0:93
                                            ///////////////////                                                               ////__DACE:0:0:93    ////__DACE:0:0:93
                                            ////__DACE:0:0:93                         ////__DACE:0:0:93
                                            __map_fusion_gtir_tmp_163_0 = __tlet_result;                                      ////__DACE:0:0:93    ////__DACE:0:0:93
                                        }                                             ////__DACE:0:0:93
                                        {                                             ////__DACE:0:0:104
                                            int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:50,104    ////__DACE:0:0:104
                                            const double* __tlet_field = &temporal_extrapolation_of_perturbed_exner[0];       ////__DACE:0:0:55,104    ////__DACE:0:0:104
                                            double __tlet_val;                                                                ////__DACE:0:0:104    ////__DACE:0:0:104
                                            ////__DACE:0:0:104                        ////__DACE:0:0:104
                                            ///////////////////                                                               ////__DACE:0:0:104    ////__DACE:0:0:104
                                            // Tasklet code (tlet_56_deref_0)                                                 ////__DACE:0:0:104    ////__DACE:0:0:104
                                            __tlet_val = __tlet_field[((__temporal_extrapolation_of_perturbed_exner_K_stride * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:0:0:104    ////__DACE:0:0:104
                                            ///////////////////                                                               ////__DACE:0:0:104    ////__DACE:0:0:104
                                            ////__DACE:0:0:104                        ////__DACE:0:0:104
                                            __map_fusion_gtir_tmp_149_0 = __tlet_val;                                         ////__DACE:0:0:104    ////__DACE:0:0:104
                                        }                                             ////__DACE:0:0:104
                                        {                                             ////__DACE:0:0:103
                                            int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:0:0:50,103    ////__DACE:0:0:103
                                            const double* __tlet_field = &temporal_extrapolation_of_perturbed_exner[0];       ////__DACE:0:0:55,103    ////__DACE:0:0:103
                                            double __tlet_val;                                                                ////__DACE:0:0:103    ////__DACE:0:0:103
                                            ////__DACE:0:0:103                        ////__DACE:0:0:103
                                            ///////////////////                                                               ////__DACE:0:0:103    ////__DACE:0:0:103
                                            // Tasklet code (tlet_57_deref_0)                                                 ////__DACE:0:0:103    ////__DACE:0:0:103
                                            __tlet_val = __tlet_field[((__temporal_extrapolation_of_perturbed_exner_K_stride * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:0:0:103    ////__DACE:0:0:103
                                            ///////////////////                                                               ////__DACE:0:0:103    ////__DACE:0:0:103
                                            ////__DACE:0:0:103                        ////__DACE:0:0:103
                                            __map_fusion_gtir_tmp_151_0 = __tlet_val;                                         ////__DACE:0:0:103    ////__DACE:0:0:103
                                        }                                             ////__DACE:0:0:103
                                        {                                             ////__DACE:0:0:102
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_149_0;                                 ////__DACE:0:0:112,102    ////__DACE:0:0:102
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_151_0;                                 ////__DACE:0:0:111,102    ////__DACE:0:0:102
                                            double __tlet_result;                                                             ////__DACE:0:0:102    ////__DACE:0:0:102
                                            ////__DACE:0:0:102                        ////__DACE:0:0:102
                                            ///////////////////                                                               ////__DACE:0:0:102    ////__DACE:0:0:102
                                            // Tasklet code (tlet_58_minus_0)                                                 ////__DACE:0:0:102    ////__DACE:0:0:102
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:102    ////__DACE:0:0:102
                                            ///////////////////                                                               ////__DACE:0:0:102    ////__DACE:0:0:102
                                            ////__DACE:0:0:102                        ////__DACE:0:0:102
                                            __map_fusion_gtir_tmp_153_0 = __tlet_result;                                      ////__DACE:0:0:102    ////__DACE:0:0:102
                                        }                                             ////__DACE:0:0:102
                                        {                                             ////__DACE:0:0:101
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_153_0;                                 ////__DACE:0:0:110,101    ////__DACE:0:0:101
                                            double __tlet_arg0 = inv_dual_edge_length[i_Edge_gtx_horizontal];                 ////__DACE:0:0:56,101    ////__DACE:0:0:101
                                            double __tlet_result;                                                             ////__DACE:0:0:101    ////__DACE:0:0:101
                                            ////__DACE:0:0:101                        ////__DACE:0:0:101
                                            ///////////////////                                                               ////__DACE:0:0:101    ////__DACE:0:0:101
                                            // Tasklet code (tlet_59_multiplies_0)                                            ////__DACE:0:0:101    ////__DACE:0:0:101
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:101    ////__DACE:0:0:101
                                            ///////////////////                                                               ////__DACE:0:0:101    ////__DACE:0:0:101
                                            ////__DACE:0:0:101                        ////__DACE:0:0:101
                                            __map_fusion_gtir_tmp_155_0 = __tlet_result;                                      ////__DACE:0:0:101    ////__DACE:0:0:101
                                        }                                             ////__DACE:0:0:101
                                        {                                             ////__DACE:0:0:92
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_155_0;                                 ////__DACE:0:0:109,92    ////__DACE:0:0:92
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_163_0;                                 ////__DACE:0:0:106,92    ////__DACE:0:0:92
                                            double __tlet_result;                                                             ////__DACE:0:0:92    ////__DACE:0:0:92
                                            ////__DACE:0:0:92                         ////__DACE:0:0:92
                                            ///////////////////                                                               ////__DACE:0:0:92    ////__DACE:0:0:92
                                            // Tasklet code (tlet_63_minus_0)                                                 ////__DACE:0:0:92    ////__DACE:0:0:92
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:92    ////__DACE:0:0:92
                                            ///////////////////                                                               ////__DACE:0:0:92    ////__DACE:0:0:92
                                            ////__DACE:0:0:92                         ////__DACE:0:0:92
                                            __map_fusion_gtir_tmp_139_split_1 = __tlet_result;                                ////__DACE:0:0:92    ////__DACE:0:0:92
                                        }                                             ////__DACE:0:0:92
                                        {                                             ////__DACE:0:0:133
                                            double __tlet_arg1 = gtir_tmp_166_1;                                              ////__DACE:0:0:284,133    ////__DACE:0:0:133
                                            double __tlet_arg0 = pg_exdist[((__pg_exdist_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:62,133    ////__DACE:0:0:133
                                            bool __tlet_result;                                                               ////__DACE:0:0:133    ////__DACE:0:0:133
                                            ////__DACE:0:0:133                        ////__DACE:0:0:133
                                            ///////////////////                                                               ////__DACE:0:0:133    ////__DACE:0:0:133
                                            // Tasklet code (tlet_65_not_eq_1)                                                ////__DACE:0:0:133    ////__DACE:0:0:133
                                            __tlet_result = (__tlet_arg0 != __tlet_arg1);                                     ////__DACE:0:0:133    ////__DACE:0:0:133
                                            ///////////////////                                                               ////__DACE:0:0:133    ////__DACE:0:0:133
                                            ////__DACE:0:0:133                        ////__DACE:0:0:133
                                            __map_fusion_gtir_tmp_168_1 = __tlet_result;                                      ////__DACE:0:0:133    ////__DACE:0:0:133
                                        }                                             ////__DACE:0:0:133
                                        if_stmt_6_0_0_132(__map_fusion_gtir_tmp_139_split_1, __map_fusion_gtir_tmp_168_1, &hydrostatic_correction_on_lowest_level[0], &pg_exdist[0], gtir_tmp_173_1, __pg_exdist_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:132
                                        {                                             ////__DACE:0:0:347
                                            double _cpy_in = gtir_tmp_173_1;                                                  ////__DACE:0:0:131,347    ////__DACE:0:0:347
                                            double _cpy_out;                                                                  ////__DACE:0:0:347    ////__DACE:0:0:347
                                            ////__DACE:0:0:347                        ////__DACE:0:0:347
                                            ///////////////////                                                               ////__DACE:0:0:347    ////__DACE:0:0:347
                                            // Tasklet code (copy_gtir_tmp_173_1_to_horizontal_pressure_gradient)             ////__DACE:0:0:347    ////__DACE:0:0:347
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:347    ////__DACE:0:0:347
                                            ///////////////////                                                               ////__DACE:0:0:347    ////__DACE:0:0:347
                                            ////__DACE:0:0:347                        ////__DACE:0:0:347
                                            horizontal_pressure_gradient[((__horizontal_pressure_gradient_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:347    ////__DACE:0:0:347
                                        }                                             ////__DACE:0:0:347
                                        {                                             ////__DACE:0:0:140
                                            double __tlet_arg0 = gtir_tmp_223_1;                                              ////__DACE:0:0:294,140    ////__DACE:0:0:140
                                            double __tlet_arg1 = theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:70,140    ////__DACE:0:0:140
                                            double __tlet_result;                                                             ////__DACE:0:0:140    ////__DACE:0:0:140
                                            ////__DACE:0:0:140                        ////__DACE:0:0:140
                                            ///////////////////                                                               ////__DACE:0:0:140    ////__DACE:0:0:140
                                            // Tasklet code (tlet_93_multiplies_1)                                            ////__DACE:0:0:140    ////__DACE:0:0:140
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:140    ////__DACE:0:0:140
                                            ///////////////////                                                               ////__DACE:0:0:140    ////__DACE:0:0:140
                                            ////__DACE:0:0:140                        ////__DACE:0:0:140
                                            __map_fusion_gtir_tmp_227_1 = __tlet_result;                                      ////__DACE:0:0:140    ////__DACE:0:0:140
                                        }                                             ////__DACE:0:0:140
                                        {                                             ////__DACE:0:0:139
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_227_1;                                 ////__DACE:0:0:146,139    ////__DACE:0:0:139
                                            double __tlet_arg1 = gtir_tmp_173_1;                                              ////__DACE:0:0:131,139    ////__DACE:0:0:139
                                            double __tlet_result;                                                             ////__DACE:0:0:139    ////__DACE:0:0:139
                                            ////__DACE:0:0:139                        ////__DACE:0:0:139
                                            ///////////////////                                                               ////__DACE:0:0:139    ////__DACE:0:0:139
                                            // Tasklet code (tlet_94_multiplies_1)                                            ////__DACE:0:0:139    ////__DACE:0:0:139
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:139    ////__DACE:0:0:139
                                            ///////////////////                                                               ////__DACE:0:0:139    ////__DACE:0:0:139
                                            ////__DACE:0:0:139                        ////__DACE:0:0:139
                                            __map_fusion_gtir_tmp_229_1 = __tlet_result;                                      ////__DACE:0:0:139    ////__DACE:0:0:139
                                        }                                             ////__DACE:0:0:139
                                        {                                             ////__DACE:0:0:138
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_229_1;                                 ////__DACE:0:0:145,138    ////__DACE:0:0:138
                                            double __tlet_arg0 = predictor_normal_wind_advective_tendency[((__predictor_normal_wind_advective_tendency_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:67,138    ////__DACE:0:0:138
                                            double __tlet_result;                                                             ////__DACE:0:0:138    ////__DACE:0:0:138
                                            ////__DACE:0:0:138                        ////__DACE:0:0:138
                                            ///////////////////                                                               ////__DACE:0:0:138    ////__DACE:0:0:138
                                            // Tasklet code (tlet_95_minus_1)                                                 ////__DACE:0:0:138    ////__DACE:0:0:138
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:138    ////__DACE:0:0:138
                                            ///////////////////                                                               ////__DACE:0:0:138    ////__DACE:0:0:138
                                            ////__DACE:0:0:138                        ////__DACE:0:0:138
                                            __map_fusion_gtir_tmp_231_1 = __tlet_result;                                      ////__DACE:0:0:138    ////__DACE:0:0:138
                                        }                                             ////__DACE:0:0:138
                                        {                                             ////__DACE:0:0:137
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_231_1;                                 ////__DACE:0:0:144,137    ////__DACE:0:0:137
                                            double __tlet_arg1 = normal_wind_tendency_due_to_slow_physics_process[((__normal_wind_tendency_due_to_slow_physics_process_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:66,137    ////__DACE:0:0:137
                                            double __tlet_result;                                                             ////__DACE:0:0:137    ////__DACE:0:0:137
                                            ////__DACE:0:0:137                        ////__DACE:0:0:137
                                            ///////////////////                                                               ////__DACE:0:0:137    ////__DACE:0:0:137
                                            // Tasklet code (tlet_96_plus_1)                                                  ////__DACE:0:0:137    ////__DACE:0:0:137
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:137    ////__DACE:0:0:137
                                            ///////////////////                                                               ////__DACE:0:0:137    ////__DACE:0:0:137
                                            ////__DACE:0:0:137                        ////__DACE:0:0:137
                                            __map_fusion_gtir_tmp_233_1 = __tlet_result;                                      ////__DACE:0:0:137    ////__DACE:0:0:137
                                        }                                             ////__DACE:0:0:137
                                        {                                             ////__DACE:0:0:136
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_233_1;                                 ////__DACE:0:0:143,136    ////__DACE:0:0:136
                                            double __tlet_arg0 = __dtime_0_1;                                                 ////__DACE:0:0:298,136    ////__DACE:0:0:136
                                            double __tlet_result;                                                             ////__DACE:0:0:136    ////__DACE:0:0:136
                                            ////__DACE:0:0:136                        ////__DACE:0:0:136
                                            ///////////////////                                                               ////__DACE:0:0:136    ////__DACE:0:0:136
                                            // Tasklet code (tlet_97_multiplies_1)                                            ////__DACE:0:0:136    ////__DACE:0:0:136
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:136    ////__DACE:0:0:136
                                            ///////////////////                                                               ////__DACE:0:0:136    ////__DACE:0:0:136
                                            ////__DACE:0:0:136                        ////__DACE:0:0:136
                                            __map_fusion_gtir_tmp_235_1 = __tlet_result;                                      ////__DACE:0:0:136    ////__DACE:0:0:136
                                        }                                             ////__DACE:0:0:136
                                        {                                             ////__DACE:0:0:135
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_235_1;                                 ////__DACE:0:0:142,135    ////__DACE:0:0:135
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,135    ////__DACE:0:0:135
                                            double __tlet_result;                                                             ////__DACE:0:0:135    ////__DACE:0:0:135
                                            ////__DACE:0:0:135                        ////__DACE:0:0:135
                                            ///////////////////                                                               ////__DACE:0:0:135    ////__DACE:0:0:135
                                            // Tasklet code (tlet_98_plus_1)                                                  ////__DACE:0:0:135    ////__DACE:0:0:135
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:135    ////__DACE:0:0:135
                                            ///////////////////                                                               ////__DACE:0:0:135    ////__DACE:0:0:135
                                            ////__DACE:0:0:135                        ////__DACE:0:0:135
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:135    ////__DACE:0:0:135
                                        }                                             ////__DACE:0:0:135
                                    }                                                 ////__DACE:0:0:250
                                }                                                     ////__DACE:0:0:250
                            }                                                         ////__DACE:0:0:134
                        }                                                             ////__DACE:0:0:134
                    }                                                                 ////__DACE:0:0:134
                }                                                                     ////__DACE:0:0:134
            }                                                                         ////__DACE:0:0:134
        }                                                                             ////__DACE:0:0:359
    }                                                                                 ////__DACE:0:0:359
}                                                                                 ////__DACE:0:0:359

                                                                                  ////__DACE:0:0:358
DACE_EXPORTED void __dace_runkernel_map_105_fieldop_1_0_0_358(theta_shared_probe_native_state_t *__state, const double * __restrict__ c_lin_e, const double * __restrict__ current_vn, const double * __restrict__ ddxn_z_full, const double * __restrict__ ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, const int * __restrict__ gt_conn_E2C, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ pg_exdist, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, const double * __restrict__ theta_v_at_edges_on_model_levels, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __pg_exdist_K_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:358
void __dace_runkernel_map_105_fieldop_1_0_0_358(theta_shared_probe_native_state_t *__state, const double * __restrict__ c_lin_e, const double * __restrict__ current_vn, const double * __restrict__ ddxn_z_full, const double * __restrict__ ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, const int * __restrict__ gt_conn_E2C, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ pg_exdist, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, const double * __restrict__ theta_v_at_edges_on_model_levels, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __pg_exdist_K_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:358
{                                                                                 ////__DACE:0:0:358
                                                                                  ////__DACE:0:0:358
    void  *map_105_fieldop_1_0_0_358_args[] = { (void *)&c_lin_e, (void *)&current_vn, (void *)&ddxn_z_full, (void *)&ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, (void *)&gt_conn_E2C, (void *)&horizontal_pressure_gradient, (void *)&hydrostatic_correction_on_lowest_level, (void *)&inv_dual_edge_length, (void *)&next_vn, (void *)&normal_wind_tendency_due_to_slow_physics_process, (void *)&pg_exdist, (void *)&predictor_normal_wind_advective_tendency, (void *)&temporal_extrapolation_of_perturbed_exner, (void *)&theta_v_at_edges_on_model_levels, (void *)&__c_lin_e_E2C_stride, (void *)&__current_vn_K_stride, (void *)&__ddxn_z_full_K_stride, (void *)&__ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, (void *)&__gt_conn_E2C_neighbor_stride, (void *)&__horizontal_pressure_gradient_K_stride, (void *)&__next_vn_K_stride, (void *)&__normal_wind_tendency_due_to_slow_physics_process_K_stride, (void *)&__pg_exdist_K_stride, (void *)&__predictor_normal_wind_advective_tendency_K_stride, (void *)&__temporal_extrapolation_of_perturbed_exner_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:358
    gpuError_t __err = hipLaunchKernel((void*)map_105_fieldop_1_0_0_358, dim3(233, 24, 1), dim3(256, 1, 1), map_105_fieldop_1_0_0_358_args, 0, nullptr);    ////__DACE:0:0:358
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_105_fieldop_1_0_0_358", 233, 24, 1, 256, 1, 1);
}

