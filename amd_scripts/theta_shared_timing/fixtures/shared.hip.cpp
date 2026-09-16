
#include <hip/hip_runtime.h>
#include <dace/dace.h>

                                                                                  ////__DACE:0
struct theta_shared_probe_shared_state_t {                                        ////__DACE:0
    dace::cuda::Context *gpu_context;                                             ////__DACE:0
};                                                                                ////__DACE:0
                                                                                  ////__DACE:0


DACE_EXPORTED int __dace_init_cuda(theta_shared_probe_shared_state_t *__state, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __grf_tend_vn_K_stride, int __gt_conn_C2E2CO_neighbor_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __ikoffset_E2C_stride, int __ikoffset_K_stride, int __next_vn_K_stride, int __normal_wind_iau_increment_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, int __zdiff_gradp_E2C_stride, int __zdiff_gradp_K_stride, double dtime);
DACE_EXPORTED int __dace_exit_cuda(theta_shared_probe_shared_state_t *__state);
DACE_EXPORTED int __dace_gpu_last_error(theta_shared_probe_shared_state_t *__state);
DACE_EXPORTED void __dace_gpu_drain_error(theta_shared_probe_shared_state_t *__state);
DACE_EXPORTED bool __dace_gpu_set_stream(theta_shared_probe_shared_state_t *__state, int streamid, gpuStream_t stream);
DACE_EXPORTED void __dace_gpu_set_all_streams(theta_shared_probe_shared_state_t *__state, gpuStream_t stream);

DACE_DFI void reduce_0_0_358(double* __restrict__ _in, double&  _out) {           ////__DACE:0:0:358
    ////__DACE:72
    {                                                                                 ////__DACE:72
        ////__DACE:72
        {                                                                                 ////__DACE:72:0:0    ////__DACE:72
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {                                       ////__DACE:72:0:0    ////__DACE:72
                {                                                                         ////__DACE:72:0:1    ////__DACE:72
                    double __out;                                                                     ////__DACE:72:0:1    ////__DACE:72:0:1    ////__DACE:72
                    ////__DACE:72:0:1                                                     ////__DACE:72:0:1    ////__DACE:72
                    ///////////////////                                                               ////__DACE:72:0:1    ////__DACE:72:0:1    ////__DACE:72
                    // Tasklet code (reduce_init)                                                     ////__DACE:72:0:1    ////__DACE:72:0:1    ////__DACE:72
                    __out = 0;                                                                        ////__DACE:72:0:1    ////__DACE:72:0:1    ////__DACE:72
                    ///////////////////                                                               ////__DACE:72:0:1    ////__DACE:72:0:1    ////__DACE:72
                    ////__DACE:72:0:1                                                     ////__DACE:72:0:1    ////__DACE:72
                    _out = __out;                                                                     ////__DACE:72:0:1    ////__DACE:72:0:1    ////__DACE:72
                }                                                                         ////__DACE:72:0:1    ////__DACE:72
            }                                                                             ////__DACE:72:0:2    ////__DACE:72
        }                                                                                 ////__DACE:72:0:2    ////__DACE:72
        ////__DACE:72
    }                                                                                 ////__DACE:72
    {                                                                                 ////__DACE:72
        ////__DACE:72
        {                                                                                 ////__DACE:72:1:0    ////__DACE:72
            for (auto _i0 = 0; _i0 < 4; _i0 += 1) {                                       ////__DACE:72:1:0    ////__DACE:72
                {                                                                         ////__DACE:72:1:2    ////__DACE:72
                    double __inp = _in[_i0];                                                          ////__DACE:72:1:3,2    ////__DACE:72:1:2    ////__DACE:72
                    double __out;                                                                     ////__DACE:72:1:2    ////__DACE:72:1:2    ////__DACE:72
                    ////__DACE:72:1:2                                                     ////__DACE:72:1:2    ////__DACE:72
                    ///////////////////                                                               ////__DACE:72:1:2    ////__DACE:72:1:2    ////__DACE:72
                    // Tasklet code (identity)                                                        ////__DACE:72:1:2    ////__DACE:72:1:2    ////__DACE:72
                    __out = __inp;                                                                    ////__DACE:72:1:2    ////__DACE:72:1:2    ////__DACE:72
                    ///////////////////                                                               ////__DACE:72:1:2    ////__DACE:72:1:2    ////__DACE:72
                    ////__DACE:72:1:2                                                     ////__DACE:72:1:2    ////__DACE:72
                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(&_out, __out);          ////__DACE:72:1:2    ////__DACE:72:1:2    ////__DACE:72
                }                                                                         ////__DACE:72:1:2    ////__DACE:72
            }                                                                             ////__DACE:72:1:1    ////__DACE:72
        }                                                                                 ////__DACE:72:1:1    ////__DACE:72
        ////__DACE:72
    }                                                                                 ////__DACE:72
}                                                                                 ////__DACE:0:0:358
////__DACE:0:0:358
DACE_DFI void if_stmt_0_0_0_158(const bool&  __arg1___, const bool&  __arg2, const bool&  __cond, bool&  __output) {    ////__DACE:0:0:158
    ////__DACE:25
    if (__cond) {                                                                     ////__DACE:25
        {                                                                             ////__DACE:25
            ////__DACE:25
            {                                                                                 ////__DACE:27:0:2    ////__DACE:25
                bool _cpy_in = __arg1___;                                                         ////__DACE:27:0:1,2    ////__DACE:27:0:2    ////__DACE:25
                bool _cpy_out;                                                                    ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                ////__DACE:27:0:2                                                             ////__DACE:27:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                _cpy_out = _cpy_in;                                                               ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
                ////__DACE:27:0:2                                                             ////__DACE:27:0:2    ////__DACE:25
                __output = _cpy_out;                                                              ////__DACE:27:0:2    ////__DACE:27:0:2    ////__DACE:25
            }                                                                                 ////__DACE:27:0:2    ////__DACE:25
            ////__DACE:25
        }                                                                             ////__DACE:25
    } else {                                                                          ////__DACE:25
        {                                                                             ////__DACE:25
            ////__DACE:25
            {                                                                                 ////__DACE:28:0:2    ////__DACE:25
                bool _cpy_in = __arg2;                                                            ////__DACE:28:0:1,2    ////__DACE:28:0:2    ////__DACE:25
                bool _cpy_out;                                                                    ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                ////__DACE:28:0:2                                                             ////__DACE:28:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                _cpy_out = _cpy_in;                                                               ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                ///////////////////                                                               ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
                ////__DACE:28:0:2                                                             ////__DACE:28:0:2    ////__DACE:25
                __output = _cpy_out;                                                              ////__DACE:28:0:2    ////__DACE:28:0:2    ////__DACE:25
            }                                                                                 ////__DACE:28:0:2    ////__DACE:25
            ////__DACE:25
        }                                                                             ////__DACE:25
    }                                                                                 ////__DACE:25
}                                                                                 ////__DACE:0:0:158
////__DACE:0:0:158
DACE_DFI void if_stmt_1_0_0_163(const double&  __arg1___, const double&  __arg1____from_cb_fusion_3, const double&  __arg2, const double&  __arg2_from_cb_fusion_3, const bool&  __cond, double&  __output, double&  __output_from_cb_fusion_3) {    ////__DACE:0:0:163
    ////__DACE:29
    if (__cond) {                                                                     ////__DACE:29
        {                                                                             ////__DACE:29
            ////__DACE:29
            {                                                                                 ////__DACE:31:0:4    ////__DACE:29
                double _cpy_in = __arg1___;                                                       ////__DACE:31:0:1,4    ////__DACE:31:0:4    ////__DACE:29
                double _cpy_out;                                                                  ////__DACE:31:0:4    ////__DACE:31:0:4    ////__DACE:29
                ////__DACE:31:0:4                                                             ////__DACE:31:0:4    ////__DACE:29
                ///////////////////                                                               ////__DACE:31:0:4    ////__DACE:31:0:4    ////__DACE:29
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:31:0:4    ////__DACE:31:0:4    ////__DACE:29
                _cpy_out = _cpy_in;                                                               ////__DACE:31:0:4    ////__DACE:31:0:4    ////__DACE:29
                ///////////////////                                                               ////__DACE:31:0:4    ////__DACE:31:0:4    ////__DACE:29
                ////__DACE:31:0:4                                                             ////__DACE:31:0:4    ////__DACE:29
                __output = _cpy_out;                                                              ////__DACE:31:0:4    ////__DACE:31:0:4    ////__DACE:29
            }                                                                                 ////__DACE:31:0:4    ////__DACE:29
            {                                                                                 ////__DACE:31:0:5    ////__DACE:29
                double _cpy_in = __arg1____from_cb_fusion_3;                                      ////__DACE:31:0:3,5    ////__DACE:31:0:5    ////__DACE:29
                double _cpy_out;                                                                  ////__DACE:31:0:5    ////__DACE:31:0:5    ////__DACE:29
                ////__DACE:31:0:5                                                             ////__DACE:31:0:5    ////__DACE:29
                ///////////////////                                                               ////__DACE:31:0:5    ////__DACE:31:0:5    ////__DACE:29
                // Tasklet code (copy___arg1____from_cb_fusion_3_to___output_from_cb_fusion_3)    ////__DACE:31:0:5    ////__DACE:31:0:5    ////__DACE:29
                _cpy_out = _cpy_in;                                                               ////__DACE:31:0:5    ////__DACE:31:0:5    ////__DACE:29
                ///////////////////                                                               ////__DACE:31:0:5    ////__DACE:31:0:5    ////__DACE:29
                ////__DACE:31:0:5                                                             ////__DACE:31:0:5    ////__DACE:29
                __output_from_cb_fusion_3 = _cpy_out;                                             ////__DACE:31:0:5    ////__DACE:31:0:5    ////__DACE:29
            }                                                                                 ////__DACE:31:0:5    ////__DACE:29
            ////__DACE:29
        }                                                                             ////__DACE:29
    } else {                                                                          ////__DACE:29
        {                                                                             ////__DACE:29
            ////__DACE:29
            {                                                                                 ////__DACE:32:0:4    ////__DACE:29
                double _cpy_in = __arg2;                                                          ////__DACE:32:0:1,4    ////__DACE:32:0:4    ////__DACE:29
                double _cpy_out;                                                                  ////__DACE:32:0:4    ////__DACE:32:0:4    ////__DACE:29
                ////__DACE:32:0:4                                                             ////__DACE:32:0:4    ////__DACE:29
                ///////////////////                                                               ////__DACE:32:0:4    ////__DACE:32:0:4    ////__DACE:29
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:32:0:4    ////__DACE:32:0:4    ////__DACE:29
                _cpy_out = _cpy_in;                                                               ////__DACE:32:0:4    ////__DACE:32:0:4    ////__DACE:29
                ///////////////////                                                               ////__DACE:32:0:4    ////__DACE:32:0:4    ////__DACE:29
                ////__DACE:32:0:4                                                             ////__DACE:32:0:4    ////__DACE:29
                __output = _cpy_out;                                                              ////__DACE:32:0:4    ////__DACE:32:0:4    ////__DACE:29
            }                                                                                 ////__DACE:32:0:4    ////__DACE:29
            {                                                                                 ////__DACE:32:0:5    ////__DACE:29
                double _cpy_in = __arg2_from_cb_fusion_3;                                         ////__DACE:32:0:3,5    ////__DACE:32:0:5    ////__DACE:29
                double _cpy_out;                                                                  ////__DACE:32:0:5    ////__DACE:32:0:5    ////__DACE:29
                ////__DACE:32:0:5                                                             ////__DACE:32:0:5    ////__DACE:29
                ///////////////////                                                               ////__DACE:32:0:5    ////__DACE:32:0:5    ////__DACE:29
                // Tasklet code (copy___arg2_from_cb_fusion_3_to___output_from_cb_fusion_3)       ////__DACE:32:0:5    ////__DACE:32:0:5    ////__DACE:29
                _cpy_out = _cpy_in;                                                               ////__DACE:32:0:5    ////__DACE:32:0:5    ////__DACE:29
                ///////////////////                                                               ////__DACE:32:0:5    ////__DACE:32:0:5    ////__DACE:29
                ////__DACE:32:0:5                                                             ////__DACE:32:0:5    ////__DACE:29
                __output_from_cb_fusion_3 = _cpy_out;                                             ////__DACE:32:0:5    ////__DACE:32:0:5    ////__DACE:29
            }                                                                                 ////__DACE:32:0:5    ////__DACE:29
            ////__DACE:29
        }                                                                             ////__DACE:29
    }                                                                                 ////__DACE:29
}                                                                                 ////__DACE:0:0:163
////__DACE:0:0:163
DACE_DFI void if_stmt_4_0_0_174(const bool&  __cond, const double&  __map_fusion_gtir_tmp_21_1_1_0_0, const double&  __map_fusion_gtir_tmp_33_1_1_0_0, const double&  gtir_tmp_34_1_0, const double&  gtir_tmp_38_1_0, const double&  gtir_tmp_44_1_0, const double&  gtir_tmp_48_1_0, const double&  gtir_tmp_56_1_0, const double&  gtir_tmp_60_1_0, const double&  gtir_tmp_66_1_0, const double&  gtir_tmp_70_1_0, double&  __output, double&  __output_from_cb_fusion_2) {    ////__DACE:0:0:174
    ////__DACE:33
    if (__cond) {                                                                     ////__DACE:33
        {                                                                             ////__DACE:33
            double __arg1____;                                                                ////__DACE:35:0:1    ////__DACE:33
            double __arg1____from_cb_fusion_2;                                                ////__DACE:35:0:3    ////__DACE:33
            double __map_fusion_gtir_tmp_63_1_0;                                              ////__DACE:35:0:5    ////__DACE:33
            double __map_fusion_gtir_tmp_59_1_0;                                              ////__DACE:35:0:7    ////__DACE:33
            double __map_fusion_gtir_tmp_41_1_0;                                              ////__DACE:35:0:10    ////__DACE:33
            double __map_fusion_gtir_tmp_37_1_0;                                              ////__DACE:35:0:12    ////__DACE:33
            ////__DACE:33
            {                                                                                 ////__DACE:35:0:6    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_60_1_0;                                             ////__DACE:35:0:14,6    ////__DACE:35:0:6    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_0_0;                            ////__DACE:35:0:15,6    ////__DACE:35:0:6    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:35:0:6    ////__DACE:35:0:6    ////__DACE:33
                ////__DACE:35:0:6                                                             ////__DACE:35:0:6    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:6    ////__DACE:35:0:6    ////__DACE:33
                // Tasklet code (tlet_21_multiplies_1_0)                                          ////__DACE:35:0:6    ////__DACE:35:0:6    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:35:0:6    ////__DACE:35:0:6    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:6    ////__DACE:35:0:6    ////__DACE:33
                ////__DACE:35:0:6                                                             ////__DACE:35:0:6    ////__DACE:33
                __map_fusion_gtir_tmp_63_1_0 = __tlet_result;                                     ////__DACE:35:0:6    ////__DACE:35:0:6    ////__DACE:33
            }                                                                                 ////__DACE:35:0:6    ////__DACE:33
            {                                                                                 ////__DACE:35:0:8    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_56_1_0;                                             ////__DACE:35:0:16,8    ////__DACE:35:0:8    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_0_0;                            ////__DACE:35:0:17,8    ////__DACE:35:0:8    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:35:0:8    ////__DACE:35:0:8    ////__DACE:33
                ////__DACE:35:0:8                                                             ////__DACE:35:0:8    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:8    ////__DACE:35:0:8    ////__DACE:33
                // Tasklet code (tlet_20_multiplies_1_0)                                          ////__DACE:35:0:8    ////__DACE:35:0:8    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:35:0:8    ////__DACE:35:0:8    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:8    ////__DACE:35:0:8    ////__DACE:33
                ////__DACE:35:0:8                                                             ////__DACE:35:0:8    ////__DACE:33
                __map_fusion_gtir_tmp_59_1_0 = __tlet_result;                                     ////__DACE:35:0:8    ////__DACE:35:0:8    ////__DACE:33
            }                                                                                 ////__DACE:35:0:8    ////__DACE:33
            {                                                                                 ////__DACE:35:0:4    ////__DACE:33
                double __tlet_arg1 = __map_fusion_gtir_tmp_63_1_0;                                ////__DACE:35:0:5,4    ////__DACE:35:0:4    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_59_1_0;                                ////__DACE:35:0:7,4    ////__DACE:35:0:4    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                ////__DACE:35:0:4                                                             ////__DACE:35:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                // Tasklet code (tlet_22_plus_1_0)                                                ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
                ////__DACE:35:0:4                                                             ////__DACE:35:0:4    ////__DACE:33
                __arg1____ = __tlet_result;                                                       ////__DACE:35:0:4    ////__DACE:35:0:4    ////__DACE:33
            }                                                                                 ////__DACE:35:0:4    ////__DACE:33
            {                                                                                 ////__DACE:35:0:20    ////__DACE:33
                double _cpy_in = __arg1____;                                                      ////__DACE:35:0:1,20    ////__DACE:35:0:20    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:35:0:20    ////__DACE:35:0:20    ////__DACE:33
                ////__DACE:35:0:20                                                            ////__DACE:35:0:20    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:20    ////__DACE:35:0:20    ////__DACE:33
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:35:0:20    ////__DACE:35:0:20    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:35:0:20    ////__DACE:35:0:20    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:20    ////__DACE:35:0:20    ////__DACE:33
                ////__DACE:35:0:20                                                            ////__DACE:35:0:20    ////__DACE:33
                __output = _cpy_out;                                                              ////__DACE:35:0:20    ////__DACE:35:0:20    ////__DACE:33
            }                                                                                 ////__DACE:35:0:20    ////__DACE:33
            {                                                                                 ////__DACE:35:0:11    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_0_0;                            ////__DACE:35:0:15,11    ////__DACE:35:0:11    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_38_1_0;                                             ////__DACE:35:0:18,11    ////__DACE:35:0:11    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:35:0:11    ////__DACE:35:0:11    ////__DACE:33
                ////__DACE:35:0:11                                                            ////__DACE:35:0:11    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:11    ////__DACE:35:0:11    ////__DACE:33
                // Tasklet code (tlet_15_multiplies_1_0)                                          ////__DACE:35:0:11    ////__DACE:35:0:11    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:35:0:11    ////__DACE:35:0:11    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:11    ////__DACE:35:0:11    ////__DACE:33
                ////__DACE:35:0:11                                                            ////__DACE:35:0:11    ////__DACE:33
                __map_fusion_gtir_tmp_41_1_0 = __tlet_result;                                     ////__DACE:35:0:11    ////__DACE:35:0:11    ////__DACE:33
            }                                                                                 ////__DACE:35:0:11    ////__DACE:33
            {                                                                                 ////__DACE:35:0:13    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_0_0;                            ////__DACE:35:0:17,13    ////__DACE:35:0:13    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_34_1_0;                                             ////__DACE:35:0:19,13    ////__DACE:35:0:13    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:35:0:13    ////__DACE:35:0:13    ////__DACE:33
                ////__DACE:35:0:13                                                            ////__DACE:35:0:13    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:13    ////__DACE:35:0:13    ////__DACE:33
                // Tasklet code (tlet_14_multiplies_1_0)                                          ////__DACE:35:0:13    ////__DACE:35:0:13    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:35:0:13    ////__DACE:35:0:13    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:13    ////__DACE:35:0:13    ////__DACE:33
                ////__DACE:35:0:13                                                            ////__DACE:35:0:13    ////__DACE:33
                __map_fusion_gtir_tmp_37_1_0 = __tlet_result;                                     ////__DACE:35:0:13    ////__DACE:35:0:13    ////__DACE:33
            }                                                                                 ////__DACE:35:0:13    ////__DACE:33
            {                                                                                 ////__DACE:35:0:9    ////__DACE:33
                double __tlet_arg1 = __map_fusion_gtir_tmp_41_1_0;                                ////__DACE:35:0:10,9    ////__DACE:35:0:9    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_37_1_0;                                ////__DACE:35:0:12,9    ////__DACE:35:0:9    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:35:0:9    ////__DACE:35:0:9    ////__DACE:33
                ////__DACE:35:0:9                                                             ////__DACE:35:0:9    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:9    ////__DACE:35:0:9    ////__DACE:33
                // Tasklet code (tlet_16_plus_1_0)                                                ////__DACE:35:0:9    ////__DACE:35:0:9    ////__DACE:33
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:35:0:9    ////__DACE:35:0:9    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:9    ////__DACE:35:0:9    ////__DACE:33
                ////__DACE:35:0:9                                                             ////__DACE:35:0:9    ////__DACE:33
                __arg1____from_cb_fusion_2 = __tlet_result;                                       ////__DACE:35:0:9    ////__DACE:35:0:9    ////__DACE:33
            }                                                                                 ////__DACE:35:0:9    ////__DACE:33
            {                                                                                 ////__DACE:35:0:21    ////__DACE:33
                double _cpy_in = __arg1____from_cb_fusion_2;                                      ////__DACE:35:0:3,21    ////__DACE:35:0:21    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:35:0:21    ////__DACE:35:0:21    ////__DACE:33
                ////__DACE:35:0:21                                                            ////__DACE:35:0:21    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:21    ////__DACE:35:0:21    ////__DACE:33
                // Tasklet code (copy___arg1____from_cb_fusion_2_to___output_from_cb_fusion_2)    ////__DACE:35:0:21    ////__DACE:35:0:21    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:35:0:21    ////__DACE:35:0:21    ////__DACE:33
                ///////////////////                                                               ////__DACE:35:0:21    ////__DACE:35:0:21    ////__DACE:33
                ////__DACE:35:0:21                                                            ////__DACE:35:0:21    ////__DACE:33
                __output_from_cb_fusion_2 = _cpy_out;                                             ////__DACE:35:0:21    ////__DACE:35:0:21    ////__DACE:33
            }                                                                                 ////__DACE:35:0:21    ////__DACE:33
            ////__DACE:33
        }                                                                             ////__DACE:33
    } else {                                                                          ////__DACE:33
        {                                                                             ////__DACE:33
            double __arg2_;                                                                   ////__DACE:36:0:1    ////__DACE:33
            double __arg2_from_cb_fusion_2;                                                   ////__DACE:36:0:3    ////__DACE:33
            double __map_fusion_gtir_tmp_73_1_0;                                              ////__DACE:36:0:5    ////__DACE:33
            double __map_fusion_gtir_tmp_69_1_0;                                              ////__DACE:36:0:7    ////__DACE:33
            double __map_fusion_gtir_tmp_51_1_0;                                              ////__DACE:36:0:10    ////__DACE:33
            double __map_fusion_gtir_tmp_47_1_0;                                              ////__DACE:36:0:12    ////__DACE:33
            ////__DACE:33
            {                                                                                 ////__DACE:36:0:6    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_70_1_0;                                             ////__DACE:36:0:14,6    ////__DACE:36:0:6    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_0_0;                            ////__DACE:36:0:15,6    ////__DACE:36:0:6    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:36:0:6    ////__DACE:36:0:6    ////__DACE:33
                ////__DACE:36:0:6                                                             ////__DACE:36:0:6    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:6    ////__DACE:36:0:6    ////__DACE:33
                // Tasklet code (tlet_24_multiplies_1_0)                                          ////__DACE:36:0:6    ////__DACE:36:0:6    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:36:0:6    ////__DACE:36:0:6    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:6    ////__DACE:36:0:6    ////__DACE:33
                ////__DACE:36:0:6                                                             ////__DACE:36:0:6    ////__DACE:33
                __map_fusion_gtir_tmp_73_1_0 = __tlet_result;                                     ////__DACE:36:0:6    ////__DACE:36:0:6    ////__DACE:33
            }                                                                                 ////__DACE:36:0:6    ////__DACE:33
            {                                                                                 ////__DACE:36:0:8    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_66_1_0;                                             ////__DACE:36:0:16,8    ////__DACE:36:0:8    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_0_0;                            ////__DACE:36:0:17,8    ////__DACE:36:0:8    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:36:0:8    ////__DACE:36:0:8    ////__DACE:33
                ////__DACE:36:0:8                                                             ////__DACE:36:0:8    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:8    ////__DACE:36:0:8    ////__DACE:33
                // Tasklet code (tlet_23_multiplies_1_0)                                          ////__DACE:36:0:8    ////__DACE:36:0:8    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:36:0:8    ////__DACE:36:0:8    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:8    ////__DACE:36:0:8    ////__DACE:33
                ////__DACE:36:0:8                                                             ////__DACE:36:0:8    ////__DACE:33
                __map_fusion_gtir_tmp_69_1_0 = __tlet_result;                                     ////__DACE:36:0:8    ////__DACE:36:0:8    ////__DACE:33
            }                                                                                 ////__DACE:36:0:8    ////__DACE:33
            {                                                                                 ////__DACE:36:0:4    ////__DACE:33
                double __tlet_arg1 = __map_fusion_gtir_tmp_73_1_0;                                ////__DACE:36:0:5,4    ////__DACE:36:0:4    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_69_1_0;                                ////__DACE:36:0:7,4    ////__DACE:36:0:4    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                ////__DACE:36:0:4                                                             ////__DACE:36:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                // Tasklet code (tlet_25_plus_1_0)                                                ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
                ////__DACE:36:0:4                                                             ////__DACE:36:0:4    ////__DACE:33
                __arg2_ = __tlet_result;                                                          ////__DACE:36:0:4    ////__DACE:36:0:4    ////__DACE:33
            }                                                                                 ////__DACE:36:0:4    ////__DACE:33
            {                                                                                 ////__DACE:36:0:20    ////__DACE:33
                double _cpy_in = __arg2_;                                                         ////__DACE:36:0:1,20    ////__DACE:36:0:20    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:36:0:20    ////__DACE:36:0:20    ////__DACE:33
                ////__DACE:36:0:20                                                            ////__DACE:36:0:20    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:20    ////__DACE:36:0:20    ////__DACE:33
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:36:0:20    ////__DACE:36:0:20    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:36:0:20    ////__DACE:36:0:20    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:20    ////__DACE:36:0:20    ////__DACE:33
                ////__DACE:36:0:20                                                            ////__DACE:36:0:20    ////__DACE:33
                __output = _cpy_out;                                                              ////__DACE:36:0:20    ////__DACE:36:0:20    ////__DACE:33
            }                                                                                 ////__DACE:36:0:20    ////__DACE:33
            {                                                                                 ////__DACE:36:0:11    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_48_1_0;                                             ////__DACE:36:0:18,11    ////__DACE:36:0:11    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_0_0;                            ////__DACE:36:0:15,11    ////__DACE:36:0:11    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:36:0:11    ////__DACE:36:0:11    ////__DACE:33
                ////__DACE:36:0:11                                                            ////__DACE:36:0:11    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:11    ////__DACE:36:0:11    ////__DACE:33
                // Tasklet code (tlet_18_multiplies_1_0)                                          ////__DACE:36:0:11    ////__DACE:36:0:11    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:36:0:11    ////__DACE:36:0:11    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:11    ////__DACE:36:0:11    ////__DACE:33
                ////__DACE:36:0:11                                                            ////__DACE:36:0:11    ////__DACE:33
                __map_fusion_gtir_tmp_51_1_0 = __tlet_result;                                     ////__DACE:36:0:11    ////__DACE:36:0:11    ////__DACE:33
            }                                                                                 ////__DACE:36:0:11    ////__DACE:33
            {                                                                                 ////__DACE:36:0:13    ////__DACE:33
                double __tlet_arg1 = gtir_tmp_44_1_0;                                             ////__DACE:36:0:19,13    ////__DACE:36:0:13    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_0_0;                            ////__DACE:36:0:17,13    ////__DACE:36:0:13    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:36:0:13    ////__DACE:36:0:13    ////__DACE:33
                ////__DACE:36:0:13                                                            ////__DACE:36:0:13    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:13    ////__DACE:36:0:13    ////__DACE:33
                // Tasklet code (tlet_17_multiplies_1_0)                                          ////__DACE:36:0:13    ////__DACE:36:0:13    ////__DACE:33
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:36:0:13    ////__DACE:36:0:13    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:13    ////__DACE:36:0:13    ////__DACE:33
                ////__DACE:36:0:13                                                            ////__DACE:36:0:13    ////__DACE:33
                __map_fusion_gtir_tmp_47_1_0 = __tlet_result;                                     ////__DACE:36:0:13    ////__DACE:36:0:13    ////__DACE:33
            }                                                                                 ////__DACE:36:0:13    ////__DACE:33
            {                                                                                 ////__DACE:36:0:9    ////__DACE:33
                double __tlet_arg1 = __map_fusion_gtir_tmp_51_1_0;                                ////__DACE:36:0:10,9    ////__DACE:36:0:9    ////__DACE:33
                double __tlet_arg0 = __map_fusion_gtir_tmp_47_1_0;                                ////__DACE:36:0:12,9    ////__DACE:36:0:9    ////__DACE:33
                double __tlet_result;                                                             ////__DACE:36:0:9    ////__DACE:36:0:9    ////__DACE:33
                ////__DACE:36:0:9                                                             ////__DACE:36:0:9    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:9    ////__DACE:36:0:9    ////__DACE:33
                // Tasklet code (tlet_19_plus_1_0)                                                ////__DACE:36:0:9    ////__DACE:36:0:9    ////__DACE:33
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:36:0:9    ////__DACE:36:0:9    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:9    ////__DACE:36:0:9    ////__DACE:33
                ////__DACE:36:0:9                                                             ////__DACE:36:0:9    ////__DACE:33
                __arg2_from_cb_fusion_2 = __tlet_result;                                          ////__DACE:36:0:9    ////__DACE:36:0:9    ////__DACE:33
            }                                                                                 ////__DACE:36:0:9    ////__DACE:33
            {                                                                                 ////__DACE:36:0:21    ////__DACE:33
                double _cpy_in = __arg2_from_cb_fusion_2;                                         ////__DACE:36:0:3,21    ////__DACE:36:0:21    ////__DACE:33
                double _cpy_out;                                                                  ////__DACE:36:0:21    ////__DACE:36:0:21    ////__DACE:33
                ////__DACE:36:0:21                                                            ////__DACE:36:0:21    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:21    ////__DACE:36:0:21    ////__DACE:33
                // Tasklet code (copy___arg2_from_cb_fusion_2_to___output_from_cb_fusion_2)       ////__DACE:36:0:21    ////__DACE:36:0:21    ////__DACE:33
                _cpy_out = _cpy_in;                                                               ////__DACE:36:0:21    ////__DACE:36:0:21    ////__DACE:33
                ///////////////////                                                               ////__DACE:36:0:21    ////__DACE:36:0:21    ////__DACE:33
                ////__DACE:36:0:21                                                            ////__DACE:36:0:21    ////__DACE:33
                __output_from_cb_fusion_2 = _cpy_out;                                             ////__DACE:36:0:21    ////__DACE:36:0:21    ////__DACE:33
            }                                                                                 ////__DACE:36:0:21    ////__DACE:33
            ////__DACE:33
        }                                                                             ////__DACE:33
    }                                                                                 ////__DACE:33
}                                                                                 ////__DACE:0:0:174
////__DACE:0:0:174
DACE_DFI void if_stmt_7_0_0_189(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double&  gtir_tmp_54_1_0, const double&  gtir_tmp_76_1_0, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double* __restrict__ perturbed_rho_at_cells_on_model_levels, const double* __restrict__ reference_rho_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_rho_at_cells_on_model_levels_K_stride_0, int __reference_rho_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:189
    ////__DACE:41
    if (__cond) {                                                                     ////__DACE:41
        {                                                                             ////__DACE:41
            double __arg1___;                                                                 ////__DACE:43:0:1    ////__DACE:41
            double __map_fusion_gtir_tmp_191_1_0;                                             ////__DACE:43:0:3    ////__DACE:41
            double __map_fusion_gtir_tmp_189_1_0;                                             ////__DACE:43:0:5    ////__DACE:41
            double __map_fusion_gtir_tmp_187_1_0;                                             ////__DACE:43:0:7    ////__DACE:41
            double __map_fusion_gtir_tmp_185_1_0;                                             ////__DACE:43:0:9    ////__DACE:41
            double __map_fusion_gtir_tmp_183_1_0;                                             ////__DACE:43:0:11    ////__DACE:41
            double __map_fusion_gtir_tmp_181_1_0;                                             ////__DACE:43:0:13    ////__DACE:41
            double __map_fusion_gtir_tmp_179_1_0;                                             ////__DACE:43:0:15    ////__DACE:41
            ////__DACE:41
            {                                                                                 ////__DACE:43:0:6    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:43:0:18,6    ////__DACE:43:0:6    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:43:0:19,6    ////__DACE:43:0:6    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                ////__DACE:43:0:6                                                             ////__DACE:43:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                // Tasklet code (tlet_75_deref_1_0)                                               ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
                ////__DACE:43:0:6                                                             ////__DACE:43:0:6    ////__DACE:41
                __map_fusion_gtir_tmp_189_1_0 = __tlet_val;                                       ////__DACE:43:0:6    ////__DACE:43:0:6    ////__DACE:41
            }                                                                                 ////__DACE:43:0:6    ////__DACE:41
            {                                                                                 ////__DACE:43:0:4    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_76_1_0;                                             ////__DACE:43:0:17,4    ////__DACE:43:0:4    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_189_1_0;                               ////__DACE:43:0:5,4    ////__DACE:43:0:4    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                ////__DACE:43:0:4                                                             ////__DACE:43:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                // Tasklet code (tlet_76_multiplies_1_0)                                          ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
                ////__DACE:43:0:4                                                             ////__DACE:43:0:4    ////__DACE:41
                __map_fusion_gtir_tmp_191_1_0 = __tlet_result;                                    ////__DACE:43:0:4    ////__DACE:43:0:4    ////__DACE:41
            }                                                                                 ////__DACE:43:0:4    ////__DACE:41
            {                                                                                 ////__DACE:43:0:12    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:43:0:21,12    ////__DACE:43:0:12    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:43:0:19,12    ////__DACE:43:0:12    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                ////__DACE:43:0:12                                                            ////__DACE:43:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                // Tasklet code (tlet_72_deref_1_0)                                               ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
                ////__DACE:43:0:12                                                            ////__DACE:43:0:12    ////__DACE:41
                __map_fusion_gtir_tmp_183_1_0 = __tlet_val;                                       ////__DACE:43:0:12    ////__DACE:43:0:12    ////__DACE:41
            }                                                                                 ////__DACE:43:0:12    ////__DACE:41
            {                                                                                 ////__DACE:43:0:10    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_54_1_0;                                             ////__DACE:43:0:20,10    ////__DACE:43:0:10    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_183_1_0;                               ////__DACE:43:0:11,10    ////__DACE:43:0:10    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                ////__DACE:43:0:10                                                            ////__DACE:43:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                // Tasklet code (tlet_73_multiplies_1_0)                                          ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
                ////__DACE:43:0:10                                                            ////__DACE:43:0:10    ////__DACE:41
                __map_fusion_gtir_tmp_185_1_0 = __tlet_result;                                    ////__DACE:43:0:10    ////__DACE:43:0:10    ////__DACE:41
            }                                                                                 ////__DACE:43:0:10    ////__DACE:41
            {                                                                                 ////__DACE:43:0:16    ////__DACE:41
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:43:0:23,16    ////__DACE:43:0:16    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:43:0:19,16    ////__DACE:43:0:16    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                ////__DACE:43:0:16                                                            ////__DACE:43:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                // Tasklet code (tlet_70_deref_1_0)                                               ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
                ////__DACE:43:0:16                                                            ////__DACE:43:0:16    ////__DACE:41
                __map_fusion_gtir_tmp_179_1_0 = __tlet_val;                                       ////__DACE:43:0:16    ////__DACE:43:0:16    ////__DACE:41
            }                                                                                 ////__DACE:43:0:16    ////__DACE:41
            {                                                                                 ////__DACE:43:0:14    ////__DACE:41
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:43:0:22,14    ////__DACE:43:0:14    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_179_1_0;                               ////__DACE:43:0:15,14    ////__DACE:43:0:14    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                ////__DACE:43:0:14                                                            ////__DACE:43:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                // Tasklet code (tlet_71_plus_1_0)                                                ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
                ////__DACE:43:0:14                                                            ////__DACE:43:0:14    ////__DACE:41
                __map_fusion_gtir_tmp_181_1_0 = __tlet_result;                                    ////__DACE:43:0:14    ////__DACE:43:0:14    ////__DACE:41
            }                                                                                 ////__DACE:43:0:14    ////__DACE:41
            {                                                                                 ////__DACE:43:0:8    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_185_1_0;                               ////__DACE:43:0:9,8    ////__DACE:43:0:8    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_181_1_0;                               ////__DACE:43:0:13,8    ////__DACE:43:0:8    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                ////__DACE:43:0:8                                                             ////__DACE:43:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                // Tasklet code (tlet_74_plus_1_0)                                                ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
                ////__DACE:43:0:8                                                             ////__DACE:43:0:8    ////__DACE:41
                __map_fusion_gtir_tmp_187_1_0 = __tlet_result;                                    ////__DACE:43:0:8    ////__DACE:43:0:8    ////__DACE:41
            }                                                                                 ////__DACE:43:0:8    ////__DACE:41
            {                                                                                 ////__DACE:43:0:2    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_191_1_0;                               ////__DACE:43:0:3,2    ////__DACE:43:0:2    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_187_1_0;                               ////__DACE:43:0:7,2    ////__DACE:43:0:2    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                ////__DACE:43:0:2                                                             ////__DACE:43:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                // Tasklet code (tlet_77_plus_1_0)                                                ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
                ////__DACE:43:0:2                                                             ////__DACE:43:0:2    ////__DACE:41
                __arg1___ = __tlet_result;                                                        ////__DACE:43:0:2    ////__DACE:43:0:2    ////__DACE:41
            }                                                                                 ////__DACE:43:0:2    ////__DACE:41
            {                                                                                 ////__DACE:43:0:24    ////__DACE:41
                double _cpy_in = __arg1___;                                                       ////__DACE:43:0:1,24    ////__DACE:43:0:24    ////__DACE:41
                double _cpy_out;                                                                  ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                ////__DACE:43:0:24                                                            ////__DACE:43:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                _cpy_out = _cpy_in;                                                               ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
                ////__DACE:43:0:24                                                            ////__DACE:43:0:24    ////__DACE:41
                __output = _cpy_out;                                                              ////__DACE:43:0:24    ////__DACE:43:0:24    ////__DACE:41
            }                                                                                 ////__DACE:43:0:24    ////__DACE:41
            ////__DACE:41
        }                                                                             ////__DACE:41
    } else {                                                                          ////__DACE:41
        {                                                                             ////__DACE:41
            double __arg2;                                                                    ////__DACE:44:0:1    ////__DACE:41
            double __map_fusion_gtir_tmp_207_1_0;                                             ////__DACE:44:0:3    ////__DACE:41
            double __map_fusion_gtir_tmp_205_1_0;                                             ////__DACE:44:0:5    ////__DACE:41
            double __map_fusion_gtir_tmp_203_1_0;                                             ////__DACE:44:0:7    ////__DACE:41
            double __map_fusion_gtir_tmp_201_1_0;                                             ////__DACE:44:0:9    ////__DACE:41
            double __map_fusion_gtir_tmp_199_1_0;                                             ////__DACE:44:0:11    ////__DACE:41
            double __map_fusion_gtir_tmp_197_1_0;                                             ////__DACE:44:0:13    ////__DACE:41
            double __map_fusion_gtir_tmp_195_1_0;                                             ////__DACE:44:0:15    ////__DACE:41
            ////__DACE:41
            {                                                                                 ////__DACE:44:0:6    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:44:0:18,6    ////__DACE:44:0:6    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:44:0:19,6    ////__DACE:44:0:6    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                ////__DACE:44:0:6                                                             ////__DACE:44:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                // Tasklet code (tlet_83_deref_1_0)                                               ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
                ////__DACE:44:0:6                                                             ////__DACE:44:0:6    ////__DACE:41
                __map_fusion_gtir_tmp_205_1_0 = __tlet_val;                                       ////__DACE:44:0:6    ////__DACE:44:0:6    ////__DACE:41
            }                                                                                 ////__DACE:44:0:6    ////__DACE:41
            {                                                                                 ////__DACE:44:0:4    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_76_1_0;                                             ////__DACE:44:0:17,4    ////__DACE:44:0:4    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_205_1_0;                               ////__DACE:44:0:5,4    ////__DACE:44:0:4    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                ////__DACE:44:0:4                                                             ////__DACE:44:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                // Tasklet code (tlet_84_multiplies_1_0)                                          ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
                ////__DACE:44:0:4                                                             ////__DACE:44:0:4    ////__DACE:41
                __map_fusion_gtir_tmp_207_1_0 = __tlet_result;                                    ////__DACE:44:0:4    ////__DACE:44:0:4    ////__DACE:41
            }                                                                                 ////__DACE:44:0:4    ////__DACE:41
            {                                                                                 ////__DACE:44:0:12    ////__DACE:41
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:44:0:21,12    ////__DACE:44:0:12    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:44:0:19,12    ////__DACE:44:0:12    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                ////__DACE:44:0:12                                                            ////__DACE:44:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                // Tasklet code (tlet_80_deref_1_0)                                               ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
                ////__DACE:44:0:12                                                            ////__DACE:44:0:12    ////__DACE:41
                __map_fusion_gtir_tmp_199_1_0 = __tlet_val;                                       ////__DACE:44:0:12    ////__DACE:44:0:12    ////__DACE:41
            }                                                                                 ////__DACE:44:0:12    ////__DACE:41
            {                                                                                 ////__DACE:44:0:10    ////__DACE:41
                double __tlet_arg0 = gtir_tmp_54_1_0;                                             ////__DACE:44:0:20,10    ////__DACE:44:0:10    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_199_1_0;                               ////__DACE:44:0:11,10    ////__DACE:44:0:10    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                ////__DACE:44:0:10                                                            ////__DACE:44:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                // Tasklet code (tlet_81_multiplies_1_0)                                          ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
                ////__DACE:44:0:10                                                            ////__DACE:44:0:10    ////__DACE:41
                __map_fusion_gtir_tmp_201_1_0 = __tlet_result;                                    ////__DACE:44:0:10    ////__DACE:44:0:10    ////__DACE:41
            }                                                                                 ////__DACE:44:0:10    ////__DACE:41
            {                                                                                 ////__DACE:44:0:16    ////__DACE:41
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:44:0:23,16    ////__DACE:44:0:16    ////__DACE:41
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:44:0:19,16    ////__DACE:44:0:16    ////__DACE:41
                double __tlet_val;                                                                ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                ////__DACE:44:0:16                                                            ////__DACE:44:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                // Tasklet code (tlet_78_deref_1_0)                                               ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
                ////__DACE:44:0:16                                                            ////__DACE:44:0:16    ////__DACE:41
                __map_fusion_gtir_tmp_195_1_0 = __tlet_val;                                       ////__DACE:44:0:16    ////__DACE:44:0:16    ////__DACE:41
            }                                                                                 ////__DACE:44:0:16    ////__DACE:41
            {                                                                                 ////__DACE:44:0:14    ////__DACE:41
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:44:0:22,14    ////__DACE:44:0:14    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_195_1_0;                               ////__DACE:44:0:15,14    ////__DACE:44:0:14    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                ////__DACE:44:0:14                                                            ////__DACE:44:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                // Tasklet code (tlet_79_plus_1_0)                                                ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
                ////__DACE:44:0:14                                                            ////__DACE:44:0:14    ////__DACE:41
                __map_fusion_gtir_tmp_197_1_0 = __tlet_result;                                    ////__DACE:44:0:14    ////__DACE:44:0:14    ////__DACE:41
            }                                                                                 ////__DACE:44:0:14    ////__DACE:41
            {                                                                                 ////__DACE:44:0:8    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_201_1_0;                               ////__DACE:44:0:9,8    ////__DACE:44:0:8    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_197_1_0;                               ////__DACE:44:0:13,8    ////__DACE:44:0:8    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                ////__DACE:44:0:8                                                             ////__DACE:44:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                // Tasklet code (tlet_82_plus_1_0)                                                ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
                ////__DACE:44:0:8                                                             ////__DACE:44:0:8    ////__DACE:41
                __map_fusion_gtir_tmp_203_1_0 = __tlet_result;                                    ////__DACE:44:0:8    ////__DACE:44:0:8    ////__DACE:41
            }                                                                                 ////__DACE:44:0:8    ////__DACE:41
            {                                                                                 ////__DACE:44:0:2    ////__DACE:41
                double __tlet_arg1 = __map_fusion_gtir_tmp_207_1_0;                               ////__DACE:44:0:3,2    ////__DACE:44:0:2    ////__DACE:41
                double __tlet_arg0 = __map_fusion_gtir_tmp_203_1_0;                               ////__DACE:44:0:7,2    ////__DACE:44:0:2    ////__DACE:41
                double __tlet_result;                                                             ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                ////__DACE:44:0:2                                                             ////__DACE:44:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                // Tasklet code (tlet_85_plus_1_0)                                                ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
                ////__DACE:44:0:2                                                             ////__DACE:44:0:2    ////__DACE:41
                __arg2 = __tlet_result;                                                           ////__DACE:44:0:2    ////__DACE:44:0:2    ////__DACE:41
            }                                                                                 ////__DACE:44:0:2    ////__DACE:41
            {                                                                                 ////__DACE:44:0:24    ////__DACE:41
                double _cpy_in = __arg2;                                                          ////__DACE:44:0:1,24    ////__DACE:44:0:24    ////__DACE:41
                double _cpy_out;                                                                  ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                ////__DACE:44:0:24                                                            ////__DACE:44:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                _cpy_out = _cpy_in;                                                               ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                ///////////////////                                                               ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
                ////__DACE:44:0:24                                                            ////__DACE:44:0:24    ////__DACE:41
                __output = _cpy_out;                                                              ////__DACE:44:0:24    ////__DACE:44:0:24    ////__DACE:41
            }                                                                                 ////__DACE:44:0:24    ////__DACE:41
            ////__DACE:41
        }                                                                             ////__DACE:41
    }                                                                                 ////__DACE:41
}                                                                                 ////__DACE:0:0:189
////__DACE:0:0:189
DACE_DFI void if_stmt_5_0_0_186(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double&  gtir_tmp_54_1_0, const double&  gtir_tmp_76_1_0, const double * __restrict__ gtir_tmp_95, const double* __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double* __restrict__ reference_theta_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_theta_v_at_cells_on_model_levels_K_stride_0, int __reference_theta_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:186
    ////__DACE:37
    if (__cond) {                                                                     ////__DACE:37
        {                                                                             ////__DACE:37
            double __arg1____;                                                                ////__DACE:39:0:1    ////__DACE:37
            double __map_fusion_gtir_tmp_118_1_0;                                             ////__DACE:39:0:3    ////__DACE:37
            double __map_fusion_gtir_tmp_116_1_0;                                             ////__DACE:39:0:5    ////__DACE:37
            double __map_fusion_gtir_tmp_114_1_0;                                             ////__DACE:39:0:7    ////__DACE:37
            double __map_fusion_gtir_tmp_112_1_0;                                             ////__DACE:39:0:9    ////__DACE:37
            double __map_fusion_gtir_tmp_110_1_0;                                             ////__DACE:39:0:11    ////__DACE:37
            double __map_fusion_gtir_tmp_108_1_0;                                             ////__DACE:39:0:13    ////__DACE:37
            double __map_fusion_gtir_tmp_106_1_0;                                             ////__DACE:39:0:15    ////__DACE:37
            ////__DACE:37
            {                                                                                 ////__DACE:39:0:6    ////__DACE:37
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:39:0:18,6    ////__DACE:39:0:6    ////__DACE:37
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:39:0:19,6    ////__DACE:39:0:6    ////__DACE:37
                double __tlet_val;                                                                ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                ////__DACE:39:0:6                                                             ////__DACE:39:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                // Tasklet code (tlet_41_deref_1_0)                                               ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
                ////__DACE:39:0:6                                                             ////__DACE:39:0:6    ////__DACE:37
                __map_fusion_gtir_tmp_116_1_0 = __tlet_val;                                       ////__DACE:39:0:6    ////__DACE:39:0:6    ////__DACE:37
            }                                                                                 ////__DACE:39:0:6    ////__DACE:37
            {                                                                                 ////__DACE:39:0:4    ////__DACE:37
                double __tlet_arg0 = gtir_tmp_76_1_0;                                             ////__DACE:39:0:17,4    ////__DACE:39:0:4    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_116_1_0;                               ////__DACE:39:0:5,4    ////__DACE:39:0:4    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                ////__DACE:39:0:4                                                             ////__DACE:39:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                // Tasklet code (tlet_42_multiplies_1_0)                                          ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
                ////__DACE:39:0:4                                                             ////__DACE:39:0:4    ////__DACE:37
                __map_fusion_gtir_tmp_118_1_0 = __tlet_result;                                    ////__DACE:39:0:4    ////__DACE:39:0:4    ////__DACE:37
            }                                                                                 ////__DACE:39:0:4    ////__DACE:37
            {                                                                                 ////__DACE:39:0:12    ////__DACE:37
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:39:0:21,12    ////__DACE:39:0:12    ////__DACE:37
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:39:0:19,12    ////__DACE:39:0:12    ////__DACE:37
                double __tlet_val;                                                                ////__DACE:39:0:12    ////__DACE:39:0:12    ////__DACE:37
                ////__DACE:39:0:12                                                            ////__DACE:39:0:12    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:12    ////__DACE:39:0:12    ////__DACE:37
                // Tasklet code (tlet_38_deref_1_0)                                               ////__DACE:39:0:12    ////__DACE:39:0:12    ////__DACE:37
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:39:0:12    ////__DACE:39:0:12    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:12    ////__DACE:39:0:12    ////__DACE:37
                ////__DACE:39:0:12                                                            ////__DACE:39:0:12    ////__DACE:37
                __map_fusion_gtir_tmp_110_1_0 = __tlet_val;                                       ////__DACE:39:0:12    ////__DACE:39:0:12    ////__DACE:37
            }                                                                                 ////__DACE:39:0:12    ////__DACE:37
            {                                                                                 ////__DACE:39:0:10    ////__DACE:37
                double __tlet_arg0 = gtir_tmp_54_1_0;                                             ////__DACE:39:0:20,10    ////__DACE:39:0:10    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_110_1_0;                               ////__DACE:39:0:11,10    ////__DACE:39:0:10    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:10    ////__DACE:39:0:10    ////__DACE:37
                ////__DACE:39:0:10                                                            ////__DACE:39:0:10    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:10    ////__DACE:39:0:10    ////__DACE:37
                // Tasklet code (tlet_39_multiplies_1_0)                                          ////__DACE:39:0:10    ////__DACE:39:0:10    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:39:0:10    ////__DACE:39:0:10    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:10    ////__DACE:39:0:10    ////__DACE:37
                ////__DACE:39:0:10                                                            ////__DACE:39:0:10    ////__DACE:37
                __map_fusion_gtir_tmp_112_1_0 = __tlet_result;                                    ////__DACE:39:0:10    ////__DACE:39:0:10    ////__DACE:37
            }                                                                                 ////__DACE:39:0:10    ////__DACE:37
            {                                                                                 ////__DACE:39:0:16    ////__DACE:37
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:39:0:23,16    ////__DACE:39:0:16    ////__DACE:37
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:39:0:19,16    ////__DACE:39:0:16    ////__DACE:37
                double __tlet_val;                                                                ////__DACE:39:0:16    ////__DACE:39:0:16    ////__DACE:37
                ////__DACE:39:0:16                                                            ////__DACE:39:0:16    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:16    ////__DACE:39:0:16    ////__DACE:37
                // Tasklet code (tlet_36_deref_1_0)                                               ////__DACE:39:0:16    ////__DACE:39:0:16    ////__DACE:37
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:39:0:16    ////__DACE:39:0:16    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:16    ////__DACE:39:0:16    ////__DACE:37
                ////__DACE:39:0:16                                                            ////__DACE:39:0:16    ////__DACE:37
                __map_fusion_gtir_tmp_106_1_0 = __tlet_val;                                       ////__DACE:39:0:16    ////__DACE:39:0:16    ////__DACE:37
            }                                                                                 ////__DACE:39:0:16    ////__DACE:37
            {                                                                                 ////__DACE:39:0:14    ////__DACE:37
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:39:0:22,14    ////__DACE:39:0:14    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_106_1_0;                               ////__DACE:39:0:15,14    ////__DACE:39:0:14    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:14    ////__DACE:39:0:14    ////__DACE:37
                ////__DACE:39:0:14                                                            ////__DACE:39:0:14    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:14    ////__DACE:39:0:14    ////__DACE:37
                // Tasklet code (tlet_37_plus_1_0)                                                ////__DACE:39:0:14    ////__DACE:39:0:14    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:39:0:14    ////__DACE:39:0:14    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:14    ////__DACE:39:0:14    ////__DACE:37
                ////__DACE:39:0:14                                                            ////__DACE:39:0:14    ////__DACE:37
                __map_fusion_gtir_tmp_108_1_0 = __tlet_result;                                    ////__DACE:39:0:14    ////__DACE:39:0:14    ////__DACE:37
            }                                                                                 ////__DACE:39:0:14    ////__DACE:37
            {                                                                                 ////__DACE:39:0:8    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_112_1_0;                               ////__DACE:39:0:9,8    ////__DACE:39:0:8    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_108_1_0;                               ////__DACE:39:0:13,8    ////__DACE:39:0:8    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                ////__DACE:39:0:8                                                             ////__DACE:39:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                // Tasklet code (tlet_40_plus_1_0)                                                ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
                ////__DACE:39:0:8                                                             ////__DACE:39:0:8    ////__DACE:37
                __map_fusion_gtir_tmp_114_1_0 = __tlet_result;                                    ////__DACE:39:0:8    ////__DACE:39:0:8    ////__DACE:37
            }                                                                                 ////__DACE:39:0:8    ////__DACE:37
            {                                                                                 ////__DACE:39:0:2    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_118_1_0;                               ////__DACE:39:0:3,2    ////__DACE:39:0:2    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_114_1_0;                               ////__DACE:39:0:7,2    ////__DACE:39:0:2    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:39:0:2    ////__DACE:39:0:2    ////__DACE:37
                ////__DACE:39:0:2                                                             ////__DACE:39:0:2    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:2    ////__DACE:39:0:2    ////__DACE:37
                // Tasklet code (tlet_43_plus_1_0)                                                ////__DACE:39:0:2    ////__DACE:39:0:2    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:39:0:2    ////__DACE:39:0:2    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:2    ////__DACE:39:0:2    ////__DACE:37
                ////__DACE:39:0:2                                                             ////__DACE:39:0:2    ////__DACE:37
                __arg1____ = __tlet_result;                                                       ////__DACE:39:0:2    ////__DACE:39:0:2    ////__DACE:37
            }                                                                                 ////__DACE:39:0:2    ////__DACE:37
            {                                                                                 ////__DACE:39:0:24    ////__DACE:37
                double _cpy_in = __arg1____;                                                      ////__DACE:39:0:1,24    ////__DACE:39:0:24    ////__DACE:37
                double _cpy_out;                                                                  ////__DACE:39:0:24    ////__DACE:39:0:24    ////__DACE:37
                ////__DACE:39:0:24                                                            ////__DACE:39:0:24    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:24    ////__DACE:39:0:24    ////__DACE:37
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:39:0:24    ////__DACE:39:0:24    ////__DACE:37
                _cpy_out = _cpy_in;                                                               ////__DACE:39:0:24    ////__DACE:39:0:24    ////__DACE:37
                ///////////////////                                                               ////__DACE:39:0:24    ////__DACE:39:0:24    ////__DACE:37
                ////__DACE:39:0:24                                                            ////__DACE:39:0:24    ////__DACE:37
                __output = _cpy_out;                                                              ////__DACE:39:0:24    ////__DACE:39:0:24    ////__DACE:37
            }                                                                                 ////__DACE:39:0:24    ////__DACE:37
            ////__DACE:37
        }                                                                             ////__DACE:37
    } else {                                                                          ////__DACE:37
        {                                                                             ////__DACE:37
            double __arg2_;                                                                   ////__DACE:40:0:1    ////__DACE:37
            double __map_fusion_gtir_tmp_134_1_0;                                             ////__DACE:40:0:3    ////__DACE:37
            double __map_fusion_gtir_tmp_132_1_0;                                             ////__DACE:40:0:5    ////__DACE:37
            double __map_fusion_gtir_tmp_130_1_0;                                             ////__DACE:40:0:7    ////__DACE:37
            double __map_fusion_gtir_tmp_128_1_0;                                             ////__DACE:40:0:9    ////__DACE:37
            double __map_fusion_gtir_tmp_126_1_0;                                             ////__DACE:40:0:11    ////__DACE:37
            double __map_fusion_gtir_tmp_124_1_0;                                             ////__DACE:40:0:13    ////__DACE:37
            double __map_fusion_gtir_tmp_122_1_0;                                             ////__DACE:40:0:15    ////__DACE:37
            ////__DACE:37
            {                                                                                 ////__DACE:40:0:6    ////__DACE:37
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:40:0:18,6    ////__DACE:40:0:6    ////__DACE:37
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:40:0:19,6    ////__DACE:40:0:6    ////__DACE:37
                double __tlet_val;                                                                ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                ////__DACE:40:0:6                                                             ////__DACE:40:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                // Tasklet code (tlet_49_deref_1_0)                                               ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
                ////__DACE:40:0:6                                                             ////__DACE:40:0:6    ////__DACE:37
                __map_fusion_gtir_tmp_132_1_0 = __tlet_val;                                       ////__DACE:40:0:6    ////__DACE:40:0:6    ////__DACE:37
            }                                                                                 ////__DACE:40:0:6    ////__DACE:37
            {                                                                                 ////__DACE:40:0:4    ////__DACE:37
                double __tlet_arg0 = gtir_tmp_76_1_0;                                             ////__DACE:40:0:17,4    ////__DACE:40:0:4    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_132_1_0;                               ////__DACE:40:0:5,4    ////__DACE:40:0:4    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                ////__DACE:40:0:4                                                             ////__DACE:40:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                // Tasklet code (tlet_50_multiplies_1_0)                                          ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
                ////__DACE:40:0:4                                                             ////__DACE:40:0:4    ////__DACE:37
                __map_fusion_gtir_tmp_134_1_0 = __tlet_result;                                    ////__DACE:40:0:4    ////__DACE:40:0:4    ////__DACE:37
            }                                                                                 ////__DACE:40:0:4    ////__DACE:37
            {                                                                                 ////__DACE:40:0:12    ////__DACE:37
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:40:0:21,12    ////__DACE:40:0:12    ////__DACE:37
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:40:0:19,12    ////__DACE:40:0:12    ////__DACE:37
                double __tlet_val;                                                                ////__DACE:40:0:12    ////__DACE:40:0:12    ////__DACE:37
                ////__DACE:40:0:12                                                            ////__DACE:40:0:12    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:12    ////__DACE:40:0:12    ////__DACE:37
                // Tasklet code (tlet_46_deref_1_0)                                               ////__DACE:40:0:12    ////__DACE:40:0:12    ////__DACE:37
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:40:0:12    ////__DACE:40:0:12    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:12    ////__DACE:40:0:12    ////__DACE:37
                ////__DACE:40:0:12                                                            ////__DACE:40:0:12    ////__DACE:37
                __map_fusion_gtir_tmp_126_1_0 = __tlet_val;                                       ////__DACE:40:0:12    ////__DACE:40:0:12    ////__DACE:37
            }                                                                                 ////__DACE:40:0:12    ////__DACE:37
            {                                                                                 ////__DACE:40:0:10    ////__DACE:37
                double __tlet_arg0 = gtir_tmp_54_1_0;                                             ////__DACE:40:0:20,10    ////__DACE:40:0:10    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_126_1_0;                               ////__DACE:40:0:11,10    ////__DACE:40:0:10    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:10    ////__DACE:40:0:10    ////__DACE:37
                ////__DACE:40:0:10                                                            ////__DACE:40:0:10    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:10    ////__DACE:40:0:10    ////__DACE:37
                // Tasklet code (tlet_47_multiplies_1_0)                                          ////__DACE:40:0:10    ////__DACE:40:0:10    ////__DACE:37
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:40:0:10    ////__DACE:40:0:10    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:10    ////__DACE:40:0:10    ////__DACE:37
                ////__DACE:40:0:10                                                            ////__DACE:40:0:10    ////__DACE:37
                __map_fusion_gtir_tmp_128_1_0 = __tlet_result;                                    ////__DACE:40:0:10    ////__DACE:40:0:10    ////__DACE:37
            }                                                                                 ////__DACE:40:0:10    ////__DACE:37
            {                                                                                 ////__DACE:40:0:16    ////__DACE:37
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:40:0:23,16    ////__DACE:40:0:16    ////__DACE:37
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:40:0:19,16    ////__DACE:40:0:16    ////__DACE:37
                double __tlet_val;                                                                ////__DACE:40:0:16    ////__DACE:40:0:16    ////__DACE:37
                ////__DACE:40:0:16                                                            ////__DACE:40:0:16    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:16    ////__DACE:40:0:16    ////__DACE:37
                // Tasklet code (tlet_44_deref_1_0)                                               ////__DACE:40:0:16    ////__DACE:40:0:16    ////__DACE:37
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:40:0:16    ////__DACE:40:0:16    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:16    ////__DACE:40:0:16    ////__DACE:37
                ////__DACE:40:0:16                                                            ////__DACE:40:0:16    ////__DACE:37
                __map_fusion_gtir_tmp_122_1_0 = __tlet_val;                                       ////__DACE:40:0:16    ////__DACE:40:0:16    ////__DACE:37
            }                                                                                 ////__DACE:40:0:16    ////__DACE:37
            {                                                                                 ////__DACE:40:0:14    ////__DACE:37
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:40:0:22,14    ////__DACE:40:0:14    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_122_1_0;                               ////__DACE:40:0:15,14    ////__DACE:40:0:14    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:14    ////__DACE:40:0:14    ////__DACE:37
                ////__DACE:40:0:14                                                            ////__DACE:40:0:14    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:14    ////__DACE:40:0:14    ////__DACE:37
                // Tasklet code (tlet_45_plus_1_0)                                                ////__DACE:40:0:14    ////__DACE:40:0:14    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:40:0:14    ////__DACE:40:0:14    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:14    ////__DACE:40:0:14    ////__DACE:37
                ////__DACE:40:0:14                                                            ////__DACE:40:0:14    ////__DACE:37
                __map_fusion_gtir_tmp_124_1_0 = __tlet_result;                                    ////__DACE:40:0:14    ////__DACE:40:0:14    ////__DACE:37
            }                                                                                 ////__DACE:40:0:14    ////__DACE:37
            {                                                                                 ////__DACE:40:0:8    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_128_1_0;                               ////__DACE:40:0:9,8    ////__DACE:40:0:8    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_124_1_0;                               ////__DACE:40:0:13,8    ////__DACE:40:0:8    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                ////__DACE:40:0:8                                                             ////__DACE:40:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                // Tasklet code (tlet_48_plus_1_0)                                                ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
                ////__DACE:40:0:8                                                             ////__DACE:40:0:8    ////__DACE:37
                __map_fusion_gtir_tmp_130_1_0 = __tlet_result;                                    ////__DACE:40:0:8    ////__DACE:40:0:8    ////__DACE:37
            }                                                                                 ////__DACE:40:0:8    ////__DACE:37
            {                                                                                 ////__DACE:40:0:2    ////__DACE:37
                double __tlet_arg1 = __map_fusion_gtir_tmp_134_1_0;                               ////__DACE:40:0:3,2    ////__DACE:40:0:2    ////__DACE:37
                double __tlet_arg0 = __map_fusion_gtir_tmp_130_1_0;                               ////__DACE:40:0:7,2    ////__DACE:40:0:2    ////__DACE:37
                double __tlet_result;                                                             ////__DACE:40:0:2    ////__DACE:40:0:2    ////__DACE:37
                ////__DACE:40:0:2                                                             ////__DACE:40:0:2    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:2    ////__DACE:40:0:2    ////__DACE:37
                // Tasklet code (tlet_51_plus_1_0)                                                ////__DACE:40:0:2    ////__DACE:40:0:2    ////__DACE:37
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:40:0:2    ////__DACE:40:0:2    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:2    ////__DACE:40:0:2    ////__DACE:37
                ////__DACE:40:0:2                                                             ////__DACE:40:0:2    ////__DACE:37
                __arg2_ = __tlet_result;                                                          ////__DACE:40:0:2    ////__DACE:40:0:2    ////__DACE:37
            }                                                                                 ////__DACE:40:0:2    ////__DACE:37
            {                                                                                 ////__DACE:40:0:24    ////__DACE:37
                double _cpy_in = __arg2_;                                                         ////__DACE:40:0:1,24    ////__DACE:40:0:24    ////__DACE:37
                double _cpy_out;                                                                  ////__DACE:40:0:24    ////__DACE:40:0:24    ////__DACE:37
                ////__DACE:40:0:24                                                            ////__DACE:40:0:24    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:24    ////__DACE:40:0:24    ////__DACE:37
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:40:0:24    ////__DACE:40:0:24    ////__DACE:37
                _cpy_out = _cpy_in;                                                               ////__DACE:40:0:24    ////__DACE:40:0:24    ////__DACE:37
                ///////////////////                                                               ////__DACE:40:0:24    ////__DACE:40:0:24    ////__DACE:37
                ////__DACE:40:0:24                                                            ////__DACE:40:0:24    ////__DACE:37
                __output = _cpy_out;                                                              ////__DACE:40:0:24    ////__DACE:40:0:24    ////__DACE:37
            }                                                                                 ////__DACE:40:0:24    ////__DACE:37
            ////__DACE:37
        }                                                                             ////__DACE:37
    }                                                                                 ////__DACE:37
}                                                                                 ////__DACE:0:0:186
////__DACE:0:0:186
DACE_DFI void if_stmt_6_0_0_91(const double&  __arg2_, const bool&  __cond, const double* __restrict__ hydrostatic_correction_on_lowest_level, const double* __restrict__ pg_exdist, double&  __output, int __pg_exdist_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:91
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
}                                                                                 ////__DACE:0:0:91
////__DACE:0:0:91
DACE_DFI void if_stmt_1_0_0_206(const double&  __arg1___, const double&  __arg1____from_cb_fusion_5, const double&  __arg2, const double&  __arg2_from_cb_fusion_5, const bool&  __cond, double&  __output, double&  __output_from_cb_fusion_5) {    ////__DACE:0:0:206
    ////__DACE:49
    if (__cond) {                                                                     ////__DACE:49
        {                                                                             ////__DACE:49
            ////__DACE:49
            {                                                                                 ////__DACE:51:0:4    ////__DACE:49
                double _cpy_in = __arg1___;                                                       ////__DACE:51:0:1,4    ////__DACE:51:0:4    ////__DACE:49
                double _cpy_out;                                                                  ////__DACE:51:0:4    ////__DACE:51:0:4    ////__DACE:49
                ////__DACE:51:0:4                                                             ////__DACE:51:0:4    ////__DACE:49
                ///////////////////                                                               ////__DACE:51:0:4    ////__DACE:51:0:4    ////__DACE:49
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:51:0:4    ////__DACE:51:0:4    ////__DACE:49
                _cpy_out = _cpy_in;                                                               ////__DACE:51:0:4    ////__DACE:51:0:4    ////__DACE:49
                ///////////////////                                                               ////__DACE:51:0:4    ////__DACE:51:0:4    ////__DACE:49
                ////__DACE:51:0:4                                                             ////__DACE:51:0:4    ////__DACE:49
                __output = _cpy_out;                                                              ////__DACE:51:0:4    ////__DACE:51:0:4    ////__DACE:49
            }                                                                                 ////__DACE:51:0:4    ////__DACE:49
            {                                                                                 ////__DACE:51:0:5    ////__DACE:49
                double _cpy_in = __arg1____from_cb_fusion_5;                                      ////__DACE:51:0:3,5    ////__DACE:51:0:5    ////__DACE:49
                double _cpy_out;                                                                  ////__DACE:51:0:5    ////__DACE:51:0:5    ////__DACE:49
                ////__DACE:51:0:5                                                             ////__DACE:51:0:5    ////__DACE:49
                ///////////////////                                                               ////__DACE:51:0:5    ////__DACE:51:0:5    ////__DACE:49
                // Tasklet code (copy___arg1____from_cb_fusion_5_to___output_from_cb_fusion_5)    ////__DACE:51:0:5    ////__DACE:51:0:5    ////__DACE:49
                _cpy_out = _cpy_in;                                                               ////__DACE:51:0:5    ////__DACE:51:0:5    ////__DACE:49
                ///////////////////                                                               ////__DACE:51:0:5    ////__DACE:51:0:5    ////__DACE:49
                ////__DACE:51:0:5                                                             ////__DACE:51:0:5    ////__DACE:49
                __output_from_cb_fusion_5 = _cpy_out;                                             ////__DACE:51:0:5    ////__DACE:51:0:5    ////__DACE:49
            }                                                                                 ////__DACE:51:0:5    ////__DACE:49
            ////__DACE:49
        }                                                                             ////__DACE:49
    } else {                                                                          ////__DACE:49
        {                                                                             ////__DACE:49
            ////__DACE:49
            {                                                                                 ////__DACE:52:0:4    ////__DACE:49
                double _cpy_in = __arg2;                                                          ////__DACE:52:0:1,4    ////__DACE:52:0:4    ////__DACE:49
                double _cpy_out;                                                                  ////__DACE:52:0:4    ////__DACE:52:0:4    ////__DACE:49
                ////__DACE:52:0:4                                                             ////__DACE:52:0:4    ////__DACE:49
                ///////////////////                                                               ////__DACE:52:0:4    ////__DACE:52:0:4    ////__DACE:49
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:52:0:4    ////__DACE:52:0:4    ////__DACE:49
                _cpy_out = _cpy_in;                                                               ////__DACE:52:0:4    ////__DACE:52:0:4    ////__DACE:49
                ///////////////////                                                               ////__DACE:52:0:4    ////__DACE:52:0:4    ////__DACE:49
                ////__DACE:52:0:4                                                             ////__DACE:52:0:4    ////__DACE:49
                __output = _cpy_out;                                                              ////__DACE:52:0:4    ////__DACE:52:0:4    ////__DACE:49
            }                                                                                 ////__DACE:52:0:4    ////__DACE:49
            {                                                                                 ////__DACE:52:0:5    ////__DACE:49
                double _cpy_in = __arg2_from_cb_fusion_5;                                         ////__DACE:52:0:3,5    ////__DACE:52:0:5    ////__DACE:49
                double _cpy_out;                                                                  ////__DACE:52:0:5    ////__DACE:52:0:5    ////__DACE:49
                ////__DACE:52:0:5                                                             ////__DACE:52:0:5    ////__DACE:49
                ///////////////////                                                               ////__DACE:52:0:5    ////__DACE:52:0:5    ////__DACE:49
                // Tasklet code (copy___arg2_from_cb_fusion_5_to___output_from_cb_fusion_5)       ////__DACE:52:0:5    ////__DACE:52:0:5    ////__DACE:49
                _cpy_out = _cpy_in;                                                               ////__DACE:52:0:5    ////__DACE:52:0:5    ////__DACE:49
                ///////////////////                                                               ////__DACE:52:0:5    ////__DACE:52:0:5    ////__DACE:49
                ////__DACE:52:0:5                                                             ////__DACE:52:0:5    ////__DACE:49
                __output_from_cb_fusion_5 = _cpy_out;                                             ////__DACE:52:0:5    ////__DACE:52:0:5    ////__DACE:49
            }                                                                                 ////__DACE:52:0:5    ////__DACE:49
            ////__DACE:49
        }                                                                             ////__DACE:49
    }                                                                                 ////__DACE:49
}                                                                                 ////__DACE:0:0:206
////__DACE:0:0:206
DACE_DFI void if_stmt_4_0_0_217(const bool&  __cond, const double&  __map_fusion_gtir_tmp_21_1_1_1, const double&  __map_fusion_gtir_tmp_33_1_1_1, const double&  gtir_tmp_34_1_1, const double&  gtir_tmp_38_1_1, const double&  gtir_tmp_44_1_1, const double&  gtir_tmp_48_1_1, const double&  gtir_tmp_56_1_1, const double&  gtir_tmp_60_1_1, const double&  gtir_tmp_66_1_1, const double&  gtir_tmp_70_1_1, double&  __output, double&  __output_from_cb_fusion_4) {    ////__DACE:0:0:217
    ////__DACE:53
    if (__cond) {                                                                     ////__DACE:53
        {                                                                             ////__DACE:53
            double __arg1____;                                                                ////__DACE:55:0:1    ////__DACE:53
            double __arg1____from_cb_fusion_4;                                                ////__DACE:55:0:3    ////__DACE:53
            double __map_fusion_gtir_tmp_63_1_1;                                              ////__DACE:55:0:5    ////__DACE:53
            double __map_fusion_gtir_tmp_59_1_1;                                              ////__DACE:55:0:7    ////__DACE:53
            double __map_fusion_gtir_tmp_41_1_1;                                              ////__DACE:55:0:10    ////__DACE:53
            double __map_fusion_gtir_tmp_37_1_1;                                              ////__DACE:55:0:12    ////__DACE:53
            ////__DACE:53
            {                                                                                 ////__DACE:55:0:6    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_60_1_1;                                             ////__DACE:55:0:14,6    ////__DACE:55:0:6    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_1;                              ////__DACE:55:0:15,6    ////__DACE:55:0:6    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:55:0:6    ////__DACE:55:0:6    ////__DACE:53
                ////__DACE:55:0:6                                                             ////__DACE:55:0:6    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:6    ////__DACE:55:0:6    ////__DACE:53
                // Tasklet code (tlet_21_multiplies_1_1)                                          ////__DACE:55:0:6    ////__DACE:55:0:6    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:55:0:6    ////__DACE:55:0:6    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:6    ////__DACE:55:0:6    ////__DACE:53
                ////__DACE:55:0:6                                                             ////__DACE:55:0:6    ////__DACE:53
                __map_fusion_gtir_tmp_63_1_1 = __tlet_result;                                     ////__DACE:55:0:6    ////__DACE:55:0:6    ////__DACE:53
            }                                                                                 ////__DACE:55:0:6    ////__DACE:53
            {                                                                                 ////__DACE:55:0:8    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_56_1_1;                                             ////__DACE:55:0:16,8    ////__DACE:55:0:8    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_1;                              ////__DACE:55:0:17,8    ////__DACE:55:0:8    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:55:0:8    ////__DACE:55:0:8    ////__DACE:53
                ////__DACE:55:0:8                                                             ////__DACE:55:0:8    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:8    ////__DACE:55:0:8    ////__DACE:53
                // Tasklet code (tlet_20_multiplies_1_1)                                          ////__DACE:55:0:8    ////__DACE:55:0:8    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:55:0:8    ////__DACE:55:0:8    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:8    ////__DACE:55:0:8    ////__DACE:53
                ////__DACE:55:0:8                                                             ////__DACE:55:0:8    ////__DACE:53
                __map_fusion_gtir_tmp_59_1_1 = __tlet_result;                                     ////__DACE:55:0:8    ////__DACE:55:0:8    ////__DACE:53
            }                                                                                 ////__DACE:55:0:8    ////__DACE:53
            {                                                                                 ////__DACE:55:0:4    ////__DACE:53
                double __tlet_arg1 = __map_fusion_gtir_tmp_63_1_1;                                ////__DACE:55:0:5,4    ////__DACE:55:0:4    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_59_1_1;                                ////__DACE:55:0:7,4    ////__DACE:55:0:4    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:55:0:4    ////__DACE:55:0:4    ////__DACE:53
                ////__DACE:55:0:4                                                             ////__DACE:55:0:4    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:4    ////__DACE:55:0:4    ////__DACE:53
                // Tasklet code (tlet_22_plus_1_1)                                                ////__DACE:55:0:4    ////__DACE:55:0:4    ////__DACE:53
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:55:0:4    ////__DACE:55:0:4    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:4    ////__DACE:55:0:4    ////__DACE:53
                ////__DACE:55:0:4                                                             ////__DACE:55:0:4    ////__DACE:53
                __arg1____ = __tlet_result;                                                       ////__DACE:55:0:4    ////__DACE:55:0:4    ////__DACE:53
            }                                                                                 ////__DACE:55:0:4    ////__DACE:53
            {                                                                                 ////__DACE:55:0:20    ////__DACE:53
                double _cpy_in = __arg1____;                                                      ////__DACE:55:0:1,20    ////__DACE:55:0:20    ////__DACE:53
                double _cpy_out;                                                                  ////__DACE:55:0:20    ////__DACE:55:0:20    ////__DACE:53
                ////__DACE:55:0:20                                                            ////__DACE:55:0:20    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:20    ////__DACE:55:0:20    ////__DACE:53
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:55:0:20    ////__DACE:55:0:20    ////__DACE:53
                _cpy_out = _cpy_in;                                                               ////__DACE:55:0:20    ////__DACE:55:0:20    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:20    ////__DACE:55:0:20    ////__DACE:53
                ////__DACE:55:0:20                                                            ////__DACE:55:0:20    ////__DACE:53
                __output = _cpy_out;                                                              ////__DACE:55:0:20    ////__DACE:55:0:20    ////__DACE:53
            }                                                                                 ////__DACE:55:0:20    ////__DACE:53
            {                                                                                 ////__DACE:55:0:11    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_1;                              ////__DACE:55:0:15,11    ////__DACE:55:0:11    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_38_1_1;                                             ////__DACE:55:0:18,11    ////__DACE:55:0:11    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:55:0:11    ////__DACE:55:0:11    ////__DACE:53
                ////__DACE:55:0:11                                                            ////__DACE:55:0:11    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:11    ////__DACE:55:0:11    ////__DACE:53
                // Tasklet code (tlet_15_multiplies_1_1)                                          ////__DACE:55:0:11    ////__DACE:55:0:11    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:55:0:11    ////__DACE:55:0:11    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:11    ////__DACE:55:0:11    ////__DACE:53
                ////__DACE:55:0:11                                                            ////__DACE:55:0:11    ////__DACE:53
                __map_fusion_gtir_tmp_41_1_1 = __tlet_result;                                     ////__DACE:55:0:11    ////__DACE:55:0:11    ////__DACE:53
            }                                                                                 ////__DACE:55:0:11    ////__DACE:53
            {                                                                                 ////__DACE:55:0:13    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_1;                              ////__DACE:55:0:17,13    ////__DACE:55:0:13    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_34_1_1;                                             ////__DACE:55:0:19,13    ////__DACE:55:0:13    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:55:0:13    ////__DACE:55:0:13    ////__DACE:53
                ////__DACE:55:0:13                                                            ////__DACE:55:0:13    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:13    ////__DACE:55:0:13    ////__DACE:53
                // Tasklet code (tlet_14_multiplies_1_1)                                          ////__DACE:55:0:13    ////__DACE:55:0:13    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:55:0:13    ////__DACE:55:0:13    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:13    ////__DACE:55:0:13    ////__DACE:53
                ////__DACE:55:0:13                                                            ////__DACE:55:0:13    ////__DACE:53
                __map_fusion_gtir_tmp_37_1_1 = __tlet_result;                                     ////__DACE:55:0:13    ////__DACE:55:0:13    ////__DACE:53
            }                                                                                 ////__DACE:55:0:13    ////__DACE:53
            {                                                                                 ////__DACE:55:0:9    ////__DACE:53
                double __tlet_arg1 = __map_fusion_gtir_tmp_41_1_1;                                ////__DACE:55:0:10,9    ////__DACE:55:0:9    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_37_1_1;                                ////__DACE:55:0:12,9    ////__DACE:55:0:9    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:55:0:9    ////__DACE:55:0:9    ////__DACE:53
                ////__DACE:55:0:9                                                             ////__DACE:55:0:9    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:9    ////__DACE:55:0:9    ////__DACE:53
                // Tasklet code (tlet_16_plus_1_1)                                                ////__DACE:55:0:9    ////__DACE:55:0:9    ////__DACE:53
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:55:0:9    ////__DACE:55:0:9    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:9    ////__DACE:55:0:9    ////__DACE:53
                ////__DACE:55:0:9                                                             ////__DACE:55:0:9    ////__DACE:53
                __arg1____from_cb_fusion_4 = __tlet_result;                                       ////__DACE:55:0:9    ////__DACE:55:0:9    ////__DACE:53
            }                                                                                 ////__DACE:55:0:9    ////__DACE:53
            {                                                                                 ////__DACE:55:0:21    ////__DACE:53
                double _cpy_in = __arg1____from_cb_fusion_4;                                      ////__DACE:55:0:3,21    ////__DACE:55:0:21    ////__DACE:53
                double _cpy_out;                                                                  ////__DACE:55:0:21    ////__DACE:55:0:21    ////__DACE:53
                ////__DACE:55:0:21                                                            ////__DACE:55:0:21    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:21    ////__DACE:55:0:21    ////__DACE:53
                // Tasklet code (copy___arg1____from_cb_fusion_4_to___output_from_cb_fusion_4)    ////__DACE:55:0:21    ////__DACE:55:0:21    ////__DACE:53
                _cpy_out = _cpy_in;                                                               ////__DACE:55:0:21    ////__DACE:55:0:21    ////__DACE:53
                ///////////////////                                                               ////__DACE:55:0:21    ////__DACE:55:0:21    ////__DACE:53
                ////__DACE:55:0:21                                                            ////__DACE:55:0:21    ////__DACE:53
                __output_from_cb_fusion_4 = _cpy_out;                                             ////__DACE:55:0:21    ////__DACE:55:0:21    ////__DACE:53
            }                                                                                 ////__DACE:55:0:21    ////__DACE:53
            ////__DACE:53
        }                                                                             ////__DACE:53
    } else {                                                                          ////__DACE:53
        {                                                                             ////__DACE:53
            double __arg2_;                                                                   ////__DACE:56:0:1    ////__DACE:53
            double __arg2_from_cb_fusion_4;                                                   ////__DACE:56:0:3    ////__DACE:53
            double __map_fusion_gtir_tmp_73_1_1;                                              ////__DACE:56:0:5    ////__DACE:53
            double __map_fusion_gtir_tmp_69_1_1;                                              ////__DACE:56:0:7    ////__DACE:53
            double __map_fusion_gtir_tmp_51_1_1;                                              ////__DACE:56:0:10    ////__DACE:53
            double __map_fusion_gtir_tmp_47_1_1;                                              ////__DACE:56:0:12    ////__DACE:53
            ////__DACE:53
            {                                                                                 ////__DACE:56:0:6    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_70_1_1;                                             ////__DACE:56:0:14,6    ////__DACE:56:0:6    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_1;                              ////__DACE:56:0:15,6    ////__DACE:56:0:6    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:56:0:6    ////__DACE:56:0:6    ////__DACE:53
                ////__DACE:56:0:6                                                             ////__DACE:56:0:6    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:6    ////__DACE:56:0:6    ////__DACE:53
                // Tasklet code (tlet_24_multiplies_1_1)                                          ////__DACE:56:0:6    ////__DACE:56:0:6    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:56:0:6    ////__DACE:56:0:6    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:6    ////__DACE:56:0:6    ////__DACE:53
                ////__DACE:56:0:6                                                             ////__DACE:56:0:6    ////__DACE:53
                __map_fusion_gtir_tmp_73_1_1 = __tlet_result;                                     ////__DACE:56:0:6    ////__DACE:56:0:6    ////__DACE:53
            }                                                                                 ////__DACE:56:0:6    ////__DACE:53
            {                                                                                 ////__DACE:56:0:8    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_66_1_1;                                             ////__DACE:56:0:16,8    ////__DACE:56:0:8    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_1;                              ////__DACE:56:0:17,8    ////__DACE:56:0:8    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:56:0:8    ////__DACE:56:0:8    ////__DACE:53
                ////__DACE:56:0:8                                                             ////__DACE:56:0:8    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:8    ////__DACE:56:0:8    ////__DACE:53
                // Tasklet code (tlet_23_multiplies_1_1)                                          ////__DACE:56:0:8    ////__DACE:56:0:8    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:56:0:8    ////__DACE:56:0:8    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:8    ////__DACE:56:0:8    ////__DACE:53
                ////__DACE:56:0:8                                                             ////__DACE:56:0:8    ////__DACE:53
                __map_fusion_gtir_tmp_69_1_1 = __tlet_result;                                     ////__DACE:56:0:8    ////__DACE:56:0:8    ////__DACE:53
            }                                                                                 ////__DACE:56:0:8    ////__DACE:53
            {                                                                                 ////__DACE:56:0:4    ////__DACE:53
                double __tlet_arg1 = __map_fusion_gtir_tmp_73_1_1;                                ////__DACE:56:0:5,4    ////__DACE:56:0:4    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_69_1_1;                                ////__DACE:56:0:7,4    ////__DACE:56:0:4    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:56:0:4    ////__DACE:56:0:4    ////__DACE:53
                ////__DACE:56:0:4                                                             ////__DACE:56:0:4    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:4    ////__DACE:56:0:4    ////__DACE:53
                // Tasklet code (tlet_25_plus_1_1)                                                ////__DACE:56:0:4    ////__DACE:56:0:4    ////__DACE:53
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:56:0:4    ////__DACE:56:0:4    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:4    ////__DACE:56:0:4    ////__DACE:53
                ////__DACE:56:0:4                                                             ////__DACE:56:0:4    ////__DACE:53
                __arg2_ = __tlet_result;                                                          ////__DACE:56:0:4    ////__DACE:56:0:4    ////__DACE:53
            }                                                                                 ////__DACE:56:0:4    ////__DACE:53
            {                                                                                 ////__DACE:56:0:20    ////__DACE:53
                double _cpy_in = __arg2_;                                                         ////__DACE:56:0:1,20    ////__DACE:56:0:20    ////__DACE:53
                double _cpy_out;                                                                  ////__DACE:56:0:20    ////__DACE:56:0:20    ////__DACE:53
                ////__DACE:56:0:20                                                            ////__DACE:56:0:20    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:20    ////__DACE:56:0:20    ////__DACE:53
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:56:0:20    ////__DACE:56:0:20    ////__DACE:53
                _cpy_out = _cpy_in;                                                               ////__DACE:56:0:20    ////__DACE:56:0:20    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:20    ////__DACE:56:0:20    ////__DACE:53
                ////__DACE:56:0:20                                                            ////__DACE:56:0:20    ////__DACE:53
                __output = _cpy_out;                                                              ////__DACE:56:0:20    ////__DACE:56:0:20    ////__DACE:53
            }                                                                                 ////__DACE:56:0:20    ////__DACE:53
            {                                                                                 ////__DACE:56:0:11    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_48_1_1;                                             ////__DACE:56:0:18,11    ////__DACE:56:0:11    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_1_1_1;                              ////__DACE:56:0:15,11    ////__DACE:56:0:11    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:56:0:11    ////__DACE:56:0:11    ////__DACE:53
                ////__DACE:56:0:11                                                            ////__DACE:56:0:11    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:11    ////__DACE:56:0:11    ////__DACE:53
                // Tasklet code (tlet_18_multiplies_1_1)                                          ////__DACE:56:0:11    ////__DACE:56:0:11    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:56:0:11    ////__DACE:56:0:11    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:11    ////__DACE:56:0:11    ////__DACE:53
                ////__DACE:56:0:11                                                            ////__DACE:56:0:11    ////__DACE:53
                __map_fusion_gtir_tmp_51_1_1 = __tlet_result;                                     ////__DACE:56:0:11    ////__DACE:56:0:11    ////__DACE:53
            }                                                                                 ////__DACE:56:0:11    ////__DACE:53
            {                                                                                 ////__DACE:56:0:13    ////__DACE:53
                double __tlet_arg1 = gtir_tmp_44_1_1;                                             ////__DACE:56:0:19,13    ////__DACE:56:0:13    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_1_1_1;                              ////__DACE:56:0:17,13    ////__DACE:56:0:13    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:56:0:13    ////__DACE:56:0:13    ////__DACE:53
                ////__DACE:56:0:13                                                            ////__DACE:56:0:13    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:13    ////__DACE:56:0:13    ////__DACE:53
                // Tasklet code (tlet_17_multiplies_1_1)                                          ////__DACE:56:0:13    ////__DACE:56:0:13    ////__DACE:53
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:56:0:13    ////__DACE:56:0:13    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:13    ////__DACE:56:0:13    ////__DACE:53
                ////__DACE:56:0:13                                                            ////__DACE:56:0:13    ////__DACE:53
                __map_fusion_gtir_tmp_47_1_1 = __tlet_result;                                     ////__DACE:56:0:13    ////__DACE:56:0:13    ////__DACE:53
            }                                                                                 ////__DACE:56:0:13    ////__DACE:53
            {                                                                                 ////__DACE:56:0:9    ////__DACE:53
                double __tlet_arg1 = __map_fusion_gtir_tmp_51_1_1;                                ////__DACE:56:0:10,9    ////__DACE:56:0:9    ////__DACE:53
                double __tlet_arg0 = __map_fusion_gtir_tmp_47_1_1;                                ////__DACE:56:0:12,9    ////__DACE:56:0:9    ////__DACE:53
                double __tlet_result;                                                             ////__DACE:56:0:9    ////__DACE:56:0:9    ////__DACE:53
                ////__DACE:56:0:9                                                             ////__DACE:56:0:9    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:9    ////__DACE:56:0:9    ////__DACE:53
                // Tasklet code (tlet_19_plus_1_1)                                                ////__DACE:56:0:9    ////__DACE:56:0:9    ////__DACE:53
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:56:0:9    ////__DACE:56:0:9    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:9    ////__DACE:56:0:9    ////__DACE:53
                ////__DACE:56:0:9                                                             ////__DACE:56:0:9    ////__DACE:53
                __arg2_from_cb_fusion_4 = __tlet_result;                                          ////__DACE:56:0:9    ////__DACE:56:0:9    ////__DACE:53
            }                                                                                 ////__DACE:56:0:9    ////__DACE:53
            {                                                                                 ////__DACE:56:0:21    ////__DACE:53
                double _cpy_in = __arg2_from_cb_fusion_4;                                         ////__DACE:56:0:3,21    ////__DACE:56:0:21    ////__DACE:53
                double _cpy_out;                                                                  ////__DACE:56:0:21    ////__DACE:56:0:21    ////__DACE:53
                ////__DACE:56:0:21                                                            ////__DACE:56:0:21    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:21    ////__DACE:56:0:21    ////__DACE:53
                // Tasklet code (copy___arg2_from_cb_fusion_4_to___output_from_cb_fusion_4)       ////__DACE:56:0:21    ////__DACE:56:0:21    ////__DACE:53
                _cpy_out = _cpy_in;                                                               ////__DACE:56:0:21    ////__DACE:56:0:21    ////__DACE:53
                ///////////////////                                                               ////__DACE:56:0:21    ////__DACE:56:0:21    ////__DACE:53
                ////__DACE:56:0:21                                                            ////__DACE:56:0:21    ////__DACE:53
                __output_from_cb_fusion_4 = _cpy_out;                                             ////__DACE:56:0:21    ////__DACE:56:0:21    ////__DACE:53
            }                                                                                 ////__DACE:56:0:21    ////__DACE:53
            ////__DACE:53
        }                                                                             ////__DACE:53
    }                                                                                 ////__DACE:53
}                                                                                 ////__DACE:0:0:217
////__DACE:0:0:217
DACE_DFI void if_stmt_7_0_0_232(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double&  gtir_tmp_54_1_1, const double&  gtir_tmp_76_1_1, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double* __restrict__ perturbed_rho_at_cells_on_model_levels, const double* __restrict__ reference_rho_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_rho_at_cells_on_model_levels_K_stride_0, int __reference_rho_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:232
    ////__DACE:61
    if (__cond) {                                                                     ////__DACE:61
        {                                                                             ////__DACE:61
            double __arg1___;                                                                 ////__DACE:63:0:1    ////__DACE:61
            double __map_fusion_gtir_tmp_191_1_1;                                             ////__DACE:63:0:3    ////__DACE:61
            double __map_fusion_gtir_tmp_189_1_1;                                             ////__DACE:63:0:5    ////__DACE:61
            double __map_fusion_gtir_tmp_187_1_1;                                             ////__DACE:63:0:7    ////__DACE:61
            double __map_fusion_gtir_tmp_185_1_1;                                             ////__DACE:63:0:9    ////__DACE:61
            double __map_fusion_gtir_tmp_183_1_1;                                             ////__DACE:63:0:11    ////__DACE:61
            double __map_fusion_gtir_tmp_181_1_1;                                             ////__DACE:63:0:13    ////__DACE:61
            double __map_fusion_gtir_tmp_179_1_1;                                             ////__DACE:63:0:15    ////__DACE:61
            ////__DACE:61
            {                                                                                 ////__DACE:63:0:6    ////__DACE:61
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:63:0:18,6    ////__DACE:63:0:6    ////__DACE:61
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:63:0:19,6    ////__DACE:63:0:6    ////__DACE:61
                double __tlet_val;                                                                ////__DACE:63:0:6    ////__DACE:63:0:6    ////__DACE:61
                ////__DACE:63:0:6                                                             ////__DACE:63:0:6    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:6    ////__DACE:63:0:6    ////__DACE:61
                // Tasklet code (tlet_75_deref_1_1)                                               ////__DACE:63:0:6    ////__DACE:63:0:6    ////__DACE:61
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:63:0:6    ////__DACE:63:0:6    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:6    ////__DACE:63:0:6    ////__DACE:61
                ////__DACE:63:0:6                                                             ////__DACE:63:0:6    ////__DACE:61
                __map_fusion_gtir_tmp_189_1_1 = __tlet_val;                                       ////__DACE:63:0:6    ////__DACE:63:0:6    ////__DACE:61
            }                                                                                 ////__DACE:63:0:6    ////__DACE:61
            {                                                                                 ////__DACE:63:0:4    ////__DACE:61
                double __tlet_arg0 = gtir_tmp_76_1_1;                                             ////__DACE:63:0:17,4    ////__DACE:63:0:4    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_189_1_1;                               ////__DACE:63:0:5,4    ////__DACE:63:0:4    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:63:0:4    ////__DACE:63:0:4    ////__DACE:61
                ////__DACE:63:0:4                                                             ////__DACE:63:0:4    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:4    ////__DACE:63:0:4    ////__DACE:61
                // Tasklet code (tlet_76_multiplies_1_1)                                          ////__DACE:63:0:4    ////__DACE:63:0:4    ////__DACE:61
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:63:0:4    ////__DACE:63:0:4    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:4    ////__DACE:63:0:4    ////__DACE:61
                ////__DACE:63:0:4                                                             ////__DACE:63:0:4    ////__DACE:61
                __map_fusion_gtir_tmp_191_1_1 = __tlet_result;                                    ////__DACE:63:0:4    ////__DACE:63:0:4    ////__DACE:61
            }                                                                                 ////__DACE:63:0:4    ////__DACE:61
            {                                                                                 ////__DACE:63:0:12    ////__DACE:61
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:63:0:21,12    ////__DACE:63:0:12    ////__DACE:61
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:63:0:19,12    ////__DACE:63:0:12    ////__DACE:61
                double __tlet_val;                                                                ////__DACE:63:0:12    ////__DACE:63:0:12    ////__DACE:61
                ////__DACE:63:0:12                                                            ////__DACE:63:0:12    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:12    ////__DACE:63:0:12    ////__DACE:61
                // Tasklet code (tlet_72_deref_1_1)                                               ////__DACE:63:0:12    ////__DACE:63:0:12    ////__DACE:61
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:63:0:12    ////__DACE:63:0:12    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:12    ////__DACE:63:0:12    ////__DACE:61
                ////__DACE:63:0:12                                                            ////__DACE:63:0:12    ////__DACE:61
                __map_fusion_gtir_tmp_183_1_1 = __tlet_val;                                       ////__DACE:63:0:12    ////__DACE:63:0:12    ////__DACE:61
            }                                                                                 ////__DACE:63:0:12    ////__DACE:61
            {                                                                                 ////__DACE:63:0:10    ////__DACE:61
                double __tlet_arg0 = gtir_tmp_54_1_1;                                             ////__DACE:63:0:20,10    ////__DACE:63:0:10    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_183_1_1;                               ////__DACE:63:0:11,10    ////__DACE:63:0:10    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:63:0:10    ////__DACE:63:0:10    ////__DACE:61
                ////__DACE:63:0:10                                                            ////__DACE:63:0:10    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:10    ////__DACE:63:0:10    ////__DACE:61
                // Tasklet code (tlet_73_multiplies_1_1)                                          ////__DACE:63:0:10    ////__DACE:63:0:10    ////__DACE:61
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:63:0:10    ////__DACE:63:0:10    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:10    ////__DACE:63:0:10    ////__DACE:61
                ////__DACE:63:0:10                                                            ////__DACE:63:0:10    ////__DACE:61
                __map_fusion_gtir_tmp_185_1_1 = __tlet_result;                                    ////__DACE:63:0:10    ////__DACE:63:0:10    ////__DACE:61
            }                                                                                 ////__DACE:63:0:10    ////__DACE:61
            {                                                                                 ////__DACE:63:0:16    ////__DACE:61
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:63:0:23,16    ////__DACE:63:0:16    ////__DACE:61
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:63:0:19,16    ////__DACE:63:0:16    ////__DACE:61
                double __tlet_val;                                                                ////__DACE:63:0:16    ////__DACE:63:0:16    ////__DACE:61
                ////__DACE:63:0:16                                                            ////__DACE:63:0:16    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:16    ////__DACE:63:0:16    ////__DACE:61
                // Tasklet code (tlet_70_deref_1_1)                                               ////__DACE:63:0:16    ////__DACE:63:0:16    ////__DACE:61
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:63:0:16    ////__DACE:63:0:16    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:16    ////__DACE:63:0:16    ////__DACE:61
                ////__DACE:63:0:16                                                            ////__DACE:63:0:16    ////__DACE:61
                __map_fusion_gtir_tmp_179_1_1 = __tlet_val;                                       ////__DACE:63:0:16    ////__DACE:63:0:16    ////__DACE:61
            }                                                                                 ////__DACE:63:0:16    ////__DACE:61
            {                                                                                 ////__DACE:63:0:14    ////__DACE:61
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:63:0:22,14    ////__DACE:63:0:14    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_179_1_1;                               ////__DACE:63:0:15,14    ////__DACE:63:0:14    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:63:0:14    ////__DACE:63:0:14    ////__DACE:61
                ////__DACE:63:0:14                                                            ////__DACE:63:0:14    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:14    ////__DACE:63:0:14    ////__DACE:61
                // Tasklet code (tlet_71_plus_1_1)                                                ////__DACE:63:0:14    ////__DACE:63:0:14    ////__DACE:61
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:63:0:14    ////__DACE:63:0:14    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:14    ////__DACE:63:0:14    ////__DACE:61
                ////__DACE:63:0:14                                                            ////__DACE:63:0:14    ////__DACE:61
                __map_fusion_gtir_tmp_181_1_1 = __tlet_result;                                    ////__DACE:63:0:14    ////__DACE:63:0:14    ////__DACE:61
            }                                                                                 ////__DACE:63:0:14    ////__DACE:61
            {                                                                                 ////__DACE:63:0:8    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_185_1_1;                               ////__DACE:63:0:9,8    ////__DACE:63:0:8    ////__DACE:61
                double __tlet_arg0 = __map_fusion_gtir_tmp_181_1_1;                               ////__DACE:63:0:13,8    ////__DACE:63:0:8    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:63:0:8    ////__DACE:63:0:8    ////__DACE:61
                ////__DACE:63:0:8                                                             ////__DACE:63:0:8    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:8    ////__DACE:63:0:8    ////__DACE:61
                // Tasklet code (tlet_74_plus_1_1)                                                ////__DACE:63:0:8    ////__DACE:63:0:8    ////__DACE:61
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:63:0:8    ////__DACE:63:0:8    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:8    ////__DACE:63:0:8    ////__DACE:61
                ////__DACE:63:0:8                                                             ////__DACE:63:0:8    ////__DACE:61
                __map_fusion_gtir_tmp_187_1_1 = __tlet_result;                                    ////__DACE:63:0:8    ////__DACE:63:0:8    ////__DACE:61
            }                                                                                 ////__DACE:63:0:8    ////__DACE:61
            {                                                                                 ////__DACE:63:0:2    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_191_1_1;                               ////__DACE:63:0:3,2    ////__DACE:63:0:2    ////__DACE:61
                double __tlet_arg0 = __map_fusion_gtir_tmp_187_1_1;                               ////__DACE:63:0:7,2    ////__DACE:63:0:2    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:63:0:2    ////__DACE:63:0:2    ////__DACE:61
                ////__DACE:63:0:2                                                             ////__DACE:63:0:2    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:2    ////__DACE:63:0:2    ////__DACE:61
                // Tasklet code (tlet_77_plus_1_1)                                                ////__DACE:63:0:2    ////__DACE:63:0:2    ////__DACE:61
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:63:0:2    ////__DACE:63:0:2    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:2    ////__DACE:63:0:2    ////__DACE:61
                ////__DACE:63:0:2                                                             ////__DACE:63:0:2    ////__DACE:61
                __arg1___ = __tlet_result;                                                        ////__DACE:63:0:2    ////__DACE:63:0:2    ////__DACE:61
            }                                                                                 ////__DACE:63:0:2    ////__DACE:61
            {                                                                                 ////__DACE:63:0:24    ////__DACE:61
                double _cpy_in = __arg1___;                                                       ////__DACE:63:0:1,24    ////__DACE:63:0:24    ////__DACE:61
                double _cpy_out;                                                                  ////__DACE:63:0:24    ////__DACE:63:0:24    ////__DACE:61
                ////__DACE:63:0:24                                                            ////__DACE:63:0:24    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:24    ////__DACE:63:0:24    ////__DACE:61
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:63:0:24    ////__DACE:63:0:24    ////__DACE:61
                _cpy_out = _cpy_in;                                                               ////__DACE:63:0:24    ////__DACE:63:0:24    ////__DACE:61
                ///////////////////                                                               ////__DACE:63:0:24    ////__DACE:63:0:24    ////__DACE:61
                ////__DACE:63:0:24                                                            ////__DACE:63:0:24    ////__DACE:61
                __output = _cpy_out;                                                              ////__DACE:63:0:24    ////__DACE:63:0:24    ////__DACE:61
            }                                                                                 ////__DACE:63:0:24    ////__DACE:61
            ////__DACE:61
        }                                                                             ////__DACE:61
    } else {                                                                          ////__DACE:61
        {                                                                             ////__DACE:61
            double __arg2;                                                                    ////__DACE:64:0:1    ////__DACE:61
            double __map_fusion_gtir_tmp_207_1_1;                                             ////__DACE:64:0:3    ////__DACE:61
            double __map_fusion_gtir_tmp_205_1_1;                                             ////__DACE:64:0:5    ////__DACE:61
            double __map_fusion_gtir_tmp_203_1_1;                                             ////__DACE:64:0:7    ////__DACE:61
            double __map_fusion_gtir_tmp_201_1_1;                                             ////__DACE:64:0:9    ////__DACE:61
            double __map_fusion_gtir_tmp_199_1_1;                                             ////__DACE:64:0:11    ////__DACE:61
            double __map_fusion_gtir_tmp_197_1_1;                                             ////__DACE:64:0:13    ////__DACE:61
            double __map_fusion_gtir_tmp_195_1_1;                                             ////__DACE:64:0:15    ////__DACE:61
            ////__DACE:61
            {                                                                                 ////__DACE:64:0:6    ////__DACE:61
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:64:0:18,6    ////__DACE:64:0:6    ////__DACE:61
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:64:0:19,6    ////__DACE:64:0:6    ////__DACE:61
                double __tlet_val;                                                                ////__DACE:64:0:6    ////__DACE:64:0:6    ////__DACE:61
                ////__DACE:64:0:6                                                             ////__DACE:64:0:6    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:6    ////__DACE:64:0:6    ////__DACE:61
                // Tasklet code (tlet_83_deref_1_1)                                               ////__DACE:64:0:6    ////__DACE:64:0:6    ////__DACE:61
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:64:0:6    ////__DACE:64:0:6    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:6    ////__DACE:64:0:6    ////__DACE:61
                ////__DACE:64:0:6                                                             ////__DACE:64:0:6    ////__DACE:61
                __map_fusion_gtir_tmp_205_1_1 = __tlet_val;                                       ////__DACE:64:0:6    ////__DACE:64:0:6    ////__DACE:61
            }                                                                                 ////__DACE:64:0:6    ////__DACE:61
            {                                                                                 ////__DACE:64:0:4    ////__DACE:61
                double __tlet_arg0 = gtir_tmp_76_1_1;                                             ////__DACE:64:0:17,4    ////__DACE:64:0:4    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_205_1_1;                               ////__DACE:64:0:5,4    ////__DACE:64:0:4    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:64:0:4    ////__DACE:64:0:4    ////__DACE:61
                ////__DACE:64:0:4                                                             ////__DACE:64:0:4    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:4    ////__DACE:64:0:4    ////__DACE:61
                // Tasklet code (tlet_84_multiplies_1_1)                                          ////__DACE:64:0:4    ////__DACE:64:0:4    ////__DACE:61
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:64:0:4    ////__DACE:64:0:4    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:4    ////__DACE:64:0:4    ////__DACE:61
                ////__DACE:64:0:4                                                             ////__DACE:64:0:4    ////__DACE:61
                __map_fusion_gtir_tmp_207_1_1 = __tlet_result;                                    ////__DACE:64:0:4    ////__DACE:64:0:4    ////__DACE:61
            }                                                                                 ////__DACE:64:0:4    ////__DACE:61
            {                                                                                 ////__DACE:64:0:12    ////__DACE:61
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:64:0:21,12    ////__DACE:64:0:12    ////__DACE:61
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:64:0:19,12    ////__DACE:64:0:12    ////__DACE:61
                double __tlet_val;                                                                ////__DACE:64:0:12    ////__DACE:64:0:12    ////__DACE:61
                ////__DACE:64:0:12                                                            ////__DACE:64:0:12    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:12    ////__DACE:64:0:12    ////__DACE:61
                // Tasklet code (tlet_80_deref_1_1)                                               ////__DACE:64:0:12    ////__DACE:64:0:12    ////__DACE:61
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:64:0:12    ////__DACE:64:0:12    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:12    ////__DACE:64:0:12    ////__DACE:61
                ////__DACE:64:0:12                                                            ////__DACE:64:0:12    ////__DACE:61
                __map_fusion_gtir_tmp_199_1_1 = __tlet_val;                                       ////__DACE:64:0:12    ////__DACE:64:0:12    ////__DACE:61
            }                                                                                 ////__DACE:64:0:12    ////__DACE:61
            {                                                                                 ////__DACE:64:0:10    ////__DACE:61
                double __tlet_arg0 = gtir_tmp_54_1_1;                                             ////__DACE:64:0:20,10    ////__DACE:64:0:10    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_199_1_1;                               ////__DACE:64:0:11,10    ////__DACE:64:0:10    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:64:0:10    ////__DACE:64:0:10    ////__DACE:61
                ////__DACE:64:0:10                                                            ////__DACE:64:0:10    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:10    ////__DACE:64:0:10    ////__DACE:61
                // Tasklet code (tlet_81_multiplies_1_1)                                          ////__DACE:64:0:10    ////__DACE:64:0:10    ////__DACE:61
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:64:0:10    ////__DACE:64:0:10    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:10    ////__DACE:64:0:10    ////__DACE:61
                ////__DACE:64:0:10                                                            ////__DACE:64:0:10    ////__DACE:61
                __map_fusion_gtir_tmp_201_1_1 = __tlet_result;                                    ////__DACE:64:0:10    ////__DACE:64:0:10    ////__DACE:61
            }                                                                                 ////__DACE:64:0:10    ////__DACE:61
            {                                                                                 ////__DACE:64:0:16    ////__DACE:61
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:64:0:23,16    ////__DACE:64:0:16    ////__DACE:61
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:64:0:19,16    ////__DACE:64:0:16    ////__DACE:61
                double __tlet_val;                                                                ////__DACE:64:0:16    ////__DACE:64:0:16    ////__DACE:61
                ////__DACE:64:0:16                                                            ////__DACE:64:0:16    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:16    ////__DACE:64:0:16    ////__DACE:61
                // Tasklet code (tlet_78_deref_1_1)                                               ////__DACE:64:0:16    ////__DACE:64:0:16    ////__DACE:61
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:64:0:16    ////__DACE:64:0:16    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:16    ////__DACE:64:0:16    ////__DACE:61
                ////__DACE:64:0:16                                                            ////__DACE:64:0:16    ////__DACE:61
                __map_fusion_gtir_tmp_195_1_1 = __tlet_val;                                       ////__DACE:64:0:16    ////__DACE:64:0:16    ////__DACE:61
            }                                                                                 ////__DACE:64:0:16    ////__DACE:61
            {                                                                                 ////__DACE:64:0:14    ////__DACE:61
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:64:0:22,14    ////__DACE:64:0:14    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_195_1_1;                               ////__DACE:64:0:15,14    ////__DACE:64:0:14    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:64:0:14    ////__DACE:64:0:14    ////__DACE:61
                ////__DACE:64:0:14                                                            ////__DACE:64:0:14    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:14    ////__DACE:64:0:14    ////__DACE:61
                // Tasklet code (tlet_79_plus_1_1)                                                ////__DACE:64:0:14    ////__DACE:64:0:14    ////__DACE:61
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:64:0:14    ////__DACE:64:0:14    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:14    ////__DACE:64:0:14    ////__DACE:61
                ////__DACE:64:0:14                                                            ////__DACE:64:0:14    ////__DACE:61
                __map_fusion_gtir_tmp_197_1_1 = __tlet_result;                                    ////__DACE:64:0:14    ////__DACE:64:0:14    ////__DACE:61
            }                                                                                 ////__DACE:64:0:14    ////__DACE:61
            {                                                                                 ////__DACE:64:0:8    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_201_1_1;                               ////__DACE:64:0:9,8    ////__DACE:64:0:8    ////__DACE:61
                double __tlet_arg0 = __map_fusion_gtir_tmp_197_1_1;                               ////__DACE:64:0:13,8    ////__DACE:64:0:8    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:64:0:8    ////__DACE:64:0:8    ////__DACE:61
                ////__DACE:64:0:8                                                             ////__DACE:64:0:8    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:8    ////__DACE:64:0:8    ////__DACE:61
                // Tasklet code (tlet_82_plus_1_1)                                                ////__DACE:64:0:8    ////__DACE:64:0:8    ////__DACE:61
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:64:0:8    ////__DACE:64:0:8    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:8    ////__DACE:64:0:8    ////__DACE:61
                ////__DACE:64:0:8                                                             ////__DACE:64:0:8    ////__DACE:61
                __map_fusion_gtir_tmp_203_1_1 = __tlet_result;                                    ////__DACE:64:0:8    ////__DACE:64:0:8    ////__DACE:61
            }                                                                                 ////__DACE:64:0:8    ////__DACE:61
            {                                                                                 ////__DACE:64:0:2    ////__DACE:61
                double __tlet_arg1 = __map_fusion_gtir_tmp_207_1_1;                               ////__DACE:64:0:3,2    ////__DACE:64:0:2    ////__DACE:61
                double __tlet_arg0 = __map_fusion_gtir_tmp_203_1_1;                               ////__DACE:64:0:7,2    ////__DACE:64:0:2    ////__DACE:61
                double __tlet_result;                                                             ////__DACE:64:0:2    ////__DACE:64:0:2    ////__DACE:61
                ////__DACE:64:0:2                                                             ////__DACE:64:0:2    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:2    ////__DACE:64:0:2    ////__DACE:61
                // Tasklet code (tlet_85_plus_1_1)                                                ////__DACE:64:0:2    ////__DACE:64:0:2    ////__DACE:61
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:64:0:2    ////__DACE:64:0:2    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:2    ////__DACE:64:0:2    ////__DACE:61
                ////__DACE:64:0:2                                                             ////__DACE:64:0:2    ////__DACE:61
                __arg2 = __tlet_result;                                                           ////__DACE:64:0:2    ////__DACE:64:0:2    ////__DACE:61
            }                                                                                 ////__DACE:64:0:2    ////__DACE:61
            {                                                                                 ////__DACE:64:0:24    ////__DACE:61
                double _cpy_in = __arg2;                                                          ////__DACE:64:0:1,24    ////__DACE:64:0:24    ////__DACE:61
                double _cpy_out;                                                                  ////__DACE:64:0:24    ////__DACE:64:0:24    ////__DACE:61
                ////__DACE:64:0:24                                                            ////__DACE:64:0:24    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:24    ////__DACE:64:0:24    ////__DACE:61
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:64:0:24    ////__DACE:64:0:24    ////__DACE:61
                _cpy_out = _cpy_in;                                                               ////__DACE:64:0:24    ////__DACE:64:0:24    ////__DACE:61
                ///////////////////                                                               ////__DACE:64:0:24    ////__DACE:64:0:24    ////__DACE:61
                ////__DACE:64:0:24                                                            ////__DACE:64:0:24    ////__DACE:61
                __output = _cpy_out;                                                              ////__DACE:64:0:24    ////__DACE:64:0:24    ////__DACE:61
            }                                                                                 ////__DACE:64:0:24    ////__DACE:61
            ////__DACE:61
        }                                                                             ////__DACE:61
    }                                                                                 ////__DACE:61
}                                                                                 ////__DACE:0:0:232
////__DACE:0:0:232
DACE_DFI void if_stmt_5_0_0_229(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double&  gtir_tmp_54_1_1, const double&  gtir_tmp_76_1_1, const double * __restrict__ gtir_tmp_95, const double* __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double* __restrict__ reference_theta_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_theta_v_at_cells_on_model_levels_K_stride_0, int __reference_theta_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:229
    ////__DACE:57
    if (__cond) {                                                                     ////__DACE:57
        {                                                                             ////__DACE:57
            double __arg1____;                                                                ////__DACE:59:0:1    ////__DACE:57
            double __map_fusion_gtir_tmp_118_1_1;                                             ////__DACE:59:0:3    ////__DACE:57
            double __map_fusion_gtir_tmp_116_1_1;                                             ////__DACE:59:0:5    ////__DACE:57
            double __map_fusion_gtir_tmp_114_1_1;                                             ////__DACE:59:0:7    ////__DACE:57
            double __map_fusion_gtir_tmp_112_1_1;                                             ////__DACE:59:0:9    ////__DACE:57
            double __map_fusion_gtir_tmp_110_1_1;                                             ////__DACE:59:0:11    ////__DACE:57
            double __map_fusion_gtir_tmp_108_1_1;                                             ////__DACE:59:0:13    ////__DACE:57
            double __map_fusion_gtir_tmp_106_1_1;                                             ////__DACE:59:0:15    ////__DACE:57
            ////__DACE:57
            {                                                                                 ////__DACE:59:0:6    ////__DACE:57
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:59:0:18,6    ////__DACE:59:0:6    ////__DACE:57
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:59:0:19,6    ////__DACE:59:0:6    ////__DACE:57
                double __tlet_val;                                                                ////__DACE:59:0:6    ////__DACE:59:0:6    ////__DACE:57
                ////__DACE:59:0:6                                                             ////__DACE:59:0:6    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:6    ////__DACE:59:0:6    ////__DACE:57
                // Tasklet code (tlet_41_deref_1_1)                                               ////__DACE:59:0:6    ////__DACE:59:0:6    ////__DACE:57
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:59:0:6    ////__DACE:59:0:6    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:6    ////__DACE:59:0:6    ////__DACE:57
                ////__DACE:59:0:6                                                             ////__DACE:59:0:6    ////__DACE:57
                __map_fusion_gtir_tmp_116_1_1 = __tlet_val;                                       ////__DACE:59:0:6    ////__DACE:59:0:6    ////__DACE:57
            }                                                                                 ////__DACE:59:0:6    ////__DACE:57
            {                                                                                 ////__DACE:59:0:4    ////__DACE:57
                double __tlet_arg0 = gtir_tmp_76_1_1;                                             ////__DACE:59:0:17,4    ////__DACE:59:0:4    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_116_1_1;                               ////__DACE:59:0:5,4    ////__DACE:59:0:4    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:59:0:4    ////__DACE:59:0:4    ////__DACE:57
                ////__DACE:59:0:4                                                             ////__DACE:59:0:4    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:4    ////__DACE:59:0:4    ////__DACE:57
                // Tasklet code (tlet_42_multiplies_1_1)                                          ////__DACE:59:0:4    ////__DACE:59:0:4    ////__DACE:57
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:59:0:4    ////__DACE:59:0:4    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:4    ////__DACE:59:0:4    ////__DACE:57
                ////__DACE:59:0:4                                                             ////__DACE:59:0:4    ////__DACE:57
                __map_fusion_gtir_tmp_118_1_1 = __tlet_result;                                    ////__DACE:59:0:4    ////__DACE:59:0:4    ////__DACE:57
            }                                                                                 ////__DACE:59:0:4    ////__DACE:57
            {                                                                                 ////__DACE:59:0:12    ////__DACE:57
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:59:0:21,12    ////__DACE:59:0:12    ////__DACE:57
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:59:0:19,12    ////__DACE:59:0:12    ////__DACE:57
                double __tlet_val;                                                                ////__DACE:59:0:12    ////__DACE:59:0:12    ////__DACE:57
                ////__DACE:59:0:12                                                            ////__DACE:59:0:12    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:12    ////__DACE:59:0:12    ////__DACE:57
                // Tasklet code (tlet_38_deref_1_1)                                               ////__DACE:59:0:12    ////__DACE:59:0:12    ////__DACE:57
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:59:0:12    ////__DACE:59:0:12    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:12    ////__DACE:59:0:12    ////__DACE:57
                ////__DACE:59:0:12                                                            ////__DACE:59:0:12    ////__DACE:57
                __map_fusion_gtir_tmp_110_1_1 = __tlet_val;                                       ////__DACE:59:0:12    ////__DACE:59:0:12    ////__DACE:57
            }                                                                                 ////__DACE:59:0:12    ////__DACE:57
            {                                                                                 ////__DACE:59:0:10    ////__DACE:57
                double __tlet_arg0 = gtir_tmp_54_1_1;                                             ////__DACE:59:0:20,10    ////__DACE:59:0:10    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_110_1_1;                               ////__DACE:59:0:11,10    ////__DACE:59:0:10    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:59:0:10    ////__DACE:59:0:10    ////__DACE:57
                ////__DACE:59:0:10                                                            ////__DACE:59:0:10    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:10    ////__DACE:59:0:10    ////__DACE:57
                // Tasklet code (tlet_39_multiplies_1_1)                                          ////__DACE:59:0:10    ////__DACE:59:0:10    ////__DACE:57
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:59:0:10    ////__DACE:59:0:10    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:10    ////__DACE:59:0:10    ////__DACE:57
                ////__DACE:59:0:10                                                            ////__DACE:59:0:10    ////__DACE:57
                __map_fusion_gtir_tmp_112_1_1 = __tlet_result;                                    ////__DACE:59:0:10    ////__DACE:59:0:10    ////__DACE:57
            }                                                                                 ////__DACE:59:0:10    ////__DACE:57
            {                                                                                 ////__DACE:59:0:16    ////__DACE:57
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:59:0:23,16    ////__DACE:59:0:16    ////__DACE:57
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:59:0:19,16    ////__DACE:59:0:16    ////__DACE:57
                double __tlet_val;                                                                ////__DACE:59:0:16    ////__DACE:59:0:16    ////__DACE:57
                ////__DACE:59:0:16                                                            ////__DACE:59:0:16    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:16    ////__DACE:59:0:16    ////__DACE:57
                // Tasklet code (tlet_36_deref_1_1)                                               ////__DACE:59:0:16    ////__DACE:59:0:16    ////__DACE:57
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:59:0:16    ////__DACE:59:0:16    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:16    ////__DACE:59:0:16    ////__DACE:57
                ////__DACE:59:0:16                                                            ////__DACE:59:0:16    ////__DACE:57
                __map_fusion_gtir_tmp_106_1_1 = __tlet_val;                                       ////__DACE:59:0:16    ////__DACE:59:0:16    ////__DACE:57
            }                                                                                 ////__DACE:59:0:16    ////__DACE:57
            {                                                                                 ////__DACE:59:0:14    ////__DACE:57
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:59:0:22,14    ////__DACE:59:0:14    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_106_1_1;                               ////__DACE:59:0:15,14    ////__DACE:59:0:14    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:59:0:14    ////__DACE:59:0:14    ////__DACE:57
                ////__DACE:59:0:14                                                            ////__DACE:59:0:14    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:14    ////__DACE:59:0:14    ////__DACE:57
                // Tasklet code (tlet_37_plus_1_1)                                                ////__DACE:59:0:14    ////__DACE:59:0:14    ////__DACE:57
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:59:0:14    ////__DACE:59:0:14    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:14    ////__DACE:59:0:14    ////__DACE:57
                ////__DACE:59:0:14                                                            ////__DACE:59:0:14    ////__DACE:57
                __map_fusion_gtir_tmp_108_1_1 = __tlet_result;                                    ////__DACE:59:0:14    ////__DACE:59:0:14    ////__DACE:57
            }                                                                                 ////__DACE:59:0:14    ////__DACE:57
            {                                                                                 ////__DACE:59:0:8    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_112_1_1;                               ////__DACE:59:0:9,8    ////__DACE:59:0:8    ////__DACE:57
                double __tlet_arg0 = __map_fusion_gtir_tmp_108_1_1;                               ////__DACE:59:0:13,8    ////__DACE:59:0:8    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:59:0:8    ////__DACE:59:0:8    ////__DACE:57
                ////__DACE:59:0:8                                                             ////__DACE:59:0:8    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:8    ////__DACE:59:0:8    ////__DACE:57
                // Tasklet code (tlet_40_plus_1_1)                                                ////__DACE:59:0:8    ////__DACE:59:0:8    ////__DACE:57
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:59:0:8    ////__DACE:59:0:8    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:8    ////__DACE:59:0:8    ////__DACE:57
                ////__DACE:59:0:8                                                             ////__DACE:59:0:8    ////__DACE:57
                __map_fusion_gtir_tmp_114_1_1 = __tlet_result;                                    ////__DACE:59:0:8    ////__DACE:59:0:8    ////__DACE:57
            }                                                                                 ////__DACE:59:0:8    ////__DACE:57
            {                                                                                 ////__DACE:59:0:2    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_118_1_1;                               ////__DACE:59:0:3,2    ////__DACE:59:0:2    ////__DACE:57
                double __tlet_arg0 = __map_fusion_gtir_tmp_114_1_1;                               ////__DACE:59:0:7,2    ////__DACE:59:0:2    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:59:0:2    ////__DACE:59:0:2    ////__DACE:57
                ////__DACE:59:0:2                                                             ////__DACE:59:0:2    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:2    ////__DACE:59:0:2    ////__DACE:57
                // Tasklet code (tlet_43_plus_1_1)                                                ////__DACE:59:0:2    ////__DACE:59:0:2    ////__DACE:57
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:59:0:2    ////__DACE:59:0:2    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:2    ////__DACE:59:0:2    ////__DACE:57
                ////__DACE:59:0:2                                                             ////__DACE:59:0:2    ////__DACE:57
                __arg1____ = __tlet_result;                                                       ////__DACE:59:0:2    ////__DACE:59:0:2    ////__DACE:57
            }                                                                                 ////__DACE:59:0:2    ////__DACE:57
            {                                                                                 ////__DACE:59:0:24    ////__DACE:57
                double _cpy_in = __arg1____;                                                      ////__DACE:59:0:1,24    ////__DACE:59:0:24    ////__DACE:57
                double _cpy_out;                                                                  ////__DACE:59:0:24    ////__DACE:59:0:24    ////__DACE:57
                ////__DACE:59:0:24                                                            ////__DACE:59:0:24    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:24    ////__DACE:59:0:24    ////__DACE:57
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:59:0:24    ////__DACE:59:0:24    ////__DACE:57
                _cpy_out = _cpy_in;                                                               ////__DACE:59:0:24    ////__DACE:59:0:24    ////__DACE:57
                ///////////////////                                                               ////__DACE:59:0:24    ////__DACE:59:0:24    ////__DACE:57
                ////__DACE:59:0:24                                                            ////__DACE:59:0:24    ////__DACE:57
                __output = _cpy_out;                                                              ////__DACE:59:0:24    ////__DACE:59:0:24    ////__DACE:57
            }                                                                                 ////__DACE:59:0:24    ////__DACE:57
            ////__DACE:57
        }                                                                             ////__DACE:57
    } else {                                                                          ////__DACE:57
        {                                                                             ////__DACE:57
            double __arg2_;                                                                   ////__DACE:60:0:1    ////__DACE:57
            double __map_fusion_gtir_tmp_134_1_1;                                             ////__DACE:60:0:3    ////__DACE:57
            double __map_fusion_gtir_tmp_132_1_1;                                             ////__DACE:60:0:5    ////__DACE:57
            double __map_fusion_gtir_tmp_130_1_1;                                             ////__DACE:60:0:7    ////__DACE:57
            double __map_fusion_gtir_tmp_128_1_1;                                             ////__DACE:60:0:9    ////__DACE:57
            double __map_fusion_gtir_tmp_126_1_1;                                             ////__DACE:60:0:11    ////__DACE:57
            double __map_fusion_gtir_tmp_124_1_1;                                             ////__DACE:60:0:13    ////__DACE:57
            double __map_fusion_gtir_tmp_122_1_1;                                             ////__DACE:60:0:15    ////__DACE:57
            ////__DACE:57
            {                                                                                 ////__DACE:60:0:6    ////__DACE:57
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:60:0:18,6    ////__DACE:60:0:6    ////__DACE:57
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:60:0:19,6    ////__DACE:60:0:6    ////__DACE:57
                double __tlet_val;                                                                ////__DACE:60:0:6    ////__DACE:60:0:6    ////__DACE:57
                ////__DACE:60:0:6                                                             ////__DACE:60:0:6    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:6    ////__DACE:60:0:6    ////__DACE:57
                // Tasklet code (tlet_49_deref_1_1)                                               ////__DACE:60:0:6    ////__DACE:60:0:6    ////__DACE:57
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:60:0:6    ////__DACE:60:0:6    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:6    ////__DACE:60:0:6    ////__DACE:57
                ////__DACE:60:0:6                                                             ////__DACE:60:0:6    ////__DACE:57
                __map_fusion_gtir_tmp_132_1_1 = __tlet_val;                                       ////__DACE:60:0:6    ////__DACE:60:0:6    ////__DACE:57
            }                                                                                 ////__DACE:60:0:6    ////__DACE:57
            {                                                                                 ////__DACE:60:0:4    ////__DACE:57
                double __tlet_arg0 = gtir_tmp_76_1_1;                                             ////__DACE:60:0:17,4    ////__DACE:60:0:4    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_132_1_1;                               ////__DACE:60:0:5,4    ////__DACE:60:0:4    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:60:0:4    ////__DACE:60:0:4    ////__DACE:57
                ////__DACE:60:0:4                                                             ////__DACE:60:0:4    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:4    ////__DACE:60:0:4    ////__DACE:57
                // Tasklet code (tlet_50_multiplies_1_1)                                          ////__DACE:60:0:4    ////__DACE:60:0:4    ////__DACE:57
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:60:0:4    ////__DACE:60:0:4    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:4    ////__DACE:60:0:4    ////__DACE:57
                ////__DACE:60:0:4                                                             ////__DACE:60:0:4    ////__DACE:57
                __map_fusion_gtir_tmp_134_1_1 = __tlet_result;                                    ////__DACE:60:0:4    ////__DACE:60:0:4    ////__DACE:57
            }                                                                                 ////__DACE:60:0:4    ////__DACE:57
            {                                                                                 ////__DACE:60:0:12    ////__DACE:57
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:60:0:21,12    ////__DACE:60:0:12    ////__DACE:57
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:60:0:19,12    ////__DACE:60:0:12    ////__DACE:57
                double __tlet_val;                                                                ////__DACE:60:0:12    ////__DACE:60:0:12    ////__DACE:57
                ////__DACE:60:0:12                                                            ////__DACE:60:0:12    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:12    ////__DACE:60:0:12    ////__DACE:57
                // Tasklet code (tlet_46_deref_1_1)                                               ////__DACE:60:0:12    ////__DACE:60:0:12    ////__DACE:57
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:60:0:12    ////__DACE:60:0:12    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:12    ////__DACE:60:0:12    ////__DACE:57
                ////__DACE:60:0:12                                                            ////__DACE:60:0:12    ////__DACE:57
                __map_fusion_gtir_tmp_126_1_1 = __tlet_val;                                       ////__DACE:60:0:12    ////__DACE:60:0:12    ////__DACE:57
            }                                                                                 ////__DACE:60:0:12    ////__DACE:57
            {                                                                                 ////__DACE:60:0:10    ////__DACE:57
                double __tlet_arg0 = gtir_tmp_54_1_1;                                             ////__DACE:60:0:20,10    ////__DACE:60:0:10    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_126_1_1;                               ////__DACE:60:0:11,10    ////__DACE:60:0:10    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:60:0:10    ////__DACE:60:0:10    ////__DACE:57
                ////__DACE:60:0:10                                                            ////__DACE:60:0:10    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:10    ////__DACE:60:0:10    ////__DACE:57
                // Tasklet code (tlet_47_multiplies_1_1)                                          ////__DACE:60:0:10    ////__DACE:60:0:10    ////__DACE:57
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:60:0:10    ////__DACE:60:0:10    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:10    ////__DACE:60:0:10    ////__DACE:57
                ////__DACE:60:0:10                                                            ////__DACE:60:0:10    ////__DACE:57
                __map_fusion_gtir_tmp_128_1_1 = __tlet_result;                                    ////__DACE:60:0:10    ////__DACE:60:0:10    ////__DACE:57
            }                                                                                 ////__DACE:60:0:10    ////__DACE:57
            {                                                                                 ////__DACE:60:0:16    ////__DACE:57
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:60:0:23,16    ////__DACE:60:0:16    ////__DACE:57
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:60:0:19,16    ////__DACE:60:0:16    ////__DACE:57
                double __tlet_val;                                                                ////__DACE:60:0:16    ////__DACE:60:0:16    ////__DACE:57
                ////__DACE:60:0:16                                                            ////__DACE:60:0:16    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:16    ////__DACE:60:0:16    ////__DACE:57
                // Tasklet code (tlet_44_deref_1_1)                                               ////__DACE:60:0:16    ////__DACE:60:0:16    ////__DACE:57
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:60:0:16    ////__DACE:60:0:16    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:16    ////__DACE:60:0:16    ////__DACE:57
                ////__DACE:60:0:16                                                            ////__DACE:60:0:16    ////__DACE:57
                __map_fusion_gtir_tmp_122_1_1 = __tlet_val;                                       ////__DACE:60:0:16    ////__DACE:60:0:16    ////__DACE:57
            }                                                                                 ////__DACE:60:0:16    ////__DACE:57
            {                                                                                 ////__DACE:60:0:14    ////__DACE:57
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:60:0:22,14    ////__DACE:60:0:14    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_122_1_1;                               ////__DACE:60:0:15,14    ////__DACE:60:0:14    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:60:0:14    ////__DACE:60:0:14    ////__DACE:57
                ////__DACE:60:0:14                                                            ////__DACE:60:0:14    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:14    ////__DACE:60:0:14    ////__DACE:57
                // Tasklet code (tlet_45_plus_1_1)                                                ////__DACE:60:0:14    ////__DACE:60:0:14    ////__DACE:57
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:60:0:14    ////__DACE:60:0:14    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:14    ////__DACE:60:0:14    ////__DACE:57
                ////__DACE:60:0:14                                                            ////__DACE:60:0:14    ////__DACE:57
                __map_fusion_gtir_tmp_124_1_1 = __tlet_result;                                    ////__DACE:60:0:14    ////__DACE:60:0:14    ////__DACE:57
            }                                                                                 ////__DACE:60:0:14    ////__DACE:57
            {                                                                                 ////__DACE:60:0:8    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_128_1_1;                               ////__DACE:60:0:9,8    ////__DACE:60:0:8    ////__DACE:57
                double __tlet_arg0 = __map_fusion_gtir_tmp_124_1_1;                               ////__DACE:60:0:13,8    ////__DACE:60:0:8    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:60:0:8    ////__DACE:60:0:8    ////__DACE:57
                ////__DACE:60:0:8                                                             ////__DACE:60:0:8    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:8    ////__DACE:60:0:8    ////__DACE:57
                // Tasklet code (tlet_48_plus_1_1)                                                ////__DACE:60:0:8    ////__DACE:60:0:8    ////__DACE:57
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:60:0:8    ////__DACE:60:0:8    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:8    ////__DACE:60:0:8    ////__DACE:57
                ////__DACE:60:0:8                                                             ////__DACE:60:0:8    ////__DACE:57
                __map_fusion_gtir_tmp_130_1_1 = __tlet_result;                                    ////__DACE:60:0:8    ////__DACE:60:0:8    ////__DACE:57
            }                                                                                 ////__DACE:60:0:8    ////__DACE:57
            {                                                                                 ////__DACE:60:0:2    ////__DACE:57
                double __tlet_arg1 = __map_fusion_gtir_tmp_134_1_1;                               ////__DACE:60:0:3,2    ////__DACE:60:0:2    ////__DACE:57
                double __tlet_arg0 = __map_fusion_gtir_tmp_130_1_1;                               ////__DACE:60:0:7,2    ////__DACE:60:0:2    ////__DACE:57
                double __tlet_result;                                                             ////__DACE:60:0:2    ////__DACE:60:0:2    ////__DACE:57
                ////__DACE:60:0:2                                                             ////__DACE:60:0:2    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:2    ////__DACE:60:0:2    ////__DACE:57
                // Tasklet code (tlet_51_plus_1_1)                                                ////__DACE:60:0:2    ////__DACE:60:0:2    ////__DACE:57
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:60:0:2    ////__DACE:60:0:2    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:2    ////__DACE:60:0:2    ////__DACE:57
                ////__DACE:60:0:2                                                             ////__DACE:60:0:2    ////__DACE:57
                __arg2_ = __tlet_result;                                                          ////__DACE:60:0:2    ////__DACE:60:0:2    ////__DACE:57
            }                                                                                 ////__DACE:60:0:2    ////__DACE:57
            {                                                                                 ////__DACE:60:0:24    ////__DACE:57
                double _cpy_in = __arg2_;                                                         ////__DACE:60:0:1,24    ////__DACE:60:0:24    ////__DACE:57
                double _cpy_out;                                                                  ////__DACE:60:0:24    ////__DACE:60:0:24    ////__DACE:57
                ////__DACE:60:0:24                                                            ////__DACE:60:0:24    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:24    ////__DACE:60:0:24    ////__DACE:57
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:60:0:24    ////__DACE:60:0:24    ////__DACE:57
                _cpy_out = _cpy_in;                                                               ////__DACE:60:0:24    ////__DACE:60:0:24    ////__DACE:57
                ///////////////////                                                               ////__DACE:60:0:24    ////__DACE:60:0:24    ////__DACE:57
                ////__DACE:60:0:24                                                            ////__DACE:60:0:24    ////__DACE:57
                __output = _cpy_out;                                                              ////__DACE:60:0:24    ////__DACE:60:0:24    ////__DACE:57
            }                                                                                 ////__DACE:60:0:24    ////__DACE:57
            ////__DACE:57
        }                                                                             ////__DACE:57
    }                                                                                 ////__DACE:57
}                                                                                 ////__DACE:0:0:229
////__DACE:0:0:229
DACE_DFI void reduce_0_0_359(double* __restrict__ _in, double&  _out) {           ////__DACE:0:0:359
    ////__DACE:73
    {                                                                                 ////__DACE:73
        ////__DACE:73
        {                                                                                 ////__DACE:73:0:0    ////__DACE:73
            for (auto _o0 = 0; _o0 < 1; _o0 += 1) {                                       ////__DACE:73:0:0    ////__DACE:73
                {                                                                         ////__DACE:73:0:1    ////__DACE:73
                    double __out;                                                                     ////__DACE:73:0:1    ////__DACE:73:0:1    ////__DACE:73
                    ////__DACE:73:0:1                                                     ////__DACE:73:0:1    ////__DACE:73
                    ///////////////////                                                               ////__DACE:73:0:1    ////__DACE:73:0:1    ////__DACE:73
                    // Tasklet code (reduce_init)                                                     ////__DACE:73:0:1    ////__DACE:73:0:1    ////__DACE:73
                    __out = 0;                                                                        ////__DACE:73:0:1    ////__DACE:73:0:1    ////__DACE:73
                    ///////////////////                                                               ////__DACE:73:0:1    ////__DACE:73:0:1    ////__DACE:73
                    ////__DACE:73:0:1                                                     ////__DACE:73:0:1    ////__DACE:73
                    _out = __out;                                                                     ////__DACE:73:0:1    ////__DACE:73:0:1    ////__DACE:73
                }                                                                         ////__DACE:73:0:1    ////__DACE:73
            }                                                                             ////__DACE:73:0:2    ////__DACE:73
        }                                                                                 ////__DACE:73:0:2    ////__DACE:73
        ////__DACE:73
    }                                                                                 ////__DACE:73
    {                                                                                 ////__DACE:73
        ////__DACE:73
        {                                                                                 ////__DACE:73:1:0    ////__DACE:73
            for (auto _i0 = 0; _i0 < 2; _i0 += 1) {                                       ////__DACE:73:1:0    ////__DACE:73
                {                                                                         ////__DACE:73:1:2    ////__DACE:73
                    double __inp = _in[_i0];                                                          ////__DACE:73:1:3,2    ////__DACE:73:1:2    ////__DACE:73
                    double __out;                                                                     ////__DACE:73:1:2    ////__DACE:73:1:2    ////__DACE:73
                    ////__DACE:73:1:2                                                     ////__DACE:73:1:2    ////__DACE:73
                    ///////////////////                                                               ////__DACE:73:1:2    ////__DACE:73:1:2    ////__DACE:73
                    // Tasklet code (identity)                                                        ////__DACE:73:1:2    ////__DACE:73:1:2    ////__DACE:73
                    __out = __inp;                                                                    ////__DACE:73:1:2    ////__DACE:73:1:2    ////__DACE:73
                    ///////////////////                                                               ////__DACE:73:1:2    ////__DACE:73:1:2    ////__DACE:73
                    ////__DACE:73:1:2                                                     ////__DACE:73:1:2    ////__DACE:73
                    dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce(&_out, __out);          ////__DACE:73:1:2    ////__DACE:73:1:2    ////__DACE:73
                }                                                                         ////__DACE:73:1:2    ////__DACE:73
            }                                                                             ////__DACE:73:1:1    ////__DACE:73
        }                                                                                 ////__DACE:73:1:1    ////__DACE:73
        ////__DACE:73
    }                                                                                 ////__DACE:73
}                                                                                 ////__DACE:0:0:359
////__DACE:0:0:359
DACE_DFI void if_stmt_6_0_0_266(const double&  __arg2_, const bool&  __cond, const double* __restrict__ hydrostatic_correction_on_lowest_level, const double* __restrict__ pg_exdist, double&  __output, int __pg_exdist_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:266
    ////__DACE:65
    if (__cond) {                                                                     ////__DACE:65
        {                                                                             ////__DACE:65
            double __arg1__;                                                                  ////__DACE:67:0:1    ////__DACE:65
            double __map_fusion_gtir_tmp_170_1_0;                                             ////__DACE:67:0:3    ////__DACE:65
            ////__DACE:65
            {                                                                                 ////__DACE:67:0:4    ////__DACE:65
                double __tlet_arg0 = hydrostatic_correction_on_lowest_level[i_Edge_gtx_horizontal];    ////__DACE:67:0:6,4    ////__DACE:67:0:4    ////__DACE:65
                double __tlet_arg1 = pg_exdist[((__pg_exdist_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:67:0:7,4    ////__DACE:67:0:4    ////__DACE:65
                double __tlet_result;                                                             ////__DACE:67:0:4    ////__DACE:67:0:4    ////__DACE:65
                ////__DACE:67:0:4                                                             ////__DACE:67:0:4    ////__DACE:65
                ///////////////////                                                               ////__DACE:67:0:4    ////__DACE:67:0:4    ////__DACE:65
                // Tasklet code (tlet_66_multiplies_1_0)                                          ////__DACE:67:0:4    ////__DACE:67:0:4    ////__DACE:65
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:67:0:4    ////__DACE:67:0:4    ////__DACE:65
                ///////////////////                                                               ////__DACE:67:0:4    ////__DACE:67:0:4    ////__DACE:65
                ////__DACE:67:0:4                                                             ////__DACE:67:0:4    ////__DACE:65
                __map_fusion_gtir_tmp_170_1_0 = __tlet_result;                                    ////__DACE:67:0:4    ////__DACE:67:0:4    ////__DACE:65
            }                                                                                 ////__DACE:67:0:4    ////__DACE:65
            {                                                                                 ////__DACE:67:0:2    ////__DACE:65
                double __tlet_arg0 = __arg2_;                                                     ////__DACE:67:0:5,2    ////__DACE:67:0:2    ////__DACE:65
                double __tlet_arg1 = __map_fusion_gtir_tmp_170_1_0;                               ////__DACE:67:0:3,2    ////__DACE:67:0:2    ////__DACE:65
                double __tlet_result;                                                             ////__DACE:67:0:2    ////__DACE:67:0:2    ////__DACE:65
                ////__DACE:67:0:2                                                             ////__DACE:67:0:2    ////__DACE:65
                ///////////////////                                                               ////__DACE:67:0:2    ////__DACE:67:0:2    ////__DACE:65
                // Tasklet code (tlet_67_plus_1_0)                                                ////__DACE:67:0:2    ////__DACE:67:0:2    ////__DACE:65
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:67:0:2    ////__DACE:67:0:2    ////__DACE:65
                ///////////////////                                                               ////__DACE:67:0:2    ////__DACE:67:0:2    ////__DACE:65
                ////__DACE:67:0:2                                                             ////__DACE:67:0:2    ////__DACE:65
                __arg1__ = __tlet_result;                                                         ////__DACE:67:0:2    ////__DACE:67:0:2    ////__DACE:65
            }                                                                                 ////__DACE:67:0:2    ////__DACE:65
            {                                                                                 ////__DACE:67:0:8    ////__DACE:65
                double _cpy_in = __arg1__;                                                        ////__DACE:67:0:1,8    ////__DACE:67:0:8    ////__DACE:65
                double _cpy_out;                                                                  ////__DACE:67:0:8    ////__DACE:67:0:8    ////__DACE:65
                ////__DACE:67:0:8                                                             ////__DACE:67:0:8    ////__DACE:65
                ///////////////////                                                               ////__DACE:67:0:8    ////__DACE:67:0:8    ////__DACE:65
                // Tasklet code (copy___arg1___to___output)                                       ////__DACE:67:0:8    ////__DACE:67:0:8    ////__DACE:65
                _cpy_out = _cpy_in;                                                               ////__DACE:67:0:8    ////__DACE:67:0:8    ////__DACE:65
                ///////////////////                                                               ////__DACE:67:0:8    ////__DACE:67:0:8    ////__DACE:65
                ////__DACE:67:0:8                                                             ////__DACE:67:0:8    ////__DACE:65
                __output = _cpy_out;                                                              ////__DACE:67:0:8    ////__DACE:67:0:8    ////__DACE:65
            }                                                                                 ////__DACE:67:0:8    ////__DACE:65
            ////__DACE:65
        }                                                                             ////__DACE:65
    } else {                                                                          ////__DACE:65
        {                                                                             ////__DACE:65
            ////__DACE:65
            {                                                                                 ////__DACE:68:0:2    ////__DACE:65
                double _cpy_in = __arg2_;                                                         ////__DACE:68:0:1,2    ////__DACE:68:0:2    ////__DACE:65
                double _cpy_out;                                                                  ////__DACE:68:0:2    ////__DACE:68:0:2    ////__DACE:65
                ////__DACE:68:0:2                                                             ////__DACE:68:0:2    ////__DACE:65
                ///////////////////                                                               ////__DACE:68:0:2    ////__DACE:68:0:2    ////__DACE:65
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:68:0:2    ////__DACE:68:0:2    ////__DACE:65
                _cpy_out = _cpy_in;                                                               ////__DACE:68:0:2    ////__DACE:68:0:2    ////__DACE:65
                ///////////////////                                                               ////__DACE:68:0:2    ////__DACE:68:0:2    ////__DACE:65
                ////__DACE:68:0:2                                                             ////__DACE:68:0:2    ////__DACE:65
                __output = _cpy_out;                                                              ////__DACE:68:0:2    ////__DACE:68:0:2    ////__DACE:65
            }                                                                                 ////__DACE:68:0:2    ////__DACE:65
            ////__DACE:65
        }                                                                             ////__DACE:65
    }                                                                                 ////__DACE:65
}                                                                                 ////__DACE:0:0:266
////__DACE:0:0:266
DACE_DFI void if_stmt_1_0_0_113(const double&  __arg1___, const double&  __arg1____from_cb_fusion_0, const double&  __arg2, const double&  __arg2_from_cb_fusion_0, const bool&  __cond, double&  __output, double&  __output_from_cb_fusion_0) {    ////__DACE:0:0:113
    ////__DACE:9
    if (__cond) {                                                                     ////__DACE:9
        {                                                                             ////__DACE:9
            ////__DACE:9
            {                                                                                 ////__DACE:11:0:4    ////__DACE:9
                double _cpy_in = __arg1___;                                                       ////__DACE:11:0:1,4    ////__DACE:11:0:4    ////__DACE:9
                double _cpy_out;                                                                  ////__DACE:11:0:4    ////__DACE:11:0:4    ////__DACE:9
                ////__DACE:11:0:4                                                             ////__DACE:11:0:4    ////__DACE:9
                ///////////////////                                                               ////__DACE:11:0:4    ////__DACE:11:0:4    ////__DACE:9
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:11:0:4    ////__DACE:11:0:4    ////__DACE:9
                _cpy_out = _cpy_in;                                                               ////__DACE:11:0:4    ////__DACE:11:0:4    ////__DACE:9
                ///////////////////                                                               ////__DACE:11:0:4    ////__DACE:11:0:4    ////__DACE:9
                ////__DACE:11:0:4                                                             ////__DACE:11:0:4    ////__DACE:9
                __output = _cpy_out;                                                              ////__DACE:11:0:4    ////__DACE:11:0:4    ////__DACE:9
            }                                                                                 ////__DACE:11:0:4    ////__DACE:9
            {                                                                                 ////__DACE:11:0:5    ////__DACE:9
                double _cpy_in = __arg1____from_cb_fusion_0;                                      ////__DACE:11:0:3,5    ////__DACE:11:0:5    ////__DACE:9
                double _cpy_out;                                                                  ////__DACE:11:0:5    ////__DACE:11:0:5    ////__DACE:9
                ////__DACE:11:0:5                                                             ////__DACE:11:0:5    ////__DACE:9
                ///////////////////                                                               ////__DACE:11:0:5    ////__DACE:11:0:5    ////__DACE:9
                // Tasklet code (copy___arg1____from_cb_fusion_0_to___output_from_cb_fusion_0)    ////__DACE:11:0:5    ////__DACE:11:0:5    ////__DACE:9
                _cpy_out = _cpy_in;                                                               ////__DACE:11:0:5    ////__DACE:11:0:5    ////__DACE:9
                ///////////////////                                                               ////__DACE:11:0:5    ////__DACE:11:0:5    ////__DACE:9
                ////__DACE:11:0:5                                                             ////__DACE:11:0:5    ////__DACE:9
                __output_from_cb_fusion_0 = _cpy_out;                                             ////__DACE:11:0:5    ////__DACE:11:0:5    ////__DACE:9
            }                                                                                 ////__DACE:11:0:5    ////__DACE:9
            ////__DACE:9
        }                                                                             ////__DACE:9
    } else {                                                                          ////__DACE:9
        {                                                                             ////__DACE:9
            ////__DACE:9
            {                                                                                 ////__DACE:12:0:4    ////__DACE:9
                double _cpy_in = __arg2;                                                          ////__DACE:12:0:1,4    ////__DACE:12:0:4    ////__DACE:9
                double _cpy_out;                                                                  ////__DACE:12:0:4    ////__DACE:12:0:4    ////__DACE:9
                ////__DACE:12:0:4                                                             ////__DACE:12:0:4    ////__DACE:9
                ///////////////////                                                               ////__DACE:12:0:4    ////__DACE:12:0:4    ////__DACE:9
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:12:0:4    ////__DACE:12:0:4    ////__DACE:9
                _cpy_out = _cpy_in;                                                               ////__DACE:12:0:4    ////__DACE:12:0:4    ////__DACE:9
                ///////////////////                                                               ////__DACE:12:0:4    ////__DACE:12:0:4    ////__DACE:9
                ////__DACE:12:0:4                                                             ////__DACE:12:0:4    ////__DACE:9
                __output = _cpy_out;                                                              ////__DACE:12:0:4    ////__DACE:12:0:4    ////__DACE:9
            }                                                                                 ////__DACE:12:0:4    ////__DACE:9
            {                                                                                 ////__DACE:12:0:5    ////__DACE:9
                double _cpy_in = __arg2_from_cb_fusion_0;                                         ////__DACE:12:0:3,5    ////__DACE:12:0:5    ////__DACE:9
                double _cpy_out;                                                                  ////__DACE:12:0:5    ////__DACE:12:0:5    ////__DACE:9
                ////__DACE:12:0:5                                                             ////__DACE:12:0:5    ////__DACE:9
                ///////////////////                                                               ////__DACE:12:0:5    ////__DACE:12:0:5    ////__DACE:9
                // Tasklet code (copy___arg2_from_cb_fusion_0_to___output_from_cb_fusion_0)       ////__DACE:12:0:5    ////__DACE:12:0:5    ////__DACE:9
                _cpy_out = _cpy_in;                                                               ////__DACE:12:0:5    ////__DACE:12:0:5    ////__DACE:9
                ///////////////////                                                               ////__DACE:12:0:5    ////__DACE:12:0:5    ////__DACE:9
                ////__DACE:12:0:5                                                             ////__DACE:12:0:5    ////__DACE:9
                __output_from_cb_fusion_0 = _cpy_out;                                             ////__DACE:12:0:5    ////__DACE:12:0:5    ////__DACE:9
            }                                                                                 ////__DACE:12:0:5    ////__DACE:9
            ////__DACE:9
        }                                                                             ////__DACE:9
    }                                                                                 ////__DACE:9
}                                                                                 ////__DACE:0:0:113
////__DACE:0:0:113
DACE_DFI void if_stmt_4_0_0_124(const bool&  __cond, const double&  __map_fusion_gtir_tmp_21_0_0, const double&  __map_fusion_gtir_tmp_33_0_0, const double&  gtir_tmp_34_0, const double&  gtir_tmp_38_0, const double&  gtir_tmp_44_0, const double&  gtir_tmp_48_0, const double&  gtir_tmp_56_0, const double&  gtir_tmp_60_0, const double&  gtir_tmp_66_0, const double&  gtir_tmp_70_0, double&  __output, double&  __output_from_cb_fusion_1) {    ////__DACE:0:0:124
    ////__DACE:13
    if (__cond) {                                                                     ////__DACE:13
        {                                                                             ////__DACE:13
            double __arg1____;                                                                ////__DACE:15:0:1    ////__DACE:13
            double __arg1____from_cb_fusion_1;                                                ////__DACE:15:0:3    ////__DACE:13
            double __map_fusion_gtir_tmp_63_0;                                                ////__DACE:15:0:5    ////__DACE:13
            double __map_fusion_gtir_tmp_59_0;                                                ////__DACE:15:0:7    ////__DACE:13
            double __map_fusion_gtir_tmp_41_0;                                                ////__DACE:15:0:10    ////__DACE:13
            double __map_fusion_gtir_tmp_37_0;                                                ////__DACE:15:0:12    ////__DACE:13
            ////__DACE:13
            {                                                                                 ////__DACE:15:0:6    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_60_0;                                               ////__DACE:15:0:14,6    ////__DACE:15:0:6    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:15:0:15,6    ////__DACE:15:0:6    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:15:0:6    ////__DACE:15:0:6    ////__DACE:13
                ////__DACE:15:0:6                                                             ////__DACE:15:0:6    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:6    ////__DACE:15:0:6    ////__DACE:13
                // Tasklet code (tlet_21_multiplies_0)                                            ////__DACE:15:0:6    ////__DACE:15:0:6    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:15:0:6    ////__DACE:15:0:6    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:6    ////__DACE:15:0:6    ////__DACE:13
                ////__DACE:15:0:6                                                             ////__DACE:15:0:6    ////__DACE:13
                __map_fusion_gtir_tmp_63_0 = __tlet_result;                                       ////__DACE:15:0:6    ////__DACE:15:0:6    ////__DACE:13
            }                                                                                 ////__DACE:15:0:6    ////__DACE:13
            {                                                                                 ////__DACE:15:0:8    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_56_0;                                               ////__DACE:15:0:16,8    ////__DACE:15:0:8    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:15:0:17,8    ////__DACE:15:0:8    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:15:0:8    ////__DACE:15:0:8    ////__DACE:13
                ////__DACE:15:0:8                                                             ////__DACE:15:0:8    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:8    ////__DACE:15:0:8    ////__DACE:13
                // Tasklet code (tlet_20_multiplies_0)                                            ////__DACE:15:0:8    ////__DACE:15:0:8    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:15:0:8    ////__DACE:15:0:8    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:8    ////__DACE:15:0:8    ////__DACE:13
                ////__DACE:15:0:8                                                             ////__DACE:15:0:8    ////__DACE:13
                __map_fusion_gtir_tmp_59_0 = __tlet_result;                                       ////__DACE:15:0:8    ////__DACE:15:0:8    ////__DACE:13
            }                                                                                 ////__DACE:15:0:8    ////__DACE:13
            {                                                                                 ////__DACE:15:0:4    ////__DACE:13
                double __tlet_arg1 = __map_fusion_gtir_tmp_63_0;                                  ////__DACE:15:0:5,4    ////__DACE:15:0:4    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_59_0;                                  ////__DACE:15:0:7,4    ////__DACE:15:0:4    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                ////__DACE:15:0:4                                                             ////__DACE:15:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                // Tasklet code (tlet_22_plus_0)                                                  ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
                ////__DACE:15:0:4                                                             ////__DACE:15:0:4    ////__DACE:13
                __arg1____ = __tlet_result;                                                       ////__DACE:15:0:4    ////__DACE:15:0:4    ////__DACE:13
            }                                                                                 ////__DACE:15:0:4    ////__DACE:13
            {                                                                                 ////__DACE:15:0:20    ////__DACE:13
                double _cpy_in = __arg1____;                                                      ////__DACE:15:0:1,20    ////__DACE:15:0:20    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:15:0:20    ////__DACE:15:0:20    ////__DACE:13
                ////__DACE:15:0:20                                                            ////__DACE:15:0:20    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:20    ////__DACE:15:0:20    ////__DACE:13
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:15:0:20    ////__DACE:15:0:20    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:15:0:20    ////__DACE:15:0:20    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:20    ////__DACE:15:0:20    ////__DACE:13
                ////__DACE:15:0:20                                                            ////__DACE:15:0:20    ////__DACE:13
                __output = _cpy_out;                                                              ////__DACE:15:0:20    ////__DACE:15:0:20    ////__DACE:13
            }                                                                                 ////__DACE:15:0:20    ////__DACE:13
            {                                                                                 ////__DACE:15:0:11    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:15:0:15,11    ////__DACE:15:0:11    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_38_0;                                               ////__DACE:15:0:18,11    ////__DACE:15:0:11    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:15:0:11    ////__DACE:15:0:11    ////__DACE:13
                ////__DACE:15:0:11                                                            ////__DACE:15:0:11    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:11    ////__DACE:15:0:11    ////__DACE:13
                // Tasklet code (tlet_15_multiplies_0)                                            ////__DACE:15:0:11    ////__DACE:15:0:11    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:15:0:11    ////__DACE:15:0:11    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:11    ////__DACE:15:0:11    ////__DACE:13
                ////__DACE:15:0:11                                                            ////__DACE:15:0:11    ////__DACE:13
                __map_fusion_gtir_tmp_41_0 = __tlet_result;                                       ////__DACE:15:0:11    ////__DACE:15:0:11    ////__DACE:13
            }                                                                                 ////__DACE:15:0:11    ////__DACE:13
            {                                                                                 ////__DACE:15:0:13    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:15:0:17,13    ////__DACE:15:0:13    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_34_0;                                               ////__DACE:15:0:19,13    ////__DACE:15:0:13    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:15:0:13    ////__DACE:15:0:13    ////__DACE:13
                ////__DACE:15:0:13                                                            ////__DACE:15:0:13    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:13    ////__DACE:15:0:13    ////__DACE:13
                // Tasklet code (tlet_14_multiplies_0)                                            ////__DACE:15:0:13    ////__DACE:15:0:13    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:15:0:13    ////__DACE:15:0:13    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:13    ////__DACE:15:0:13    ////__DACE:13
                ////__DACE:15:0:13                                                            ////__DACE:15:0:13    ////__DACE:13
                __map_fusion_gtir_tmp_37_0 = __tlet_result;                                       ////__DACE:15:0:13    ////__DACE:15:0:13    ////__DACE:13
            }                                                                                 ////__DACE:15:0:13    ////__DACE:13
            {                                                                                 ////__DACE:15:0:9    ////__DACE:13
                double __tlet_arg1 = __map_fusion_gtir_tmp_41_0;                                  ////__DACE:15:0:10,9    ////__DACE:15:0:9    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_37_0;                                  ////__DACE:15:0:12,9    ////__DACE:15:0:9    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:15:0:9    ////__DACE:15:0:9    ////__DACE:13
                ////__DACE:15:0:9                                                             ////__DACE:15:0:9    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:9    ////__DACE:15:0:9    ////__DACE:13
                // Tasklet code (tlet_16_plus_0)                                                  ////__DACE:15:0:9    ////__DACE:15:0:9    ////__DACE:13
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:15:0:9    ////__DACE:15:0:9    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:9    ////__DACE:15:0:9    ////__DACE:13
                ////__DACE:15:0:9                                                             ////__DACE:15:0:9    ////__DACE:13
                __arg1____from_cb_fusion_1 = __tlet_result;                                       ////__DACE:15:0:9    ////__DACE:15:0:9    ////__DACE:13
            }                                                                                 ////__DACE:15:0:9    ////__DACE:13
            {                                                                                 ////__DACE:15:0:21    ////__DACE:13
                double _cpy_in = __arg1____from_cb_fusion_1;                                      ////__DACE:15:0:3,21    ////__DACE:15:0:21    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:15:0:21    ////__DACE:15:0:21    ////__DACE:13
                ////__DACE:15:0:21                                                            ////__DACE:15:0:21    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:21    ////__DACE:15:0:21    ////__DACE:13
                // Tasklet code (copy___arg1____from_cb_fusion_1_to___output_from_cb_fusion_1)    ////__DACE:15:0:21    ////__DACE:15:0:21    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:15:0:21    ////__DACE:15:0:21    ////__DACE:13
                ///////////////////                                                               ////__DACE:15:0:21    ////__DACE:15:0:21    ////__DACE:13
                ////__DACE:15:0:21                                                            ////__DACE:15:0:21    ////__DACE:13
                __output_from_cb_fusion_1 = _cpy_out;                                             ////__DACE:15:0:21    ////__DACE:15:0:21    ////__DACE:13
            }                                                                                 ////__DACE:15:0:21    ////__DACE:13
            ////__DACE:13
        }                                                                             ////__DACE:13
    } else {                                                                          ////__DACE:13
        {                                                                             ////__DACE:13
            double __arg2_;                                                                   ////__DACE:16:0:1    ////__DACE:13
            double __arg2_from_cb_fusion_1;                                                   ////__DACE:16:0:3    ////__DACE:13
            double __map_fusion_gtir_tmp_73_0;                                                ////__DACE:16:0:5    ////__DACE:13
            double __map_fusion_gtir_tmp_69_0;                                                ////__DACE:16:0:7    ////__DACE:13
            double __map_fusion_gtir_tmp_51_0;                                                ////__DACE:16:0:10    ////__DACE:13
            double __map_fusion_gtir_tmp_47_0;                                                ////__DACE:16:0:12    ////__DACE:13
            ////__DACE:13
            {                                                                                 ////__DACE:16:0:6    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_70_0;                                               ////__DACE:16:0:14,6    ////__DACE:16:0:6    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:16:0:15,6    ////__DACE:16:0:6    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:16:0:6    ////__DACE:16:0:6    ////__DACE:13
                ////__DACE:16:0:6                                                             ////__DACE:16:0:6    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:6    ////__DACE:16:0:6    ////__DACE:13
                // Tasklet code (tlet_24_multiplies_0)                                            ////__DACE:16:0:6    ////__DACE:16:0:6    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:16:0:6    ////__DACE:16:0:6    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:6    ////__DACE:16:0:6    ////__DACE:13
                ////__DACE:16:0:6                                                             ////__DACE:16:0:6    ////__DACE:13
                __map_fusion_gtir_tmp_73_0 = __tlet_result;                                       ////__DACE:16:0:6    ////__DACE:16:0:6    ////__DACE:13
            }                                                                                 ////__DACE:16:0:6    ////__DACE:13
            {                                                                                 ////__DACE:16:0:8    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_66_0;                                               ////__DACE:16:0:16,8    ////__DACE:16:0:8    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:16:0:17,8    ////__DACE:16:0:8    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:16:0:8    ////__DACE:16:0:8    ////__DACE:13
                ////__DACE:16:0:8                                                             ////__DACE:16:0:8    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:8    ////__DACE:16:0:8    ////__DACE:13
                // Tasklet code (tlet_23_multiplies_0)                                            ////__DACE:16:0:8    ////__DACE:16:0:8    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:16:0:8    ////__DACE:16:0:8    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:8    ////__DACE:16:0:8    ////__DACE:13
                ////__DACE:16:0:8                                                             ////__DACE:16:0:8    ////__DACE:13
                __map_fusion_gtir_tmp_69_0 = __tlet_result;                                       ////__DACE:16:0:8    ////__DACE:16:0:8    ////__DACE:13
            }                                                                                 ////__DACE:16:0:8    ////__DACE:13
            {                                                                                 ////__DACE:16:0:4    ////__DACE:13
                double __tlet_arg1 = __map_fusion_gtir_tmp_73_0;                                  ////__DACE:16:0:5,4    ////__DACE:16:0:4    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_69_0;                                  ////__DACE:16:0:7,4    ////__DACE:16:0:4    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                ////__DACE:16:0:4                                                             ////__DACE:16:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                // Tasklet code (tlet_25_plus_0)                                                  ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
                ////__DACE:16:0:4                                                             ////__DACE:16:0:4    ////__DACE:13
                __arg2_ = __tlet_result;                                                          ////__DACE:16:0:4    ////__DACE:16:0:4    ////__DACE:13
            }                                                                                 ////__DACE:16:0:4    ////__DACE:13
            {                                                                                 ////__DACE:16:0:20    ////__DACE:13
                double _cpy_in = __arg2_;                                                         ////__DACE:16:0:1,20    ////__DACE:16:0:20    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:16:0:20    ////__DACE:16:0:20    ////__DACE:13
                ////__DACE:16:0:20                                                            ////__DACE:16:0:20    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:20    ////__DACE:16:0:20    ////__DACE:13
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:16:0:20    ////__DACE:16:0:20    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:16:0:20    ////__DACE:16:0:20    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:20    ////__DACE:16:0:20    ////__DACE:13
                ////__DACE:16:0:20                                                            ////__DACE:16:0:20    ////__DACE:13
                __output = _cpy_out;                                                              ////__DACE:16:0:20    ////__DACE:16:0:20    ////__DACE:13
            }                                                                                 ////__DACE:16:0:20    ////__DACE:13
            {                                                                                 ////__DACE:16:0:11    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_48_0;                                               ////__DACE:16:0:18,11    ////__DACE:16:0:11    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_33_0_0;                                ////__DACE:16:0:15,11    ////__DACE:16:0:11    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:16:0:11    ////__DACE:16:0:11    ////__DACE:13
                ////__DACE:16:0:11                                                            ////__DACE:16:0:11    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:11    ////__DACE:16:0:11    ////__DACE:13
                // Tasklet code (tlet_18_multiplies_0)                                            ////__DACE:16:0:11    ////__DACE:16:0:11    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:16:0:11    ////__DACE:16:0:11    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:11    ////__DACE:16:0:11    ////__DACE:13
                ////__DACE:16:0:11                                                            ////__DACE:16:0:11    ////__DACE:13
                __map_fusion_gtir_tmp_51_0 = __tlet_result;                                       ////__DACE:16:0:11    ////__DACE:16:0:11    ////__DACE:13
            }                                                                                 ////__DACE:16:0:11    ////__DACE:13
            {                                                                                 ////__DACE:16:0:13    ////__DACE:13
                double __tlet_arg1 = gtir_tmp_44_0;                                               ////__DACE:16:0:19,13    ////__DACE:16:0:13    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_21_0_0;                                ////__DACE:16:0:17,13    ////__DACE:16:0:13    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:16:0:13    ////__DACE:16:0:13    ////__DACE:13
                ////__DACE:16:0:13                                                            ////__DACE:16:0:13    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:13    ////__DACE:16:0:13    ////__DACE:13
                // Tasklet code (tlet_17_multiplies_0)                                            ////__DACE:16:0:13    ////__DACE:16:0:13    ////__DACE:13
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:16:0:13    ////__DACE:16:0:13    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:13    ////__DACE:16:0:13    ////__DACE:13
                ////__DACE:16:0:13                                                            ////__DACE:16:0:13    ////__DACE:13
                __map_fusion_gtir_tmp_47_0 = __tlet_result;                                       ////__DACE:16:0:13    ////__DACE:16:0:13    ////__DACE:13
            }                                                                                 ////__DACE:16:0:13    ////__DACE:13
            {                                                                                 ////__DACE:16:0:9    ////__DACE:13
                double __tlet_arg1 = __map_fusion_gtir_tmp_51_0;                                  ////__DACE:16:0:10,9    ////__DACE:16:0:9    ////__DACE:13
                double __tlet_arg0 = __map_fusion_gtir_tmp_47_0;                                  ////__DACE:16:0:12,9    ////__DACE:16:0:9    ////__DACE:13
                double __tlet_result;                                                             ////__DACE:16:0:9    ////__DACE:16:0:9    ////__DACE:13
                ////__DACE:16:0:9                                                             ////__DACE:16:0:9    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:9    ////__DACE:16:0:9    ////__DACE:13
                // Tasklet code (tlet_19_plus_0)                                                  ////__DACE:16:0:9    ////__DACE:16:0:9    ////__DACE:13
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:16:0:9    ////__DACE:16:0:9    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:9    ////__DACE:16:0:9    ////__DACE:13
                ////__DACE:16:0:9                                                             ////__DACE:16:0:9    ////__DACE:13
                __arg2_from_cb_fusion_1 = __tlet_result;                                          ////__DACE:16:0:9    ////__DACE:16:0:9    ////__DACE:13
            }                                                                                 ////__DACE:16:0:9    ////__DACE:13
            {                                                                                 ////__DACE:16:0:21    ////__DACE:13
                double _cpy_in = __arg2_from_cb_fusion_1;                                         ////__DACE:16:0:3,21    ////__DACE:16:0:21    ////__DACE:13
                double _cpy_out;                                                                  ////__DACE:16:0:21    ////__DACE:16:0:21    ////__DACE:13
                ////__DACE:16:0:21                                                            ////__DACE:16:0:21    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:21    ////__DACE:16:0:21    ////__DACE:13
                // Tasklet code (copy___arg2_from_cb_fusion_1_to___output_from_cb_fusion_1)       ////__DACE:16:0:21    ////__DACE:16:0:21    ////__DACE:13
                _cpy_out = _cpy_in;                                                               ////__DACE:16:0:21    ////__DACE:16:0:21    ////__DACE:13
                ///////////////////                                                               ////__DACE:16:0:21    ////__DACE:16:0:21    ////__DACE:13
                ////__DACE:16:0:21                                                            ////__DACE:16:0:21    ////__DACE:13
                __output_from_cb_fusion_1 = _cpy_out;                                             ////__DACE:16:0:21    ////__DACE:16:0:21    ////__DACE:13
            }                                                                                 ////__DACE:16:0:21    ////__DACE:13
            ////__DACE:13
        }                                                                             ////__DACE:13
    }                                                                                 ////__DACE:13
}                                                                                 ////__DACE:0:0:124
////__DACE:0:0:124
DACE_DFI void if_stmt_7_0_0_140(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double&  gtir_tmp_54_0, const double&  gtir_tmp_76_0, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double* __restrict__ perturbed_rho_at_cells_on_model_levels, const double* __restrict__ reference_rho_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_rho_at_cells_on_model_levels_K_stride_0, int __reference_rho_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:140
    ////__DACE:21
    if (__cond) {                                                                     ////__DACE:21
        {                                                                             ////__DACE:21
            double __arg1___;                                                                 ////__DACE:23:0:1    ////__DACE:21
            double __map_fusion_gtir_tmp_191_0;                                               ////__DACE:23:0:3    ////__DACE:21
            double __map_fusion_gtir_tmp_189_0;                                               ////__DACE:23:0:5    ////__DACE:21
            double __map_fusion_gtir_tmp_187_0;                                               ////__DACE:23:0:7    ////__DACE:21
            double __map_fusion_gtir_tmp_185_0;                                               ////__DACE:23:0:9    ////__DACE:21
            double __map_fusion_gtir_tmp_183_0;                                               ////__DACE:23:0:11    ////__DACE:21
            double __map_fusion_gtir_tmp_181_0;                                               ////__DACE:23:0:13    ////__DACE:21
            double __map_fusion_gtir_tmp_179_0;                                               ////__DACE:23:0:15    ////__DACE:21
            ////__DACE:21
            {                                                                                 ////__DACE:23:0:6    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:23:0:18,6    ////__DACE:23:0:6    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:23:0:19,6    ////__DACE:23:0:6    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                ////__DACE:23:0:6                                                             ////__DACE:23:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                // Tasklet code (tlet_75_deref_0)                                                 ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
                ////__DACE:23:0:6                                                             ////__DACE:23:0:6    ////__DACE:21
                __map_fusion_gtir_tmp_189_0 = __tlet_val;                                         ////__DACE:23:0:6    ////__DACE:23:0:6    ////__DACE:21
            }                                                                                 ////__DACE:23:0:6    ////__DACE:21
            {                                                                                 ////__DACE:23:0:4    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:23:0:17,4    ////__DACE:23:0:4    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_189_0;                                 ////__DACE:23:0:5,4    ////__DACE:23:0:4    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                ////__DACE:23:0:4                                                             ////__DACE:23:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                // Tasklet code (tlet_76_multiplies_0)                                            ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
                ////__DACE:23:0:4                                                             ////__DACE:23:0:4    ////__DACE:21
                __map_fusion_gtir_tmp_191_0 = __tlet_result;                                      ////__DACE:23:0:4    ////__DACE:23:0:4    ////__DACE:21
            }                                                                                 ////__DACE:23:0:4    ////__DACE:21
            {                                                                                 ////__DACE:23:0:12    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:23:0:21,12    ////__DACE:23:0:12    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:23:0:19,12    ////__DACE:23:0:12    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                ////__DACE:23:0:12                                                            ////__DACE:23:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                // Tasklet code (tlet_72_deref_0)                                                 ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
                ////__DACE:23:0:12                                                            ////__DACE:23:0:12    ////__DACE:21
                __map_fusion_gtir_tmp_183_0 = __tlet_val;                                         ////__DACE:23:0:12    ////__DACE:23:0:12    ////__DACE:21
            }                                                                                 ////__DACE:23:0:12    ////__DACE:21
            {                                                                                 ////__DACE:23:0:10    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:23:0:20,10    ////__DACE:23:0:10    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_183_0;                                 ////__DACE:23:0:11,10    ////__DACE:23:0:10    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                ////__DACE:23:0:10                                                            ////__DACE:23:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                // Tasklet code (tlet_73_multiplies_0)                                            ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
                ////__DACE:23:0:10                                                            ////__DACE:23:0:10    ////__DACE:21
                __map_fusion_gtir_tmp_185_0 = __tlet_result;                                      ////__DACE:23:0:10    ////__DACE:23:0:10    ////__DACE:21
            }                                                                                 ////__DACE:23:0:10    ////__DACE:21
            {                                                                                 ////__DACE:23:0:16    ////__DACE:21
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:23:0:23,16    ////__DACE:23:0:16    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:23:0:19,16    ////__DACE:23:0:16    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                ////__DACE:23:0:16                                                            ////__DACE:23:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                // Tasklet code (tlet_70_deref_0)                                                 ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
                ////__DACE:23:0:16                                                            ////__DACE:23:0:16    ////__DACE:21
                __map_fusion_gtir_tmp_179_0 = __tlet_val;                                         ////__DACE:23:0:16    ////__DACE:23:0:16    ////__DACE:21
            }                                                                                 ////__DACE:23:0:16    ////__DACE:21
            {                                                                                 ////__DACE:23:0:14    ////__DACE:21
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:23:0:22,14    ////__DACE:23:0:14    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_179_0;                                 ////__DACE:23:0:15,14    ////__DACE:23:0:14    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                ////__DACE:23:0:14                                                            ////__DACE:23:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                // Tasklet code (tlet_71_plus_0)                                                  ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
                ////__DACE:23:0:14                                                            ////__DACE:23:0:14    ////__DACE:21
                __map_fusion_gtir_tmp_181_0 = __tlet_result;                                      ////__DACE:23:0:14    ////__DACE:23:0:14    ////__DACE:21
            }                                                                                 ////__DACE:23:0:14    ////__DACE:21
            {                                                                                 ////__DACE:23:0:8    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_185_0;                                 ////__DACE:23:0:9,8    ////__DACE:23:0:8    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_181_0;                                 ////__DACE:23:0:13,8    ////__DACE:23:0:8    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                ////__DACE:23:0:8                                                             ////__DACE:23:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                // Tasklet code (tlet_74_plus_0)                                                  ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
                ////__DACE:23:0:8                                                             ////__DACE:23:0:8    ////__DACE:21
                __map_fusion_gtir_tmp_187_0 = __tlet_result;                                      ////__DACE:23:0:8    ////__DACE:23:0:8    ////__DACE:21
            }                                                                                 ////__DACE:23:0:8    ////__DACE:21
            {                                                                                 ////__DACE:23:0:2    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_191_0;                                 ////__DACE:23:0:3,2    ////__DACE:23:0:2    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_187_0;                                 ////__DACE:23:0:7,2    ////__DACE:23:0:2    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                ////__DACE:23:0:2                                                             ////__DACE:23:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                // Tasklet code (tlet_77_plus_0)                                                  ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
                ////__DACE:23:0:2                                                             ////__DACE:23:0:2    ////__DACE:21
                __arg1___ = __tlet_result;                                                        ////__DACE:23:0:2    ////__DACE:23:0:2    ////__DACE:21
            }                                                                                 ////__DACE:23:0:2    ////__DACE:21
            {                                                                                 ////__DACE:23:0:24    ////__DACE:21
                double _cpy_in = __arg1___;                                                       ////__DACE:23:0:1,24    ////__DACE:23:0:24    ////__DACE:21
                double _cpy_out;                                                                  ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                ////__DACE:23:0:24                                                            ////__DACE:23:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                // Tasklet code (copy___arg1____to___output)                                      ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                _cpy_out = _cpy_in;                                                               ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
                ////__DACE:23:0:24                                                            ////__DACE:23:0:24    ////__DACE:21
                __output = _cpy_out;                                                              ////__DACE:23:0:24    ////__DACE:23:0:24    ////__DACE:21
            }                                                                                 ////__DACE:23:0:24    ////__DACE:21
            ////__DACE:21
        }                                                                             ////__DACE:21
    } else {                                                                          ////__DACE:21
        {                                                                             ////__DACE:21
            double __arg2;                                                                    ////__DACE:24:0:1    ////__DACE:21
            double __map_fusion_gtir_tmp_207_0;                                               ////__DACE:24:0:3    ////__DACE:21
            double __map_fusion_gtir_tmp_205_0;                                               ////__DACE:24:0:5    ////__DACE:21
            double __map_fusion_gtir_tmp_203_0;                                               ////__DACE:24:0:7    ////__DACE:21
            double __map_fusion_gtir_tmp_201_0;                                               ////__DACE:24:0:9    ////__DACE:21
            double __map_fusion_gtir_tmp_199_0;                                               ////__DACE:24:0:11    ////__DACE:21
            double __map_fusion_gtir_tmp_197_0;                                               ////__DACE:24:0:13    ////__DACE:21
            double __map_fusion_gtir_tmp_195_0;                                               ////__DACE:24:0:15    ////__DACE:21
            ////__DACE:21
            {                                                                                 ////__DACE:24:0:6    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_89[0];                                    ////__DACE:24:0:18,6    ////__DACE:24:0:6    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:24:0:19,6    ////__DACE:24:0:6    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                ////__DACE:24:0:6                                                             ////__DACE:24:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                // Tasklet code (tlet_83_deref_0)                                                 ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
                ////__DACE:24:0:6                                                             ////__DACE:24:0:6    ////__DACE:21
                __map_fusion_gtir_tmp_205_0 = __tlet_val;                                         ////__DACE:24:0:6    ////__DACE:24:0:6    ////__DACE:21
            }                                                                                 ////__DACE:24:0:6    ////__DACE:21
            {                                                                                 ////__DACE:24:0:4    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:24:0:17,4    ////__DACE:24:0:4    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_205_0;                                 ////__DACE:24:0:5,4    ////__DACE:24:0:4    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                ////__DACE:24:0:4                                                             ////__DACE:24:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                // Tasklet code (tlet_84_multiplies_0)                                            ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
                ////__DACE:24:0:4                                                             ////__DACE:24:0:4    ////__DACE:21
                __map_fusion_gtir_tmp_207_0 = __tlet_result;                                      ////__DACE:24:0:4    ////__DACE:24:0:4    ////__DACE:21
            }                                                                                 ////__DACE:24:0:4    ////__DACE:21
            {                                                                                 ////__DACE:24:0:12    ////__DACE:21
                const double * __tlet_field = &gtir_tmp_83[0];                                    ////__DACE:24:0:21,12    ////__DACE:24:0:12    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:24:0:19,12    ////__DACE:24:0:12    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                ////__DACE:24:0:12                                                            ////__DACE:24:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                // Tasklet code (tlet_80_deref_0)                                                 ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
                ////__DACE:24:0:12                                                            ////__DACE:24:0:12    ////__DACE:21
                __map_fusion_gtir_tmp_199_0 = __tlet_val;                                         ////__DACE:24:0:12    ////__DACE:24:0:12    ////__DACE:21
            }                                                                                 ////__DACE:24:0:12    ////__DACE:21
            {                                                                                 ////__DACE:24:0:10    ////__DACE:21
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:24:0:20,10    ////__DACE:24:0:10    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_199_0;                                 ////__DACE:24:0:11,10    ////__DACE:24:0:10    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                ////__DACE:24:0:10                                                            ////__DACE:24:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                // Tasklet code (tlet_81_multiplies_0)                                            ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
                ////__DACE:24:0:10                                                            ////__DACE:24:0:10    ////__DACE:21
                __map_fusion_gtir_tmp_201_0 = __tlet_result;                                      ////__DACE:24:0:10    ////__DACE:24:0:10    ////__DACE:21
            }                                                                                 ////__DACE:24:0:10    ////__DACE:21
            {                                                                                 ////__DACE:24:0:16    ////__DACE:21
                const double* __tlet_field = &perturbed_rho_at_cells_on_model_levels[0];          ////__DACE:24:0:23,16    ////__DACE:24:0:16    ////__DACE:21
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:24:0:19,16    ////__DACE:24:0:16    ////__DACE:21
                double __tlet_val;                                                                ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                ////__DACE:24:0:16                                                            ////__DACE:24:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                // Tasklet code (tlet_78_deref_0)                                                 ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                __tlet_val = __tlet_field[((__perturbed_rho_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
                ////__DACE:24:0:16                                                            ////__DACE:24:0:16    ////__DACE:21
                __map_fusion_gtir_tmp_195_0 = __tlet_val;                                         ////__DACE:24:0:16    ////__DACE:24:0:16    ////__DACE:21
            }                                                                                 ////__DACE:24:0:16    ////__DACE:21
            {                                                                                 ////__DACE:24:0:14    ////__DACE:21
                double __tlet_arg0 = reference_rho_at_edges_on_model_levels[((__reference_rho_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:24:0:22,14    ////__DACE:24:0:14    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_195_0;                                 ////__DACE:24:0:15,14    ////__DACE:24:0:14    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                ////__DACE:24:0:14                                                            ////__DACE:24:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                // Tasklet code (tlet_79_plus_0)                                                  ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
                ////__DACE:24:0:14                                                            ////__DACE:24:0:14    ////__DACE:21
                __map_fusion_gtir_tmp_197_0 = __tlet_result;                                      ////__DACE:24:0:14    ////__DACE:24:0:14    ////__DACE:21
            }                                                                                 ////__DACE:24:0:14    ////__DACE:21
            {                                                                                 ////__DACE:24:0:8    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_201_0;                                 ////__DACE:24:0:9,8    ////__DACE:24:0:8    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_197_0;                                 ////__DACE:24:0:13,8    ////__DACE:24:0:8    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                ////__DACE:24:0:8                                                             ////__DACE:24:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                // Tasklet code (tlet_82_plus_0)                                                  ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
                ////__DACE:24:0:8                                                             ////__DACE:24:0:8    ////__DACE:21
                __map_fusion_gtir_tmp_203_0 = __tlet_result;                                      ////__DACE:24:0:8    ////__DACE:24:0:8    ////__DACE:21
            }                                                                                 ////__DACE:24:0:8    ////__DACE:21
            {                                                                                 ////__DACE:24:0:2    ////__DACE:21
                double __tlet_arg1 = __map_fusion_gtir_tmp_207_0;                                 ////__DACE:24:0:3,2    ////__DACE:24:0:2    ////__DACE:21
                double __tlet_arg0 = __map_fusion_gtir_tmp_203_0;                                 ////__DACE:24:0:7,2    ////__DACE:24:0:2    ////__DACE:21
                double __tlet_result;                                                             ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                ////__DACE:24:0:2                                                             ////__DACE:24:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                // Tasklet code (tlet_85_plus_0)                                                  ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
                ////__DACE:24:0:2                                                             ////__DACE:24:0:2    ////__DACE:21
                __arg2 = __tlet_result;                                                           ////__DACE:24:0:2    ////__DACE:24:0:2    ////__DACE:21
            }                                                                                 ////__DACE:24:0:2    ////__DACE:21
            {                                                                                 ////__DACE:24:0:24    ////__DACE:21
                double _cpy_in = __arg2;                                                          ////__DACE:24:0:1,24    ////__DACE:24:0:24    ////__DACE:21
                double _cpy_out;                                                                  ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                ////__DACE:24:0:24                                                            ////__DACE:24:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                // Tasklet code (copy___arg2_to___output)                                         ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                _cpy_out = _cpy_in;                                                               ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                ///////////////////                                                               ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
                ////__DACE:24:0:24                                                            ////__DACE:24:0:24    ////__DACE:21
                __output = _cpy_out;                                                              ////__DACE:24:0:24    ////__DACE:24:0:24    ////__DACE:21
            }                                                                                 ////__DACE:24:0:24    ////__DACE:21
            ////__DACE:21
        }                                                                             ////__DACE:21
    }                                                                                 ////__DACE:21
}                                                                                 ////__DACE:0:0:140
////__DACE:0:0:140
DACE_DFI void if_stmt_5_0_0_136(const bool&  __cond, const int* __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double&  gtir_tmp_54_0, const double&  gtir_tmp_76_0, const double * __restrict__ gtir_tmp_95, const double* __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double* __restrict__ reference_theta_at_edges_on_model_levels, double&  __output, int __gt_conn_E2C_neighbor_stride_0, int __perturbed_theta_v_at_cells_on_model_levels_K_stride_0, int __reference_theta_at_edges_on_model_levels_K_stride_0, int64_t i_Edge_gtx_horizontal, int64_t i_K_gtx_vertical) {    ////__DACE:0:0:136
    ////__DACE:17
    if (__cond) {                                                                     ////__DACE:17
        {                                                                             ////__DACE:17
            double __arg1____;                                                                ////__DACE:19:0:1    ////__DACE:17
            double __map_fusion_gtir_tmp_118_0;                                               ////__DACE:19:0:3    ////__DACE:17
            double __map_fusion_gtir_tmp_116_0;                                               ////__DACE:19:0:5    ////__DACE:17
            double __map_fusion_gtir_tmp_114_0;                                               ////__DACE:19:0:7    ////__DACE:17
            double __map_fusion_gtir_tmp_112_0;                                               ////__DACE:19:0:9    ////__DACE:17
            double __map_fusion_gtir_tmp_110_0;                                               ////__DACE:19:0:11    ////__DACE:17
            double __map_fusion_gtir_tmp_108_0;                                               ////__DACE:19:0:13    ////__DACE:17
            double __map_fusion_gtir_tmp_106_0;                                               ////__DACE:19:0:15    ////__DACE:17
            ////__DACE:17
            {                                                                                 ////__DACE:19:0:6    ////__DACE:17
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:19:0:18,6    ////__DACE:19:0:6    ////__DACE:17
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:19:0:19,6    ////__DACE:19:0:6    ////__DACE:17
                double __tlet_val;                                                                ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                ////__DACE:19:0:6                                                             ////__DACE:19:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                // Tasklet code (tlet_41_deref_0)                                                 ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
                ////__DACE:19:0:6                                                             ////__DACE:19:0:6    ////__DACE:17
                __map_fusion_gtir_tmp_116_0 = __tlet_val;                                         ////__DACE:19:0:6    ////__DACE:19:0:6    ////__DACE:17
            }                                                                                 ////__DACE:19:0:6    ////__DACE:17
            {                                                                                 ////__DACE:19:0:4    ////__DACE:17
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:19:0:17,4    ////__DACE:19:0:4    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_116_0;                                 ////__DACE:19:0:5,4    ////__DACE:19:0:4    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                ////__DACE:19:0:4                                                             ////__DACE:19:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                // Tasklet code (tlet_42_multiplies_0)                                            ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
                ////__DACE:19:0:4                                                             ////__DACE:19:0:4    ////__DACE:17
                __map_fusion_gtir_tmp_118_0 = __tlet_result;                                      ////__DACE:19:0:4    ////__DACE:19:0:4    ////__DACE:17
            }                                                                                 ////__DACE:19:0:4    ////__DACE:17
            {                                                                                 ////__DACE:19:0:12    ////__DACE:17
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:19:0:21,12    ////__DACE:19:0:12    ////__DACE:17
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:19:0:19,12    ////__DACE:19:0:12    ////__DACE:17
                double __tlet_val;                                                                ////__DACE:19:0:12    ////__DACE:19:0:12    ////__DACE:17
                ////__DACE:19:0:12                                                            ////__DACE:19:0:12    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:12    ////__DACE:19:0:12    ////__DACE:17
                // Tasklet code (tlet_38_deref_0)                                                 ////__DACE:19:0:12    ////__DACE:19:0:12    ////__DACE:17
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:19:0:12    ////__DACE:19:0:12    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:12    ////__DACE:19:0:12    ////__DACE:17
                ////__DACE:19:0:12                                                            ////__DACE:19:0:12    ////__DACE:17
                __map_fusion_gtir_tmp_110_0 = __tlet_val;                                         ////__DACE:19:0:12    ////__DACE:19:0:12    ////__DACE:17
            }                                                                                 ////__DACE:19:0:12    ////__DACE:17
            {                                                                                 ////__DACE:19:0:10    ////__DACE:17
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:19:0:20,10    ////__DACE:19:0:10    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_110_0;                                 ////__DACE:19:0:11,10    ////__DACE:19:0:10    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:10    ////__DACE:19:0:10    ////__DACE:17
                ////__DACE:19:0:10                                                            ////__DACE:19:0:10    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:10    ////__DACE:19:0:10    ////__DACE:17
                // Tasklet code (tlet_39_multiplies_0)                                            ////__DACE:19:0:10    ////__DACE:19:0:10    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:19:0:10    ////__DACE:19:0:10    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:10    ////__DACE:19:0:10    ////__DACE:17
                ////__DACE:19:0:10                                                            ////__DACE:19:0:10    ////__DACE:17
                __map_fusion_gtir_tmp_112_0 = __tlet_result;                                      ////__DACE:19:0:10    ////__DACE:19:0:10    ////__DACE:17
            }                                                                                 ////__DACE:19:0:10    ////__DACE:17
            {                                                                                 ////__DACE:19:0:16    ////__DACE:17
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:19:0:23,16    ////__DACE:19:0:16    ////__DACE:17
                int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:19:0:19,16    ////__DACE:19:0:16    ////__DACE:17
                double __tlet_val;                                                                ////__DACE:19:0:16    ////__DACE:19:0:16    ////__DACE:17
                ////__DACE:19:0:16                                                            ////__DACE:19:0:16    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:16    ////__DACE:19:0:16    ////__DACE:17
                // Tasklet code (tlet_36_deref_0)                                                 ////__DACE:19:0:16    ////__DACE:19:0:16    ////__DACE:17
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:19:0:16    ////__DACE:19:0:16    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:16    ////__DACE:19:0:16    ////__DACE:17
                ////__DACE:19:0:16                                                            ////__DACE:19:0:16    ////__DACE:17
                __map_fusion_gtir_tmp_106_0 = __tlet_val;                                         ////__DACE:19:0:16    ////__DACE:19:0:16    ////__DACE:17
            }                                                                                 ////__DACE:19:0:16    ////__DACE:17
            {                                                                                 ////__DACE:19:0:14    ////__DACE:17
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:19:0:22,14    ////__DACE:19:0:14    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_106_0;                                 ////__DACE:19:0:15,14    ////__DACE:19:0:14    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:14    ////__DACE:19:0:14    ////__DACE:17
                ////__DACE:19:0:14                                                            ////__DACE:19:0:14    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:14    ////__DACE:19:0:14    ////__DACE:17
                // Tasklet code (tlet_37_plus_0)                                                  ////__DACE:19:0:14    ////__DACE:19:0:14    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:19:0:14    ////__DACE:19:0:14    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:14    ////__DACE:19:0:14    ////__DACE:17
                ////__DACE:19:0:14                                                            ////__DACE:19:0:14    ////__DACE:17
                __map_fusion_gtir_tmp_108_0 = __tlet_result;                                      ////__DACE:19:0:14    ////__DACE:19:0:14    ////__DACE:17
            }                                                                                 ////__DACE:19:0:14    ////__DACE:17
            {                                                                                 ////__DACE:19:0:8    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_112_0;                                 ////__DACE:19:0:9,8    ////__DACE:19:0:8    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_108_0;                                 ////__DACE:19:0:13,8    ////__DACE:19:0:8    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                ////__DACE:19:0:8                                                             ////__DACE:19:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                // Tasklet code (tlet_40_plus_0)                                                  ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
                ////__DACE:19:0:8                                                             ////__DACE:19:0:8    ////__DACE:17
                __map_fusion_gtir_tmp_114_0 = __tlet_result;                                      ////__DACE:19:0:8    ////__DACE:19:0:8    ////__DACE:17
            }                                                                                 ////__DACE:19:0:8    ////__DACE:17
            {                                                                                 ////__DACE:19:0:2    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_118_0;                                 ////__DACE:19:0:3,2    ////__DACE:19:0:2    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_114_0;                                 ////__DACE:19:0:7,2    ////__DACE:19:0:2    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:19:0:2    ////__DACE:19:0:2    ////__DACE:17
                ////__DACE:19:0:2                                                             ////__DACE:19:0:2    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:2    ////__DACE:19:0:2    ////__DACE:17
                // Tasklet code (tlet_43_plus_0)                                                  ////__DACE:19:0:2    ////__DACE:19:0:2    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:19:0:2    ////__DACE:19:0:2    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:2    ////__DACE:19:0:2    ////__DACE:17
                ////__DACE:19:0:2                                                             ////__DACE:19:0:2    ////__DACE:17
                __arg1____ = __tlet_result;                                                       ////__DACE:19:0:2    ////__DACE:19:0:2    ////__DACE:17
            }                                                                                 ////__DACE:19:0:2    ////__DACE:17
            {                                                                                 ////__DACE:19:0:24    ////__DACE:17
                double _cpy_in = __arg1____;                                                      ////__DACE:19:0:1,24    ////__DACE:19:0:24    ////__DACE:17
                double _cpy_out;                                                                  ////__DACE:19:0:24    ////__DACE:19:0:24    ////__DACE:17
                ////__DACE:19:0:24                                                            ////__DACE:19:0:24    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:24    ////__DACE:19:0:24    ////__DACE:17
                // Tasklet code (copy___arg1_____to___output)                                     ////__DACE:19:0:24    ////__DACE:19:0:24    ////__DACE:17
                _cpy_out = _cpy_in;                                                               ////__DACE:19:0:24    ////__DACE:19:0:24    ////__DACE:17
                ///////////////////                                                               ////__DACE:19:0:24    ////__DACE:19:0:24    ////__DACE:17
                ////__DACE:19:0:24                                                            ////__DACE:19:0:24    ////__DACE:17
                __output = _cpy_out;                                                              ////__DACE:19:0:24    ////__DACE:19:0:24    ////__DACE:17
            }                                                                                 ////__DACE:19:0:24    ////__DACE:17
            ////__DACE:17
        }                                                                             ////__DACE:17
    } else {                                                                          ////__DACE:17
        {                                                                             ////__DACE:17
            double __arg2_;                                                                   ////__DACE:20:0:1    ////__DACE:17
            double __map_fusion_gtir_tmp_134_0;                                               ////__DACE:20:0:3    ////__DACE:17
            double __map_fusion_gtir_tmp_132_0;                                               ////__DACE:20:0:5    ////__DACE:17
            double __map_fusion_gtir_tmp_130_0;                                               ////__DACE:20:0:7    ////__DACE:17
            double __map_fusion_gtir_tmp_128_0;                                               ////__DACE:20:0:9    ////__DACE:17
            double __map_fusion_gtir_tmp_126_0;                                               ////__DACE:20:0:11    ////__DACE:17
            double __map_fusion_gtir_tmp_124_0;                                               ////__DACE:20:0:13    ////__DACE:17
            double __map_fusion_gtir_tmp_122_0;                                               ////__DACE:20:0:15    ////__DACE:17
            ////__DACE:17
            {                                                                                 ////__DACE:20:0:6    ////__DACE:17
                const double * __tlet_field = &gtir_tmp_101[0];                                   ////__DACE:20:0:18,6    ////__DACE:20:0:6    ////__DACE:17
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:20:0:19,6    ////__DACE:20:0:6    ////__DACE:17
                double __tlet_val;                                                                ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                ////__DACE:20:0:6                                                             ////__DACE:20:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                // Tasklet code (tlet_49_deref_0)                                                 ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
                ////__DACE:20:0:6                                                             ////__DACE:20:0:6    ////__DACE:17
                __map_fusion_gtir_tmp_132_0 = __tlet_val;                                         ////__DACE:20:0:6    ////__DACE:20:0:6    ////__DACE:17
            }                                                                                 ////__DACE:20:0:6    ////__DACE:17
            {                                                                                 ////__DACE:20:0:4    ////__DACE:17
                double __tlet_arg0 = gtir_tmp_76_0;                                               ////__DACE:20:0:17,4    ////__DACE:20:0:4    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_132_0;                                 ////__DACE:20:0:5,4    ////__DACE:20:0:4    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                ////__DACE:20:0:4                                                             ////__DACE:20:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                // Tasklet code (tlet_50_multiplies_0)                                            ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
                ////__DACE:20:0:4                                                             ////__DACE:20:0:4    ////__DACE:17
                __map_fusion_gtir_tmp_134_0 = __tlet_result;                                      ////__DACE:20:0:4    ////__DACE:20:0:4    ////__DACE:17
            }                                                                                 ////__DACE:20:0:4    ////__DACE:17
            {                                                                                 ////__DACE:20:0:12    ////__DACE:17
                const double * __tlet_field = &gtir_tmp_95[0];                                    ////__DACE:20:0:21,12    ////__DACE:20:0:12    ////__DACE:17
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:20:0:19,12    ////__DACE:20:0:12    ////__DACE:17
                double __tlet_val;                                                                ////__DACE:20:0:12    ////__DACE:20:0:12    ////__DACE:17
                ////__DACE:20:0:12                                                            ////__DACE:20:0:12    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:12    ////__DACE:20:0:12    ////__DACE:17
                // Tasklet code (tlet_46_deref_0)                                                 ////__DACE:20:0:12    ////__DACE:20:0:12    ////__DACE:17
                __tlet_val = __tlet_field[((__tlet_index_Cell + (42122 * i_K_gtx_vertical)) - 2406)];    ////__DACE:20:0:12    ////__DACE:20:0:12    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:12    ////__DACE:20:0:12    ////__DACE:17
                ////__DACE:20:0:12                                                            ////__DACE:20:0:12    ////__DACE:17
                __map_fusion_gtir_tmp_126_0 = __tlet_val;                                         ////__DACE:20:0:12    ////__DACE:20:0:12    ////__DACE:17
            }                                                                                 ////__DACE:20:0:12    ////__DACE:17
            {                                                                                 ////__DACE:20:0:10    ////__DACE:17
                double __tlet_arg0 = gtir_tmp_54_0;                                               ////__DACE:20:0:20,10    ////__DACE:20:0:10    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_126_0;                                 ////__DACE:20:0:11,10    ////__DACE:20:0:10    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:10    ////__DACE:20:0:10    ////__DACE:17
                ////__DACE:20:0:10                                                            ////__DACE:20:0:10    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:10    ////__DACE:20:0:10    ////__DACE:17
                // Tasklet code (tlet_47_multiplies_0)                                            ////__DACE:20:0:10    ////__DACE:20:0:10    ////__DACE:17
                __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:20:0:10    ////__DACE:20:0:10    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:10    ////__DACE:20:0:10    ////__DACE:17
                ////__DACE:20:0:10                                                            ////__DACE:20:0:10    ////__DACE:17
                __map_fusion_gtir_tmp_128_0 = __tlet_result;                                      ////__DACE:20:0:10    ////__DACE:20:0:10    ////__DACE:17
            }                                                                                 ////__DACE:20:0:10    ////__DACE:17
            {                                                                                 ////__DACE:20:0:16    ////__DACE:17
                const double* __tlet_field = &perturbed_theta_v_at_cells_on_model_levels[0];      ////__DACE:20:0:23,16    ////__DACE:20:0:16    ////__DACE:17
                int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride_0 + i_Edge_gtx_horizontal)];    ////__DACE:20:0:19,16    ////__DACE:20:0:16    ////__DACE:17
                double __tlet_val;                                                                ////__DACE:20:0:16    ////__DACE:20:0:16    ////__DACE:17
                ////__DACE:20:0:16                                                            ////__DACE:20:0:16    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:16    ////__DACE:20:0:16    ////__DACE:17
                // Tasklet code (tlet_44_deref_0)                                                 ////__DACE:20:0:16    ////__DACE:20:0:16    ////__DACE:17
                __tlet_val = __tlet_field[((__perturbed_theta_v_at_cells_on_model_levels_K_stride_0 * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:20:0:16    ////__DACE:20:0:16    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:16    ////__DACE:20:0:16    ////__DACE:17
                ////__DACE:20:0:16                                                            ////__DACE:20:0:16    ////__DACE:17
                __map_fusion_gtir_tmp_122_0 = __tlet_val;                                         ////__DACE:20:0:16    ////__DACE:20:0:16    ////__DACE:17
            }                                                                                 ////__DACE:20:0:16    ////__DACE:17
            {                                                                                 ////__DACE:20:0:14    ////__DACE:17
                double __tlet_arg0 = reference_theta_at_edges_on_model_levels[((__reference_theta_at_edges_on_model_levels_K_stride_0 * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:20:0:22,14    ////__DACE:20:0:14    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_122_0;                                 ////__DACE:20:0:15,14    ////__DACE:20:0:14    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:14    ////__DACE:20:0:14    ////__DACE:17
                ////__DACE:20:0:14                                                            ////__DACE:20:0:14    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:14    ////__DACE:20:0:14    ////__DACE:17
                // Tasklet code (tlet_45_plus_0)                                                  ////__DACE:20:0:14    ////__DACE:20:0:14    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:20:0:14    ////__DACE:20:0:14    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:14    ////__DACE:20:0:14    ////__DACE:17
                ////__DACE:20:0:14                                                            ////__DACE:20:0:14    ////__DACE:17
                __map_fusion_gtir_tmp_124_0 = __tlet_result;                                      ////__DACE:20:0:14    ////__DACE:20:0:14    ////__DACE:17
            }                                                                                 ////__DACE:20:0:14    ////__DACE:17
            {                                                                                 ////__DACE:20:0:8    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_128_0;                                 ////__DACE:20:0:9,8    ////__DACE:20:0:8    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_124_0;                                 ////__DACE:20:0:13,8    ////__DACE:20:0:8    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                ////__DACE:20:0:8                                                             ////__DACE:20:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                // Tasklet code (tlet_48_plus_0)                                                  ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
                ////__DACE:20:0:8                                                             ////__DACE:20:0:8    ////__DACE:17
                __map_fusion_gtir_tmp_130_0 = __tlet_result;                                      ////__DACE:20:0:8    ////__DACE:20:0:8    ////__DACE:17
            }                                                                                 ////__DACE:20:0:8    ////__DACE:17
            {                                                                                 ////__DACE:20:0:2    ////__DACE:17
                double __tlet_arg1 = __map_fusion_gtir_tmp_134_0;                                 ////__DACE:20:0:3,2    ////__DACE:20:0:2    ////__DACE:17
                double __tlet_arg0 = __map_fusion_gtir_tmp_130_0;                                 ////__DACE:20:0:7,2    ////__DACE:20:0:2    ////__DACE:17
                double __tlet_result;                                                             ////__DACE:20:0:2    ////__DACE:20:0:2    ////__DACE:17
                ////__DACE:20:0:2                                                             ////__DACE:20:0:2    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:2    ////__DACE:20:0:2    ////__DACE:17
                // Tasklet code (tlet_51_plus_0)                                                  ////__DACE:20:0:2    ////__DACE:20:0:2    ////__DACE:17
                __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:20:0:2    ////__DACE:20:0:2    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:2    ////__DACE:20:0:2    ////__DACE:17
                ////__DACE:20:0:2                                                             ////__DACE:20:0:2    ////__DACE:17
                __arg2_ = __tlet_result;                                                          ////__DACE:20:0:2    ////__DACE:20:0:2    ////__DACE:17
            }                                                                                 ////__DACE:20:0:2    ////__DACE:17
            {                                                                                 ////__DACE:20:0:24    ////__DACE:17
                double _cpy_in = __arg2_;                                                         ////__DACE:20:0:1,24    ////__DACE:20:0:24    ////__DACE:17
                double _cpy_out;                                                                  ////__DACE:20:0:24    ////__DACE:20:0:24    ////__DACE:17
                ////__DACE:20:0:24                                                            ////__DACE:20:0:24    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:24    ////__DACE:20:0:24    ////__DACE:17
                // Tasklet code (copy___arg2__to___output)                                        ////__DACE:20:0:24    ////__DACE:20:0:24    ////__DACE:17
                _cpy_out = _cpy_in;                                                               ////__DACE:20:0:24    ////__DACE:20:0:24    ////__DACE:17
                ///////////////////                                                               ////__DACE:20:0:24    ////__DACE:20:0:24    ////__DACE:17
                ////__DACE:20:0:24                                                            ////__DACE:20:0:24    ////__DACE:17
                __output = _cpy_out;                                                              ////__DACE:20:0:24    ////__DACE:20:0:24    ////__DACE:17
            }                                                                                 ////__DACE:20:0:24    ////__DACE:17
            ////__DACE:17
        }                                                                             ////__DACE:17
    }                                                                                 ////__DACE:17
}                                                                                 ////__DACE:0:0:136
////__DACE:0:0:136


int __dace_init_cuda(theta_shared_probe_shared_state_t *__state, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __grf_tend_vn_K_stride, int __gt_conn_C2E2CO_neighbor_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __ikoffset_E2C_stride, int __ikoffset_K_stride, int __next_vn_K_stride, int __normal_wind_iau_increment_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, int __zdiff_gradp_E2C_stride, int __zdiff_gradp_K_stride, double dtime) {
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

int __dace_exit_cuda(theta_shared_probe_shared_state_t *__state) {
    

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
void __dace_gpu_drain_error(theta_shared_probe_shared_state_t *__state) {
    (void)__state;
    gpuError_t __pre_existing = hipGetLastError();
    if (__pre_existing != (gpuError_t)0) {
        printf("WARNING: a GPU error was already pending on entry to a DaCe program and has been "
               "discarded: %s (%d). It was not caused by this SDFG.\n",
               gpuGetErrorString(__pre_existing), __pre_existing);
    }
}

// Returns what the generated code recorded, not the runtime's shared slot, and clears it.
int __dace_gpu_last_error(theta_shared_probe_shared_state_t *__state) {
    int __err = static_cast<int>(__state->gpu_context->lasterror);
    __state->gpu_context->lasterror = (gpuError_t)0;
    return __err;
}

bool __dace_gpu_set_stream(theta_shared_probe_shared_state_t *__state, int streamid, gpuStream_t stream)
{
    if (streamid < 0 || streamid >= 1)
        return false;

    __state->gpu_context->streams[streamid] = stream;

    return true;
}

void __dace_gpu_set_all_streams(theta_shared_probe_shared_state_t *__state, gpuStream_t stream)
{
    for (int i = 0; i < 1; ++i)
        __state->gpu_context->streams[i] = stream;
}

__global__ void  __launch_bounds__(256) map_37_fieldop_0_0_420(const double * __restrict__ geofac_grg_x, const double * __restrict__ geofac_grg_y, const int * __restrict__ gt_conn_C2E2CO, double * __restrict__ gtir_tmp_101, double * __restrict__ gtir_tmp_83, double * __restrict__ gtir_tmp_89, double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __gt_conn_C2E2CO_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride) {    ////__DACE:0:0:420
    {                                                                                 ////__DACE:0:0:420
        {                                                                             ////__DACE:0:0:420
            int b_i_Cell_gtx_horizontal = ((256 * blockIdx.x) + 2406);                ////__DACE:0:0:420
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:420
            {                                                                         ////__DACE:0:0:20
                {                                                                     ////__DACE:0:0:20
                    {                                                                 ////__DACE:0:0:20
                        int i_Cell_gtx_horizontal = (threadIdx.x + b_i_Cell_gtx_horizontal);    ////__DACE:0:0:20
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:20
                        if (i_Cell_gtx_horizontal >= b_i_Cell_gtx_horizontal && i_Cell_gtx_horizontal < (Min(44527, (b_i_Cell_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:20
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(29, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:20
                                {                                                     ////__DACE:0:0:285
                                    #pragma unroll 4                                  ////__DACE:0:0:285
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:285
                                        double gtir_tmp_100;                          ////__DACE:0:0:13
                                        double gtir_tmp_94;                           ////__DACE:0:0:25
                                        double gtir_tmp_88;                           ////__DACE:0:0:34
                                        double gtir_tmp_82;                           ////__DACE:0:0:43
                                        double __map_fusion_gtir_tmp_99[4]  DACE_ALIGN(64);    ////__DACE:0:0:71
                                        double __map_fusion_gtir_tmp_97[4]  DACE_ALIGN(64);    ////__DACE:0:0:72
                                        double __map_fusion_gtir_tmp_93[4]  DACE_ALIGN(64);    ////__DACE:0:0:73
                                        double __map_fusion_gtir_tmp_91[4]  DACE_ALIGN(64);    ////__DACE:0:0:74
                                        double __map_fusion_gtir_tmp_87[4]  DACE_ALIGN(64);    ////__DACE:0:0:75
                                        double __map_fusion_gtir_tmp_85[4]  DACE_ALIGN(64);    ////__DACE:0:0:76
                                        double __map_fusion_gtir_tmp_81[4]  DACE_ALIGN(64);    ////__DACE:0:0:77
                                        double __map_fusion_gtir_tmp_79[4]  DACE_ALIGN(64);    ////__DACE:0:0:78
                                        {                                             ////__DACE:0:0:49
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:49
                                                double __gtx_double_write_remover_inner_inner_distribution_node_6;    ////__DACE:0:0:89
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
                                                {                                     ////__DACE:0:0:367
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_6;      ////__DACE:0:0:89,367    ////__DACE:0:0:367
                                                    double _cpy_out;                                                                  ////__DACE:0:0:367    ////__DACE:0:0:367
                                                    ////__DACE:0:0:367                ////__DACE:0:0:367
                                                    ///////////////////                                                               ////__DACE:0:0:367    ////__DACE:0:0:367
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_6_to___map_fusion_gtir_tmp_79)    ////__DACE:0:0:367    ////__DACE:0:0:367
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:367    ////__DACE:0:0:367
                                                    ///////////////////                                                               ////__DACE:0:0:367    ////__DACE:0:0:367
                                                    ////__DACE:0:0:367                ////__DACE:0:0:367
                                                    __map_fusion_gtir_tmp_79[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:367    ////__DACE:0:0:367
                                                }                                     ////__DACE:0:0:367
                                            }                                         ////__DACE:0:0:47
                                        }                                             ////__DACE:0:0:47
                                        {                                             ////__DACE:0:0:46
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:46
                                                double __gtx_double_write_remover_inner_inner_distribution_node_5;    ////__DACE:0:0:88
                                                {                                     ////__DACE:0:0:45
                                                    double __tlet_arg0 = geofac_grg_x[((__geofac_grg_x_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:32,45    ////__DACE:0:0:45
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_79[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:78,45    ////__DACE:0:0:45
                                                    double __tlet_out;                                                                ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    ////__DACE:0:0:45                 ////__DACE:0:0:45
                                                    ///////////////////                                                               ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    // Tasklet code (tlet_27_map)                                                     ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    ///////////////////                                                               ////__DACE:0:0:45    ////__DACE:0:0:45
                                                    ////__DACE:0:0:45                 ////__DACE:0:0:45
                                                    __gtx_double_write_remover_inner_inner_distribution_node_5 = __tlet_out;          ////__DACE:0:0:45    ////__DACE:0:0:45
                                                }                                     ////__DACE:0:0:45
                                                {                                     ////__DACE:0:0:366
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_5;      ////__DACE:0:0:88,366    ////__DACE:0:0:366
                                                    double _cpy_out;                                                                  ////__DACE:0:0:366    ////__DACE:0:0:366
                                                    ////__DACE:0:0:366                ////__DACE:0:0:366
                                                    ///////////////////                                                               ////__DACE:0:0:366    ////__DACE:0:0:366
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_5_to___map_fusion_gtir_tmp_81)    ////__DACE:0:0:366    ////__DACE:0:0:366
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:366    ////__DACE:0:0:366
                                                    ///////////////////                                                               ////__DACE:0:0:366    ////__DACE:0:0:366
                                                    ////__DACE:0:0:366                ////__DACE:0:0:366
                                                    __map_fusion_gtir_tmp_81[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:366    ////__DACE:0:0:366
                                                }                                     ////__DACE:0:0:366
                                            }                                         ////__DACE:0:0:44
                                        }                                             ////__DACE:0:0:44
                                        reduce_0_0_358(&__map_fusion_gtir_tmp_81[0], gtir_tmp_82);    ////__DACE:0:0:358
                                        {                                             ////__DACE:0:0:407
                                            double _cpy_in = gtir_tmp_82;                                                     ////__DACE:0:0:43,407    ////__DACE:0:0:407
                                            double _cpy_out;                                                                  ////__DACE:0:0:407    ////__DACE:0:0:407
                                            ////__DACE:0:0:407                        ////__DACE:0:0:407
                                            ///////////////////                                                               ////__DACE:0:0:407    ////__DACE:0:0:407
                                            // Tasklet code (copy_gtir_tmp_82_to_gtir_tmp_83)                                 ////__DACE:0:0:407    ////__DACE:0:0:407
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:407    ////__DACE:0:0:407
                                            ///////////////////                                                               ////__DACE:0:0:407    ////__DACE:0:0:407
                                            ////__DACE:0:0:407                        ////__DACE:0:0:407
                                            gtir_tmp_83[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:407    ////__DACE:0:0:407
                                        }                                             ////__DACE:0:0:407
                                        {                                             ////__DACE:0:0:40
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:40
                                                double __gtx_double_write_remover_inner_inner_distribution_node_4;    ////__DACE:0:0:87
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
                                                {                                     ////__DACE:0:0:365
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_4;      ////__DACE:0:0:87,365    ////__DACE:0:0:365
                                                    double _cpy_out;                                                                  ////__DACE:0:0:365    ////__DACE:0:0:365
                                                    ////__DACE:0:0:365                ////__DACE:0:0:365
                                                    ///////////////////                                                               ////__DACE:0:0:365    ////__DACE:0:0:365
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_4_to___map_fusion_gtir_tmp_85)    ////__DACE:0:0:365    ////__DACE:0:0:365
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:365    ////__DACE:0:0:365
                                                    ///////////////////                                                               ////__DACE:0:0:365    ////__DACE:0:0:365
                                                    ////__DACE:0:0:365                ////__DACE:0:0:365
                                                    __map_fusion_gtir_tmp_85[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:365    ////__DACE:0:0:365
                                                }                                     ////__DACE:0:0:365
                                            }                                         ////__DACE:0:0:38
                                        }                                             ////__DACE:0:0:38
                                        {                                             ////__DACE:0:0:37
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:37
                                                double __gtx_double_write_remover_inner_inner_distribution_node_3;    ////__DACE:0:0:86
                                                {                                     ////__DACE:0:0:36
                                                    double __tlet_arg0 = geofac_grg_y[((__geofac_grg_y_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:23,36    ////__DACE:0:0:36
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_85[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:76,36    ////__DACE:0:0:36
                                                    double __tlet_out;                                                                ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    ////__DACE:0:0:36                 ////__DACE:0:0:36
                                                    ///////////////////                                                               ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    // Tasklet code (tlet_29_map)                                                     ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    ///////////////////                                                               ////__DACE:0:0:36    ////__DACE:0:0:36
                                                    ////__DACE:0:0:36                 ////__DACE:0:0:36
                                                    __gtx_double_write_remover_inner_inner_distribution_node_3 = __tlet_out;          ////__DACE:0:0:36    ////__DACE:0:0:36
                                                }                                     ////__DACE:0:0:36
                                                {                                     ////__DACE:0:0:364
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_3;      ////__DACE:0:0:86,364    ////__DACE:0:0:364
                                                    double _cpy_out;                                                                  ////__DACE:0:0:364    ////__DACE:0:0:364
                                                    ////__DACE:0:0:364                ////__DACE:0:0:364
                                                    ///////////////////                                                               ////__DACE:0:0:364    ////__DACE:0:0:364
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_3_to___map_fusion_gtir_tmp_87)    ////__DACE:0:0:364    ////__DACE:0:0:364
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:364    ////__DACE:0:0:364
                                                    ///////////////////                                                               ////__DACE:0:0:364    ////__DACE:0:0:364
                                                    ////__DACE:0:0:364                ////__DACE:0:0:364
                                                    __map_fusion_gtir_tmp_87[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:364    ////__DACE:0:0:364
                                                }                                     ////__DACE:0:0:364
                                            }                                         ////__DACE:0:0:35
                                        }                                             ////__DACE:0:0:35
                                        reduce_0_0_358(&__map_fusion_gtir_tmp_87[0], gtir_tmp_88);    ////__DACE:0:0:357
                                        {                                             ////__DACE:0:0:408
                                            double _cpy_in = gtir_tmp_88;                                                     ////__DACE:0:0:34,408    ////__DACE:0:0:408
                                            double _cpy_out;                                                                  ////__DACE:0:0:408    ////__DACE:0:0:408
                                            ////__DACE:0:0:408                        ////__DACE:0:0:408
                                            ///////////////////                                                               ////__DACE:0:0:408    ////__DACE:0:0:408
                                            // Tasklet code (copy_gtir_tmp_88_to_gtir_tmp_89)                                 ////__DACE:0:0:408    ////__DACE:0:0:408
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:408    ////__DACE:0:0:408
                                            ///////////////////                                                               ////__DACE:0:0:408    ////__DACE:0:0:408
                                            ////__DACE:0:0:408                        ////__DACE:0:0:408
                                            gtir_tmp_89[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:408    ////__DACE:0:0:408
                                        }                                             ////__DACE:0:0:408
                                        {                                             ////__DACE:0:0:31
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:31
                                                double __gtx_double_write_remover_inner_inner_distribution_node_2;    ////__DACE:0:0:85
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
                                                {                                     ////__DACE:0:0:363
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_2;      ////__DACE:0:0:85,363    ////__DACE:0:0:363
                                                    double _cpy_out;                                                                  ////__DACE:0:0:363    ////__DACE:0:0:363
                                                    ////__DACE:0:0:363                ////__DACE:0:0:363
                                                    ///////////////////                                                               ////__DACE:0:0:363    ////__DACE:0:0:363
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_2_to___map_fusion_gtir_tmp_91)    ////__DACE:0:0:363    ////__DACE:0:0:363
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:363    ////__DACE:0:0:363
                                                    ///////////////////                                                               ////__DACE:0:0:363    ////__DACE:0:0:363
                                                    ////__DACE:0:0:363                ////__DACE:0:0:363
                                                    __map_fusion_gtir_tmp_91[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:363    ////__DACE:0:0:363
                                                }                                     ////__DACE:0:0:363
                                            }                                         ////__DACE:0:0:29
                                        }                                             ////__DACE:0:0:29
                                        {                                             ////__DACE:0:0:28
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:28
                                                double __gtx_double_write_remover_inner_inner_distribution_node_1;    ////__DACE:0:0:84
                                                {                                     ////__DACE:0:0:27
                                                    double __tlet_arg0 = geofac_grg_x[((__geofac_grg_x_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:32,27    ////__DACE:0:0:27
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_91[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:74,27    ////__DACE:0:0:27
                                                    double __tlet_out;                                                                ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    ////__DACE:0:0:27                 ////__DACE:0:0:27
                                                    ///////////////////                                                               ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    // Tasklet code (tlet_31_map)                                                     ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    ///////////////////                                                               ////__DACE:0:0:27    ////__DACE:0:0:27
                                                    ////__DACE:0:0:27                 ////__DACE:0:0:27
                                                    __gtx_double_write_remover_inner_inner_distribution_node_1 = __tlet_out;          ////__DACE:0:0:27    ////__DACE:0:0:27
                                                }                                     ////__DACE:0:0:27
                                                {                                     ////__DACE:0:0:362
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_1;      ////__DACE:0:0:84,362    ////__DACE:0:0:362
                                                    double _cpy_out;                                                                  ////__DACE:0:0:362    ////__DACE:0:0:362
                                                    ////__DACE:0:0:362                ////__DACE:0:0:362
                                                    ///////////////////                                                               ////__DACE:0:0:362    ////__DACE:0:0:362
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_1_to___map_fusion_gtir_tmp_93)    ////__DACE:0:0:362    ////__DACE:0:0:362
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:362    ////__DACE:0:0:362
                                                    ///////////////////                                                               ////__DACE:0:0:362    ////__DACE:0:0:362
                                                    ////__DACE:0:0:362                ////__DACE:0:0:362
                                                    __map_fusion_gtir_tmp_93[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:362    ////__DACE:0:0:362
                                                }                                     ////__DACE:0:0:362
                                            }                                         ////__DACE:0:0:26
                                        }                                             ////__DACE:0:0:26
                                        reduce_0_0_358(&__map_fusion_gtir_tmp_93[0], gtir_tmp_94);    ////__DACE:0:0:356
                                        {                                             ////__DACE:0:0:409
                                            double _cpy_in = gtir_tmp_94;                                                     ////__DACE:0:0:25,409    ////__DACE:0:0:409
                                            double _cpy_out;                                                                  ////__DACE:0:0:409    ////__DACE:0:0:409
                                            ////__DACE:0:0:409                        ////__DACE:0:0:409
                                            ///////////////////                                                               ////__DACE:0:0:409    ////__DACE:0:0:409
                                            // Tasklet code (copy_gtir_tmp_94_to_gtir_tmp_95)                                 ////__DACE:0:0:409    ////__DACE:0:0:409
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:409    ////__DACE:0:0:409
                                            ///////////////////                                                               ////__DACE:0:0:409    ////__DACE:0:0:409
                                            ////__DACE:0:0:409                        ////__DACE:0:0:409
                                            gtir_tmp_95[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:409    ////__DACE:0:0:409
                                        }                                             ////__DACE:0:0:409
                                        {                                             ////__DACE:0:0:19
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:19
                                                double __gtx_double_write_remover_inner_inner_distribution_node_0;    ////__DACE:0:0:83
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
                                                {                                     ////__DACE:0:0:361
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_0;      ////__DACE:0:0:83,361    ////__DACE:0:0:361
                                                    double _cpy_out;                                                                  ////__DACE:0:0:361    ////__DACE:0:0:361
                                                    ////__DACE:0:0:361                ////__DACE:0:0:361
                                                    ///////////////////                                                               ////__DACE:0:0:361    ////__DACE:0:0:361
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_0_to___map_fusion_gtir_tmp_97)    ////__DACE:0:0:361    ////__DACE:0:0:361
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:361    ////__DACE:0:0:361
                                                    ///////////////////                                                               ////__DACE:0:0:361    ////__DACE:0:0:361
                                                    ////__DACE:0:0:361                ////__DACE:0:0:361
                                                    __map_fusion_gtir_tmp_97[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:361    ////__DACE:0:0:361
                                                }                                     ////__DACE:0:0:361
                                            }                                         ////__DACE:0:0:17
                                        }                                             ////__DACE:0:0:17
                                        {                                             ////__DACE:0:0:16
                                            for (auto i_C2E2CO_gtx_localdim = 0; i_C2E2CO_gtx_localdim < 4; i_C2E2CO_gtx_localdim += 1) {    ////__DACE:0:0:16
                                                double __gtx_double_write_remover_inner_inner_distribution_node;    ////__DACE:0:0:82
                                                {                                     ////__DACE:0:0:15
                                                    double __tlet_arg0 = geofac_grg_y[((__geofac_grg_y_C2E2CO_stride * i_C2E2CO_gtx_localdim) + i_Cell_gtx_horizontal)];    ////__DACE:0:0:23,15    ////__DACE:0:0:15
                                                    double __tlet_arg1 = __map_fusion_gtir_tmp_97[i_C2E2CO_gtx_localdim];             ////__DACE:0:0:72,15    ////__DACE:0:0:15
                                                    double __tlet_out;                                                                ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    ////__DACE:0:0:15                 ////__DACE:0:0:15
                                                    ///////////////////                                                               ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    // Tasklet code (tlet_33_map)                                                     ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    ///////////////////                                                               ////__DACE:0:0:15    ////__DACE:0:0:15
                                                    ////__DACE:0:0:15                 ////__DACE:0:0:15
                                                    __gtx_double_write_remover_inner_inner_distribution_node = __tlet_out;            ////__DACE:0:0:15    ////__DACE:0:0:15
                                                }                                     ////__DACE:0:0:15
                                                {                                     ////__DACE:0:0:360
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node;        ////__DACE:0:0:82,360    ////__DACE:0:0:360
                                                    double _cpy_out;                                                                  ////__DACE:0:0:360    ////__DACE:0:0:360
                                                    ////__DACE:0:0:360                ////__DACE:0:0:360
                                                    ///////////////////                                                               ////__DACE:0:0:360    ////__DACE:0:0:360
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_to___map_fusion_gtir_tmp_99)    ////__DACE:0:0:360    ////__DACE:0:0:360
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:360    ////__DACE:0:0:360
                                                    ///////////////////                                                               ////__DACE:0:0:360    ////__DACE:0:0:360
                                                    ////__DACE:0:0:360                ////__DACE:0:0:360
                                                    __map_fusion_gtir_tmp_99[i_C2E2CO_gtx_localdim] = _cpy_out;                       ////__DACE:0:0:360    ////__DACE:0:0:360
                                                }                                     ////__DACE:0:0:360
                                            }                                         ////__DACE:0:0:14
                                        }                                             ////__DACE:0:0:14
                                        reduce_0_0_358(&__map_fusion_gtir_tmp_99[0], gtir_tmp_100);    ////__DACE:0:0:355
                                        {                                             ////__DACE:0:0:406
                                            double _cpy_in = gtir_tmp_100;                                                    ////__DACE:0:0:13,406    ////__DACE:0:0:406
                                            double _cpy_out;                                                                  ////__DACE:0:0:406    ////__DACE:0:0:406
                                            ////__DACE:0:0:406                        ////__DACE:0:0:406
                                            ///////////////////                                                               ////__DACE:0:0:406    ////__DACE:0:0:406
                                            // Tasklet code (copy_gtir_tmp_100_to_gtir_tmp_101)                               ////__DACE:0:0:406    ////__DACE:0:0:406
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:406    ////__DACE:0:0:406
                                            ///////////////////                                                               ////__DACE:0:0:406    ////__DACE:0:0:406
                                            ////__DACE:0:0:406                        ////__DACE:0:0:406
                                            gtir_tmp_101[((i_Cell_gtx_horizontal + (42122 * i_K_gtx_vertical)) - 2406)] = _cpy_out;    ////__DACE:0:0:406    ////__DACE:0:0:406
                                        }                                             ////__DACE:0:0:406
                                    }                                                 ////__DACE:0:0:286
                                }                                                     ////__DACE:0:0:286
                            }                                                         ////__DACE:0:0:12
                        }                                                             ////__DACE:0:0:12
                    }                                                                 ////__DACE:0:0:12
                }                                                                     ////__DACE:0:0:12
            }                                                                         ////__DACE:0:0:12
        }                                                                             ////__DACE:0:0:421
    }                                                                                 ////__DACE:0:0:421
}                                                                                 ////__DACE:0:0:421

                                                                                  ////__DACE:0:0:420
DACE_EXPORTED void __dace_runkernel_map_37_fieldop_0_0_420(theta_shared_probe_shared_state_t *__state, const double * __restrict__ geofac_grg_x, const double * __restrict__ geofac_grg_y, const int * __restrict__ gt_conn_C2E2CO, double * __restrict__ gtir_tmp_101, double * __restrict__ gtir_tmp_83, double * __restrict__ gtir_tmp_89, double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __gt_conn_C2E2CO_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride);    ////__DACE:0:0:420
void __dace_runkernel_map_37_fieldop_0_0_420(theta_shared_probe_shared_state_t *__state, const double * __restrict__ geofac_grg_x, const double * __restrict__ geofac_grg_y, const int * __restrict__ gt_conn_C2E2CO, double * __restrict__ gtir_tmp_101, double * __restrict__ gtir_tmp_83, double * __restrict__ gtir_tmp_89, double * __restrict__ gtir_tmp_95, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, int __geofac_grg_x_C2E2CO_stride, int __geofac_grg_y_C2E2CO_stride, int __gt_conn_C2E2CO_neighbor_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride)    ////__DACE:0:0:420
{                                                                                 ////__DACE:0:0:420
                                                                                  ////__DACE:0:0:420
    void  *map_37_fieldop_0_0_420_args[] = { (void *)&geofac_grg_x, (void *)&geofac_grg_y, (void *)&gt_conn_C2E2CO, (void *)&gtir_tmp_101, (void *)&gtir_tmp_83, (void *)&gtir_tmp_89, (void *)&gtir_tmp_95, (void *)&perturbed_rho_at_cells_on_model_levels, (void *)&perturbed_theta_v_at_cells_on_model_levels, (void *)&__geofac_grg_x_C2E2CO_stride, (void *)&__geofac_grg_y_C2E2CO_stride, (void *)&__gt_conn_C2E2CO_neighbor_stride, (void *)&__perturbed_rho_at_cells_on_model_levels_K_stride, (void *)&__perturbed_theta_v_at_cells_on_model_levels_K_stride };    ////__DACE:0:0:420
    gpuError_t __err = hipLaunchKernel((void*)map_37_fieldop_0_0_420, dim3(165, 30, 1), dim3(256, 1, 1), map_37_fieldop_0_0_420_args, 0, nullptr);    ////__DACE:0:0:420
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_37_fieldop_0_0_420", 165, 30, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_100_fieldop_1_0_0_0_424(const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pg_exdist, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:424
    {                                                                                 ////__DACE:0:0:424
        {                                                                             ////__DACE:0:0:424
            int b_i_Edge_gtx_horizontal = ((256 * blockIdx.x) + 7701);                ////__DACE:0:0:424
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:424
            {                                                                         ////__DACE:0:0:184
                {                                                                     ////__DACE:0:0:184
                    {                                                                 ////__DACE:0:0:184
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:184
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:184
                        double gtir_tmp_14_1_0;                                       ////__DACE:0:0:164
                        double gtir_tmp_12_1_0;                                       ////__DACE:0:0:165
                        double gtir_tmp_26_1_0;                                       ////__DACE:0:0:170
                        double gtir_tmp_24_1_0;                                       ////__DACE:0:0:171
                        double gtir_tmp_70_1_0;                                       ////__DACE:0:0:175
                        double gtir_tmp_66_1_0;                                       ////__DACE:0:0:176
                        double gtir_tmp_60_1_0;                                       ////__DACE:0:0:177
                        double gtir_tmp_56_1_0;                                       ////__DACE:0:0:178
                        double gtir_tmp_48_1_0;                                       ////__DACE:0:0:180
                        double gtir_tmp_44_1_0;                                       ////__DACE:0:0:181
                        double gtir_tmp_38_1_0;                                       ////__DACE:0:0:182
                        double gtir_tmp_34_1_0;                                       ////__DACE:0:0:183
                        bool gtir_tmp_7_2;                                            ////__DACE:0:0:300
                        bool gtir_tmp_6_2;                                            ////__DACE:0:0:306
                        double gtir_tmp_3_1;                                          ////__DACE:0:0:310
                        double __p_dthalf_0;                                          ////__DACE:0:0:314
                        double lambda_4___p_dthalf_2;                                 ////__DACE:0:0:324
                        double gtir_tmp_102_2;                                        ////__DACE:0:0:330
                        double gtir_tmp_166_0;                                        ////__DACE:0:0:332
                        double gtir_tmp_175_2;                                        ////__DACE:0:0:342
                        double gtir_tmp_223_1;                                        ////__DACE:0:0:346
                        double __dtime_0_0;                                           ////__DACE:0:0:348
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(67095, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:184
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(6, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:184
                                {                                                     ////__DACE:0:0:299
                                    bool __tlet_out;                                                                  ////__DACE:0:0:299    ////__DACE:0:0:299
                                    ////__DACE:0:0:299                                ////__DACE:0:0:299
                                    ///////////////////                                                               ////__DACE:0:0:299    ////__DACE:0:0:299
                                    // Tasklet code (tlet_5_get_value__clone_2)                                       ////__DACE:0:0:299    ////__DACE:0:0:299
                                    __tlet_out = false;                                                               ////__DACE:0:0:299    ////__DACE:0:0:299
                                    ///////////////////                                                               ////__DACE:0:0:299    ////__DACE:0:0:299
                                    ////__DACE:0:0:299                                ////__DACE:0:0:299
                                    gtir_tmp_7_2 = __tlet_out;                                                        ////__DACE:0:0:299    ////__DACE:0:0:299
                                }                                                     ////__DACE:0:0:299
                                {                                                     ////__DACE:0:0:305
                                    bool __tlet_out;                                                                  ////__DACE:0:0:305    ////__DACE:0:0:305
                                    ////__DACE:0:0:305                                ////__DACE:0:0:305
                                    ///////////////////                                                               ////__DACE:0:0:305    ////__DACE:0:0:305
                                    // Tasklet code (tlet_4_get_value__clone_2)                                       ////__DACE:0:0:305    ////__DACE:0:0:305
                                    __tlet_out = true;                                                                ////__DACE:0:0:305    ////__DACE:0:0:305
                                    ///////////////////                                                               ////__DACE:0:0:305    ////__DACE:0:0:305
                                    ////__DACE:0:0:305                                ////__DACE:0:0:305
                                    gtir_tmp_6_2 = __tlet_out;                                                        ////__DACE:0:0:305    ////__DACE:0:0:305
                                }                                                     ////__DACE:0:0:305
                                {                                                     ////__DACE:0:0:309
                                    double __tlet_out;                                                                ////__DACE:0:0:309    ////__DACE:0:0:309
                                    ////__DACE:0:0:309                                ////__DACE:0:0:309
                                    ///////////////////                                                               ////__DACE:0:0:309    ////__DACE:0:0:309
                                    // Tasklet code (tlet_2_get_value__clone_1)                                       ////__DACE:0:0:309    ////__DACE:0:0:309
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:309    ////__DACE:0:0:309
                                    ///////////////////                                                               ////__DACE:0:0:309    ////__DACE:0:0:309
                                    ////__DACE:0:0:309                                ////__DACE:0:0:309
                                    gtir_tmp_3_1 = __tlet_out;                                                        ////__DACE:0:0:309    ////__DACE:0:0:309
                                }                                                     ////__DACE:0:0:309
                                {                                                     ////__DACE:0:0:313
                                    double __tlet_out;                                                                ////__DACE:0:0:313    ////__DACE:0:0:313
                                    ////__DACE:0:0:313                                ////__DACE:0:0:313
                                    ///////////////////                                                               ////__DACE:0:0:313    ////__DACE:0:0:313
                                    // Tasklet code (tlet_6_get_value__clone_0)                                       ////__DACE:0:0:313    ////__DACE:0:0:313
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:313    ////__DACE:0:0:313
                                    ///////////////////                                                               ////__DACE:0:0:313    ////__DACE:0:0:313
                                    ////__DACE:0:0:313                                ////__DACE:0:0:313
                                    __p_dthalf_0 = __tlet_out;                                                        ////__DACE:0:0:313    ////__DACE:0:0:313
                                }                                                     ////__DACE:0:0:313
                                {                                                     ////__DACE:0:0:323
                                    double __tlet_out;                                                                ////__DACE:0:0:323    ////__DACE:0:0:323
                                    ////__DACE:0:0:323                                ////__DACE:0:0:323
                                    ///////////////////                                                               ////__DACE:0:0:323    ////__DACE:0:0:323
                                    // Tasklet code (tlet_10_get_value__clone_2)                                      ////__DACE:0:0:323    ////__DACE:0:0:323
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:323    ////__DACE:0:0:323
                                    ///////////////////                                                               ////__DACE:0:0:323    ////__DACE:0:0:323
                                    ////__DACE:0:0:323                                ////__DACE:0:0:323
                                    lambda_4___p_dthalf_2 = __tlet_out;                                               ////__DACE:0:0:323    ////__DACE:0:0:323
                                }                                                     ////__DACE:0:0:323
                                {                                                     ////__DACE:0:0:329
                                    double __tlet_out;                                                                ////__DACE:0:0:329    ////__DACE:0:0:329
                                    ////__DACE:0:0:329                                ////__DACE:0:0:329
                                    ///////////////////                                                               ////__DACE:0:0:329    ////__DACE:0:0:329
                                    // Tasklet code (tlet_34_get_value__clone_2)                                      ////__DACE:0:0:329    ////__DACE:0:0:329
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:329    ////__DACE:0:0:329
                                    ///////////////////                                                               ////__DACE:0:0:329    ////__DACE:0:0:329
                                    ////__DACE:0:0:329                                ////__DACE:0:0:329
                                    gtir_tmp_102_2 = __tlet_out;                                                      ////__DACE:0:0:329    ////__DACE:0:0:329
                                }                                                     ////__DACE:0:0:329
                                {                                                     ////__DACE:0:0:331
                                    double __tlet_out;                                                                ////__DACE:0:0:331    ////__DACE:0:0:331
                                    ////__DACE:0:0:331                                ////__DACE:0:0:331
                                    ///////////////////                                                               ////__DACE:0:0:331    ////__DACE:0:0:331
                                    // Tasklet code (tlet_64_get_value__clone_0)                                      ////__DACE:0:0:331    ////__DACE:0:0:331
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:331    ////__DACE:0:0:331
                                    ///////////////////                                                               ////__DACE:0:0:331    ////__DACE:0:0:331
                                    ////__DACE:0:0:331                                ////__DACE:0:0:331
                                    gtir_tmp_166_0 = __tlet_out;                                                      ////__DACE:0:0:331    ////__DACE:0:0:331
                                }                                                     ////__DACE:0:0:331
                                {                                                     ////__DACE:0:0:341
                                    double __tlet_out;                                                                ////__DACE:0:0:341    ////__DACE:0:0:341
                                    ////__DACE:0:0:341                                ////__DACE:0:0:341
                                    ///////////////////                                                               ////__DACE:0:0:341    ////__DACE:0:0:341
                                    // Tasklet code (tlet_68_get_value__clone_2)                                      ////__DACE:0:0:341    ////__DACE:0:0:341
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:341    ////__DACE:0:0:341
                                    ///////////////////                                                               ////__DACE:0:0:341    ////__DACE:0:0:341
                                    ////__DACE:0:0:341                                ////__DACE:0:0:341
                                    gtir_tmp_175_2 = __tlet_out;                                                      ////__DACE:0:0:341    ////__DACE:0:0:341
                                }                                                     ////__DACE:0:0:341
                                {                                                     ////__DACE:0:0:345
                                    double __tlet_out;                                                                ////__DACE:0:0:345    ////__DACE:0:0:345
                                    ////__DACE:0:0:345                                ////__DACE:0:0:345
                                    ///////////////////                                                               ////__DACE:0:0:345    ////__DACE:0:0:345
                                    // Tasklet code (tlet_92_get_value__clone_1)                                      ////__DACE:0:0:345    ////__DACE:0:0:345
                                    __tlet_out = 1004.64;                                                             ////__DACE:0:0:345    ////__DACE:0:0:345
                                    ///////////////////                                                               ////__DACE:0:0:345    ////__DACE:0:0:345
                                    ////__DACE:0:0:345                                ////__DACE:0:0:345
                                    gtir_tmp_223_1 = __tlet_out;                                                      ////__DACE:0:0:345    ////__DACE:0:0:345
                                }                                                     ////__DACE:0:0:345
                                {                                                     ////__DACE:0:0:347
                                    double __tlet_out;                                                                ////__DACE:0:0:347    ////__DACE:0:0:347
                                    ////__DACE:0:0:347                                ////__DACE:0:0:347
                                    ///////////////////                                                               ////__DACE:0:0:347    ////__DACE:0:0:347
                                    // Tasklet code (tlet_91_get_value__clone_0)                                      ////__DACE:0:0:347    ////__DACE:0:0:347
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:347    ////__DACE:0:0:347
                                    ///////////////////                                                               ////__DACE:0:0:347    ////__DACE:0:0:347
                                    ////__DACE:0:0:347                                ////__DACE:0:0:347
                                    __dtime_0_0 = __tlet_out;                                                         ////__DACE:0:0:347    ////__DACE:0:0:347
                                }                                                     ////__DACE:0:0:347
                                {                                                     ////__DACE:0:0:380
                                    double _cpy_in = dual_normal_cell_x[i_Edge_gtx_horizontal];                       ////__DACE:0:0:9,380    ////__DACE:0:0:380
                                    double _cpy_out;                                                                  ////__DACE:0:0:380    ////__DACE:0:0:380
                                    ////__DACE:0:0:380                                ////__DACE:0:0:380
                                    ///////////////////                                                               ////__DACE:0:0:380    ////__DACE:0:0:380
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_38_1_0)                      ////__DACE:0:0:380    ////__DACE:0:0:380
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:380    ////__DACE:0:0:380
                                    ///////////////////                                                               ////__DACE:0:0:380    ////__DACE:0:0:380
                                    ////__DACE:0:0:380                                ////__DACE:0:0:380
                                    gtir_tmp_38_1_0 = _cpy_out;                                                       ////__DACE:0:0:380    ////__DACE:0:0:380
                                }                                                     ////__DACE:0:0:380
                                {                                                     ////__DACE:0:0:381
                                    double _cpy_in = dual_normal_cell_x[(__dual_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:9,381    ////__DACE:0:0:381
                                    double _cpy_out;                                                                  ////__DACE:0:0:381    ////__DACE:0:0:381
                                    ////__DACE:0:0:381                                ////__DACE:0:0:381
                                    ///////////////////                                                               ////__DACE:0:0:381    ////__DACE:0:0:381
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_48_1_0)                      ////__DACE:0:0:381    ////__DACE:0:0:381
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:381    ////__DACE:0:0:381
                                    ///////////////////                                                               ////__DACE:0:0:381    ////__DACE:0:0:381
                                    ////__DACE:0:0:381                                ////__DACE:0:0:381
                                    gtir_tmp_48_1_0 = _cpy_out;                                                       ////__DACE:0:0:381    ////__DACE:0:0:381
                                }                                                     ////__DACE:0:0:381
                                {                                                     ////__DACE:0:0:382
                                    double _cpy_in = dual_normal_cell_y[i_Edge_gtx_horizontal];                       ////__DACE:0:0:7,382    ////__DACE:0:0:382
                                    double _cpy_out;                                                                  ////__DACE:0:0:382    ////__DACE:0:0:382
                                    ////__DACE:0:0:382                                ////__DACE:0:0:382
                                    ///////////////////                                                               ////__DACE:0:0:382    ////__DACE:0:0:382
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_60_1_0)                      ////__DACE:0:0:382    ////__DACE:0:0:382
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:382    ////__DACE:0:0:382
                                    ///////////////////                                                               ////__DACE:0:0:382    ////__DACE:0:0:382
                                    ////__DACE:0:0:382                                ////__DACE:0:0:382
                                    gtir_tmp_60_1_0 = _cpy_out;                                                       ////__DACE:0:0:382    ////__DACE:0:0:382
                                }                                                     ////__DACE:0:0:382
                                {                                                     ////__DACE:0:0:383
                                    double _cpy_in = dual_normal_cell_y[(__dual_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:7,383    ////__DACE:0:0:383
                                    double _cpy_out;                                                                  ////__DACE:0:0:383    ////__DACE:0:0:383
                                    ////__DACE:0:0:383                                ////__DACE:0:0:383
                                    ///////////////////                                                               ////__DACE:0:0:383    ////__DACE:0:0:383
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_70_1_0)                      ////__DACE:0:0:383    ////__DACE:0:0:383
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:383    ////__DACE:0:0:383
                                    ///////////////////                                                               ////__DACE:0:0:383    ////__DACE:0:0:383
                                    ////__DACE:0:0:383                                ////__DACE:0:0:383
                                    gtir_tmp_70_1_0 = _cpy_out;                                                       ////__DACE:0:0:383    ////__DACE:0:0:383
                                }                                                     ////__DACE:0:0:383
                                {                                                     ////__DACE:0:0:384
                                    double _cpy_in = pos_on_tplane_e_x[i_Edge_gtx_horizontal];                        ////__DACE:0:0:4,384    ////__DACE:0:0:384
                                    double _cpy_out;                                                                  ////__DACE:0:0:384    ////__DACE:0:0:384
                                    ////__DACE:0:0:384                                ////__DACE:0:0:384
                                    ///////////////////                                                               ////__DACE:0:0:384    ////__DACE:0:0:384
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_12_1_0)                       ////__DACE:0:0:384    ////__DACE:0:0:384
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:384    ////__DACE:0:0:384
                                    ///////////////////                                                               ////__DACE:0:0:384    ////__DACE:0:0:384
                                    ////__DACE:0:0:384                                ////__DACE:0:0:384
                                    gtir_tmp_12_1_0 = _cpy_out;                                                       ////__DACE:0:0:384    ////__DACE:0:0:384
                                }                                                     ////__DACE:0:0:384
                                {                                                     ////__DACE:0:0:385
                                    double _cpy_in = pos_on_tplane_e_x[(__pos_on_tplane_e_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:4,385    ////__DACE:0:0:385
                                    double _cpy_out;                                                                  ////__DACE:0:0:385    ////__DACE:0:0:385
                                    ////__DACE:0:0:385                                ////__DACE:0:0:385
                                    ///////////////////                                                               ////__DACE:0:0:385    ////__DACE:0:0:385
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_14_1_0)                       ////__DACE:0:0:385    ////__DACE:0:0:385
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:385    ////__DACE:0:0:385
                                    ///////////////////                                                               ////__DACE:0:0:385    ////__DACE:0:0:385
                                    ////__DACE:0:0:385                                ////__DACE:0:0:385
                                    gtir_tmp_14_1_0 = _cpy_out;                                                       ////__DACE:0:0:385    ////__DACE:0:0:385
                                }                                                     ////__DACE:0:0:385
                                {                                                     ////__DACE:0:0:386
                                    double _cpy_in = pos_on_tplane_e_y[i_Edge_gtx_horizontal];                        ////__DACE:0:0:5,386    ////__DACE:0:0:386
                                    double _cpy_out;                                                                  ////__DACE:0:0:386    ////__DACE:0:0:386
                                    ////__DACE:0:0:386                                ////__DACE:0:0:386
                                    ///////////////////                                                               ////__DACE:0:0:386    ////__DACE:0:0:386
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_24_1_0)                       ////__DACE:0:0:386    ////__DACE:0:0:386
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:386    ////__DACE:0:0:386
                                    ///////////////////                                                               ////__DACE:0:0:386    ////__DACE:0:0:386
                                    ////__DACE:0:0:386                                ////__DACE:0:0:386
                                    gtir_tmp_24_1_0 = _cpy_out;                                                       ////__DACE:0:0:386    ////__DACE:0:0:386
                                }                                                     ////__DACE:0:0:386
                                {                                                     ////__DACE:0:0:387
                                    double _cpy_in = pos_on_tplane_e_y[(__pos_on_tplane_e_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:5,387    ////__DACE:0:0:387
                                    double _cpy_out;                                                                  ////__DACE:0:0:387    ////__DACE:0:0:387
                                    ////__DACE:0:0:387                                ////__DACE:0:0:387
                                    ///////////////////                                                               ////__DACE:0:0:387    ////__DACE:0:0:387
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_26_1_0)                       ////__DACE:0:0:387    ////__DACE:0:0:387
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:387    ////__DACE:0:0:387
                                    ///////////////////                                                               ////__DACE:0:0:387    ////__DACE:0:0:387
                                    ////__DACE:0:0:387                                ////__DACE:0:0:387
                                    gtir_tmp_26_1_0 = _cpy_out;                                                       ////__DACE:0:0:387    ////__DACE:0:0:387
                                }                                                     ////__DACE:0:0:387
                                {                                                     ////__DACE:0:0:388
                                    double _cpy_in = primal_normal_cell_x[i_Edge_gtx_horizontal];                     ////__DACE:0:0:10,388    ////__DACE:0:0:388
                                    double _cpy_out;                                                                  ////__DACE:0:0:388    ////__DACE:0:0:388
                                    ////__DACE:0:0:388                                ////__DACE:0:0:388
                                    ///////////////////                                                               ////__DACE:0:0:388    ////__DACE:0:0:388
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_34_1_0)                    ////__DACE:0:0:388    ////__DACE:0:0:388
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:388    ////__DACE:0:0:388
                                    ///////////////////                                                               ////__DACE:0:0:388    ////__DACE:0:0:388
                                    ////__DACE:0:0:388                                ////__DACE:0:0:388
                                    gtir_tmp_34_1_0 = _cpy_out;                                                       ////__DACE:0:0:388    ////__DACE:0:0:388
                                }                                                     ////__DACE:0:0:388
                                {                                                     ////__DACE:0:0:389
                                    double _cpy_in = primal_normal_cell_x[(__primal_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:10,389    ////__DACE:0:0:389
                                    double _cpy_out;                                                                  ////__DACE:0:0:389    ////__DACE:0:0:389
                                    ////__DACE:0:0:389                                ////__DACE:0:0:389
                                    ///////////////////                                                               ////__DACE:0:0:389    ////__DACE:0:0:389
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_44_1_0)                    ////__DACE:0:0:389    ////__DACE:0:0:389
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:389    ////__DACE:0:0:389
                                    ///////////////////                                                               ////__DACE:0:0:389    ////__DACE:0:0:389
                                    ////__DACE:0:0:389                                ////__DACE:0:0:389
                                    gtir_tmp_44_1_0 = _cpy_out;                                                       ////__DACE:0:0:389    ////__DACE:0:0:389
                                }                                                     ////__DACE:0:0:389
                                {                                                     ////__DACE:0:0:390
                                    double _cpy_in = primal_normal_cell_y[i_Edge_gtx_horizontal];                     ////__DACE:0:0:8,390    ////__DACE:0:0:390
                                    double _cpy_out;                                                                  ////__DACE:0:0:390    ////__DACE:0:0:390
                                    ////__DACE:0:0:390                                ////__DACE:0:0:390
                                    ///////////////////                                                               ////__DACE:0:0:390    ////__DACE:0:0:390
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_56_1_0)                    ////__DACE:0:0:390    ////__DACE:0:0:390
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:390    ////__DACE:0:0:390
                                    ///////////////////                                                               ////__DACE:0:0:390    ////__DACE:0:0:390
                                    ////__DACE:0:0:390                                ////__DACE:0:0:390
                                    gtir_tmp_56_1_0 = _cpy_out;                                                       ////__DACE:0:0:390    ////__DACE:0:0:390
                                }                                                     ////__DACE:0:0:390
                                {                                                     ////__DACE:0:0:391
                                    double _cpy_in = primal_normal_cell_y[(__primal_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:8,391    ////__DACE:0:0:391
                                    double _cpy_out;                                                                  ////__DACE:0:0:391    ////__DACE:0:0:391
                                    ////__DACE:0:0:391                                ////__DACE:0:0:391
                                    ///////////////////                                                               ////__DACE:0:0:391    ////__DACE:0:0:391
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_66_1_0)                    ////__DACE:0:0:391    ////__DACE:0:0:391
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:391    ////__DACE:0:0:391
                                    ///////////////////                                                               ////__DACE:0:0:391    ////__DACE:0:0:391
                                    ////__DACE:0:0:391                                ////__DACE:0:0:391
                                    gtir_tmp_66_1_0 = _cpy_out;                                                       ////__DACE:0:0:391    ////__DACE:0:0:391
                                }                                                     ////__DACE:0:0:391
                                {                                                     ////__DACE:0:0:289
                                    #pragma unroll 4                                  ////__DACE:0:0:289
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(27, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:289
                                        double __map_fusion_gtir_tmp_145;             ////__DACE:0:0:79
                                        double __map_fusion_gtir_tmp_143;             ////__DACE:0:0:80
                                        double __map_fusion_gtir_tmp_141;             ////__DACE:0:0:81
                                        double gtir_tmp_173_0;                        ////__DACE:0:0:90
                                        bool __map_fusion_gtir_tmp_168_0;             ////__DACE:0:0:100
                                        double __map_fusion_gtir_tmp_235_0;           ////__DACE:0:0:101
                                        double __map_fusion_gtir_tmp_233_0;           ////__DACE:0:0:102
                                        double __map_fusion_gtir_tmp_231_0;           ////__DACE:0:0:103
                                        double __map_fusion_gtir_tmp_229_0;           ////__DACE:0:0:104
                                        double __map_fusion_gtir_tmp_227_0;           ////__DACE:0:0:105
                                        double __map_fusion_gtir_tmp_139_split_0;     ////__DACE:0:0:106
                                        bool gtir_tmp_8_1_0;                          ////__DACE:0:0:157
                                        double gtir_tmp_16_1_0;                       ////__DACE:0:0:162
                                        double gtir_tmp_28_1_0;                       ////__DACE:0:0:169
                                        double gtir_tmp_76_1_0;                       ////__DACE:0:0:173
                                        double gtir_tmp_54_1_0;                       ////__DACE:0:0:179
                                        double gtir_tmp_137_1_0;                      ////__DACE:0:0:185
                                        double gtir_tmp_210_1_0;                      ////__DACE:0:0:188
                                        bool __map_fusion_gtir_tmp_5_1_0;             ////__DACE:0:0:191
                                        double __map_fusion_gtir_tmp_21_1_1_0_0;      ////__DACE:0:0:192
                                        double __map_fusion_gtir_tmp_19_1_0;          ////__DACE:0:0:193
                                        double __map_fusion_gtir_tmp_11_1_0;          ////__DACE:0:0:194
                                        double __map_fusion_gtir_tmp_33_1_1_0_0;      ////__DACE:0:0:195
                                        double __map_fusion_gtir_tmp_31_1_0;          ////__DACE:0:0:196
                                        double __map_fusion_gtir_tmp_23_1_0;          ////__DACE:0:0:197
                                        bool __map_fusion_gtir_tmp_104_1_0;           ////__DACE:0:0:198
                                        bool __map_fusion_gtir_tmp_177_1_0;           ////__DACE:0:0:199
                                        {                                             ////__DACE:0:0:166
                                            double __tlet_arg1 = __p_dthalf_0;                                                ////__DACE:0:0:314,166    ////__DACE:0:0:166
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,166    ////__DACE:0:0:166
                                            double __tlet_result;                                                             ////__DACE:0:0:166    ////__DACE:0:0:166
                                            ////__DACE:0:0:166                        ////__DACE:0:0:166
                                            ///////////////////                                                               ////__DACE:0:0:166    ////__DACE:0:0:166
                                            // Tasklet code (tlet_7_multiplies_1_0)                                           ////__DACE:0:0:166    ////__DACE:0:0:166
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:166    ////__DACE:0:0:166
                                            ///////////////////                                                               ////__DACE:0:0:166    ////__DACE:0:0:166
                                            ////__DACE:0:0:166                        ////__DACE:0:0:166
                                            __map_fusion_gtir_tmp_11_1_0 = __tlet_result;                                     ////__DACE:0:0:166    ////__DACE:0:0:166
                                        }                                             ////__DACE:0:0:166
                                        {                                             ////__DACE:0:0:172
                                            double __tlet_arg1 = lambda_4___p_dthalf_2;                                       ////__DACE:0:0:324,172    ////__DACE:0:0:172
                                            double __tlet_arg0 = tangential_wind[((__tangential_wind_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:6,172    ////__DACE:0:0:172
                                            double __tlet_result;                                                             ////__DACE:0:0:172    ////__DACE:0:0:172
                                            ////__DACE:0:0:172                        ////__DACE:0:0:172
                                            ///////////////////                                                               ////__DACE:0:0:172    ////__DACE:0:0:172
                                            // Tasklet code (tlet_11_multiplies_1_0)                                          ////__DACE:0:0:172    ////__DACE:0:0:172
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:172    ////__DACE:0:0:172
                                            ///////////////////                                                               ////__DACE:0:0:172    ////__DACE:0:0:172
                                            ////__DACE:0:0:172                        ////__DACE:0:0:172
                                            __map_fusion_gtir_tmp_23_1_0 = __tlet_result;                                     ////__DACE:0:0:172    ////__DACE:0:0:172
                                        }                                             ////__DACE:0:0:172
                                        {                                             ////__DACE:0:0:190
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,190    ////__DACE:0:0:190
                                            double __tlet_arg1 = gtir_tmp_175_2;                                              ////__DACE:0:0:342,190    ////__DACE:0:0:190
                                            bool __tlet_result;                                                               ////__DACE:0:0:190    ////__DACE:0:0:190
                                            ////__DACE:0:0:190                        ////__DACE:0:0:190
                                            ///////////////////                                                               ////__DACE:0:0:190    ////__DACE:0:0:190
                                            // Tasklet code (tlet_69_greater_equal_1_0)                                       ////__DACE:0:0:190    ////__DACE:0:0:190
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:190    ////__DACE:0:0:190
                                            ///////////////////                                                               ////__DACE:0:0:190    ////__DACE:0:0:190
                                            ////__DACE:0:0:190                        ////__DACE:0:0:190
                                            __map_fusion_gtir_tmp_177_1_0 = __tlet_result;                                    ////__DACE:0:0:190    ////__DACE:0:0:190
                                        }                                             ////__DACE:0:0:190
                                        {                                             ////__DACE:0:0:159
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,159    ////__DACE:0:0:159
                                            double __tlet_arg1 = gtir_tmp_3_1;                                                ////__DACE:0:0:310,159    ////__DACE:0:0:159
                                            bool __tlet_result;                                                               ////__DACE:0:0:159    ////__DACE:0:0:159
                                            ////__DACE:0:0:159                        ////__DACE:0:0:159
                                            ///////////////////                                                               ////__DACE:0:0:159    ////__DACE:0:0:159
                                            // Tasklet code (tlet_3_greater_equal_1_0)                                        ////__DACE:0:0:159    ////__DACE:0:0:159
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:159    ////__DACE:0:0:159
                                            ///////////////////                                                               ////__DACE:0:0:159    ////__DACE:0:0:159
                                            ////__DACE:0:0:159                        ////__DACE:0:0:159
                                            __map_fusion_gtir_tmp_5_1_0 = __tlet_result;                                      ////__DACE:0:0:159    ////__DACE:0:0:159
                                        }                                             ////__DACE:0:0:159
                                        if_stmt_0_0_0_158(gtir_tmp_6_2, gtir_tmp_7_2, __map_fusion_gtir_tmp_5_1_0, gtir_tmp_8_1_0);    ////__DACE:0:0:158
                                        if_stmt_1_0_0_163(gtir_tmp_12_1_0, gtir_tmp_24_1_0, gtir_tmp_14_1_0, gtir_tmp_26_1_0, gtir_tmp_8_1_0, gtir_tmp_16_1_0, gtir_tmp_28_1_0);    ////__DACE:0:0:163
                                        {                                             ////__DACE:0:0:161
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_11_1_0;                                ////__DACE:0:0:194,161    ////__DACE:0:0:161
                                            double __tlet_arg1 = gtir_tmp_16_1_0;                                             ////__DACE:0:0:162,161    ////__DACE:0:0:161
                                            double __tlet_result;                                                             ////__DACE:0:0:161    ////__DACE:0:0:161
                                            ////__DACE:0:0:161                        ////__DACE:0:0:161
                                            ///////////////////                                                               ////__DACE:0:0:161    ////__DACE:0:0:161
                                            // Tasklet code (tlet_8_plus_1_0)                                                 ////__DACE:0:0:161    ////__DACE:0:0:161
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:161    ////__DACE:0:0:161
                                            ///////////////////                                                               ////__DACE:0:0:161    ////__DACE:0:0:161
                                            ////__DACE:0:0:161                        ////__DACE:0:0:161
                                            __map_fusion_gtir_tmp_19_1_0 = __tlet_result;                                     ////__DACE:0:0:161    ////__DACE:0:0:161
                                        }                                             ////__DACE:0:0:161
                                        {                                             ////__DACE:0:0:160
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_19_1_0;                                ////__DACE:0:0:193,160    ////__DACE:0:0:160
                                            double __tlet_result;                                                             ////__DACE:0:0:160    ////__DACE:0:0:160
                                            ////__DACE:0:0:160                        ////__DACE:0:0:160
                                            ///////////////////                                                               ////__DACE:0:0:160    ////__DACE:0:0:160
                                            // Tasklet code (tlet_9_neg_1_0)                                                  ////__DACE:0:0:160    ////__DACE:0:0:160
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:160    ////__DACE:0:0:160
                                            ///////////////////                                                               ////__DACE:0:0:160    ////__DACE:0:0:160
                                            ////__DACE:0:0:160                        ////__DACE:0:0:160
                                            __map_fusion_gtir_tmp_21_1_1_0_0 = __tlet_result;                                 ////__DACE:0:0:160    ////__DACE:0:0:160
                                        }                                             ////__DACE:0:0:160
                                        {                                             ////__DACE:0:0:168
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_23_1_0;                                ////__DACE:0:0:197,168    ////__DACE:0:0:168
                                            double __tlet_arg1 = gtir_tmp_28_1_0;                                             ////__DACE:0:0:169,168    ////__DACE:0:0:168
                                            double __tlet_result;                                                             ////__DACE:0:0:168    ////__DACE:0:0:168
                                            ////__DACE:0:0:168                        ////__DACE:0:0:168
                                            ///////////////////                                                               ////__DACE:0:0:168    ////__DACE:0:0:168
                                            // Tasklet code (tlet_12_plus_1_0)                                                ////__DACE:0:0:168    ////__DACE:0:0:168
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:168    ////__DACE:0:0:168
                                            ///////////////////                                                               ////__DACE:0:0:168    ////__DACE:0:0:168
                                            ////__DACE:0:0:168                        ////__DACE:0:0:168
                                            __map_fusion_gtir_tmp_31_1_0 = __tlet_result;                                     ////__DACE:0:0:168    ////__DACE:0:0:168
                                        }                                             ////__DACE:0:0:168
                                        {                                             ////__DACE:0:0:167
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_31_1_0;                                ////__DACE:0:0:196,167    ////__DACE:0:0:167
                                            double __tlet_result;                                                             ////__DACE:0:0:167    ////__DACE:0:0:167
                                            ////__DACE:0:0:167                        ////__DACE:0:0:167
                                            ///////////////////                                                               ////__DACE:0:0:167    ////__DACE:0:0:167
                                            // Tasklet code (tlet_13_neg_1_0)                                                 ////__DACE:0:0:167    ////__DACE:0:0:167
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:167    ////__DACE:0:0:167
                                            ///////////////////                                                               ////__DACE:0:0:167    ////__DACE:0:0:167
                                            ////__DACE:0:0:167                        ////__DACE:0:0:167
                                            __map_fusion_gtir_tmp_33_1_1_0_0 = __tlet_result;                                 ////__DACE:0:0:167    ////__DACE:0:0:167
                                        }                                             ////__DACE:0:0:167
                                        if_stmt_4_0_0_174(gtir_tmp_8_1_0, __map_fusion_gtir_tmp_21_1_1_0_0, __map_fusion_gtir_tmp_33_1_1_0_0, gtir_tmp_34_1_0, gtir_tmp_38_1_0, gtir_tmp_44_1_0, gtir_tmp_48_1_0, gtir_tmp_56_1_0, gtir_tmp_60_1_0, gtir_tmp_66_1_0, gtir_tmp_70_1_0, gtir_tmp_76_1_0, gtir_tmp_54_1_0);    ////__DACE:0:0:174
                                        if_stmt_7_0_0_189(__map_fusion_gtir_tmp_177_1_0, &gt_conn_E2C[0], gtir_tmp_54_1_0, gtir_tmp_76_1_0, &gtir_tmp_83[0], &gtir_tmp_89[0], &perturbed_rho_at_cells_on_model_levels[0], &reference_rho_at_edges_on_model_levels[0], gtir_tmp_210_1_0, __gt_conn_E2C_neighbor_stride, __perturbed_rho_at_cells_on_model_levels_K_stride, __reference_rho_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:189
                                        {                                             ////__DACE:0:0:412
                                            double _cpy_in = gtir_tmp_210_1_0;                                                ////__DACE:0:0:188,412    ////__DACE:0:0:412
                                            double _cpy_out;                                                                  ////__DACE:0:0:412    ////__DACE:0:0:412
                                            ////__DACE:0:0:412                        ////__DACE:0:0:412
                                            ///////////////////                                                               ////__DACE:0:0:412    ////__DACE:0:0:412
                                            // Tasklet code (copy_gtir_tmp_210_1_0_to_rho_at_edges_on_model_levels)           ////__DACE:0:0:412    ////__DACE:0:0:412
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:412    ////__DACE:0:0:412
                                            ///////////////////                                                               ////__DACE:0:0:412    ////__DACE:0:0:412
                                            ////__DACE:0:0:412                        ////__DACE:0:0:412
                                            rho_at_edges_on_model_levels[((__rho_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:412    ////__DACE:0:0:412
                                        }                                             ////__DACE:0:0:412
                                        {                                             ////__DACE:0:0:187
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,187    ////__DACE:0:0:187
                                            double __tlet_arg1 = gtir_tmp_102_2;                                              ////__DACE:0:0:330,187    ////__DACE:0:0:187
                                            bool __tlet_result;                                                               ////__DACE:0:0:187    ////__DACE:0:0:187
                                            ////__DACE:0:0:187                        ////__DACE:0:0:187
                                            ///////////////////                                                               ////__DACE:0:0:187    ////__DACE:0:0:187
                                            // Tasklet code (tlet_35_greater_equal_1_0)                                       ////__DACE:0:0:187    ////__DACE:0:0:187
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:187    ////__DACE:0:0:187
                                            ///////////////////                                                               ////__DACE:0:0:187    ////__DACE:0:0:187
                                            ////__DACE:0:0:187                        ////__DACE:0:0:187
                                            __map_fusion_gtir_tmp_104_1_0 = __tlet_result;                                    ////__DACE:0:0:187    ////__DACE:0:0:187
                                        }                                             ////__DACE:0:0:187
                                        if_stmt_5_0_0_186(__map_fusion_gtir_tmp_104_1_0, &gt_conn_E2C[0], &gtir_tmp_101[0], gtir_tmp_54_1_0, gtir_tmp_76_1_0, &gtir_tmp_95[0], &perturbed_theta_v_at_cells_on_model_levels[0], &reference_theta_at_edges_on_model_levels[0], gtir_tmp_137_1_0, __gt_conn_E2C_neighbor_stride, __perturbed_theta_v_at_cells_on_model_levels_K_stride, __reference_theta_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:186
                                        {                                             ////__DACE:0:0:99
                                            double __tlet_arg0 = gtir_tmp_223_1;                                              ////__DACE:0:0:346,99    ////__DACE:0:0:99
                                            double __tlet_arg1 = gtir_tmp_137_1_0;                                            ////__DACE:0:0:185,99    ////__DACE:0:0:99
                                            double __tlet_result;                                                             ////__DACE:0:0:99    ////__DACE:0:0:99
                                            ////__DACE:0:0:99                         ////__DACE:0:0:99
                                            ///////////////////                                                               ////__DACE:0:0:99    ////__DACE:0:0:99
                                            // Tasklet code (tlet_93_multiplies_0)                                            ////__DACE:0:0:99    ////__DACE:0:0:99
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:99    ////__DACE:0:0:99
                                            ///////////////////                                                               ////__DACE:0:0:99    ////__DACE:0:0:99
                                            ////__DACE:0:0:99                         ////__DACE:0:0:99
                                            __map_fusion_gtir_tmp_227_0 = __tlet_result;                                      ////__DACE:0:0:99    ////__DACE:0:0:99
                                        }                                             ////__DACE:0:0:99
                                        {                                             ////__DACE:0:0:414
                                            double _cpy_in = gtir_tmp_137_1_0;                                                ////__DACE:0:0:185,414    ////__DACE:0:0:414
                                            double _cpy_out;                                                                  ////__DACE:0:0:414    ////__DACE:0:0:414
                                            ////__DACE:0:0:414                        ////__DACE:0:0:414
                                            ///////////////////                                                               ////__DACE:0:0:414    ////__DACE:0:0:414
                                            // Tasklet code (copy_gtir_tmp_137_1_0_to_theta_v_at_edges_on_model_levels)       ////__DACE:0:0:414    ////__DACE:0:0:414
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:414    ////__DACE:0:0:414
                                            ///////////////////                                                               ////__DACE:0:0:414    ////__DACE:0:0:414
                                            ////__DACE:0:0:414                        ////__DACE:0:0:414
                                            theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:414    ////__DACE:0:0:414
                                        }                                             ////__DACE:0:0:414
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
                                        {                                             ////__DACE:0:0:58
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_141;                                   ////__DACE:0:0:81,58    ////__DACE:0:0:58
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_143;                                   ////__DACE:0:0:80,58    ////__DACE:0:0:58
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
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_145;                                   ////__DACE:0:0:79,57    ////__DACE:0:0:57
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
                                        {                                             ////__DACE:0:0:92
                                            double __tlet_arg1 = gtir_tmp_166_0;                                              ////__DACE:0:0:332,92    ////__DACE:0:0:92
                                            double __tlet_arg0 = pg_exdist[((__pg_exdist_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:61,92    ////__DACE:0:0:92
                                            bool __tlet_result;                                                               ////__DACE:0:0:92    ////__DACE:0:0:92
                                            ////__DACE:0:0:92                         ////__DACE:0:0:92
                                            ///////////////////                                                               ////__DACE:0:0:92    ////__DACE:0:0:92
                                            // Tasklet code (tlet_65_not_eq_0)                                                ////__DACE:0:0:92    ////__DACE:0:0:92
                                            __tlet_result = (__tlet_arg0 != __tlet_arg1);                                     ////__DACE:0:0:92    ////__DACE:0:0:92
                                            ///////////////////                                                               ////__DACE:0:0:92    ////__DACE:0:0:92
                                            ////__DACE:0:0:92                         ////__DACE:0:0:92
                                            __map_fusion_gtir_tmp_168_0 = __tlet_result;                                      ////__DACE:0:0:92    ////__DACE:0:0:92
                                        }                                             ////__DACE:0:0:92
                                        if_stmt_6_0_0_91(__map_fusion_gtir_tmp_139_split_0, __map_fusion_gtir_tmp_168_0, &hydrostatic_correction_on_lowest_level[0], &pg_exdist[0], gtir_tmp_173_0, __pg_exdist_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:91
                                        {                                             ////__DACE:0:0:98
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_227_0;                                 ////__DACE:0:0:105,98    ////__DACE:0:0:98
                                            double __tlet_arg1 = gtir_tmp_173_0;                                              ////__DACE:0:0:90,98    ////__DACE:0:0:98
                                            double __tlet_result;                                                             ////__DACE:0:0:98    ////__DACE:0:0:98
                                            ////__DACE:0:0:98                         ////__DACE:0:0:98
                                            ///////////////////                                                               ////__DACE:0:0:98    ////__DACE:0:0:98
                                            // Tasklet code (tlet_94_multiplies_0)                                            ////__DACE:0:0:98    ////__DACE:0:0:98
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:98    ////__DACE:0:0:98
                                            ///////////////////                                                               ////__DACE:0:0:98    ////__DACE:0:0:98
                                            ////__DACE:0:0:98                         ////__DACE:0:0:98
                                            __map_fusion_gtir_tmp_229_0 = __tlet_result;                                      ////__DACE:0:0:98    ////__DACE:0:0:98
                                        }                                             ////__DACE:0:0:98
                                        {                                             ////__DACE:0:0:97
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_229_0;                                 ////__DACE:0:0:104,97    ////__DACE:0:0:97
                                            double __tlet_arg0 = predictor_normal_wind_advective_tendency[((__predictor_normal_wind_advective_tendency_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:66,97    ////__DACE:0:0:97
                                            double __tlet_result;                                                             ////__DACE:0:0:97    ////__DACE:0:0:97
                                            ////__DACE:0:0:97                         ////__DACE:0:0:97
                                            ///////////////////                                                               ////__DACE:0:0:97    ////__DACE:0:0:97
                                            // Tasklet code (tlet_95_minus_0)                                                 ////__DACE:0:0:97    ////__DACE:0:0:97
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:97    ////__DACE:0:0:97
                                            ///////////////////                                                               ////__DACE:0:0:97    ////__DACE:0:0:97
                                            ////__DACE:0:0:97                         ////__DACE:0:0:97
                                            __map_fusion_gtir_tmp_231_0 = __tlet_result;                                      ////__DACE:0:0:97    ////__DACE:0:0:97
                                        }                                             ////__DACE:0:0:97
                                        {                                             ////__DACE:0:0:96
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_231_0;                                 ////__DACE:0:0:103,96    ////__DACE:0:0:96
                                            double __tlet_arg1 = normal_wind_tendency_due_to_slow_physics_process[((__normal_wind_tendency_due_to_slow_physics_process_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:65,96    ////__DACE:0:0:96
                                            double __tlet_result;                                                             ////__DACE:0:0:96    ////__DACE:0:0:96
                                            ////__DACE:0:0:96                         ////__DACE:0:0:96
                                            ///////////////////                                                               ////__DACE:0:0:96    ////__DACE:0:0:96
                                            // Tasklet code (tlet_96_plus_0)                                                  ////__DACE:0:0:96    ////__DACE:0:0:96
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:96    ////__DACE:0:0:96
                                            ///////////////////                                                               ////__DACE:0:0:96    ////__DACE:0:0:96
                                            ////__DACE:0:0:96                         ////__DACE:0:0:96
                                            __map_fusion_gtir_tmp_233_0 = __tlet_result;                                      ////__DACE:0:0:96    ////__DACE:0:0:96
                                        }                                             ////__DACE:0:0:96
                                        {                                             ////__DACE:0:0:95
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_233_0;                                 ////__DACE:0:0:102,95    ////__DACE:0:0:95
                                            double __tlet_arg0 = __dtime_0_0;                                                 ////__DACE:0:0:348,95    ////__DACE:0:0:95
                                            double __tlet_result;                                                             ////__DACE:0:0:95    ////__DACE:0:0:95
                                            ////__DACE:0:0:95                         ////__DACE:0:0:95
                                            ///////////////////                                                               ////__DACE:0:0:95    ////__DACE:0:0:95
                                            // Tasklet code (tlet_97_multiplies_0)                                            ////__DACE:0:0:95    ////__DACE:0:0:95
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:95    ////__DACE:0:0:95
                                            ///////////////////                                                               ////__DACE:0:0:95    ////__DACE:0:0:95
                                            ////__DACE:0:0:95                         ////__DACE:0:0:95
                                            __map_fusion_gtir_tmp_235_0 = __tlet_result;                                      ////__DACE:0:0:95    ////__DACE:0:0:95
                                        }                                             ////__DACE:0:0:95
                                        {                                             ////__DACE:0:0:94
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_235_0;                                 ////__DACE:0:0:101,94    ////__DACE:0:0:94
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,94    ////__DACE:0:0:94
                                            double __tlet_result;                                                             ////__DACE:0:0:94    ////__DACE:0:0:94
                                            ////__DACE:0:0:94                         ////__DACE:0:0:94
                                            ///////////////////                                                               ////__DACE:0:0:94    ////__DACE:0:0:94
                                            // Tasklet code (tlet_98_plus_0)                                                  ////__DACE:0:0:94    ////__DACE:0:0:94
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:94    ////__DACE:0:0:94
                                            ///////////////////                                                               ////__DACE:0:0:94    ////__DACE:0:0:94
                                            ////__DACE:0:0:94                         ////__DACE:0:0:94
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:94    ////__DACE:0:0:94
                                        }                                             ////__DACE:0:0:94
                                        {                                             ////__DACE:0:0:413
                                            double _cpy_in = gtir_tmp_173_0;                                                  ////__DACE:0:0:90,413    ////__DACE:0:0:413
                                            double _cpy_out;                                                                  ////__DACE:0:0:413    ////__DACE:0:0:413
                                            ////__DACE:0:0:413                        ////__DACE:0:0:413
                                            ///////////////////                                                               ////__DACE:0:0:413    ////__DACE:0:0:413
                                            // Tasklet code (copy_gtir_tmp_173_0_to_horizontal_pressure_gradient)             ////__DACE:0:0:413    ////__DACE:0:0:413
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:413    ////__DACE:0:0:413
                                            ///////////////////                                                               ////__DACE:0:0:413    ////__DACE:0:0:413
                                            ////__DACE:0:0:413                        ////__DACE:0:0:413
                                            horizontal_pressure_gradient[((__horizontal_pressure_gradient_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:413    ////__DACE:0:0:413
                                        }                                             ////__DACE:0:0:413
                                    }                                                 ////__DACE:0:0:290
                                }                                                     ////__DACE:0:0:290
                            }                                                         ////__DACE:0:0:93
                        }                                                             ////__DACE:0:0:93
                    }                                                                 ////__DACE:0:0:93
                }                                                                     ////__DACE:0:0:93
            }                                                                         ////__DACE:0:0:93
        }                                                                             ////__DACE:0:0:425
    }                                                                                 ////__DACE:0:0:425
}                                                                                 ////__DACE:0:0:425

                                                                                  ////__DACE:0:0:424
DACE_EXPORTED void __dace_runkernel_map_100_fieldop_1_0_0_0_424(theta_shared_probe_shared_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pg_exdist, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:424
void __dace_runkernel_map_100_fieldop_1_0_0_0_424(theta_shared_probe_shared_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pg_exdist, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:424
{                                                                                 ////__DACE:0:0:424
                                                                                  ////__DACE:0:0:424
    void  *map_100_fieldop_1_0_0_0_424_args[] = { (void *)&current_vn, (void *)&dual_normal_cell_x, (void *)&dual_normal_cell_y, (void *)&gt_conn_E2C, (void *)&gtir_tmp_101, (void *)&gtir_tmp_83, (void *)&gtir_tmp_89, (void *)&gtir_tmp_95, (void *)&horizontal_pressure_gradient, (void *)&hydrostatic_correction_on_lowest_level, (void *)&inv_dual_edge_length, (void *)&next_vn, (void *)&normal_wind_tendency_due_to_slow_physics_process, (void *)&perturbed_rho_at_cells_on_model_levels, (void *)&perturbed_theta_v_at_cells_on_model_levels, (void *)&pg_exdist, (void *)&pos_on_tplane_e_x, (void *)&pos_on_tplane_e_y, (void *)&predictor_normal_wind_advective_tendency, (void *)&primal_normal_cell_x, (void *)&primal_normal_cell_y, (void *)&reference_rho_at_edges_on_model_levels, (void *)&reference_theta_at_edges_on_model_levels, (void *)&rho_at_edges_on_model_levels, (void *)&tangential_wind, (void *)&temporal_extrapolation_of_perturbed_exner, (void *)&theta_v_at_edges_on_model_levels, (void *)&__current_vn_K_stride, (void *)&__dual_normal_cell_x_E2C_stride, (void *)&__dual_normal_cell_y_E2C_stride, (void *)&__gt_conn_E2C_neighbor_stride, (void *)&__horizontal_pressure_gradient_K_stride, (void *)&__next_vn_K_stride, (void *)&__normal_wind_tendency_due_to_slow_physics_process_K_stride, (void *)&__perturbed_rho_at_cells_on_model_levels_K_stride, (void *)&__perturbed_theta_v_at_cells_on_model_levels_K_stride, (void *)&__pg_exdist_K_stride, (void *)&__pos_on_tplane_e_x_E2C_stride, (void *)&__pos_on_tplane_e_y_E2C_stride, (void *)&__predictor_normal_wind_advective_tendency_K_stride, (void *)&__primal_normal_cell_x_E2C_stride, (void *)&__primal_normal_cell_y_E2C_stride, (void *)&__reference_rho_at_edges_on_model_levels_K_stride, (void *)&__reference_theta_at_edges_on_model_levels_K_stride, (void *)&__rho_at_edges_on_model_levels_K_stride, (void *)&__tangential_wind_K_stride, (void *)&__temporal_extrapolation_of_perturbed_exner_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:424
    gpuError_t __err = hipLaunchKernel((void*)map_100_fieldop_1_0_0_0_424, dim3(233, 7, 1), dim3(256, 1, 1), map_100_fieldop_1_0_0_0_424_args, 0, nullptr);    ////__DACE:0:0:424
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_100_fieldop_1_0_0_0_424", 233, 7, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_100_fieldop_1_1_0_0_426(const double * __restrict__ c_lin_e, const double * __restrict__ current_vn, const double * __restrict__ ddxn_z_full, const double * __restrict__ ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pg_exdist, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, double * __restrict__ theta_v_at_edges_on_model_levels, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:426
    {                                                                                 ////__DACE:0:0:426
        {                                                                             ////__DACE:0:0:426
            int b_i_Edge_gtx_horizontal = ((256 * blockIdx.x) + 7701);                ////__DACE:0:0:426
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:426
            {                                                                         ////__DACE:0:0:227
                {                                                                     ////__DACE:0:0:227
                    {                                                                 ////__DACE:0:0:227
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:227
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:227
                        double gtir_tmp_14_1_1;                                       ////__DACE:0:0:207
                        double gtir_tmp_12_1_1;                                       ////__DACE:0:0:208
                        double gtir_tmp_26_1_1;                                       ////__DACE:0:0:213
                        double gtir_tmp_24_1_1;                                       ////__DACE:0:0:214
                        double gtir_tmp_70_1_1;                                       ////__DACE:0:0:218
                        double gtir_tmp_66_1_1;                                       ////__DACE:0:0:219
                        double gtir_tmp_60_1_1;                                       ////__DACE:0:0:220
                        double gtir_tmp_56_1_1;                                       ////__DACE:0:0:221
                        double gtir_tmp_48_1_1;                                       ////__DACE:0:0:223
                        double gtir_tmp_44_1_1;                                       ////__DACE:0:0:224
                        double gtir_tmp_38_1_1;                                       ////__DACE:0:0:225
                        double gtir_tmp_34_1_1;                                       ////__DACE:0:0:226
                        bool gtir_tmp_7_0;                                            ////__DACE:0:0:296
                        bool gtir_tmp_6_0;                                            ////__DACE:0:0:302
                        double gtir_tmp_3_2;                                          ////__DACE:0:0:312
                        double __p_dthalf_1;                                          ////__DACE:0:0:316
                        double lambda_4___p_dthalf_0;                                 ////__DACE:0:0:320
                        double gtir_tmp_102_0;                                        ////__DACE:0:0:326
                        double gtir_tmp_166_1;                                        ////__DACE:0:0:334
                        double gtir_tmp_175_0;                                        ////__DACE:0:0:338
                        double gtir_tmp_223_0;                                        ////__DACE:0:0:344
                        double __dtime_0_1;                                           ////__DACE:0:0:350
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(67095, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:227
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(23, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:227
                                {                                                     ////__DACE:0:0:295
                                    bool __tlet_out;                                                                  ////__DACE:0:0:295    ////__DACE:0:0:295
                                    ////__DACE:0:0:295                                ////__DACE:0:0:295
                                    ///////////////////                                                               ////__DACE:0:0:295    ////__DACE:0:0:295
                                    // Tasklet code (tlet_5_get_value__clone_0)                                       ////__DACE:0:0:295    ////__DACE:0:0:295
                                    __tlet_out = false;                                                               ////__DACE:0:0:295    ////__DACE:0:0:295
                                    ///////////////////                                                               ////__DACE:0:0:295    ////__DACE:0:0:295
                                    ////__DACE:0:0:295                                ////__DACE:0:0:295
                                    gtir_tmp_7_0 = __tlet_out;                                                        ////__DACE:0:0:295    ////__DACE:0:0:295
                                }                                                     ////__DACE:0:0:295
                                {                                                     ////__DACE:0:0:301
                                    bool __tlet_out;                                                                  ////__DACE:0:0:301    ////__DACE:0:0:301
                                    ////__DACE:0:0:301                                ////__DACE:0:0:301
                                    ///////////////////                                                               ////__DACE:0:0:301    ////__DACE:0:0:301
                                    // Tasklet code (tlet_4_get_value__clone_0)                                       ////__DACE:0:0:301    ////__DACE:0:0:301
                                    __tlet_out = true;                                                                ////__DACE:0:0:301    ////__DACE:0:0:301
                                    ///////////////////                                                               ////__DACE:0:0:301    ////__DACE:0:0:301
                                    ////__DACE:0:0:301                                ////__DACE:0:0:301
                                    gtir_tmp_6_0 = __tlet_out;                                                        ////__DACE:0:0:301    ////__DACE:0:0:301
                                }                                                     ////__DACE:0:0:301
                                {                                                     ////__DACE:0:0:311
                                    double __tlet_out;                                                                ////__DACE:0:0:311    ////__DACE:0:0:311
                                    ////__DACE:0:0:311                                ////__DACE:0:0:311
                                    ///////////////////                                                               ////__DACE:0:0:311    ////__DACE:0:0:311
                                    // Tasklet code (tlet_2_get_value__clone_2)                                       ////__DACE:0:0:311    ////__DACE:0:0:311
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:311    ////__DACE:0:0:311
                                    ///////////////////                                                               ////__DACE:0:0:311    ////__DACE:0:0:311
                                    ////__DACE:0:0:311                                ////__DACE:0:0:311
                                    gtir_tmp_3_2 = __tlet_out;                                                        ////__DACE:0:0:311    ////__DACE:0:0:311
                                }                                                     ////__DACE:0:0:311
                                {                                                     ////__DACE:0:0:315
                                    double __tlet_out;                                                                ////__DACE:0:0:315    ////__DACE:0:0:315
                                    ////__DACE:0:0:315                                ////__DACE:0:0:315
                                    ///////////////////                                                               ////__DACE:0:0:315    ////__DACE:0:0:315
                                    // Tasklet code (tlet_6_get_value__clone_1)                                       ////__DACE:0:0:315    ////__DACE:0:0:315
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:315    ////__DACE:0:0:315
                                    ///////////////////                                                               ////__DACE:0:0:315    ////__DACE:0:0:315
                                    ////__DACE:0:0:315                                ////__DACE:0:0:315
                                    __p_dthalf_1 = __tlet_out;                                                        ////__DACE:0:0:315    ////__DACE:0:0:315
                                }                                                     ////__DACE:0:0:315
                                {                                                     ////__DACE:0:0:319
                                    double __tlet_out;                                                                ////__DACE:0:0:319    ////__DACE:0:0:319
                                    ////__DACE:0:0:319                                ////__DACE:0:0:319
                                    ///////////////////                                                               ////__DACE:0:0:319    ////__DACE:0:0:319
                                    // Tasklet code (tlet_10_get_value__clone_0)                                      ////__DACE:0:0:319    ////__DACE:0:0:319
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:319    ////__DACE:0:0:319
                                    ///////////////////                                                               ////__DACE:0:0:319    ////__DACE:0:0:319
                                    ////__DACE:0:0:319                                ////__DACE:0:0:319
                                    lambda_4___p_dthalf_0 = __tlet_out;                                               ////__DACE:0:0:319    ////__DACE:0:0:319
                                }                                                     ////__DACE:0:0:319
                                {                                                     ////__DACE:0:0:325
                                    double __tlet_out;                                                                ////__DACE:0:0:325    ////__DACE:0:0:325
                                    ////__DACE:0:0:325                                ////__DACE:0:0:325
                                    ///////////////////                                                               ////__DACE:0:0:325    ////__DACE:0:0:325
                                    // Tasklet code (tlet_34_get_value__clone_0)                                      ////__DACE:0:0:325    ////__DACE:0:0:325
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:325    ////__DACE:0:0:325
                                    ///////////////////                                                               ////__DACE:0:0:325    ////__DACE:0:0:325
                                    ////__DACE:0:0:325                                ////__DACE:0:0:325
                                    gtir_tmp_102_0 = __tlet_out;                                                      ////__DACE:0:0:325    ////__DACE:0:0:325
                                }                                                     ////__DACE:0:0:325
                                {                                                     ////__DACE:0:0:333
                                    double __tlet_out;                                                                ////__DACE:0:0:333    ////__DACE:0:0:333
                                    ////__DACE:0:0:333                                ////__DACE:0:0:333
                                    ///////////////////                                                               ////__DACE:0:0:333    ////__DACE:0:0:333
                                    // Tasklet code (tlet_64_get_value__clone_1)                                      ////__DACE:0:0:333    ////__DACE:0:0:333
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:333    ////__DACE:0:0:333
                                    ///////////////////                                                               ////__DACE:0:0:333    ////__DACE:0:0:333
                                    ////__DACE:0:0:333                                ////__DACE:0:0:333
                                    gtir_tmp_166_1 = __tlet_out;                                                      ////__DACE:0:0:333    ////__DACE:0:0:333
                                }                                                     ////__DACE:0:0:333
                                {                                                     ////__DACE:0:0:337
                                    double __tlet_out;                                                                ////__DACE:0:0:337    ////__DACE:0:0:337
                                    ////__DACE:0:0:337                                ////__DACE:0:0:337
                                    ///////////////////                                                               ////__DACE:0:0:337    ////__DACE:0:0:337
                                    // Tasklet code (tlet_68_get_value__clone_0)                                      ////__DACE:0:0:337    ////__DACE:0:0:337
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:337    ////__DACE:0:0:337
                                    ///////////////////                                                               ////__DACE:0:0:337    ////__DACE:0:0:337
                                    ////__DACE:0:0:337                                ////__DACE:0:0:337
                                    gtir_tmp_175_0 = __tlet_out;                                                      ////__DACE:0:0:337    ////__DACE:0:0:337
                                }                                                     ////__DACE:0:0:337
                                {                                                     ////__DACE:0:0:343
                                    double __tlet_out;                                                                ////__DACE:0:0:343    ////__DACE:0:0:343
                                    ////__DACE:0:0:343                                ////__DACE:0:0:343
                                    ///////////////////                                                               ////__DACE:0:0:343    ////__DACE:0:0:343
                                    // Tasklet code (tlet_92_get_value__clone_0)                                      ////__DACE:0:0:343    ////__DACE:0:0:343
                                    __tlet_out = 1004.64;                                                             ////__DACE:0:0:343    ////__DACE:0:0:343
                                    ///////////////////                                                               ////__DACE:0:0:343    ////__DACE:0:0:343
                                    ////__DACE:0:0:343                                ////__DACE:0:0:343
                                    gtir_tmp_223_0 = __tlet_out;                                                      ////__DACE:0:0:343    ////__DACE:0:0:343
                                }                                                     ////__DACE:0:0:343
                                {                                                     ////__DACE:0:0:349
                                    double __tlet_out;                                                                ////__DACE:0:0:349    ////__DACE:0:0:349
                                    ////__DACE:0:0:349                                ////__DACE:0:0:349
                                    ///////////////////                                                               ////__DACE:0:0:349    ////__DACE:0:0:349
                                    // Tasklet code (tlet_91_get_value__clone_1)                                      ////__DACE:0:0:349    ////__DACE:0:0:349
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:349    ////__DACE:0:0:349
                                    ///////////////////                                                               ////__DACE:0:0:349    ////__DACE:0:0:349
                                    ////__DACE:0:0:349                                ////__DACE:0:0:349
                                    __dtime_0_1 = __tlet_out;                                                         ////__DACE:0:0:349    ////__DACE:0:0:349
                                }                                                     ////__DACE:0:0:349
                                {                                                     ////__DACE:0:0:392
                                    double _cpy_in = dual_normal_cell_x[i_Edge_gtx_horizontal];                       ////__DACE:0:0:9,392    ////__DACE:0:0:392
                                    double _cpy_out;                                                                  ////__DACE:0:0:392    ////__DACE:0:0:392
                                    ////__DACE:0:0:392                                ////__DACE:0:0:392
                                    ///////////////////                                                               ////__DACE:0:0:392    ////__DACE:0:0:392
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_38_1_1)                      ////__DACE:0:0:392    ////__DACE:0:0:392
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:392    ////__DACE:0:0:392
                                    ///////////////////                                                               ////__DACE:0:0:392    ////__DACE:0:0:392
                                    ////__DACE:0:0:392                                ////__DACE:0:0:392
                                    gtir_tmp_38_1_1 = _cpy_out;                                                       ////__DACE:0:0:392    ////__DACE:0:0:392
                                }                                                     ////__DACE:0:0:392
                                {                                                     ////__DACE:0:0:393
                                    double _cpy_in = dual_normal_cell_x[(__dual_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:9,393    ////__DACE:0:0:393
                                    double _cpy_out;                                                                  ////__DACE:0:0:393    ////__DACE:0:0:393
                                    ////__DACE:0:0:393                                ////__DACE:0:0:393
                                    ///////////////////                                                               ////__DACE:0:0:393    ////__DACE:0:0:393
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_48_1_1)                      ////__DACE:0:0:393    ////__DACE:0:0:393
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:393    ////__DACE:0:0:393
                                    ///////////////////                                                               ////__DACE:0:0:393    ////__DACE:0:0:393
                                    ////__DACE:0:0:393                                ////__DACE:0:0:393
                                    gtir_tmp_48_1_1 = _cpy_out;                                                       ////__DACE:0:0:393    ////__DACE:0:0:393
                                }                                                     ////__DACE:0:0:393
                                {                                                     ////__DACE:0:0:394
                                    double _cpy_in = dual_normal_cell_y[i_Edge_gtx_horizontal];                       ////__DACE:0:0:7,394    ////__DACE:0:0:394
                                    double _cpy_out;                                                                  ////__DACE:0:0:394    ////__DACE:0:0:394
                                    ////__DACE:0:0:394                                ////__DACE:0:0:394
                                    ///////////////////                                                               ////__DACE:0:0:394    ////__DACE:0:0:394
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_60_1_1)                      ////__DACE:0:0:394    ////__DACE:0:0:394
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:394    ////__DACE:0:0:394
                                    ///////////////////                                                               ////__DACE:0:0:394    ////__DACE:0:0:394
                                    ////__DACE:0:0:394                                ////__DACE:0:0:394
                                    gtir_tmp_60_1_1 = _cpy_out;                                                       ////__DACE:0:0:394    ////__DACE:0:0:394
                                }                                                     ////__DACE:0:0:394
                                {                                                     ////__DACE:0:0:395
                                    double _cpy_in = dual_normal_cell_y[(__dual_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:7,395    ////__DACE:0:0:395
                                    double _cpy_out;                                                                  ////__DACE:0:0:395    ////__DACE:0:0:395
                                    ////__DACE:0:0:395                                ////__DACE:0:0:395
                                    ///////////////////                                                               ////__DACE:0:0:395    ////__DACE:0:0:395
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_70_1_1)                      ////__DACE:0:0:395    ////__DACE:0:0:395
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:395    ////__DACE:0:0:395
                                    ///////////////////                                                               ////__DACE:0:0:395    ////__DACE:0:0:395
                                    ////__DACE:0:0:395                                ////__DACE:0:0:395
                                    gtir_tmp_70_1_1 = _cpy_out;                                                       ////__DACE:0:0:395    ////__DACE:0:0:395
                                }                                                     ////__DACE:0:0:395
                                {                                                     ////__DACE:0:0:396
                                    double _cpy_in = pos_on_tplane_e_x[i_Edge_gtx_horizontal];                        ////__DACE:0:0:4,396    ////__DACE:0:0:396
                                    double _cpy_out;                                                                  ////__DACE:0:0:396    ////__DACE:0:0:396
                                    ////__DACE:0:0:396                                ////__DACE:0:0:396
                                    ///////////////////                                                               ////__DACE:0:0:396    ////__DACE:0:0:396
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_12_1_1)                       ////__DACE:0:0:396    ////__DACE:0:0:396
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:396    ////__DACE:0:0:396
                                    ///////////////////                                                               ////__DACE:0:0:396    ////__DACE:0:0:396
                                    ////__DACE:0:0:396                                ////__DACE:0:0:396
                                    gtir_tmp_12_1_1 = _cpy_out;                                                       ////__DACE:0:0:396    ////__DACE:0:0:396
                                }                                                     ////__DACE:0:0:396
                                {                                                     ////__DACE:0:0:397
                                    double _cpy_in = pos_on_tplane_e_x[(__pos_on_tplane_e_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:4,397    ////__DACE:0:0:397
                                    double _cpy_out;                                                                  ////__DACE:0:0:397    ////__DACE:0:0:397
                                    ////__DACE:0:0:397                                ////__DACE:0:0:397
                                    ///////////////////                                                               ////__DACE:0:0:397    ////__DACE:0:0:397
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_14_1_1)                       ////__DACE:0:0:397    ////__DACE:0:0:397
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:397    ////__DACE:0:0:397
                                    ///////////////////                                                               ////__DACE:0:0:397    ////__DACE:0:0:397
                                    ////__DACE:0:0:397                                ////__DACE:0:0:397
                                    gtir_tmp_14_1_1 = _cpy_out;                                                       ////__DACE:0:0:397    ////__DACE:0:0:397
                                }                                                     ////__DACE:0:0:397
                                {                                                     ////__DACE:0:0:398
                                    double _cpy_in = pos_on_tplane_e_y[i_Edge_gtx_horizontal];                        ////__DACE:0:0:5,398    ////__DACE:0:0:398
                                    double _cpy_out;                                                                  ////__DACE:0:0:398    ////__DACE:0:0:398
                                    ////__DACE:0:0:398                                ////__DACE:0:0:398
                                    ///////////////////                                                               ////__DACE:0:0:398    ////__DACE:0:0:398
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_24_1_1)                       ////__DACE:0:0:398    ////__DACE:0:0:398
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:398    ////__DACE:0:0:398
                                    ///////////////////                                                               ////__DACE:0:0:398    ////__DACE:0:0:398
                                    ////__DACE:0:0:398                                ////__DACE:0:0:398
                                    gtir_tmp_24_1_1 = _cpy_out;                                                       ////__DACE:0:0:398    ////__DACE:0:0:398
                                }                                                     ////__DACE:0:0:398
                                {                                                     ////__DACE:0:0:399
                                    double _cpy_in = pos_on_tplane_e_y[(__pos_on_tplane_e_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:5,399    ////__DACE:0:0:399
                                    double _cpy_out;                                                                  ////__DACE:0:0:399    ////__DACE:0:0:399
                                    ////__DACE:0:0:399                                ////__DACE:0:0:399
                                    ///////////////////                                                               ////__DACE:0:0:399    ////__DACE:0:0:399
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_26_1_1)                       ////__DACE:0:0:399    ////__DACE:0:0:399
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:399    ////__DACE:0:0:399
                                    ///////////////////                                                               ////__DACE:0:0:399    ////__DACE:0:0:399
                                    ////__DACE:0:0:399                                ////__DACE:0:0:399
                                    gtir_tmp_26_1_1 = _cpy_out;                                                       ////__DACE:0:0:399    ////__DACE:0:0:399
                                }                                                     ////__DACE:0:0:399
                                {                                                     ////__DACE:0:0:400
                                    double _cpy_in = primal_normal_cell_x[i_Edge_gtx_horizontal];                     ////__DACE:0:0:10,400    ////__DACE:0:0:400
                                    double _cpy_out;                                                                  ////__DACE:0:0:400    ////__DACE:0:0:400
                                    ////__DACE:0:0:400                                ////__DACE:0:0:400
                                    ///////////////////                                                               ////__DACE:0:0:400    ////__DACE:0:0:400
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_34_1_1)                    ////__DACE:0:0:400    ////__DACE:0:0:400
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:400    ////__DACE:0:0:400
                                    ///////////////////                                                               ////__DACE:0:0:400    ////__DACE:0:0:400
                                    ////__DACE:0:0:400                                ////__DACE:0:0:400
                                    gtir_tmp_34_1_1 = _cpy_out;                                                       ////__DACE:0:0:400    ////__DACE:0:0:400
                                }                                                     ////__DACE:0:0:400
                                {                                                     ////__DACE:0:0:401
                                    double _cpy_in = primal_normal_cell_x[(__primal_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:10,401    ////__DACE:0:0:401
                                    double _cpy_out;                                                                  ////__DACE:0:0:401    ////__DACE:0:0:401
                                    ////__DACE:0:0:401                                ////__DACE:0:0:401
                                    ///////////////////                                                               ////__DACE:0:0:401    ////__DACE:0:0:401
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_44_1_1)                    ////__DACE:0:0:401    ////__DACE:0:0:401
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:401    ////__DACE:0:0:401
                                    ///////////////////                                                               ////__DACE:0:0:401    ////__DACE:0:0:401
                                    ////__DACE:0:0:401                                ////__DACE:0:0:401
                                    gtir_tmp_44_1_1 = _cpy_out;                                                       ////__DACE:0:0:401    ////__DACE:0:0:401
                                }                                                     ////__DACE:0:0:401
                                {                                                     ////__DACE:0:0:402
                                    double _cpy_in = primal_normal_cell_y[i_Edge_gtx_horizontal];                     ////__DACE:0:0:8,402    ////__DACE:0:0:402
                                    double _cpy_out;                                                                  ////__DACE:0:0:402    ////__DACE:0:0:402
                                    ////__DACE:0:0:402                                ////__DACE:0:0:402
                                    ///////////////////                                                               ////__DACE:0:0:402    ////__DACE:0:0:402
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_56_1_1)                    ////__DACE:0:0:402    ////__DACE:0:0:402
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:402    ////__DACE:0:0:402
                                    ///////////////////                                                               ////__DACE:0:0:402    ////__DACE:0:0:402
                                    ////__DACE:0:0:402                                ////__DACE:0:0:402
                                    gtir_tmp_56_1_1 = _cpy_out;                                                       ////__DACE:0:0:402    ////__DACE:0:0:402
                                }                                                     ////__DACE:0:0:402
                                {                                                     ////__DACE:0:0:403
                                    double _cpy_in = primal_normal_cell_y[(__primal_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:8,403    ////__DACE:0:0:403
                                    double _cpy_out;                                                                  ////__DACE:0:0:403    ////__DACE:0:0:403
                                    ////__DACE:0:0:403                                ////__DACE:0:0:403
                                    ///////////////////                                                               ////__DACE:0:0:403    ////__DACE:0:0:403
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_66_1_1)                    ////__DACE:0:0:403    ////__DACE:0:0:403
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:403    ////__DACE:0:0:403
                                    ///////////////////                                                               ////__DACE:0:0:403    ////__DACE:0:0:403
                                    ////__DACE:0:0:403                                ////__DACE:0:0:403
                                    gtir_tmp_66_1_1 = _cpy_out;                                                       ////__DACE:0:0:403    ////__DACE:0:0:403
                                }                                                     ////__DACE:0:0:403
                                {                                                     ////__DACE:0:0:291
                                    #pragma unroll 4                                  ////__DACE:0:0:291
                                    for (auto i_K_gtx_vertical = ((4 * __gtx_coarse_i_K_gtx_vertical) + 27); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 31)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:291
                                        bool gtir_tmp_8_1_1;                          ////__DACE:0:0:200
                                        double gtir_tmp_16_1_1;                       ////__DACE:0:0:205
                                        double gtir_tmp_28_1_1;                       ////__DACE:0:0:212
                                        double gtir_tmp_76_1_1;                       ////__DACE:0:0:216
                                        double gtir_tmp_54_1_1;                       ////__DACE:0:0:222
                                        double gtir_tmp_137_1_1;                      ////__DACE:0:0:228
                                        double gtir_tmp_210_1_1;                      ////__DACE:0:0:231
                                        bool __map_fusion_gtir_tmp_5_1_1;             ////__DACE:0:0:234
                                        double __map_fusion_gtir_tmp_21_1_1_1;        ////__DACE:0:0:235
                                        double __map_fusion_gtir_tmp_19_1_1;          ////__DACE:0:0:236
                                        double __map_fusion_gtir_tmp_11_1_1;          ////__DACE:0:0:237
                                        double __map_fusion_gtir_tmp_33_1_1_1;        ////__DACE:0:0:238
                                        double __map_fusion_gtir_tmp_31_1_1;          ////__DACE:0:0:239
                                        double __map_fusion_gtir_tmp_23_1_1;          ////__DACE:0:0:240
                                        bool __map_fusion_gtir_tmp_104_1_1;           ////__DACE:0:0:241
                                        bool __map_fusion_gtir_tmp_177_1_1;           ////__DACE:0:0:242
                                        double gtir_tmp_160_0_0;                      ////__DACE:0:0:245
                                        double __map_fusion_gtir_tmp_163_0_0;         ////__DACE:0:0:256
                                        double __map_fusion_gtir_tmp_159_0_0[2]  DACE_ALIGN(64);    ////__DACE:0:0:257
                                        double __map_fusion_gtir_tmp_157_0_0[2]  DACE_ALIGN(64);    ////__DACE:0:0:258
                                        double __map_fusion_gtir_tmp_155_0_0;         ////__DACE:0:0:259
                                        double __map_fusion_gtir_tmp_153_0_0;         ////__DACE:0:0:260
                                        double __map_fusion_gtir_tmp_151_0_0;         ////__DACE:0:0:261
                                        double __map_fusion_gtir_tmp_149_0_0;         ////__DACE:0:0:262
                                        double gtir_tmp_173_1_0;                      ////__DACE:0:0:265
                                        bool __map_fusion_gtir_tmp_168_1_0;           ////__DACE:0:0:275
                                        double __map_fusion_gtir_tmp_235_1_0;         ////__DACE:0:0:276
                                        double __map_fusion_gtir_tmp_233_1_0;         ////__DACE:0:0:277
                                        double __map_fusion_gtir_tmp_231_1_0;         ////__DACE:0:0:278
                                        double __map_fusion_gtir_tmp_229_1_0;         ////__DACE:0:0:279
                                        double __map_fusion_gtir_tmp_227_1_0;         ////__DACE:0:0:280
                                        double __map_fusion_gtir_tmp_139_split_1_0;    ////__DACE:0:0:281
                                        {                                             ////__DACE:0:0:209
                                            double __tlet_arg1 = __p_dthalf_1;                                                ////__DACE:0:0:316,209    ////__DACE:0:0:209
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,209    ////__DACE:0:0:209
                                            double __tlet_result;                                                             ////__DACE:0:0:209    ////__DACE:0:0:209
                                            ////__DACE:0:0:209                        ////__DACE:0:0:209
                                            ///////////////////                                                               ////__DACE:0:0:209    ////__DACE:0:0:209
                                            // Tasklet code (tlet_7_multiplies_1_1)                                           ////__DACE:0:0:209    ////__DACE:0:0:209
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:209    ////__DACE:0:0:209
                                            ///////////////////                                                               ////__DACE:0:0:209    ////__DACE:0:0:209
                                            ////__DACE:0:0:209                        ////__DACE:0:0:209
                                            __map_fusion_gtir_tmp_11_1_1 = __tlet_result;                                     ////__DACE:0:0:209    ////__DACE:0:0:209
                                        }                                             ////__DACE:0:0:209
                                        {                                             ////__DACE:0:0:215
                                            double __tlet_arg1 = lambda_4___p_dthalf_0;                                       ////__DACE:0:0:320,215    ////__DACE:0:0:215
                                            double __tlet_arg0 = tangential_wind[((__tangential_wind_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:6,215    ////__DACE:0:0:215
                                            double __tlet_result;                                                             ////__DACE:0:0:215    ////__DACE:0:0:215
                                            ////__DACE:0:0:215                        ////__DACE:0:0:215
                                            ///////////////////                                                               ////__DACE:0:0:215    ////__DACE:0:0:215
                                            // Tasklet code (tlet_11_multiplies_1_1)                                          ////__DACE:0:0:215    ////__DACE:0:0:215
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:215    ////__DACE:0:0:215
                                            ///////////////////                                                               ////__DACE:0:0:215    ////__DACE:0:0:215
                                            ////__DACE:0:0:215                        ////__DACE:0:0:215
                                            __map_fusion_gtir_tmp_23_1_1 = __tlet_result;                                     ////__DACE:0:0:215    ////__DACE:0:0:215
                                        }                                             ////__DACE:0:0:215
                                        {                                             ////__DACE:0:0:233
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,233    ////__DACE:0:0:233
                                            double __tlet_arg1 = gtir_tmp_175_0;                                              ////__DACE:0:0:338,233    ////__DACE:0:0:233
                                            bool __tlet_result;                                                               ////__DACE:0:0:233    ////__DACE:0:0:233
                                            ////__DACE:0:0:233                        ////__DACE:0:0:233
                                            ///////////////////                                                               ////__DACE:0:0:233    ////__DACE:0:0:233
                                            // Tasklet code (tlet_69_greater_equal_1_1)                                       ////__DACE:0:0:233    ////__DACE:0:0:233
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:233    ////__DACE:0:0:233
                                            ///////////////////                                                               ////__DACE:0:0:233    ////__DACE:0:0:233
                                            ////__DACE:0:0:233                        ////__DACE:0:0:233
                                            __map_fusion_gtir_tmp_177_1_1 = __tlet_result;                                    ////__DACE:0:0:233    ////__DACE:0:0:233
                                        }                                             ////__DACE:0:0:233
                                        {                                             ////__DACE:0:0:202
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,202    ////__DACE:0:0:202
                                            double __tlet_arg1 = gtir_tmp_3_2;                                                ////__DACE:0:0:312,202    ////__DACE:0:0:202
                                            bool __tlet_result;                                                               ////__DACE:0:0:202    ////__DACE:0:0:202
                                            ////__DACE:0:0:202                        ////__DACE:0:0:202
                                            ///////////////////                                                               ////__DACE:0:0:202    ////__DACE:0:0:202
                                            // Tasklet code (tlet_3_greater_equal_1_1)                                        ////__DACE:0:0:202    ////__DACE:0:0:202
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:202    ////__DACE:0:0:202
                                            ///////////////////                                                               ////__DACE:0:0:202    ////__DACE:0:0:202
                                            ////__DACE:0:0:202                        ////__DACE:0:0:202
                                            __map_fusion_gtir_tmp_5_1_1 = __tlet_result;                                      ////__DACE:0:0:202    ////__DACE:0:0:202
                                        }                                             ////__DACE:0:0:202
                                        if_stmt_0_0_0_158(gtir_tmp_6_0, gtir_tmp_7_0, __map_fusion_gtir_tmp_5_1_1, gtir_tmp_8_1_1);    ////__DACE:0:0:201
                                        if_stmt_1_0_0_206(gtir_tmp_12_1_1, gtir_tmp_24_1_1, gtir_tmp_14_1_1, gtir_tmp_26_1_1, gtir_tmp_8_1_1, gtir_tmp_16_1_1, gtir_tmp_28_1_1);    ////__DACE:0:0:206
                                        {                                             ////__DACE:0:0:204
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_11_1_1;                                ////__DACE:0:0:237,204    ////__DACE:0:0:204
                                            double __tlet_arg1 = gtir_tmp_16_1_1;                                             ////__DACE:0:0:205,204    ////__DACE:0:0:204
                                            double __tlet_result;                                                             ////__DACE:0:0:204    ////__DACE:0:0:204
                                            ////__DACE:0:0:204                        ////__DACE:0:0:204
                                            ///////////////////                                                               ////__DACE:0:0:204    ////__DACE:0:0:204
                                            // Tasklet code (tlet_8_plus_1_1)                                                 ////__DACE:0:0:204    ////__DACE:0:0:204
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:204    ////__DACE:0:0:204
                                            ///////////////////                                                               ////__DACE:0:0:204    ////__DACE:0:0:204
                                            ////__DACE:0:0:204                        ////__DACE:0:0:204
                                            __map_fusion_gtir_tmp_19_1_1 = __tlet_result;                                     ////__DACE:0:0:204    ////__DACE:0:0:204
                                        }                                             ////__DACE:0:0:204
                                        {                                             ////__DACE:0:0:203
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_19_1_1;                                ////__DACE:0:0:236,203    ////__DACE:0:0:203
                                            double __tlet_result;                                                             ////__DACE:0:0:203    ////__DACE:0:0:203
                                            ////__DACE:0:0:203                        ////__DACE:0:0:203
                                            ///////////////////                                                               ////__DACE:0:0:203    ////__DACE:0:0:203
                                            // Tasklet code (tlet_9_neg_1_1)                                                  ////__DACE:0:0:203    ////__DACE:0:0:203
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:203    ////__DACE:0:0:203
                                            ///////////////////                                                               ////__DACE:0:0:203    ////__DACE:0:0:203
                                            ////__DACE:0:0:203                        ////__DACE:0:0:203
                                            __map_fusion_gtir_tmp_21_1_1_1 = __tlet_result;                                   ////__DACE:0:0:203    ////__DACE:0:0:203
                                        }                                             ////__DACE:0:0:203
                                        {                                             ////__DACE:0:0:211
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_23_1_1;                                ////__DACE:0:0:240,211    ////__DACE:0:0:211
                                            double __tlet_arg1 = gtir_tmp_28_1_1;                                             ////__DACE:0:0:212,211    ////__DACE:0:0:211
                                            double __tlet_result;                                                             ////__DACE:0:0:211    ////__DACE:0:0:211
                                            ////__DACE:0:0:211                        ////__DACE:0:0:211
                                            ///////////////////                                                               ////__DACE:0:0:211    ////__DACE:0:0:211
                                            // Tasklet code (tlet_12_plus_1_1)                                                ////__DACE:0:0:211    ////__DACE:0:0:211
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:211    ////__DACE:0:0:211
                                            ///////////////////                                                               ////__DACE:0:0:211    ////__DACE:0:0:211
                                            ////__DACE:0:0:211                        ////__DACE:0:0:211
                                            __map_fusion_gtir_tmp_31_1_1 = __tlet_result;                                     ////__DACE:0:0:211    ////__DACE:0:0:211
                                        }                                             ////__DACE:0:0:211
                                        {                                             ////__DACE:0:0:210
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_31_1_1;                                ////__DACE:0:0:239,210    ////__DACE:0:0:210
                                            double __tlet_result;                                                             ////__DACE:0:0:210    ////__DACE:0:0:210
                                            ////__DACE:0:0:210                        ////__DACE:0:0:210
                                            ///////////////////                                                               ////__DACE:0:0:210    ////__DACE:0:0:210
                                            // Tasklet code (tlet_13_neg_1_1)                                                 ////__DACE:0:0:210    ////__DACE:0:0:210
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:210    ////__DACE:0:0:210
                                            ///////////////////                                                               ////__DACE:0:0:210    ////__DACE:0:0:210
                                            ////__DACE:0:0:210                        ////__DACE:0:0:210
                                            __map_fusion_gtir_tmp_33_1_1_1 = __tlet_result;                                   ////__DACE:0:0:210    ////__DACE:0:0:210
                                        }                                             ////__DACE:0:0:210
                                        if_stmt_4_0_0_217(gtir_tmp_8_1_1, __map_fusion_gtir_tmp_21_1_1_1, __map_fusion_gtir_tmp_33_1_1_1, gtir_tmp_34_1_1, gtir_tmp_38_1_1, gtir_tmp_44_1_1, gtir_tmp_48_1_1, gtir_tmp_56_1_1, gtir_tmp_60_1_1, gtir_tmp_66_1_1, gtir_tmp_70_1_1, gtir_tmp_76_1_1, gtir_tmp_54_1_1);    ////__DACE:0:0:217
                                        if_stmt_7_0_0_232(__map_fusion_gtir_tmp_177_1_1, &gt_conn_E2C[0], gtir_tmp_54_1_1, gtir_tmp_76_1_1, &gtir_tmp_83[0], &gtir_tmp_89[0], &perturbed_rho_at_cells_on_model_levels[0], &reference_rho_at_edges_on_model_levels[0], gtir_tmp_210_1_1, __gt_conn_E2C_neighbor_stride, __perturbed_rho_at_cells_on_model_levels_K_stride, __reference_rho_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:232
                                        {                                             ////__DACE:0:0:415
                                            double _cpy_in = gtir_tmp_210_1_1;                                                ////__DACE:0:0:231,415    ////__DACE:0:0:415
                                            double _cpy_out;                                                                  ////__DACE:0:0:415    ////__DACE:0:0:415
                                            ////__DACE:0:0:415                        ////__DACE:0:0:415
                                            ///////////////////                                                               ////__DACE:0:0:415    ////__DACE:0:0:415
                                            // Tasklet code (copy_gtir_tmp_210_1_1_to_rho_at_edges_on_model_levels)           ////__DACE:0:0:415    ////__DACE:0:0:415
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:415    ////__DACE:0:0:415
                                            ///////////////////                                                               ////__DACE:0:0:415    ////__DACE:0:0:415
                                            ////__DACE:0:0:415                        ////__DACE:0:0:415
                                            rho_at_edges_on_model_levels[((__rho_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:415    ////__DACE:0:0:415
                                        }                                             ////__DACE:0:0:415
                                        {                                             ////__DACE:0:0:230
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,230    ////__DACE:0:0:230
                                            double __tlet_arg1 = gtir_tmp_102_0;                                              ////__DACE:0:0:326,230    ////__DACE:0:0:230
                                            bool __tlet_result;                                                               ////__DACE:0:0:230    ////__DACE:0:0:230
                                            ////__DACE:0:0:230                        ////__DACE:0:0:230
                                            ///////////////////                                                               ////__DACE:0:0:230    ////__DACE:0:0:230
                                            // Tasklet code (tlet_35_greater_equal_1_1)                                       ////__DACE:0:0:230    ////__DACE:0:0:230
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:230    ////__DACE:0:0:230
                                            ///////////////////                                                               ////__DACE:0:0:230    ////__DACE:0:0:230
                                            ////__DACE:0:0:230                        ////__DACE:0:0:230
                                            __map_fusion_gtir_tmp_104_1_1 = __tlet_result;                                    ////__DACE:0:0:230    ////__DACE:0:0:230
                                        }                                             ////__DACE:0:0:230
                                        if_stmt_5_0_0_229(__map_fusion_gtir_tmp_104_1_1, &gt_conn_E2C[0], &gtir_tmp_101[0], gtir_tmp_54_1_1, gtir_tmp_76_1_1, &gtir_tmp_95[0], &perturbed_theta_v_at_cells_on_model_levels[0], &reference_theta_at_edges_on_model_levels[0], gtir_tmp_137_1_1, __gt_conn_E2C_neighbor_stride, __perturbed_theta_v_at_cells_on_model_levels_K_stride, __reference_theta_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:229
                                        {                                             ////__DACE:0:0:274
                                            double __tlet_arg0 = gtir_tmp_223_0;                                              ////__DACE:0:0:344,274    ////__DACE:0:0:274
                                            double __tlet_arg1 = gtir_tmp_137_1_1;                                            ////__DACE:0:0:228,274    ////__DACE:0:0:274
                                            double __tlet_result;                                                             ////__DACE:0:0:274    ////__DACE:0:0:274
                                            ////__DACE:0:0:274                        ////__DACE:0:0:274
                                            ///////////////////                                                               ////__DACE:0:0:274    ////__DACE:0:0:274
                                            // Tasklet code (tlet_93_multiplies_1_0)                                          ////__DACE:0:0:274    ////__DACE:0:0:274
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:274    ////__DACE:0:0:274
                                            ///////////////////                                                               ////__DACE:0:0:274    ////__DACE:0:0:274
                                            ////__DACE:0:0:274                        ////__DACE:0:0:274
                                            __map_fusion_gtir_tmp_227_1_0 = __tlet_result;                                    ////__DACE:0:0:274    ////__DACE:0:0:274
                                        }                                             ////__DACE:0:0:274
                                        {                                             ////__DACE:0:0:416
                                            double _cpy_in = gtir_tmp_137_1_1;                                                ////__DACE:0:0:228,416    ////__DACE:0:0:416
                                            double _cpy_out;                                                                  ////__DACE:0:0:416    ////__DACE:0:0:416
                                            ////__DACE:0:0:416                        ////__DACE:0:0:416
                                            ///////////////////                                                               ////__DACE:0:0:416    ////__DACE:0:0:416
                                            // Tasklet code (copy_gtir_tmp_137_1_1_to_theta_v_at_edges_on_model_levels)       ////__DACE:0:0:416    ////__DACE:0:0:416
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:416    ////__DACE:0:0:416
                                            ///////////////////                                                               ////__DACE:0:0:416    ////__DACE:0:0:416
                                            ////__DACE:0:0:416                        ////__DACE:0:0:416
                                            theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:416    ////__DACE:0:0:416
                                        }                                             ////__DACE:0:0:416
                                        {                                             ////__DACE:0:0:251
                                            for (auto i_E2C_gtx_localdim = 0; i_E2C_gtx_localdim < 2; i_E2C_gtx_localdim += 1) {    ////__DACE:0:0:251
                                                double __gtx_double_write_remover_inner_inner_distribution_node_8_0_0;    ////__DACE:0:0:264
                                                {                                     ////__DACE:0:0:250
                                                    const double* __tlet_field = &ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels[(__ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride * i_K_gtx_vertical)];    ////__DACE:0:0:53,250    ////__DACE:0:0:250
                                                    int __tlet_index = gt_conn_E2C[((__gt_conn_E2C_neighbor_stride * i_E2C_gtx_localdim) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:50,250    ////__DACE:0:0:250
                                                    double __tlet_val;                                                                ////__DACE:0:0:250    ////__DACE:0:0:250
                                                    ////__DACE:0:0:250                ////__DACE:0:0:250
                                                    ///////////////////                                                               ////__DACE:0:0:250    ////__DACE:0:0:250
                                                    // Tasklet code (tlet_60_E2C_neighbors_0_0)                                       ////__DACE:0:0:250    ////__DACE:0:0:250
                                                    __tlet_val = __tlet_field[__tlet_index];                                          ////__DACE:0:0:250    ////__DACE:0:0:250
                                                    ///////////////////                                                               ////__DACE:0:0:250    ////__DACE:0:0:250
                                                    ////__DACE:0:0:250                ////__DACE:0:0:250
                                                    __gtx_double_write_remover_inner_inner_distribution_node_8_0_0 = __tlet_val;      ////__DACE:0:0:250    ////__DACE:0:0:250
                                                }                                     ////__DACE:0:0:250
                                                {                                     ////__DACE:0:0:405
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_8_0_0;    ////__DACE:0:0:264,405    ////__DACE:0:0:405
                                                    double _cpy_out;                                                                  ////__DACE:0:0:405    ////__DACE:0:0:405
                                                    ////__DACE:0:0:405                ////__DACE:0:0:405
                                                    ///////////////////                                                               ////__DACE:0:0:405    ////__DACE:0:0:405
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_8_0_0_to___map_fusion_gtir_tmp_157_0_0)    ////__DACE:0:0:405    ////__DACE:0:0:405
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:405    ////__DACE:0:0:405
                                                    ///////////////////                                                               ////__DACE:0:0:405    ////__DACE:0:0:405
                                                    ////__DACE:0:0:405                ////__DACE:0:0:405
                                                    __map_fusion_gtir_tmp_157_0_0[i_E2C_gtx_localdim] = _cpy_out;                     ////__DACE:0:0:405    ////__DACE:0:0:405
                                                }                                     ////__DACE:0:0:405
                                            }                                         ////__DACE:0:0:249
                                        }                                             ////__DACE:0:0:249
                                        {                                             ////__DACE:0:0:248
                                            for (auto i_E2C_gtx_localdim = 0; i_E2C_gtx_localdim < 2; i_E2C_gtx_localdim += 1) {    ////__DACE:0:0:248
                                                double __gtx_double_write_remover_inner_inner_distribution_node_7_0_0;    ////__DACE:0:0:263
                                                {                                     ////__DACE:0:0:247
                                                    double __tlet_arg1 = c_lin_e[((__c_lin_e_E2C_stride * i_E2C_gtx_localdim) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:52,247    ////__DACE:0:0:247
                                                    double __tlet_arg0 = __map_fusion_gtir_tmp_157_0_0[i_E2C_gtx_localdim];           ////__DACE:0:0:258,247    ////__DACE:0:0:247
                                                    double __tlet_out;                                                                ////__DACE:0:0:247    ////__DACE:0:0:247
                                                    ////__DACE:0:0:247                ////__DACE:0:0:247
                                                    ///////////////////                                                               ////__DACE:0:0:247    ////__DACE:0:0:247
                                                    // Tasklet code (tlet_61_map_0_0)                                                 ////__DACE:0:0:247    ////__DACE:0:0:247
                                                    __tlet_out = (__tlet_arg0 * __tlet_arg1);                                         ////__DACE:0:0:247    ////__DACE:0:0:247
                                                    ///////////////////                                                               ////__DACE:0:0:247    ////__DACE:0:0:247
                                                    ////__DACE:0:0:247                ////__DACE:0:0:247
                                                    __gtx_double_write_remover_inner_inner_distribution_node_7_0_0 = __tlet_out;      ////__DACE:0:0:247    ////__DACE:0:0:247
                                                }                                     ////__DACE:0:0:247
                                                {                                     ////__DACE:0:0:404
                                                    double _cpy_in = __gtx_double_write_remover_inner_inner_distribution_node_7_0_0;    ////__DACE:0:0:263,404    ////__DACE:0:0:404
                                                    double _cpy_out;                                                                  ////__DACE:0:0:404    ////__DACE:0:0:404
                                                    ////__DACE:0:0:404                ////__DACE:0:0:404
                                                    ///////////////////                                                               ////__DACE:0:0:404    ////__DACE:0:0:404
                                                    // Tasklet code (copy___gtx_double_write_remover_inner_inner_distribution_node_7_0_0_to___map_fusion_gtir_tmp_159_0_0)    ////__DACE:0:0:404    ////__DACE:0:0:404
                                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:404    ////__DACE:0:0:404
                                                    ///////////////////                                                               ////__DACE:0:0:404    ////__DACE:0:0:404
                                                    ////__DACE:0:0:404                ////__DACE:0:0:404
                                                    __map_fusion_gtir_tmp_159_0_0[i_E2C_gtx_localdim] = _cpy_out;                     ////__DACE:0:0:404    ////__DACE:0:0:404
                                                }                                     ////__DACE:0:0:404
                                            }                                         ////__DACE:0:0:246
                                        }                                             ////__DACE:0:0:246
                                        reduce_0_0_359(&__map_fusion_gtir_tmp_159_0_0[0], gtir_tmp_160_0_0);    ////__DACE:0:0:359
                                        {                                             ////__DACE:0:0:244
                                            double __tlet_arg0 = ddxn_z_full[((__ddxn_z_full_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:54,244    ////__DACE:0:0:244
                                            double __tlet_arg1 = gtir_tmp_160_0_0;                                            ////__DACE:0:0:245,244    ////__DACE:0:0:244
                                            double __tlet_result;                                                             ////__DACE:0:0:244    ////__DACE:0:0:244
                                            ////__DACE:0:0:244                        ////__DACE:0:0:244
                                            ///////////////////                                                               ////__DACE:0:0:244    ////__DACE:0:0:244
                                            // Tasklet code (tlet_62_multiplies_0_0)                                          ////__DACE:0:0:244    ////__DACE:0:0:244
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:244    ////__DACE:0:0:244
                                            ///////////////////                                                               ////__DACE:0:0:244    ////__DACE:0:0:244
                                            ////__DACE:0:0:244                        ////__DACE:0:0:244
                                            __map_fusion_gtir_tmp_163_0_0 = __tlet_result;                                    ////__DACE:0:0:244    ////__DACE:0:0:244
                                        }                                             ////__DACE:0:0:244
                                        {                                             ////__DACE:0:0:255
                                            int __tlet_index_Cell = gt_conn_E2C[(__gt_conn_E2C_neighbor_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:50,255    ////__DACE:0:0:255
                                            const double* __tlet_field = &temporal_extrapolation_of_perturbed_exner[0];       ////__DACE:0:0:55,255    ////__DACE:0:0:255
                                            double __tlet_val;                                                                ////__DACE:0:0:255    ////__DACE:0:0:255
                                            ////__DACE:0:0:255                        ////__DACE:0:0:255
                                            ///////////////////                                                               ////__DACE:0:0:255    ////__DACE:0:0:255
                                            // Tasklet code (tlet_56_deref_0_0)                                               ////__DACE:0:0:255    ////__DACE:0:0:255
                                            __tlet_val = __tlet_field[((__temporal_extrapolation_of_perturbed_exner_K_stride * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:0:0:255    ////__DACE:0:0:255
                                            ///////////////////                                                               ////__DACE:0:0:255    ////__DACE:0:0:255
                                            ////__DACE:0:0:255                        ////__DACE:0:0:255
                                            __map_fusion_gtir_tmp_149_0_0 = __tlet_val;                                       ////__DACE:0:0:255    ////__DACE:0:0:255
                                        }                                             ////__DACE:0:0:255
                                        {                                             ////__DACE:0:0:254
                                            int __tlet_index_Cell = gt_conn_E2C[i_Edge_gtx_horizontal];                       ////__DACE:0:0:50,254    ////__DACE:0:0:254
                                            const double* __tlet_field = &temporal_extrapolation_of_perturbed_exner[0];       ////__DACE:0:0:55,254    ////__DACE:0:0:254
                                            double __tlet_val;                                                                ////__DACE:0:0:254    ////__DACE:0:0:254
                                            ////__DACE:0:0:254                        ////__DACE:0:0:254
                                            ///////////////////                                                               ////__DACE:0:0:254    ////__DACE:0:0:254
                                            // Tasklet code (tlet_57_deref_0_0)                                               ////__DACE:0:0:254    ////__DACE:0:0:254
                                            __tlet_val = __tlet_field[((__temporal_extrapolation_of_perturbed_exner_K_stride * i_K_gtx_vertical) + __tlet_index_Cell)];    ////__DACE:0:0:254    ////__DACE:0:0:254
                                            ///////////////////                                                               ////__DACE:0:0:254    ////__DACE:0:0:254
                                            ////__DACE:0:0:254                        ////__DACE:0:0:254
                                            __map_fusion_gtir_tmp_151_0_0 = __tlet_val;                                       ////__DACE:0:0:254    ////__DACE:0:0:254
                                        }                                             ////__DACE:0:0:254
                                        {                                             ////__DACE:0:0:253
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_149_0_0;                               ////__DACE:0:0:262,253    ////__DACE:0:0:253
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_151_0_0;                               ////__DACE:0:0:261,253    ////__DACE:0:0:253
                                            double __tlet_result;                                                             ////__DACE:0:0:253    ////__DACE:0:0:253
                                            ////__DACE:0:0:253                        ////__DACE:0:0:253
                                            ///////////////////                                                               ////__DACE:0:0:253    ////__DACE:0:0:253
                                            // Tasklet code (tlet_58_minus_0_0)                                               ////__DACE:0:0:253    ////__DACE:0:0:253
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:253    ////__DACE:0:0:253
                                            ///////////////////                                                               ////__DACE:0:0:253    ////__DACE:0:0:253
                                            ////__DACE:0:0:253                        ////__DACE:0:0:253
                                            __map_fusion_gtir_tmp_153_0_0 = __tlet_result;                                    ////__DACE:0:0:253    ////__DACE:0:0:253
                                        }                                             ////__DACE:0:0:253
                                        {                                             ////__DACE:0:0:252
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_153_0_0;                               ////__DACE:0:0:260,252    ////__DACE:0:0:252
                                            double __tlet_arg0 = inv_dual_edge_length[i_Edge_gtx_horizontal];                 ////__DACE:0:0:56,252    ////__DACE:0:0:252
                                            double __tlet_result;                                                             ////__DACE:0:0:252    ////__DACE:0:0:252
                                            ////__DACE:0:0:252                        ////__DACE:0:0:252
                                            ///////////////////                                                               ////__DACE:0:0:252    ////__DACE:0:0:252
                                            // Tasklet code (tlet_59_multiplies_0_0)                                          ////__DACE:0:0:252    ////__DACE:0:0:252
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:252    ////__DACE:0:0:252
                                            ///////////////////                                                               ////__DACE:0:0:252    ////__DACE:0:0:252
                                            ////__DACE:0:0:252                        ////__DACE:0:0:252
                                            __map_fusion_gtir_tmp_155_0_0 = __tlet_result;                                    ////__DACE:0:0:252    ////__DACE:0:0:252
                                        }                                             ////__DACE:0:0:252
                                        {                                             ////__DACE:0:0:243
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_155_0_0;                               ////__DACE:0:0:259,243    ////__DACE:0:0:243
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_163_0_0;                               ////__DACE:0:0:256,243    ////__DACE:0:0:243
                                            double __tlet_result;                                                             ////__DACE:0:0:243    ////__DACE:0:0:243
                                            ////__DACE:0:0:243                        ////__DACE:0:0:243
                                            ///////////////////                                                               ////__DACE:0:0:243    ////__DACE:0:0:243
                                            // Tasklet code (tlet_63_minus_0_0)                                               ////__DACE:0:0:243    ////__DACE:0:0:243
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:243    ////__DACE:0:0:243
                                            ///////////////////                                                               ////__DACE:0:0:243    ////__DACE:0:0:243
                                            ////__DACE:0:0:243                        ////__DACE:0:0:243
                                            __map_fusion_gtir_tmp_139_split_1_0 = __tlet_result;                              ////__DACE:0:0:243    ////__DACE:0:0:243
                                        }                                             ////__DACE:0:0:243
                                        {                                             ////__DACE:0:0:267
                                            double __tlet_arg1 = gtir_tmp_166_1;                                              ////__DACE:0:0:334,267    ////__DACE:0:0:267
                                            double __tlet_arg0 = pg_exdist[((__pg_exdist_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:61,267    ////__DACE:0:0:267
                                            bool __tlet_result;                                                               ////__DACE:0:0:267    ////__DACE:0:0:267
                                            ////__DACE:0:0:267                        ////__DACE:0:0:267
                                            ///////////////////                                                               ////__DACE:0:0:267    ////__DACE:0:0:267
                                            // Tasklet code (tlet_65_not_eq_1_0)                                              ////__DACE:0:0:267    ////__DACE:0:0:267
                                            __tlet_result = (__tlet_arg0 != __tlet_arg1);                                     ////__DACE:0:0:267    ////__DACE:0:0:267
                                            ///////////////////                                                               ////__DACE:0:0:267    ////__DACE:0:0:267
                                            ////__DACE:0:0:267                        ////__DACE:0:0:267
                                            __map_fusion_gtir_tmp_168_1_0 = __tlet_result;                                    ////__DACE:0:0:267    ////__DACE:0:0:267
                                        }                                             ////__DACE:0:0:267
                                        if_stmt_6_0_0_266(__map_fusion_gtir_tmp_139_split_1_0, __map_fusion_gtir_tmp_168_1_0, &hydrostatic_correction_on_lowest_level[0], &pg_exdist[0], gtir_tmp_173_1_0, __pg_exdist_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:266
                                        {                                             ////__DACE:0:0:273
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_227_1_0;                               ////__DACE:0:0:280,273    ////__DACE:0:0:273
                                            double __tlet_arg1 = gtir_tmp_173_1_0;                                            ////__DACE:0:0:265,273    ////__DACE:0:0:273
                                            double __tlet_result;                                                             ////__DACE:0:0:273    ////__DACE:0:0:273
                                            ////__DACE:0:0:273                        ////__DACE:0:0:273
                                            ///////////////////                                                               ////__DACE:0:0:273    ////__DACE:0:0:273
                                            // Tasklet code (tlet_94_multiplies_1_0)                                          ////__DACE:0:0:273    ////__DACE:0:0:273
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:273    ////__DACE:0:0:273
                                            ///////////////////                                                               ////__DACE:0:0:273    ////__DACE:0:0:273
                                            ////__DACE:0:0:273                        ////__DACE:0:0:273
                                            __map_fusion_gtir_tmp_229_1_0 = __tlet_result;                                    ////__DACE:0:0:273    ////__DACE:0:0:273
                                        }                                             ////__DACE:0:0:273
                                        {                                             ////__DACE:0:0:272
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_229_1_0;                               ////__DACE:0:0:279,272    ////__DACE:0:0:272
                                            double __tlet_arg0 = predictor_normal_wind_advective_tendency[((__predictor_normal_wind_advective_tendency_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:66,272    ////__DACE:0:0:272
                                            double __tlet_result;                                                             ////__DACE:0:0:272    ////__DACE:0:0:272
                                            ////__DACE:0:0:272                        ////__DACE:0:0:272
                                            ///////////////////                                                               ////__DACE:0:0:272    ////__DACE:0:0:272
                                            // Tasklet code (tlet_95_minus_1_0)                                               ////__DACE:0:0:272    ////__DACE:0:0:272
                                            __tlet_result = (__tlet_arg0 - __tlet_arg1);                                      ////__DACE:0:0:272    ////__DACE:0:0:272
                                            ///////////////////                                                               ////__DACE:0:0:272    ////__DACE:0:0:272
                                            ////__DACE:0:0:272                        ////__DACE:0:0:272
                                            __map_fusion_gtir_tmp_231_1_0 = __tlet_result;                                    ////__DACE:0:0:272    ////__DACE:0:0:272
                                        }                                             ////__DACE:0:0:272
                                        {                                             ////__DACE:0:0:271
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_231_1_0;                               ////__DACE:0:0:278,271    ////__DACE:0:0:271
                                            double __tlet_arg1 = normal_wind_tendency_due_to_slow_physics_process[((__normal_wind_tendency_due_to_slow_physics_process_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:65,271    ////__DACE:0:0:271
                                            double __tlet_result;                                                             ////__DACE:0:0:271    ////__DACE:0:0:271
                                            ////__DACE:0:0:271                        ////__DACE:0:0:271
                                            ///////////////////                                                               ////__DACE:0:0:271    ////__DACE:0:0:271
                                            // Tasklet code (tlet_96_plus_1_0)                                                ////__DACE:0:0:271    ////__DACE:0:0:271
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:271    ////__DACE:0:0:271
                                            ///////////////////                                                               ////__DACE:0:0:271    ////__DACE:0:0:271
                                            ////__DACE:0:0:271                        ////__DACE:0:0:271
                                            __map_fusion_gtir_tmp_233_1_0 = __tlet_result;                                    ////__DACE:0:0:271    ////__DACE:0:0:271
                                        }                                             ////__DACE:0:0:271
                                        {                                             ////__DACE:0:0:270
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_233_1_0;                               ////__DACE:0:0:277,270    ////__DACE:0:0:270
                                            double __tlet_arg0 = __dtime_0_1;                                                 ////__DACE:0:0:350,270    ////__DACE:0:0:270
                                            double __tlet_result;                                                             ////__DACE:0:0:270    ////__DACE:0:0:270
                                            ////__DACE:0:0:270                        ////__DACE:0:0:270
                                            ///////////////////                                                               ////__DACE:0:0:270    ////__DACE:0:0:270
                                            // Tasklet code (tlet_97_multiplies_1_0)                                          ////__DACE:0:0:270    ////__DACE:0:0:270
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:270    ////__DACE:0:0:270
                                            ///////////////////                                                               ////__DACE:0:0:270    ////__DACE:0:0:270
                                            ////__DACE:0:0:270                        ////__DACE:0:0:270
                                            __map_fusion_gtir_tmp_235_1_0 = __tlet_result;                                    ////__DACE:0:0:270    ////__DACE:0:0:270
                                        }                                             ////__DACE:0:0:270
                                        {                                             ////__DACE:0:0:269
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_235_1_0;                               ////__DACE:0:0:276,269    ////__DACE:0:0:269
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,269    ////__DACE:0:0:269
                                            double __tlet_result;                                                             ////__DACE:0:0:269    ////__DACE:0:0:269
                                            ////__DACE:0:0:269                        ////__DACE:0:0:269
                                            ///////////////////                                                               ////__DACE:0:0:269    ////__DACE:0:0:269
                                            // Tasklet code (tlet_98_plus_1_0)                                                ////__DACE:0:0:269    ////__DACE:0:0:269
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:269    ////__DACE:0:0:269
                                            ///////////////////                                                               ////__DACE:0:0:269    ////__DACE:0:0:269
                                            ////__DACE:0:0:269                        ////__DACE:0:0:269
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:269    ////__DACE:0:0:269
                                        }                                             ////__DACE:0:0:269
                                        {                                             ////__DACE:0:0:417
                                            double _cpy_in = gtir_tmp_173_1_0;                                                ////__DACE:0:0:265,417    ////__DACE:0:0:417
                                            double _cpy_out;                                                                  ////__DACE:0:0:417    ////__DACE:0:0:417
                                            ////__DACE:0:0:417                        ////__DACE:0:0:417
                                            ///////////////////                                                               ////__DACE:0:0:417    ////__DACE:0:0:417
                                            // Tasklet code (copy_gtir_tmp_173_1_0_to_horizontal_pressure_gradient)           ////__DACE:0:0:417    ////__DACE:0:0:417
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:417    ////__DACE:0:0:417
                                            ///////////////////                                                               ////__DACE:0:0:417    ////__DACE:0:0:417
                                            ////__DACE:0:0:417                        ////__DACE:0:0:417
                                            horizontal_pressure_gradient[((__horizontal_pressure_gradient_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:417    ////__DACE:0:0:417
                                        }                                             ////__DACE:0:0:417
                                    }                                                 ////__DACE:0:0:292
                                }                                                     ////__DACE:0:0:292
                            }                                                         ////__DACE:0:0:268
                        }                                                             ////__DACE:0:0:268
                    }                                                                 ////__DACE:0:0:268
                }                                                                     ////__DACE:0:0:268
            }                                                                         ////__DACE:0:0:268
        }                                                                             ////__DACE:0:0:427
    }                                                                                 ////__DACE:0:0:427
}                                                                                 ////__DACE:0:0:427

                                                                                  ////__DACE:0:0:426
DACE_EXPORTED void __dace_runkernel_map_100_fieldop_1_1_0_0_426(theta_shared_probe_shared_state_t *__state, const double * __restrict__ c_lin_e, const double * __restrict__ current_vn, const double * __restrict__ ddxn_z_full, const double * __restrict__ ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pg_exdist, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, double * __restrict__ theta_v_at_edges_on_model_levels, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:426
void __dace_runkernel_map_100_fieldop_1_1_0_0_426(theta_shared_probe_shared_state_t *__state, const double * __restrict__ c_lin_e, const double * __restrict__ current_vn, const double * __restrict__ ddxn_z_full, const double * __restrict__ ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ horizontal_pressure_gradient, const double * __restrict__ hydrostatic_correction_on_lowest_level, const double * __restrict__ inv_dual_edge_length, double * __restrict__ next_vn, const double * __restrict__ normal_wind_tendency_due_to_slow_physics_process, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pg_exdist, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ predictor_normal_wind_advective_tendency, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, const double * __restrict__ temporal_extrapolation_of_perturbed_exner, double * __restrict__ theta_v_at_edges_on_model_levels, int __c_lin_e_E2C_stride, int __current_vn_K_stride, int __ddxn_z_full_K_stride, int __ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __gt_conn_E2C_neighbor_stride, int __horizontal_pressure_gradient_K_stride, int __next_vn_K_stride, int __normal_wind_tendency_due_to_slow_physics_process_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pg_exdist_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __predictor_normal_wind_advective_tendency_K_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __temporal_extrapolation_of_perturbed_exner_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:426
{                                                                                 ////__DACE:0:0:426
                                                                                  ////__DACE:0:0:426
    void  *map_100_fieldop_1_1_0_0_426_args[] = { (void *)&c_lin_e, (void *)&current_vn, (void *)&ddxn_z_full, (void *)&ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels, (void *)&dual_normal_cell_x, (void *)&dual_normal_cell_y, (void *)&gt_conn_E2C, (void *)&gtir_tmp_101, (void *)&gtir_tmp_83, (void *)&gtir_tmp_89, (void *)&gtir_tmp_95, (void *)&horizontal_pressure_gradient, (void *)&hydrostatic_correction_on_lowest_level, (void *)&inv_dual_edge_length, (void *)&next_vn, (void *)&normal_wind_tendency_due_to_slow_physics_process, (void *)&perturbed_rho_at_cells_on_model_levels, (void *)&perturbed_theta_v_at_cells_on_model_levels, (void *)&pg_exdist, (void *)&pos_on_tplane_e_x, (void *)&pos_on_tplane_e_y, (void *)&predictor_normal_wind_advective_tendency, (void *)&primal_normal_cell_x, (void *)&primal_normal_cell_y, (void *)&reference_rho_at_edges_on_model_levels, (void *)&reference_theta_at_edges_on_model_levels, (void *)&rho_at_edges_on_model_levels, (void *)&tangential_wind, (void *)&temporal_extrapolation_of_perturbed_exner, (void *)&theta_v_at_edges_on_model_levels, (void *)&__c_lin_e_E2C_stride, (void *)&__current_vn_K_stride, (void *)&__ddxn_z_full_K_stride, (void *)&__ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels_K_stride, (void *)&__dual_normal_cell_x_E2C_stride, (void *)&__dual_normal_cell_y_E2C_stride, (void *)&__gt_conn_E2C_neighbor_stride, (void *)&__horizontal_pressure_gradient_K_stride, (void *)&__next_vn_K_stride, (void *)&__normal_wind_tendency_due_to_slow_physics_process_K_stride, (void *)&__perturbed_rho_at_cells_on_model_levels_K_stride, (void *)&__perturbed_theta_v_at_cells_on_model_levels_K_stride, (void *)&__pg_exdist_K_stride, (void *)&__pos_on_tplane_e_x_E2C_stride, (void *)&__pos_on_tplane_e_y_E2C_stride, (void *)&__predictor_normal_wind_advective_tendency_K_stride, (void *)&__primal_normal_cell_x_E2C_stride, (void *)&__primal_normal_cell_y_E2C_stride, (void *)&__reference_rho_at_edges_on_model_levels_K_stride, (void *)&__reference_theta_at_edges_on_model_levels_K_stride, (void *)&__rho_at_edges_on_model_levels_K_stride, (void *)&__tangential_wind_K_stride, (void *)&__temporal_extrapolation_of_perturbed_exner_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:426
    gpuError_t __err = hipLaunchKernel((void*)map_100_fieldop_1_1_0_0_426, dim3(233, 24, 1), dim3(256, 1, 1), map_100_fieldop_1_1_0_0_426_args, 0, nullptr);    ////__DACE:0:0:426
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_100_fieldop_1_1_0_0_426", 233, 24, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_0_fieldop_0_0_418(const double * __restrict__ current_vn, const double * __restrict__ grf_tend_vn, double * __restrict__ next_vn, double * __restrict__ rho_at_edges_on_model_levels, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __grf_tend_vn_K_stride, int __next_vn_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:418
    {                                                                                 ////__DACE:0:0:418
        {                                                                             ////__DACE:0:0:418
            int b_i_Edge_gtx_horizontal = (256 * blockIdx.x);                         ////__DACE:0:0:418
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:418
            {                                                                         ////__DACE:0:0:2
                {                                                                     ////__DACE:0:0:2
                    {                                                                 ////__DACE:0:0:2
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:2
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:2
                        double gtir_tmp_0_0;                                          ////__DACE:0:0:294
                        double gtir_tmp_212_0;                                        ////__DACE:0:0:336
                        double __dtime_0;                                             ////__DACE:0:0:352
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(5386, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:2
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(29, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:2
                                {                                                     ////__DACE:0:0:293
                                    double __tlet_out;                                                                ////__DACE:0:0:293    ////__DACE:0:0:293
                                    ////__DACE:0:0:293                                ////__DACE:0:0:293
                                    ///////////////////                                                               ////__DACE:0:0:293    ////__DACE:0:0:293
                                    // Tasklet code (tlet_0_get_value__clone_0)                                       ////__DACE:0:0:293    ////__DACE:0:0:293
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:293    ////__DACE:0:0:293
                                    ///////////////////                                                               ////__DACE:0:0:293    ////__DACE:0:0:293
                                    ////__DACE:0:0:293                                ////__DACE:0:0:293
                                    gtir_tmp_0_0 = __tlet_out;                                                        ////__DACE:0:0:293    ////__DACE:0:0:293
                                }                                                     ////__DACE:0:0:293
                                {                                                     ////__DACE:0:0:335
                                    double __tlet_out;                                                                ////__DACE:0:0:335    ////__DACE:0:0:335
                                    ////__DACE:0:0:335                                ////__DACE:0:0:335
                                    ///////////////////                                                               ////__DACE:0:0:335    ////__DACE:0:0:335
                                    // Tasklet code (tlet_86_get_value__clone_0)                                      ////__DACE:0:0:335    ////__DACE:0:0:335
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:335    ////__DACE:0:0:335
                                    ///////////////////                                                               ////__DACE:0:0:335    ////__DACE:0:0:335
                                    ////__DACE:0:0:335                                ////__DACE:0:0:335
                                    gtir_tmp_212_0 = __tlet_out;                                                      ////__DACE:0:0:335    ////__DACE:0:0:335
                                }                                                     ////__DACE:0:0:335
                                {                                                     ////__DACE:0:0:351
                                    double __tlet_out;                                                                ////__DACE:0:0:351    ////__DACE:0:0:351
                                    ////__DACE:0:0:351                                ////__DACE:0:0:351
                                    ///////////////////                                                               ////__DACE:0:0:351    ////__DACE:0:0:351
                                    // Tasklet code (tlet_88_get_value__clone_0)                                      ////__DACE:0:0:351    ////__DACE:0:0:351
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:351    ////__DACE:0:0:351
                                    ///////////////////                                                               ////__DACE:0:0:351    ////__DACE:0:0:351
                                    ////__DACE:0:0:351                                ////__DACE:0:0:351
                                    __dtime_0 = __tlet_out;                                                           ////__DACE:0:0:351    ////__DACE:0:0:351
                                }                                                     ////__DACE:0:0:351
                                {                                                     ////__DACE:0:0:283
                                    #pragma unroll 4                                  ////__DACE:0:0:283
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:283
                                        double __map_fusion_gtir_tmp_220_0;           ////__DACE:0:0:153
                                        {                                             ////__DACE:0:0:152
                                            double __tlet_arg0 = __dtime_0;                                                   ////__DACE:0:0:352,152    ////__DACE:0:0:152
                                            double __tlet_arg1 = grf_tend_vn[((__grf_tend_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:67,152    ////__DACE:0:0:152
                                            double __tlet_result;                                                             ////__DACE:0:0:152    ////__DACE:0:0:152
                                            ////__DACE:0:0:152                        ////__DACE:0:0:152
                                            ///////////////////                                                               ////__DACE:0:0:152    ////__DACE:0:0:152
                                            // Tasklet code (tlet_89_multiplies_0)                                            ////__DACE:0:0:152    ////__DACE:0:0:152
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:152    ////__DACE:0:0:152
                                            ///////////////////                                                               ////__DACE:0:0:152    ////__DACE:0:0:152
                                            ////__DACE:0:0:152                        ////__DACE:0:0:152
                                            __map_fusion_gtir_tmp_220_0 = __tlet_result;                                      ////__DACE:0:0:152    ////__DACE:0:0:152
                                        }                                             ////__DACE:0:0:152
                                        {                                             ////__DACE:0:0:151
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_220_0;                                 ////__DACE:0:0:153,151    ////__DACE:0:0:151
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,151    ////__DACE:0:0:151
                                            double __tlet_result;                                                             ////__DACE:0:0:151    ////__DACE:0:0:151
                                            ////__DACE:0:0:151                        ////__DACE:0:0:151
                                            ///////////////////                                                               ////__DACE:0:0:151    ////__DACE:0:0:151
                                            // Tasklet code (tlet_90_plus_0)                                                  ////__DACE:0:0:151    ////__DACE:0:0:151
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:151    ////__DACE:0:0:151
                                            ///////////////////                                                               ////__DACE:0:0:151    ////__DACE:0:0:151
                                            ////__DACE:0:0:151                        ////__DACE:0:0:151
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:151    ////__DACE:0:0:151
                                        }                                             ////__DACE:0:0:151
                                        {                                             ////__DACE:0:0:1
                                            double __tlet_inp = gtir_tmp_0_0;                                                 ////__DACE:0:0:294,1    ////__DACE:0:0:1
                                            double __tlet_out;                                                                ////__DACE:0:0:1    ////__DACE:0:0:1
                                            ////__DACE:0:0:1                          ////__DACE:0:0:1
                                            ///////////////////                                                               ////__DACE:0:0:1    ////__DACE:0:0:1
                                            // Tasklet code (tlet_1_copy)                                                     ////__DACE:0:0:1    ////__DACE:0:0:1
                                            __tlet_out = __tlet_inp;                                                          ////__DACE:0:0:1    ////__DACE:0:0:1
                                            ///////////////////                                                               ////__DACE:0:0:1    ////__DACE:0:0:1
                                            ////__DACE:0:0:1                          ////__DACE:0:0:1
                                            theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_out;    ////__DACE:0:0:1    ////__DACE:0:0:1
                                        }                                             ////__DACE:0:0:1
                                        {                                             ////__DACE:0:0:63
                                            double __tlet_inp = gtir_tmp_212_0;                                               ////__DACE:0:0:336,63    ////__DACE:0:0:63
                                            double __tlet_out;                                                                ////__DACE:0:0:63    ////__DACE:0:0:63
                                            ////__DACE:0:0:63                         ////__DACE:0:0:63
                                            ///////////////////                                                               ////__DACE:0:0:63    ////__DACE:0:0:63
                                            // Tasklet code (tlet_87_copy)                                                    ////__DACE:0:0:63    ////__DACE:0:0:63
                                            __tlet_out = __tlet_inp;                                                          ////__DACE:0:0:63    ////__DACE:0:0:63
                                            ///////////////////                                                               ////__DACE:0:0:63    ////__DACE:0:0:63
                                            ////__DACE:0:0:63                         ////__DACE:0:0:63
                                            rho_at_edges_on_model_levels[((__rho_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_out;    ////__DACE:0:0:63    ////__DACE:0:0:63
                                        }                                             ////__DACE:0:0:63
                                    }                                                 ////__DACE:0:0:284
                                }                                                     ////__DACE:0:0:284
                            }                                                         ////__DACE:0:0:0
                        }                                                             ////__DACE:0:0:0
                    }                                                                 ////__DACE:0:0:0
                }                                                                     ////__DACE:0:0:0
            }                                                                         ////__DACE:0:0:0
        }                                                                             ////__DACE:0:0:419
    }                                                                                 ////__DACE:0:0:419
}                                                                                 ////__DACE:0:0:419

                                                                                  ////__DACE:0:0:418
DACE_EXPORTED void __dace_runkernel_map_0_fieldop_0_0_418(theta_shared_probe_shared_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ grf_tend_vn, double * __restrict__ next_vn, double * __restrict__ rho_at_edges_on_model_levels, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __grf_tend_vn_K_stride, int __next_vn_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:418
void __dace_runkernel_map_0_fieldop_0_0_418(theta_shared_probe_shared_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ grf_tend_vn, double * __restrict__ next_vn, double * __restrict__ rho_at_edges_on_model_levels, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __grf_tend_vn_K_stride, int __next_vn_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:418
{                                                                                 ////__DACE:0:0:418
                                                                                  ////__DACE:0:0:418
    void  *map_0_fieldop_0_0_418_args[] = { (void *)&current_vn, (void *)&grf_tend_vn, (void *)&next_vn, (void *)&rho_at_edges_on_model_levels, (void *)&theta_v_at_edges_on_model_levels, (void *)&__current_vn_K_stride, (void *)&__grf_tend_vn_K_stride, (void *)&__next_vn_K_stride, (void *)&__rho_at_edges_on_model_levels_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:418
    gpuError_t __err = hipLaunchKernel((void*)map_0_fieldop_0_0_418, dim3(22, 30, 1), dim3(256, 1, 1), map_0_fieldop_0_0_418_args, 0, nullptr);    ////__DACE:0:0:418
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_0_fieldop_0_0_418", 22, 30, 1, 256, 1, 1);
}
__global__ void  __launch_bounds__(256) map_100_fieldop_0_0_0_422(const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const double * __restrict__ grf_tend_vn, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ next_vn, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __grf_tend_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __next_vn_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime) {    ////__DACE:0:0:422
    {                                                                                 ////__DACE:0:0:422
        {                                                                             ////__DACE:0:0:422
            int b_i_Edge_gtx_horizontal = ((256 * blockIdx.x) + 5387);                ////__DACE:0:0:422
            int b___gtx_coarse_i_K_gtx_vertical = blockIdx.y;                         ////__DACE:0:0:422
            {                                                                         ////__DACE:0:0:134
                {                                                                     ////__DACE:0:0:134
                    {                                                                 ////__DACE:0:0:134
                        int i_Edge_gtx_horizontal = (threadIdx.x + b_i_Edge_gtx_horizontal);    ////__DACE:0:0:134
                        int __gtx_coarse_i_K_gtx_vertical = (threadIdx.y + b___gtx_coarse_i_K_gtx_vertical);    ////__DACE:0:0:134
                        double gtir_tmp_14_0;                                         ////__DACE:0:0:114
                        double gtir_tmp_12_0;                                         ////__DACE:0:0:115
                        double gtir_tmp_26_0;                                         ////__DACE:0:0:120
                        double gtir_tmp_24_0;                                         ////__DACE:0:0:121
                        double gtir_tmp_70_0;                                         ////__DACE:0:0:125
                        double gtir_tmp_66_0;                                         ////__DACE:0:0:126
                        double gtir_tmp_60_0;                                         ////__DACE:0:0:127
                        double gtir_tmp_56_0;                                         ////__DACE:0:0:128
                        double gtir_tmp_48_0;                                         ////__DACE:0:0:130
                        double gtir_tmp_44_0;                                         ////__DACE:0:0:131
                        double gtir_tmp_38_0;                                         ////__DACE:0:0:132
                        double gtir_tmp_34_0;                                         ////__DACE:0:0:133
                        bool gtir_tmp_7_1;                                            ////__DACE:0:0:298
                        bool gtir_tmp_6_1;                                            ////__DACE:0:0:304
                        double gtir_tmp_3_0;                                          ////__DACE:0:0:308
                        double __p_dthalf_2;                                          ////__DACE:0:0:318
                        double lambda_4___p_dthalf_1;                                 ////__DACE:0:0:322
                        double gtir_tmp_102_1;                                        ////__DACE:0:0:328
                        double gtir_tmp_175_1;                                        ////__DACE:0:0:340
                        double __dtime_1;                                             ////__DACE:0:0:354
                        if (i_Edge_gtx_horizontal >= b_i_Edge_gtx_horizontal && i_Edge_gtx_horizontal < (Min(7700, (b_i_Edge_gtx_horizontal + 255)) + 1)) {    ////__DACE:0:0:134
                            if (__gtx_coarse_i_K_gtx_vertical >= b___gtx_coarse_i_K_gtx_vertical && __gtx_coarse_i_K_gtx_vertical < (Min(29, b___gtx_coarse_i_K_gtx_vertical) + 1)) {    ////__DACE:0:0:134
                                {                                                     ////__DACE:0:0:297
                                    bool __tlet_out;                                                                  ////__DACE:0:0:297    ////__DACE:0:0:297
                                    ////__DACE:0:0:297                                ////__DACE:0:0:297
                                    ///////////////////                                                               ////__DACE:0:0:297    ////__DACE:0:0:297
                                    // Tasklet code (tlet_5_get_value__clone_1)                                       ////__DACE:0:0:297    ////__DACE:0:0:297
                                    __tlet_out = false;                                                               ////__DACE:0:0:297    ////__DACE:0:0:297
                                    ///////////////////                                                               ////__DACE:0:0:297    ////__DACE:0:0:297
                                    ////__DACE:0:0:297                                ////__DACE:0:0:297
                                    gtir_tmp_7_1 = __tlet_out;                                                        ////__DACE:0:0:297    ////__DACE:0:0:297
                                }                                                     ////__DACE:0:0:297
                                {                                                     ////__DACE:0:0:303
                                    bool __tlet_out;                                                                  ////__DACE:0:0:303    ////__DACE:0:0:303
                                    ////__DACE:0:0:303                                ////__DACE:0:0:303
                                    ///////////////////                                                               ////__DACE:0:0:303    ////__DACE:0:0:303
                                    // Tasklet code (tlet_4_get_value__clone_1)                                       ////__DACE:0:0:303    ////__DACE:0:0:303
                                    __tlet_out = true;                                                                ////__DACE:0:0:303    ////__DACE:0:0:303
                                    ///////////////////                                                               ////__DACE:0:0:303    ////__DACE:0:0:303
                                    ////__DACE:0:0:303                                ////__DACE:0:0:303
                                    gtir_tmp_6_1 = __tlet_out;                                                        ////__DACE:0:0:303    ////__DACE:0:0:303
                                }                                                     ////__DACE:0:0:303
                                {                                                     ////__DACE:0:0:307
                                    double __tlet_out;                                                                ////__DACE:0:0:307    ////__DACE:0:0:307
                                    ////__DACE:0:0:307                                ////__DACE:0:0:307
                                    ///////////////////                                                               ////__DACE:0:0:307    ////__DACE:0:0:307
                                    // Tasklet code (tlet_2_get_value__clone_0)                                       ////__DACE:0:0:307    ////__DACE:0:0:307
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:307    ////__DACE:0:0:307
                                    ///////////////////                                                               ////__DACE:0:0:307    ////__DACE:0:0:307
                                    ////__DACE:0:0:307                                ////__DACE:0:0:307
                                    gtir_tmp_3_0 = __tlet_out;                                                        ////__DACE:0:0:307    ////__DACE:0:0:307
                                }                                                     ////__DACE:0:0:307
                                {                                                     ////__DACE:0:0:317
                                    double __tlet_out;                                                                ////__DACE:0:0:317    ////__DACE:0:0:317
                                    ////__DACE:0:0:317                                ////__DACE:0:0:317
                                    ///////////////////                                                               ////__DACE:0:0:317    ////__DACE:0:0:317
                                    // Tasklet code (tlet_6_get_value__clone_2)                                       ////__DACE:0:0:317    ////__DACE:0:0:317
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:317    ////__DACE:0:0:317
                                    ///////////////////                                                               ////__DACE:0:0:317    ////__DACE:0:0:317
                                    ////__DACE:0:0:317                                ////__DACE:0:0:317
                                    __p_dthalf_2 = __tlet_out;                                                        ////__DACE:0:0:317    ////__DACE:0:0:317
                                }                                                     ////__DACE:0:0:317
                                {                                                     ////__DACE:0:0:321
                                    double __tlet_out;                                                                ////__DACE:0:0:321    ////__DACE:0:0:321
                                    ////__DACE:0:0:321                                ////__DACE:0:0:321
                                    ///////////////////                                                               ////__DACE:0:0:321    ////__DACE:0:0:321
                                    // Tasklet code (tlet_10_get_value__clone_1)                                      ////__DACE:0:0:321    ////__DACE:0:0:321
                                    __tlet_out = (0.5 * dtime);                                                       ////__DACE:0:0:321    ////__DACE:0:0:321
                                    ///////////////////                                                               ////__DACE:0:0:321    ////__DACE:0:0:321
                                    ////__DACE:0:0:321                                ////__DACE:0:0:321
                                    lambda_4___p_dthalf_1 = __tlet_out;                                               ////__DACE:0:0:321    ////__DACE:0:0:321
                                }                                                     ////__DACE:0:0:321
                                {                                                     ////__DACE:0:0:327
                                    double __tlet_out;                                                                ////__DACE:0:0:327    ////__DACE:0:0:327
                                    ////__DACE:0:0:327                                ////__DACE:0:0:327
                                    ///////////////////                                                               ////__DACE:0:0:327    ////__DACE:0:0:327
                                    // Tasklet code (tlet_34_get_value__clone_1)                                      ////__DACE:0:0:327    ////__DACE:0:0:327
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:327    ////__DACE:0:0:327
                                    ///////////////////                                                               ////__DACE:0:0:327    ////__DACE:0:0:327
                                    ////__DACE:0:0:327                                ////__DACE:0:0:327
                                    gtir_tmp_102_1 = __tlet_out;                                                      ////__DACE:0:0:327    ////__DACE:0:0:327
                                }                                                     ////__DACE:0:0:327
                                {                                                     ////__DACE:0:0:339
                                    double __tlet_out;                                                                ////__DACE:0:0:339    ////__DACE:0:0:339
                                    ////__DACE:0:0:339                                ////__DACE:0:0:339
                                    ///////////////////                                                               ////__DACE:0:0:339    ////__DACE:0:0:339
                                    // Tasklet code (tlet_68_get_value__clone_1)                                      ////__DACE:0:0:339    ////__DACE:0:0:339
                                    __tlet_out = 0.0;                                                                 ////__DACE:0:0:339    ////__DACE:0:0:339
                                    ///////////////////                                                               ////__DACE:0:0:339    ////__DACE:0:0:339
                                    ////__DACE:0:0:339                                ////__DACE:0:0:339
                                    gtir_tmp_175_1 = __tlet_out;                                                      ////__DACE:0:0:339    ////__DACE:0:0:339
                                }                                                     ////__DACE:0:0:339
                                {                                                     ////__DACE:0:0:353
                                    double __tlet_out;                                                                ////__DACE:0:0:353    ////__DACE:0:0:353
                                    ////__DACE:0:0:353                                ////__DACE:0:0:353
                                    ///////////////////                                                               ////__DACE:0:0:353    ////__DACE:0:0:353
                                    // Tasklet code (tlet_88_get_value__clone_1)                                      ////__DACE:0:0:353    ////__DACE:0:0:353
                                    __tlet_out = dtime;                                                               ////__DACE:0:0:353    ////__DACE:0:0:353
                                    ///////////////////                                                               ////__DACE:0:0:353    ////__DACE:0:0:353
                                    ////__DACE:0:0:353                                ////__DACE:0:0:353
                                    __dtime_1 = __tlet_out;                                                           ////__DACE:0:0:353    ////__DACE:0:0:353
                                }                                                     ////__DACE:0:0:353
                                {                                                     ////__DACE:0:0:368
                                    double _cpy_in = dual_normal_cell_x[i_Edge_gtx_horizontal];                       ////__DACE:0:0:9,368    ////__DACE:0:0:368
                                    double _cpy_out;                                                                  ////__DACE:0:0:368    ////__DACE:0:0:368
                                    ////__DACE:0:0:368                                ////__DACE:0:0:368
                                    ///////////////////                                                               ////__DACE:0:0:368    ////__DACE:0:0:368
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_38_0)                        ////__DACE:0:0:368    ////__DACE:0:0:368
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:368    ////__DACE:0:0:368
                                    ///////////////////                                                               ////__DACE:0:0:368    ////__DACE:0:0:368
                                    ////__DACE:0:0:368                                ////__DACE:0:0:368
                                    gtir_tmp_38_0 = _cpy_out;                                                         ////__DACE:0:0:368    ////__DACE:0:0:368
                                }                                                     ////__DACE:0:0:368
                                {                                                     ////__DACE:0:0:369
                                    double _cpy_in = dual_normal_cell_x[(__dual_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:9,369    ////__DACE:0:0:369
                                    double _cpy_out;                                                                  ////__DACE:0:0:369    ////__DACE:0:0:369
                                    ////__DACE:0:0:369                                ////__DACE:0:0:369
                                    ///////////////////                                                               ////__DACE:0:0:369    ////__DACE:0:0:369
                                    // Tasklet code (copy_dual_normal_cell_x_to_gtir_tmp_48_0)                        ////__DACE:0:0:369    ////__DACE:0:0:369
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:369    ////__DACE:0:0:369
                                    ///////////////////                                                               ////__DACE:0:0:369    ////__DACE:0:0:369
                                    ////__DACE:0:0:369                                ////__DACE:0:0:369
                                    gtir_tmp_48_0 = _cpy_out;                                                         ////__DACE:0:0:369    ////__DACE:0:0:369
                                }                                                     ////__DACE:0:0:369
                                {                                                     ////__DACE:0:0:370
                                    double _cpy_in = dual_normal_cell_y[i_Edge_gtx_horizontal];                       ////__DACE:0:0:7,370    ////__DACE:0:0:370
                                    double _cpy_out;                                                                  ////__DACE:0:0:370    ////__DACE:0:0:370
                                    ////__DACE:0:0:370                                ////__DACE:0:0:370
                                    ///////////////////                                                               ////__DACE:0:0:370    ////__DACE:0:0:370
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_60_0)                        ////__DACE:0:0:370    ////__DACE:0:0:370
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:370    ////__DACE:0:0:370
                                    ///////////////////                                                               ////__DACE:0:0:370    ////__DACE:0:0:370
                                    ////__DACE:0:0:370                                ////__DACE:0:0:370
                                    gtir_tmp_60_0 = _cpy_out;                                                         ////__DACE:0:0:370    ////__DACE:0:0:370
                                }                                                     ////__DACE:0:0:370
                                {                                                     ////__DACE:0:0:371
                                    double _cpy_in = dual_normal_cell_y[(__dual_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:7,371    ////__DACE:0:0:371
                                    double _cpy_out;                                                                  ////__DACE:0:0:371    ////__DACE:0:0:371
                                    ////__DACE:0:0:371                                ////__DACE:0:0:371
                                    ///////////////////                                                               ////__DACE:0:0:371    ////__DACE:0:0:371
                                    // Tasklet code (copy_dual_normal_cell_y_to_gtir_tmp_70_0)                        ////__DACE:0:0:371    ////__DACE:0:0:371
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:371    ////__DACE:0:0:371
                                    ///////////////////                                                               ////__DACE:0:0:371    ////__DACE:0:0:371
                                    ////__DACE:0:0:371                                ////__DACE:0:0:371
                                    gtir_tmp_70_0 = _cpy_out;                                                         ////__DACE:0:0:371    ////__DACE:0:0:371
                                }                                                     ////__DACE:0:0:371
                                {                                                     ////__DACE:0:0:372
                                    double _cpy_in = pos_on_tplane_e_x[i_Edge_gtx_horizontal];                        ////__DACE:0:0:4,372    ////__DACE:0:0:372
                                    double _cpy_out;                                                                  ////__DACE:0:0:372    ////__DACE:0:0:372
                                    ////__DACE:0:0:372                                ////__DACE:0:0:372
                                    ///////////////////                                                               ////__DACE:0:0:372    ////__DACE:0:0:372
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_12_0)                         ////__DACE:0:0:372    ////__DACE:0:0:372
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:372    ////__DACE:0:0:372
                                    ///////////////////                                                               ////__DACE:0:0:372    ////__DACE:0:0:372
                                    ////__DACE:0:0:372                                ////__DACE:0:0:372
                                    gtir_tmp_12_0 = _cpy_out;                                                         ////__DACE:0:0:372    ////__DACE:0:0:372
                                }                                                     ////__DACE:0:0:372
                                {                                                     ////__DACE:0:0:373
                                    double _cpy_in = pos_on_tplane_e_x[(__pos_on_tplane_e_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:4,373    ////__DACE:0:0:373
                                    double _cpy_out;                                                                  ////__DACE:0:0:373    ////__DACE:0:0:373
                                    ////__DACE:0:0:373                                ////__DACE:0:0:373
                                    ///////////////////                                                               ////__DACE:0:0:373    ////__DACE:0:0:373
                                    // Tasklet code (copy_pos_on_tplane_e_x_to_gtir_tmp_14_0)                         ////__DACE:0:0:373    ////__DACE:0:0:373
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:373    ////__DACE:0:0:373
                                    ///////////////////                                                               ////__DACE:0:0:373    ////__DACE:0:0:373
                                    ////__DACE:0:0:373                                ////__DACE:0:0:373
                                    gtir_tmp_14_0 = _cpy_out;                                                         ////__DACE:0:0:373    ////__DACE:0:0:373
                                }                                                     ////__DACE:0:0:373
                                {                                                     ////__DACE:0:0:374
                                    double _cpy_in = pos_on_tplane_e_y[i_Edge_gtx_horizontal];                        ////__DACE:0:0:5,374    ////__DACE:0:0:374
                                    double _cpy_out;                                                                  ////__DACE:0:0:374    ////__DACE:0:0:374
                                    ////__DACE:0:0:374                                ////__DACE:0:0:374
                                    ///////////////////                                                               ////__DACE:0:0:374    ////__DACE:0:0:374
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_24_0)                         ////__DACE:0:0:374    ////__DACE:0:0:374
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:374    ////__DACE:0:0:374
                                    ///////////////////                                                               ////__DACE:0:0:374    ////__DACE:0:0:374
                                    ////__DACE:0:0:374                                ////__DACE:0:0:374
                                    gtir_tmp_24_0 = _cpy_out;                                                         ////__DACE:0:0:374    ////__DACE:0:0:374
                                }                                                     ////__DACE:0:0:374
                                {                                                     ////__DACE:0:0:375
                                    double _cpy_in = pos_on_tplane_e_y[(__pos_on_tplane_e_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:5,375    ////__DACE:0:0:375
                                    double _cpy_out;                                                                  ////__DACE:0:0:375    ////__DACE:0:0:375
                                    ////__DACE:0:0:375                                ////__DACE:0:0:375
                                    ///////////////////                                                               ////__DACE:0:0:375    ////__DACE:0:0:375
                                    // Tasklet code (copy_pos_on_tplane_e_y_to_gtir_tmp_26_0)                         ////__DACE:0:0:375    ////__DACE:0:0:375
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:375    ////__DACE:0:0:375
                                    ///////////////////                                                               ////__DACE:0:0:375    ////__DACE:0:0:375
                                    ////__DACE:0:0:375                                ////__DACE:0:0:375
                                    gtir_tmp_26_0 = _cpy_out;                                                         ////__DACE:0:0:375    ////__DACE:0:0:375
                                }                                                     ////__DACE:0:0:375
                                {                                                     ////__DACE:0:0:376
                                    double _cpy_in = primal_normal_cell_x[i_Edge_gtx_horizontal];                     ////__DACE:0:0:10,376    ////__DACE:0:0:376
                                    double _cpy_out;                                                                  ////__DACE:0:0:376    ////__DACE:0:0:376
                                    ////__DACE:0:0:376                                ////__DACE:0:0:376
                                    ///////////////////                                                               ////__DACE:0:0:376    ////__DACE:0:0:376
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_34_0)                      ////__DACE:0:0:376    ////__DACE:0:0:376
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:376    ////__DACE:0:0:376
                                    ///////////////////                                                               ////__DACE:0:0:376    ////__DACE:0:0:376
                                    ////__DACE:0:0:376                                ////__DACE:0:0:376
                                    gtir_tmp_34_0 = _cpy_out;                                                         ////__DACE:0:0:376    ////__DACE:0:0:376
                                }                                                     ////__DACE:0:0:376
                                {                                                     ////__DACE:0:0:377
                                    double _cpy_in = primal_normal_cell_x[(__primal_normal_cell_x_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:10,377    ////__DACE:0:0:377
                                    double _cpy_out;                                                                  ////__DACE:0:0:377    ////__DACE:0:0:377
                                    ////__DACE:0:0:377                                ////__DACE:0:0:377
                                    ///////////////////                                                               ////__DACE:0:0:377    ////__DACE:0:0:377
                                    // Tasklet code (copy_primal_normal_cell_x_to_gtir_tmp_44_0)                      ////__DACE:0:0:377    ////__DACE:0:0:377
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:377    ////__DACE:0:0:377
                                    ///////////////////                                                               ////__DACE:0:0:377    ////__DACE:0:0:377
                                    ////__DACE:0:0:377                                ////__DACE:0:0:377
                                    gtir_tmp_44_0 = _cpy_out;                                                         ////__DACE:0:0:377    ////__DACE:0:0:377
                                }                                                     ////__DACE:0:0:377
                                {                                                     ////__DACE:0:0:378
                                    double _cpy_in = primal_normal_cell_y[i_Edge_gtx_horizontal];                     ////__DACE:0:0:8,378    ////__DACE:0:0:378
                                    double _cpy_out;                                                                  ////__DACE:0:0:378    ////__DACE:0:0:378
                                    ////__DACE:0:0:378                                ////__DACE:0:0:378
                                    ///////////////////                                                               ////__DACE:0:0:378    ////__DACE:0:0:378
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_56_0)                      ////__DACE:0:0:378    ////__DACE:0:0:378
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:378    ////__DACE:0:0:378
                                    ///////////////////                                                               ////__DACE:0:0:378    ////__DACE:0:0:378
                                    ////__DACE:0:0:378                                ////__DACE:0:0:378
                                    gtir_tmp_56_0 = _cpy_out;                                                         ////__DACE:0:0:378    ////__DACE:0:0:378
                                }                                                     ////__DACE:0:0:378
                                {                                                     ////__DACE:0:0:379
                                    double _cpy_in = primal_normal_cell_y[(__primal_normal_cell_y_E2C_stride + i_Edge_gtx_horizontal)];    ////__DACE:0:0:8,379    ////__DACE:0:0:379
                                    double _cpy_out;                                                                  ////__DACE:0:0:379    ////__DACE:0:0:379
                                    ////__DACE:0:0:379                                ////__DACE:0:0:379
                                    ///////////////////                                                               ////__DACE:0:0:379    ////__DACE:0:0:379
                                    // Tasklet code (copy_primal_normal_cell_y_to_gtir_tmp_66_0)                      ////__DACE:0:0:379    ////__DACE:0:0:379
                                    _cpy_out = _cpy_in;                                                               ////__DACE:0:0:379    ////__DACE:0:0:379
                                    ///////////////////                                                               ////__DACE:0:0:379    ////__DACE:0:0:379
                                    ////__DACE:0:0:379                                ////__DACE:0:0:379
                                    gtir_tmp_66_0 = _cpy_out;                                                         ////__DACE:0:0:379    ////__DACE:0:0:379
                                }                                                     ////__DACE:0:0:379
                                {                                                     ////__DACE:0:0:287
                                    #pragma unroll 4                                  ////__DACE:0:0:287
                                    for (auto i_K_gtx_vertical = (4 * __gtx_coarse_i_K_gtx_vertical); i_K_gtx_vertical < Min(120, ((4 * __gtx_coarse_i_K_gtx_vertical) + 4)); i_K_gtx_vertical += 1) {    ////__DACE:0:0:287
                                        bool gtir_tmp_8_0;                            ////__DACE:0:0:107
                                        double gtir_tmp_16_0;                         ////__DACE:0:0:112
                                        double gtir_tmp_28_0;                         ////__DACE:0:0:119
                                        double gtir_tmp_76_0;                         ////__DACE:0:0:123
                                        double gtir_tmp_54_0;                         ////__DACE:0:0:129
                                        double gtir_tmp_137_0;                        ////__DACE:0:0:135
                                        double gtir_tmp_210_0;                        ////__DACE:0:0:139
                                        bool __map_fusion_gtir_tmp_5_0;               ////__DACE:0:0:142
                                        double __map_fusion_gtir_tmp_21_0_0;          ////__DACE:0:0:143
                                        double __map_fusion_gtir_tmp_19_0;            ////__DACE:0:0:144
                                        double __map_fusion_gtir_tmp_11_0;            ////__DACE:0:0:145
                                        double __map_fusion_gtir_tmp_33_0_0;          ////__DACE:0:0:146
                                        double __map_fusion_gtir_tmp_31_0;            ////__DACE:0:0:147
                                        double __map_fusion_gtir_tmp_23_0;            ////__DACE:0:0:148
                                        bool __map_fusion_gtir_tmp_104_0;             ////__DACE:0:0:149
                                        bool __map_fusion_gtir_tmp_177_0;             ////__DACE:0:0:150
                                        double __map_fusion_gtir_tmp_220_1;           ////__DACE:0:0:156
                                        {                                             ////__DACE:0:0:116
                                            double __tlet_arg1 = __p_dthalf_2;                                                ////__DACE:0:0:318,116    ////__DACE:0:0:116
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,116    ////__DACE:0:0:116
                                            double __tlet_result;                                                             ////__DACE:0:0:116    ////__DACE:0:0:116
                                            ////__DACE:0:0:116                        ////__DACE:0:0:116
                                            ///////////////////                                                               ////__DACE:0:0:116    ////__DACE:0:0:116
                                            // Tasklet code (tlet_7_multiplies_0)                                             ////__DACE:0:0:116    ////__DACE:0:0:116
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:116    ////__DACE:0:0:116
                                            ///////////////////                                                               ////__DACE:0:0:116    ////__DACE:0:0:116
                                            ////__DACE:0:0:116                        ////__DACE:0:0:116
                                            __map_fusion_gtir_tmp_11_0 = __tlet_result;                                       ////__DACE:0:0:116    ////__DACE:0:0:116
                                        }                                             ////__DACE:0:0:116
                                        {                                             ////__DACE:0:0:155
                                            double __tlet_arg0 = __dtime_1;                                                   ////__DACE:0:0:354,155    ////__DACE:0:0:155
                                            double __tlet_arg1 = grf_tend_vn[((__grf_tend_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:67,155    ////__DACE:0:0:155
                                            double __tlet_result;                                                             ////__DACE:0:0:155    ////__DACE:0:0:155
                                            ////__DACE:0:0:155                        ////__DACE:0:0:155
                                            ///////////////////                                                               ////__DACE:0:0:155    ////__DACE:0:0:155
                                            // Tasklet code (tlet_89_multiplies_1)                                            ////__DACE:0:0:155    ////__DACE:0:0:155
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:155    ////__DACE:0:0:155
                                            ///////////////////                                                               ////__DACE:0:0:155    ////__DACE:0:0:155
                                            ////__DACE:0:0:155                        ////__DACE:0:0:155
                                            __map_fusion_gtir_tmp_220_1 = __tlet_result;                                      ////__DACE:0:0:155    ////__DACE:0:0:155
                                        }                                             ////__DACE:0:0:155
                                        {                                             ////__DACE:0:0:154
                                            double __tlet_arg1 = __map_fusion_gtir_tmp_220_1;                                 ////__DACE:0:0:156,154    ////__DACE:0:0:154
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,154    ////__DACE:0:0:154
                                            double __tlet_result;                                                             ////__DACE:0:0:154    ////__DACE:0:0:154
                                            ////__DACE:0:0:154                        ////__DACE:0:0:154
                                            ///////////////////                                                               ////__DACE:0:0:154    ////__DACE:0:0:154
                                            // Tasklet code (tlet_90_plus_1)                                                  ////__DACE:0:0:154    ////__DACE:0:0:154
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:154    ////__DACE:0:0:154
                                            ///////////////////                                                               ////__DACE:0:0:154    ////__DACE:0:0:154
                                            ////__DACE:0:0:154                        ////__DACE:0:0:154
                                            next_vn[((__next_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = __tlet_result;    ////__DACE:0:0:154    ////__DACE:0:0:154
                                        }                                             ////__DACE:0:0:154
                                        {                                             ////__DACE:0:0:122
                                            double __tlet_arg1 = lambda_4___p_dthalf_1;                                       ////__DACE:0:0:322,122    ////__DACE:0:0:122
                                            double __tlet_arg0 = tangential_wind[((__tangential_wind_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:6,122    ////__DACE:0:0:122
                                            double __tlet_result;                                                             ////__DACE:0:0:122    ////__DACE:0:0:122
                                            ////__DACE:0:0:122                        ////__DACE:0:0:122
                                            ///////////////////                                                               ////__DACE:0:0:122    ////__DACE:0:0:122
                                            // Tasklet code (tlet_11_multiplies_0)                                            ////__DACE:0:0:122    ////__DACE:0:0:122
                                            __tlet_result = (__tlet_arg0 * __tlet_arg1);                                      ////__DACE:0:0:122    ////__DACE:0:0:122
                                            ///////////////////                                                               ////__DACE:0:0:122    ////__DACE:0:0:122
                                            ////__DACE:0:0:122                        ////__DACE:0:0:122
                                            __map_fusion_gtir_tmp_23_0 = __tlet_result;                                       ////__DACE:0:0:122    ////__DACE:0:0:122
                                        }                                             ////__DACE:0:0:122
                                        {                                             ////__DACE:0:0:141
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,141    ////__DACE:0:0:141
                                            double __tlet_arg1 = gtir_tmp_175_1;                                              ////__DACE:0:0:340,141    ////__DACE:0:0:141
                                            bool __tlet_result;                                                               ////__DACE:0:0:141    ////__DACE:0:0:141
                                            ////__DACE:0:0:141                        ////__DACE:0:0:141
                                            ///////////////////                                                               ////__DACE:0:0:141    ////__DACE:0:0:141
                                            // Tasklet code (tlet_69_greater_equal_0)                                         ////__DACE:0:0:141    ////__DACE:0:0:141
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:141    ////__DACE:0:0:141
                                            ///////////////////                                                               ////__DACE:0:0:141    ////__DACE:0:0:141
                                            ////__DACE:0:0:141                        ////__DACE:0:0:141
                                            __map_fusion_gtir_tmp_177_0 = __tlet_result;                                      ////__DACE:0:0:141    ////__DACE:0:0:141
                                        }                                             ////__DACE:0:0:141
                                        {                                             ////__DACE:0:0:109
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,109    ////__DACE:0:0:109
                                            double __tlet_arg1 = gtir_tmp_3_0;                                                ////__DACE:0:0:308,109    ////__DACE:0:0:109
                                            bool __tlet_result;                                                               ////__DACE:0:0:109    ////__DACE:0:0:109
                                            ////__DACE:0:0:109                        ////__DACE:0:0:109
                                            ///////////////////                                                               ////__DACE:0:0:109    ////__DACE:0:0:109
                                            // Tasklet code (tlet_3_greater_equal_0)                                          ////__DACE:0:0:109    ////__DACE:0:0:109
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:109    ////__DACE:0:0:109
                                            ///////////////////                                                               ////__DACE:0:0:109    ////__DACE:0:0:109
                                            ////__DACE:0:0:109                        ////__DACE:0:0:109
                                            __map_fusion_gtir_tmp_5_0 = __tlet_result;                                        ////__DACE:0:0:109    ////__DACE:0:0:109
                                        }                                             ////__DACE:0:0:109
                                        if_stmt_0_0_0_158(gtir_tmp_6_1, gtir_tmp_7_1, __map_fusion_gtir_tmp_5_0, gtir_tmp_8_0);    ////__DACE:0:0:108
                                        if_stmt_1_0_0_113(gtir_tmp_12_0, gtir_tmp_24_0, gtir_tmp_14_0, gtir_tmp_26_0, gtir_tmp_8_0, gtir_tmp_16_0, gtir_tmp_28_0);    ////__DACE:0:0:113
                                        {                                             ////__DACE:0:0:111
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_11_0;                                  ////__DACE:0:0:145,111    ////__DACE:0:0:111
                                            double __tlet_arg1 = gtir_tmp_16_0;                                               ////__DACE:0:0:112,111    ////__DACE:0:0:111
                                            double __tlet_result;                                                             ////__DACE:0:0:111    ////__DACE:0:0:111
                                            ////__DACE:0:0:111                        ////__DACE:0:0:111
                                            ///////////////////                                                               ////__DACE:0:0:111    ////__DACE:0:0:111
                                            // Tasklet code (tlet_8_plus_0)                                                   ////__DACE:0:0:111    ////__DACE:0:0:111
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:111    ////__DACE:0:0:111
                                            ///////////////////                                                               ////__DACE:0:0:111    ////__DACE:0:0:111
                                            ////__DACE:0:0:111                        ////__DACE:0:0:111
                                            __map_fusion_gtir_tmp_19_0 = __tlet_result;                                       ////__DACE:0:0:111    ////__DACE:0:0:111
                                        }                                             ////__DACE:0:0:111
                                        {                                             ////__DACE:0:0:110
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_19_0;                                  ////__DACE:0:0:144,110    ////__DACE:0:0:110
                                            double __tlet_result;                                                             ////__DACE:0:0:110    ////__DACE:0:0:110
                                            ////__DACE:0:0:110                        ////__DACE:0:0:110
                                            ///////////////////                                                               ////__DACE:0:0:110    ////__DACE:0:0:110
                                            // Tasklet code (tlet_9_neg_0)                                                    ////__DACE:0:0:110    ////__DACE:0:0:110
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:110    ////__DACE:0:0:110
                                            ///////////////////                                                               ////__DACE:0:0:110    ////__DACE:0:0:110
                                            ////__DACE:0:0:110                        ////__DACE:0:0:110
                                            __map_fusion_gtir_tmp_21_0_0 = __tlet_result;                                     ////__DACE:0:0:110    ////__DACE:0:0:110
                                        }                                             ////__DACE:0:0:110
                                        {                                             ////__DACE:0:0:118
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_23_0;                                  ////__DACE:0:0:148,118    ////__DACE:0:0:118
                                            double __tlet_arg1 = gtir_tmp_28_0;                                               ////__DACE:0:0:119,118    ////__DACE:0:0:118
                                            double __tlet_result;                                                             ////__DACE:0:0:118    ////__DACE:0:0:118
                                            ////__DACE:0:0:118                        ////__DACE:0:0:118
                                            ///////////////////                                                               ////__DACE:0:0:118    ////__DACE:0:0:118
                                            // Tasklet code (tlet_12_plus_0)                                                  ////__DACE:0:0:118    ////__DACE:0:0:118
                                            __tlet_result = (__tlet_arg0 + __tlet_arg1);                                      ////__DACE:0:0:118    ////__DACE:0:0:118
                                            ///////////////////                                                               ////__DACE:0:0:118    ////__DACE:0:0:118
                                            ////__DACE:0:0:118                        ////__DACE:0:0:118
                                            __map_fusion_gtir_tmp_31_0 = __tlet_result;                                       ////__DACE:0:0:118    ////__DACE:0:0:118
                                        }                                             ////__DACE:0:0:118
                                        {                                             ////__DACE:0:0:117
                                            double __tlet_arg0 = __map_fusion_gtir_tmp_31_0;                                  ////__DACE:0:0:147,117    ////__DACE:0:0:117
                                            double __tlet_result;                                                             ////__DACE:0:0:117    ////__DACE:0:0:117
                                            ////__DACE:0:0:117                        ////__DACE:0:0:117
                                            ///////////////////                                                               ////__DACE:0:0:117    ////__DACE:0:0:117
                                            // Tasklet code (tlet_13_neg_0)                                                   ////__DACE:0:0:117    ////__DACE:0:0:117
                                            __tlet_result = (- __tlet_arg0);                                                  ////__DACE:0:0:117    ////__DACE:0:0:117
                                            ///////////////////                                                               ////__DACE:0:0:117    ////__DACE:0:0:117
                                            ////__DACE:0:0:117                        ////__DACE:0:0:117
                                            __map_fusion_gtir_tmp_33_0_0 = __tlet_result;                                     ////__DACE:0:0:117    ////__DACE:0:0:117
                                        }                                             ////__DACE:0:0:117
                                        if_stmt_4_0_0_124(gtir_tmp_8_0, __map_fusion_gtir_tmp_21_0_0, __map_fusion_gtir_tmp_33_0_0, gtir_tmp_34_0, gtir_tmp_38_0, gtir_tmp_44_0, gtir_tmp_48_0, gtir_tmp_56_0, gtir_tmp_60_0, gtir_tmp_66_0, gtir_tmp_70_0, gtir_tmp_76_0, gtir_tmp_54_0);    ////__DACE:0:0:124
                                        if_stmt_7_0_0_140(__map_fusion_gtir_tmp_177_0, &gt_conn_E2C[0], gtir_tmp_54_0, gtir_tmp_76_0, &gtir_tmp_83[0], &gtir_tmp_89[0], &perturbed_rho_at_cells_on_model_levels[0], &reference_rho_at_edges_on_model_levels[0], gtir_tmp_210_0, __gt_conn_E2C_neighbor_stride, __perturbed_rho_at_cells_on_model_levels_K_stride, __reference_rho_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:140
                                        {                                             ////__DACE:0:0:411
                                            double _cpy_in = gtir_tmp_210_0;                                                  ////__DACE:0:0:139,411    ////__DACE:0:0:411
                                            double _cpy_out;                                                                  ////__DACE:0:0:411    ////__DACE:0:0:411
                                            ////__DACE:0:0:411                        ////__DACE:0:0:411
                                            ///////////////////                                                               ////__DACE:0:0:411    ////__DACE:0:0:411
                                            // Tasklet code (copy_gtir_tmp_210_0_to_rho_at_edges_on_model_levels)             ////__DACE:0:0:411    ////__DACE:0:0:411
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:411    ////__DACE:0:0:411
                                            ///////////////////                                                               ////__DACE:0:0:411    ////__DACE:0:0:411
                                            ////__DACE:0:0:411                        ////__DACE:0:0:411
                                            rho_at_edges_on_model_levels[((__rho_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:411    ////__DACE:0:0:411
                                        }                                             ////__DACE:0:0:411
                                        {                                             ////__DACE:0:0:137
                                            double __tlet_arg0 = current_vn[((__current_vn_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)];    ////__DACE:0:0:3,137    ////__DACE:0:0:137
                                            double __tlet_arg1 = gtir_tmp_102_1;                                              ////__DACE:0:0:328,137    ////__DACE:0:0:137
                                            bool __tlet_result;                                                               ////__DACE:0:0:137    ////__DACE:0:0:137
                                            ////__DACE:0:0:137                        ////__DACE:0:0:137
                                            ///////////////////                                                               ////__DACE:0:0:137    ////__DACE:0:0:137
                                            // Tasklet code (tlet_35_greater_equal_0)                                         ////__DACE:0:0:137    ////__DACE:0:0:137
                                            __tlet_result = (__tlet_arg0 >= __tlet_arg1);                                     ////__DACE:0:0:137    ////__DACE:0:0:137
                                            ///////////////////                                                               ////__DACE:0:0:137    ////__DACE:0:0:137
                                            ////__DACE:0:0:137                        ////__DACE:0:0:137
                                            __map_fusion_gtir_tmp_104_0 = __tlet_result;                                      ////__DACE:0:0:137    ////__DACE:0:0:137
                                        }                                             ////__DACE:0:0:137
                                        if_stmt_5_0_0_136(__map_fusion_gtir_tmp_104_0, &gt_conn_E2C[0], &gtir_tmp_101[0], gtir_tmp_54_0, gtir_tmp_76_0, &gtir_tmp_95[0], &perturbed_theta_v_at_cells_on_model_levels[0], &reference_theta_at_edges_on_model_levels[0], gtir_tmp_137_0, __gt_conn_E2C_neighbor_stride, __perturbed_theta_v_at_cells_on_model_levels_K_stride, __reference_theta_at_edges_on_model_levels_K_stride, i_Edge_gtx_horizontal, i_K_gtx_vertical);    ////__DACE:0:0:136
                                        {                                             ////__DACE:0:0:410
                                            double _cpy_in = gtir_tmp_137_0;                                                  ////__DACE:0:0:135,410    ////__DACE:0:0:410
                                            double _cpy_out;                                                                  ////__DACE:0:0:410    ////__DACE:0:0:410
                                            ////__DACE:0:0:410                        ////__DACE:0:0:410
                                            ///////////////////                                                               ////__DACE:0:0:410    ////__DACE:0:0:410
                                            // Tasklet code (copy_gtir_tmp_137_0_to_theta_v_at_edges_on_model_levels)         ////__DACE:0:0:410    ////__DACE:0:0:410
                                            _cpy_out = _cpy_in;                                                               ////__DACE:0:0:410    ////__DACE:0:0:410
                                            ///////////////////                                                               ////__DACE:0:0:410    ////__DACE:0:0:410
                                            ////__DACE:0:0:410                        ////__DACE:0:0:410
                                            theta_v_at_edges_on_model_levels[((__theta_v_at_edges_on_model_levels_K_stride * i_K_gtx_vertical) + i_Edge_gtx_horizontal)] = _cpy_out;    ////__DACE:0:0:410    ////__DACE:0:0:410
                                        }                                             ////__DACE:0:0:410
                                    }                                                 ////__DACE:0:0:288
                                }                                                     ////__DACE:0:0:288
                            }                                                         ////__DACE:0:0:138
                        }                                                             ////__DACE:0:0:138
                    }                                                                 ////__DACE:0:0:138
                }                                                                     ////__DACE:0:0:138
            }                                                                         ////__DACE:0:0:138
        }                                                                             ////__DACE:0:0:423
    }                                                                                 ////__DACE:0:0:423
}                                                                                 ////__DACE:0:0:423

                                                                                  ////__DACE:0:0:422
DACE_EXPORTED void __dace_runkernel_map_100_fieldop_0_0_0_422(theta_shared_probe_shared_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const double * __restrict__ grf_tend_vn, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ next_vn, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __grf_tend_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __next_vn_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime);    ////__DACE:0:0:422
void __dace_runkernel_map_100_fieldop_0_0_0_422(theta_shared_probe_shared_state_t *__state, const double * __restrict__ current_vn, const double * __restrict__ dual_normal_cell_x, const double * __restrict__ dual_normal_cell_y, const double * __restrict__ grf_tend_vn, const int * __restrict__ gt_conn_E2C, const double * __restrict__ gtir_tmp_101, const double * __restrict__ gtir_tmp_83, const double * __restrict__ gtir_tmp_89, const double * __restrict__ gtir_tmp_95, double * __restrict__ next_vn, const double * __restrict__ perturbed_rho_at_cells_on_model_levels, const double * __restrict__ perturbed_theta_v_at_cells_on_model_levels, const double * __restrict__ pos_on_tplane_e_x, const double * __restrict__ pos_on_tplane_e_y, const double * __restrict__ primal_normal_cell_x, const double * __restrict__ primal_normal_cell_y, const double * __restrict__ reference_rho_at_edges_on_model_levels, const double * __restrict__ reference_theta_at_edges_on_model_levels, double * __restrict__ rho_at_edges_on_model_levels, const double * __restrict__ tangential_wind, double * __restrict__ theta_v_at_edges_on_model_levels, int __current_vn_K_stride, int __dual_normal_cell_x_E2C_stride, int __dual_normal_cell_y_E2C_stride, int __grf_tend_vn_K_stride, int __gt_conn_E2C_neighbor_stride, int __next_vn_K_stride, int __perturbed_rho_at_cells_on_model_levels_K_stride, int __perturbed_theta_v_at_cells_on_model_levels_K_stride, int __pos_on_tplane_e_x_E2C_stride, int __pos_on_tplane_e_y_E2C_stride, int __primal_normal_cell_x_E2C_stride, int __primal_normal_cell_y_E2C_stride, int __reference_rho_at_edges_on_model_levels_K_stride, int __reference_theta_at_edges_on_model_levels_K_stride, int __rho_at_edges_on_model_levels_K_stride, int __tangential_wind_K_stride, int __theta_v_at_edges_on_model_levels_K_stride, double dtime)    ////__DACE:0:0:422
{                                                                                 ////__DACE:0:0:422
                                                                                  ////__DACE:0:0:422
    void  *map_100_fieldop_0_0_0_422_args[] = { (void *)&current_vn, (void *)&dual_normal_cell_x, (void *)&dual_normal_cell_y, (void *)&grf_tend_vn, (void *)&gt_conn_E2C, (void *)&gtir_tmp_101, (void *)&gtir_tmp_83, (void *)&gtir_tmp_89, (void *)&gtir_tmp_95, (void *)&next_vn, (void *)&perturbed_rho_at_cells_on_model_levels, (void *)&perturbed_theta_v_at_cells_on_model_levels, (void *)&pos_on_tplane_e_x, (void *)&pos_on_tplane_e_y, (void *)&primal_normal_cell_x, (void *)&primal_normal_cell_y, (void *)&reference_rho_at_edges_on_model_levels, (void *)&reference_theta_at_edges_on_model_levels, (void *)&rho_at_edges_on_model_levels, (void *)&tangential_wind, (void *)&theta_v_at_edges_on_model_levels, (void *)&__current_vn_K_stride, (void *)&__dual_normal_cell_x_E2C_stride, (void *)&__dual_normal_cell_y_E2C_stride, (void *)&__grf_tend_vn_K_stride, (void *)&__gt_conn_E2C_neighbor_stride, (void *)&__next_vn_K_stride, (void *)&__perturbed_rho_at_cells_on_model_levels_K_stride, (void *)&__perturbed_theta_v_at_cells_on_model_levels_K_stride, (void *)&__pos_on_tplane_e_x_E2C_stride, (void *)&__pos_on_tplane_e_y_E2C_stride, (void *)&__primal_normal_cell_x_E2C_stride, (void *)&__primal_normal_cell_y_E2C_stride, (void *)&__reference_rho_at_edges_on_model_levels_K_stride, (void *)&__reference_theta_at_edges_on_model_levels_K_stride, (void *)&__rho_at_edges_on_model_levels_K_stride, (void *)&__tangential_wind_K_stride, (void *)&__theta_v_at_edges_on_model_levels_K_stride, (void *)&dtime };    ////__DACE:0:0:422
    gpuError_t __err = hipLaunchKernel((void*)map_100_fieldop_0_0_0_422, dim3(10, 30, 1), dim3(256, 1, 1), map_100_fieldop_0_0_0_422_args, 0, nullptr);    ////__DACE:0:0:422
    DACE_KERNEL_LAUNCH_CHECK(__err, "map_100_fieldop_0_0_0_422", 10, 30, 1, 256, 1, 1);
}

