import gt4py.next as gtx
from gt4py.next import astype
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.scan_operator(axis=dims.KDim, forward=True, init=(vpfloat("0.0"), 0.0))
def _coefficient_forward_scan(
    state_kminus1: tuple[vpfloat, float],
    vwind_impl_wgt: wpfloat,
    theta_v_ic: wpfloat,
    ddqz_z_half: vpfloat,
    alpha_prev: vpfloat,
    alpha: vpfloat,
    alpha_next: vpfloat,
    beta_prev: vpfloat,
    beta: vpfloat,
    w_explicit: wpfloat,
    exner_prev: wpfloat,
    exner: wpfloat,
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[wpfloat, wpfloat]:
    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)
    z_gamma_vp = astype(dtime * cpd * vwind_impl_wgt * theta_v_ic / ddqz_z_half_wp, vpfloat)
    a = (vpfloat("0.0") - z_gamma_vp) * beta_prev * alpha_prev
    c = (vpfloat("0.0") - z_gamma_vp) * beta * alpha_next
    b = vpfloat("1.0") + z_gamma_vp * alpha * (beta_prev + beta)
    z_gamma_wp = astype(z_gamma_vp, wpfloat)
    d = w_explicit - z_gamma_wp * (exner_prev - exner)
    c_kminus1 = astype(state_kminus1[0], vpfloat)
    d_kminus1 = state_kminus1[1]
    normalization = vpfloat("1.0") / (b + a * c_kminus1)
    c_new = (vpfloat("0.0") - c) * normalization
    d_new = (d - astype(a, wpfloat) * d_kminus1) * astype(normalization, wpfloat)
    return (c_new, d_new)


@gtx.field_operator
def _solve_tridiagonal_matrix_for_w_forward_sweep(
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_w_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[wpfloat]]:
    return _coefficient_forward_scan(
        vwind_impl_wgt,
        theta_v_ic,
        ddqz_z_half,
        z_alpha(dims.KDim - 1),
        z_alpha,
        z_alpha(dims.KDim + 1),
        z_beta(dims.KDim - 1),
        z_beta,
        z_w_expl,
        z_exner_expl(dims.KDim - 1),
        z_exner_expl,
        dtime,
        cpd,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def solve_tridiagonal_matrix_for_w_forward_sweep(
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_w_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    z_q: fa.CellKField[vpfloat],
    w: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _solve_tridiagonal_matrix_for_w_forward_sweep(
        vwind_impl_wgt=vwind_impl_wgt,
        theta_v_ic=theta_v_ic,
        ddqz_z_half=ddqz_z_half,
        z_alpha=z_alpha,
        z_beta=z_beta,
        z_w_expl=z_w_expl,
        z_exner_expl=z_exner_expl,
        dtime=dtime,
        cpd=cpd,
        out=(z_q, w),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
