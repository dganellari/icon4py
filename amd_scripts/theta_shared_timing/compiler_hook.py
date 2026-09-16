"""Opt in only to theta's interior vertical split in the original program."""

THETA_OUTPUT = "theta_v_at_edges_on_model_levels"
EDGE = "i_Edge_gtx_horizontal"
LEVEL = "i_K_gtx_vertical"


def select_theta_split(transformation, first_map, second_map, state, sdfg):
    # The matcher reuses a transformation instance across different candidates.
    transformation.allow_shared_data = False
    if transformation.access_node.data != THETA_OUTPUT:
        return True
    first = dict(zip(first_map.params, first_map.range))
    second = dict(zip(second_map.params, second_map.range))
    transformation.allow_shared_data = (
        set(first) == set(second) == {EDGE, LEVEL}
        and first[EDGE] == second[EDGE]
        and first[LEVEL] != second[LEVEL]
    )
    return True
