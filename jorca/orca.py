import jax.numpy as jnp
from jax import jit, vmap, lax, debug

RVO_EPSILON = 0.00001
NAN_ORCA_LINE = jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]])

@jit
def det(vector1:jnp.ndarray, vector2:jnp.ndarray) -> float:
    """
    This function computes the determinant of two vectors.

    args:
    - vector1: shape is (2,) in the form (x, y)
    - vector2: shape is (2,) in the form (x, y)

    output:
    - determinant: float
    """
    return vector1[0] * vector2[1] - vector1[1] * vector2[0]

@jit
def del_idx_from_arr(arr:jnp.ndarray, idx:int) -> jnp.ndarray:
    """
    Removes one row from a given 2D array.
    """
    return jnp.where(
            jnp.tile(jnp.transpose(jnp.expand_dims(jnp.arange(len(arr)-1) < idx, axis=0)), (1,arr.shape[1])),
            arr[:-1],
            arr[1:])

@jit
def linearProgram1(
    orca_lines:jnp.ndarray,
    idx:int,
    max_speed:float,
    pref_velocity:jnp.ndarray
) -> tuple:
    """
    """
    dot_product = jnp.dot(orca_lines[idx][1],orca_lines[idx][0])
    discriminant = dot_product * dot_product + max_speed * max_speed - jnp.dot(orca_lines[idx][1], orca_lines[idx][1])

    @jit
    def _discriminant_greater_than_zero(val:tuple):
        discriminant, dot_product, orca_lines, idx, max_speed, pref_velocity = val
        sqrt_discriminant = jnp.sqrt(discriminant)
        t_left = -dot_product - sqrt_discriminant
        t_right = -dot_product + sqrt_discriminant

        @jit
        def _while_loop_body(val:tuple):

            @jit
            def _not_nan_orca_line(val:tuple):
                while_idx, fail, discriminant, dot_product, orca_lines, idx, max_speed, pref_velocity, t_right, t_left = val
                denominator = det(orca_lines[idx][0], orca_lines[while_idx][0])
                numerator = det(orca_lines[while_idx][0], orca_lines[idx][1] - orca_lines[while_idx][1])
                @jit
                def _if_denominator_not_zero(val:tuple):
                    numerator, denominator, t_right, t_left = val
                    t = numerator / denominator
                    t_right, t_left = lax.cond(
                        denominator >= 0, 
                        lambda _: (jnp.min(jnp.array([t_right, t])), t_left),
                        lambda _: (t_right, jnp.max(jnp.array([t_left, t]))),
                        None)
                    fail = t_left > t_right
                    return fail, t_right, t_left

                fail, t_right, t_left = lax.cond(
                    jnp.abs(denominator) <= RVO_EPSILON,
                    lambda x: (numerator < 0, x[2], x[3]),
                    _if_denominator_not_zero,
                    (numerator, denominator, t_right, t_left))

                while_idx += 1
                return while_idx, fail, discriminant, dot_product, orca_lines, idx, max_speed, pref_velocity, t_right, t_left
            
            while_idx, fail, discriminant, dot_product, orca_lines, idx, max_speed, pref_velocity, t_right, t_left = lax.cond(
                jnp.any(jnp.isnan(val[4][val[0]])),
                lambda _: (val[0]+1, val[1], val[2], val[3], val[4], val[5], val[6], val[7], val[8], val[9]),
                _not_nan_orca_line,
                val)
            
            return while_idx, fail, discriminant, dot_product, orca_lines, idx, max_speed, pref_velocity, t_right, t_left

        _, fail, _, _, _, _, _, _, t_right, t_left = lax.while_loop(
            lambda x: ((x[0] < idx) & (~x[1])),
            _while_loop_body,
            (0, False, discriminant, dot_product, orca_lines, idx, max_speed, pref_velocity, t_right, t_left))
        
        @jit
        def _if_not_fail(val:tuple):
            t_right, t_left, orca_lines, idx, pref_velocity = val
            t = jnp.dot(orca_lines[idx][0], (pref_velocity - orca_lines[idx][1]))
            switch_case = jnp.argmax(jnp.array([t < t_left, t > t_right, (t_left <= t) & (t <= t_right)], dtype=int))
            result = lax.switch(
                switch_case,
                [lambda _: orca_lines[idx][1] + t_left * orca_lines[idx][0], 
                lambda _: orca_lines[idx][1] + t_right * orca_lines[idx][0], 
                lambda _: orca_lines[idx][1] + t * orca_lines[idx][0]],
                None)
            return result
            
        result = lax.cond(
            fail,
            lambda _: jnp.zeros((2, )),
            _if_not_fail,
            (t_right, t_left, orca_lines, idx, pref_velocity))
        
        return result, fail
    
    result, fail = lax.cond(
        discriminant < 0, 
        lambda _: (jnp.zeros((2, )), True), 
        _discriminant_greater_than_zero, 
        (discriminant, dot_product, orca_lines, idx, max_speed, pref_velocity))

    return result, fail

@jit
def linearProgram2(
    orca_lines:jnp.ndarray,
    max_speed:jnp.ndarray,
    pref_velocity:jnp.ndarray,
) -> tuple:
    """
    This function finds the closest velocity to the preferred velocity that is collision free and whose norm is below max_speed.

    args:
    - orca_lines: shape is (n_humans-1, 2, 2) where each row is (direction, point)
    - max_speed: float
    - pref_velocity: shape is (2,) in the form (vx, vy)

    output:
    - result: shape is (2,) in the form (vx, vy)
    - line_fail: int, line at which the linear program failed
    """
    result = pref_velocity

    @jit
    def _orca_line_loop(val:tuple):
        
        @jit
        def _not_nan_orca_line(val:tuple):
            result, i, lines, max_speed, fail, pref_velocity = val
            does_not_satisfy_constraint = det(lines[i, 0], lines[i, 1] - result) > 0.

            @jit
            def _go_to_linear_program_1(val:tuple):
                temp_result, i, lines, max_speed, pref_velocity = val
                result, fail = linearProgram1(lines, i, max_speed, pref_velocity)
                result, i = lax.cond(fail, lambda _: (temp_result, i-1), lambda _: (result, i), None)
                return result, fail, i
            
            result, fail, i = lax.cond(
                does_not_satisfy_constraint,
                _go_to_linear_program_1,
                lambda _: (result, False, i),
                (result, i, lines, max_speed, pref_velocity))
            i += 1
            return result, i, lines, max_speed, fail, pref_velocity

        result, i, lines, max_speed, line_fail, pref_velocity = lax.cond(
            jnp.any(jnp.isnan(val[2][val[1]])),
            lambda _: (val[0], val[1]+1, val[2], val[3], val[4], val[5]),
            _not_nan_orca_line,
            val)
        return result, i, lines, max_speed, line_fail, pref_velocity
    
    result, line_fail, _, _, _, _ = lax.while_loop(
        lambda x: ((x[1] < len(orca_lines)) & ~(x[4])),
        _orca_line_loop, 
        (result, 0, orca_lines, max_speed, False, pref_velocity))
    return result, line_fail

@jit
def linearProgram3(
    orca_lines:jnp.ndarray,
    begin_line:int,
    max_speed:float,
    result:jnp.ndarray
) -> jnp.ndarray:
    """
    This function in the ORCA library is responsible for refining the velocity solution for collision avoidance. 
    It solves a higher-dimensional optimization problem where multiple linear constraints must be satisfied while minimizing the deviation from a preferred velocity.
    """

    @jit
    def _fori_body(i:int, val:tuple):
        result, distance, orca_lines, max_speed = val

        @jit
        def _does_not_satisfy_constraint_i(val:tuple):
            result, distance, orca_lines, max_speed = val

            @jit
            def _second_fori_body(j:int, val:tuple):
                proj_lines, orca_lines = val
                determinant = det(orca_lines[i][0], orca_lines[j][0])
                conditions = jnp.array([
                    (jnp.abs(determinant) <= RVO_EPSILON) & (jnp.dot(orca_lines[i][0], orca_lines[j][0]) > 0),
                    (jnp.abs(determinant) <= RVO_EPSILON) & (jnp.dot(orca_lines[i][0], orca_lines[j][0]) <= 0),
                    jnp.abs(determinant) > RVO_EPSILON], dtype=int)
                switch_case = jnp.argmax(conditions)
                point, do_not_push = lax.switch(
                    switch_case,
                    [lambda _: (jnp.zeros((2,)), True),
                    lambda _: (0.5 * (orca_lines[i][1] + orca_lines[j][1]), False),
                    lambda _: (orca_lines[i][1] + (det(orca_lines[j][0], orca_lines[i][1] - orca_lines[j][0]) / determinant * orca_lines[i][0]), False)],
                    None)
                proj_lines = lax.cond(
                    do_not_push,
                    lambda _: proj_lines,
                    lambda _: proj_lines.at[j].set(jnp.array([(orca_lines[j][0] - orca_lines[i][0]) / jnp.linalg.norm(orca_lines[j][0] - orca_lines[i][0]), point])),
                    None)
                return proj_lines, orca_lines
            
            proj_lines, orca_lines = lax.fori_loop(
                0,
                i,
                _second_fori_body,
                (jnp.tile(NAN_ORCA_LINE, (len(orca_lines), 1, 1)), orca_lines))
            temp_result, line_fail = linearProgram2(proj_lines, max_speed, jnp.array([-orca_lines[i][0][1], orca_lines[i][0][0]]))
            result = lax.cond(
                line_fail < len(proj_lines), # Failure here is only due to numerical error, it should not happen, so the result is overwritten.
                lambda _: result,
                lambda _: temp_result,
                None)
            distance = det(orca_lines[i][0], orca_lines[i][1] - result)
            return result, distance, orca_lines, max_speed
            
        result, distance, _, _ = lax.cond(
            det(orca_lines[i][0], orca_lines[i][1] - result) > distance,
            _does_not_satisfy_constraint_i,
            lambda x: x,
            (result, distance, orca_lines, max_speed))
        return result, distance, orca_lines, max_speed

    result, _, _ ,_ = lax.fori_loop(
        begin_line, 
        len(orca_lines), 
        _fori_body, 
        (result, 0., orca_lines, max_speed))
    return result

@jit
def compute_single_human_orca_line(human_state:jnp.ndarray, other_human_state:jnp.ndarray, parameters:jnp.ndarray, other_human_parameters:jnp.ndarray, dt:float) -> jnp.ndarray:
    """
    This function computes the ORCA lines for a single human.

    args:
    - human_state: shape is (4,) in the form (px, py, vx, vy)
    - other_human_state: shape is (4,) in the form (px, py, vx, vy)
    - parameters: shape is (..,) in the form (radius, time_horizon, v_max, ..., safety_space)
    - other_human_parameters: shape is (..,) in the form (radius, time_horizon, v_max, ..., safety_space)
    - dt: sampling time for the update

    output:

    """
    relative_position = other_human_state[:2] - human_state[:2]
    relative_velocity = human_state[2:] - other_human_state[2:]
    squared_distance = jnp.dot(relative_position, relative_position)
    combined_radius = (parameters[0] + parameters[-1]) + (other_human_parameters[0] + other_human_parameters[-1])
    combined_radius_squared = combined_radius * combined_radius
    inverted_time_horizon = 1.0 / parameters[1]

    @jit
    def _no_collision(val:tuple):
        rel_pos, rel_vel, sq_dist, comb_rad, comb_rad_sq, inv_time_hor, _ = val
        w = rel_vel - inv_time_hor * rel_pos
        w_length_squared = jnp.dot(w, w)
        dot_product = jnp.dot(w, relative_position)

        @jit
        def _project_on_cutoff_circle(val:tuple):
            # w, w_length_sq, comb_rad, inv_time_hor, sq_dist, comb_rad_sq, rel_pos, rel_vel = val
            w, w_length_sq, comb_rad, inv_time_hor, _, _, _, _ = val
            w_length = jnp.sqrt(w_length_sq)
            unit_w = w / w_length
            direction = jnp.array([unit_w[1], -unit_w[0]])
            u = (comb_rad * inv_time_hor - w_length) * unit_w
            return direction, u
        @jit
        def _project_on_left_leg(val:tuple):
            # w, w_length_sq, comb_rad, inv_time_hor, sq_dist, comb_rad_sq, rel_pos, rel_vel = val
            _, _, _, _, sq_dist, comb_rad_sq, rel_pos, rel_vel = val
            leg = jnp.sqrt(sq_dist - comb_rad_sq)
            direction = jnp.array([
                rel_pos[0] * leg - rel_pos[1] * comb_rad, 
                rel_pos[0] * comb_rad + rel_pos[1] * leg]) / sq_dist
            u = jnp.dot(jnp.dot(rel_vel, direction), direction) - rel_vel
            return direction, u
        @jit
        def _project_on_right_leg(val:tuple):
            # w, w_length_sq, comb_rad, inv_time_hor, sq_dist, comb_rad_sq, rel_pos, rel_vel = val
            _, _, _, _, sq_dist, comb_rad_sq, rel_pos, rel_vel = val
            leg = jnp.sqrt(sq_dist - comb_rad_sq)
            direction = jnp.array([
                rel_pos[0] * leg + rel_pos[1] * comb_rad, 
                -rel_pos[0] * comb_rad + rel_pos[1] * leg]) / sq_dist
            u = jnp.dot(jnp.dot(rel_vel, direction), direction) - rel_vel
            return direction, u
        
        conditions = jnp.array([
            (dot_product < 0.0) & (dot_product * dot_product > comb_rad_sq * w_length_squared),
            ~((dot_product < 0.0) & (dot_product * dot_product > comb_rad_sq * w_length_squared)) & (det(relative_position, w) > 0.0),
            ~((dot_product < 0.0) & (dot_product * dot_product > comb_rad_sq * w_length_squared)) & (det(relative_position, w) <= 0.0)], dtype=int)
        
        switch_case = jnp.argmax(conditions) # There will be all 0 and one 1

        direction, u = lax.switch(
            switch_case, 
            [_project_on_cutoff_circle, _project_on_left_leg, _project_on_right_leg],
            (w, w_length_squared, comb_rad, inv_time_hor, sq_dist, comb_rad_sq, relative_position, relative_velocity))
        
        return direction, u
    
    @jit
    def _collision(val:tuple):
        rel_pos, rel_vel, _, comb_rad, _, _, dt = val
        inverted_time_step = 1 / dt
        w = rel_vel - inverted_time_step * rel_pos
        w_length = jnp.sqrt(jnp.dot(w, w))
        unit_w = w / w_length
        direction = jnp.array([unit_w[1], -unit_w[0]])
        u = (comb_rad * inverted_time_step - w_length) * unit_w
        
        return direction, u
    
    direction, u = lax.cond(
        squared_distance > combined_radius_squared,
        _no_collision,
        _collision,
        (relative_position, relative_velocity, squared_distance, combined_radius, combined_radius_squared, inverted_time_horizon, dt))
    point = human_state[2:4] + 0.5 * u
    orca_line = jnp.array([direction, point])
    return orca_line

@jit
def compute_single_human_neighbors(human_state:jnp.ndarray, other_humans_state:jnp.ndarray, parameters:jnp.ndarray, other_humans_parameters:jnp.ndarray) -> jnp.ndarray:
    """
    This function computes the neighbors of a single human given a max distance and a max number of neighbors.
    """
    neighbor_dist = parameters[3]
    squared_neighbor_dist = neighbor_dist**2
    max_neighbors = parameters[4]
    relative_positions = other_humans_state[:, :2] - human_state[:2]
    squared_distances = jnp.sum(relative_positions * relative_positions, axis=1)
    sorted_indices = jnp.argsort(squared_distances)
    # TODO: Vmap this function
    @jit
    def _set_neighbor(idx, sorted_idx, squared_distances, squared_neighbor_dist, max_neighbors, humans_state, humans_parameters):
        neighbor, neighbor_parameters = lax.cond(
            (squared_distances[sorted_idx] < squared_neighbor_dist) & (idx < max_neighbors),
            lambda _: (humans_state[sorted_idx], humans_parameters[sorted_idx]),
            lambda _: (jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan]), jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan])),
            None)
        return neighbor, neighbor_parameters
    neighbors, neighbor_parameters = vmap(_set_neighbor, in_axes=(0, 0, None, None, None, None, None))(
        jnp.arange(len(sorted_indices)),
        sorted_indices, 
        squared_distances, 
        squared_neighbor_dist, 
        max_neighbors, 
        other_humans_state, 
        other_humans_parameters)
    # neighbors, neighbor_parameters = lax.fori_loop(
    #     0,
    #     len(sorted_indices),
    #     lambda i, val: lax.cond(
    #         (squared_distances[sorted_indices[i]] < squared_neighbor_dist) & (i < max_neighbors),
    #         lambda _: (val[0].at[i].set(other_humans_state[sorted_indices[i]]), val[1].at[i].set(other_humans_parameters[sorted_indices[i]])),
    #         lambda _: (val[0].at[i].set(jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan])), val[1].at[i].set(jnp.array([jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan, jnp.nan]))),
    #         None),
    #     (jnp.zeros((len(other_humans_state), 4)), jnp.zeros((len(other_humans_state), 6)))
    # )
    return neighbors, neighbor_parameters

@jit
def single_update(idx:int, humans_state:jnp.ndarray, human_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for a single human using the Optimal Reciprocal Collision Avoidance (ORCA) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - idx: human index in the state, goal and parameter vectors
    - humans_state: shape is (n_humans, 4) in the form is (px, py, vx, vy)
    - humans_goal: shape is (2,) in the form (gx, gy)
    - parameters: shape is (n_humans, ..) in the form (radius, time_horizon, v_max, ..., safety_space)
    - obstacles: shape is (n_obstacles, n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_human_state: shape is (4,) in the form (px, py, vx, vy)
    """
    ### Compute neighbors
    neighbors, neighbors_parameters = compute_single_human_neighbors(humans_state[idx], del_idx_from_arr(humans_state, idx), parameters[idx], del_idx_from_arr(parameters, idx))
    ### Compute ORCA lines for Humans
    orca_lines = vmap(compute_single_human_orca_line, in_axes=(None, 0, None, 0, None))(
        humans_state[idx], 
        neighbors, # del_idx_from_arr(humans_state, idx), 
        parameters[idx], 
        neighbors_parameters, # del_idx_from_arr(parameters, idx), 
        dt)
    ### Compute ORCA lines for Obstacles
    # TODO: Implement static obstacles avoidance in ORCA
    ### Compute preferred velocity
    goal_relative_position = human_goal - humans_state[idx][:2]
    preferred_velocity = goal_relative_position / jnp.linalg.norm(goal_relative_position) * parameters[idx][2]
    ### Compute new velocity
    ## First 2D-linear program (computes collision free velocity closest to the preferred one, if exists)
    new_velocity, line_fail = linearProgram2(orca_lines, parameters[idx][2], preferred_velocity)
    ## Second 3D-linear program (computes safest velocity if the first linear program fails)
    new_velocity = lax.cond(
        line_fail < len(orca_lines),
        lambda _: linearProgram3(orca_lines, line_fail, parameters[idx][2], new_velocity),
        lambda _: new_velocity,
        None)
    ### Update human state
    updated_human_state = jnp.array([
        humans_state[idx, 0] + dt * humans_state[idx, 2],
        humans_state[idx, 1] + dt * humans_state[idx, 3],
        new_velocity[0],
        new_velocity[1]
    ])
    return updated_human_state

@jit
def step(humans_state:jnp.ndarray, humans_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for the humans' state using the Optimal Reciprocal Collision Avoidance (ORCA) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - humans_state: shape is (n_humans, 4) where each row is (px, py, vx, vy)
    - humans_goal: shape is (n_humans, 2) where each row is (gx, gy)
    - parameters: shape is (n_humans, ..) where each row is (radius, time_horizon, v_max, ..., safety_space)
    - obstacles: shape is (n_obstacles, n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_humans_state: shape is (n_humans, 4) where each row is (px, py, vx, vy)
    """
    updated_humans_state = vmap(single_update, in_axes=(0, None, 0, None, None, None))(
        jnp.arange(len(humans_state)),
        humans_state, 
        humans_goal, 
        parameters, 
        obstacles, 
        dt)
    return updated_humans_state
