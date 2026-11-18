/* city-drone-goal-v4, C ABI v7.
 *
 * All tensor storage is CUDA device memory on the stream returned by
 * sim_env_stream(). Buffers are valid until sim_env_destroy(). Calls must be
 * serialized by the caller. Functions return NULL/0 on failure; the message is
 * available from sim_env_error() on the calling thread.
 */
#ifndef SIM_ENV_H
#define SIM_ENV_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SIM_ENV_ABI_VERSION 7

/* Field identifiers for sim_env_buffer(). */
enum sim_env_field {
    /* float32 [n, 3, H, W]: RGB observation, values in [0, 1]. */
    SIM_ENV_FIELD_RGB = 0,
    /* float32 [n, 19]: proprioceptive sensors (see metadata "sensors"). */
    SIM_ENV_FIELD_SENSORS = 1,
    /* float32 [n]: reward of the transition just taken. */
    SIM_ENV_FIELD_REWARD = 2,
    /* float32 [n]: 1 if the episode terminated on this transition, else 0. */
    SIM_ENV_FIELD_DONE = 3,
    /* float32 [n]: 1 if the episode succeeded on this transition, else 0. */
    SIM_ENV_FIELD_SUCCESS = 4,
    /* float32 [n, 29]: privileged state, for critics only (see metadata). */
    SIM_ENV_FIELD_PRIVILEGED = 5,
    /* float32 [n]: 1 if the drone crashed (collision or hard landing), else 0. */
    SIM_ENV_FIELD_COLLISION = 6,
    /* uint64 [n]: current episode index per environment (after autoreset). */
    SIM_ENV_FIELD_EPISODES = 7,
    /* float32 [n, 28]: diagnostics (see metadata "diagnostics"). */
    SIM_ENV_FIELD_DIAGNOSTICS = 8,
    /* int32 [n]: catalog index of the current episode (row of a learner's goal table). */
    SIM_ENV_FIELD_CATALOG_INDEX = 9,
    /* float32 [n, 3, H, W]: goal reference views written by sim_env_render_goals(). */
    SIM_ENV_FIELD_GOAL_VIEWS = 10,
    /* float32 [5]: [w_visible, w_partial, w_hidden, p_no_gps, appearance] bucket sampling
     * weights, per-episode GPS-outage probability and appearance randomization strength
     * in [0, 1]; writable by the caller. */
    SIM_ENV_FIELD_CURRICULUM = 11,
    /* float32 [n]: bucket of the episode ended on this transition (-1 if not done). */
    SIM_ENV_FIELD_BUCKET = 12,
    /* float32 [n]: 1 if the drone left the district or ceiling, else 0. */
    SIM_ENV_FIELD_OOB = 13,
    /* float32 [n]: 1 if GPS was available for the episode ended on this transition, else 0; -1 if not done. */
    SIM_ENV_FIELD_GPS_AVAILABLE = 14,
    /* float32 [n, 4]: normalized goal crops [x0, y0, x1, y1] written by sim_env_render_goals(). */
    SIM_ENV_FIELD_GOAL_CROPS = 15,
    /* float32 [n, 3]: current/terminal unique tiles, terminal consecutive hold steps; diagnostic only. */
    SIM_ENV_FIELD_EPISODE_DIAGNOSTICS = 16,
    /* float32 [n, P], P = (H/16)(W/16): mean planar depth (m) per 16x16 patch, 0 if no geometry.
     * Training label only, never actor input. */
    SIM_ENV_FIELD_PATCH_DEPTH = 17,
    /* int32 [n, P]: city layout object most of the patch shows, -1 if none. Training label only. */
    SIM_ENV_FIELD_PATCH_OBJECT = 18,
    SIM_ENV_FIELD_COUNT = 19
};

/* Creates an environment batch of n drones.
 * width * 3 must equal height * 4 (fixed 4:3 visibility and camera projection).
 * style: 0 training, 1 validation, 2 test.
 * control_mode: 0 velocity (PX4-style offboard), 1 body rates (v3 inner loop).
 * Returns NULL on failure. */
void *sim_env_create(uint32_t n, uint64_t seed, uint32_t width, uint32_t height,
                     uint32_t style, uint32_t control_mode, const char *city_path);

/* Same task with a different catalog start-box-distance range (metres), for far-target cohorts.
 * min_distance > 6, max_distance > min_distance, both finite. */
void *sim_env_create_with_start_distance(uint32_t n, uint64_t seed, uint32_t width, uint32_t height,
                     uint32_t style, uint32_t control_mode, const char *city_path,
                     float min_distance, float max_distance);

/* Null-terminated UTF-8 JSON describing layouts, dynamics and sign conventions.
 * Owned by the environment; valid until sim_env_destroy(). */
const char *sim_env_metadata(void *env);

/* Resets all environments with a new seed; episode counters restart at 0. */
int sim_env_reset(void *env, uint64_t seed);

/* Steps all environments. actions: float32 [n, 4] on the environment stream,
 * each component clamped to [-1, 1]. Layout per env is given by metadata
 * "action" (velocity: forward, right, up, yaw_rate; rates: thrust, roll_rate,
 * pitch_rate, yaw_rate). Caller must order producers before sim_env_stream(). */
int sim_env_step(void *env, const float *actions);

/* Renders the goal reference views of catalog entries first .. first + n - 1 (clamped to the
 * last entry) into SIM_ENV_FIELD_GOAL_VIEWS and their crops into SIM_ENV_FIELD_GOAL_CROPS,
 * then restores the current observation. Intended for one-time goal-table construction. */
int sim_env_render_goals(void *env, uint32_t first);

/* Replaces current-episode targets. targets: float32 [n, 8] on the environment stream, rows
 * [centre x, y, z, layout object, half x, y, z, layout class]; rows with object < 0 keep the
 * catalog target. Success, reward and privileged labels then refer to the given box until the
 * episode ends (autoreset returns to catalog targets). Observations are refreshed. */
int sim_env_retarget(void *env, const float *targets);

/* Free flight for interactive use (never training): enabled != 0 bounds flight only by the city
 * edge and a 100 m ceiling, reports success once without ending the episode, and disables the
 * timeout. Takes effect on the next step. */
int sim_env_set_free_flight(void *env, uint32_t enabled);


/* Device pointer to field storage (see enum), or 0 on error. */
uint64_t sim_env_buffer(void *env, int32_t field);

/* CUDA stream handle (CUstream as uint64) on which all fields are produced. */
uint64_t sim_env_stream(void *env);

void sim_env_destroy(void *env);

/* Last error message for this thread. Empty string if none. */
const char *sim_env_error(void);

#ifdef __cplusplus
}
#endif

#endif /* SIM_ENV_H */
