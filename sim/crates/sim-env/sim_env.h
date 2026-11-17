/* city-drone-goal-v1, C ABI v3.
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

#define SIM_ENV_ABI_VERSION 3

/* Field identifiers for sim_env_buffer(). */
enum sim_env_field {
    /* float32 [n, 3, H, W]: RGB observation, values in [0, 1]. */
    SIM_ENV_FIELD_RGB = 0,
    /* float32 [n, 18]: proprioceptive sensors (see metadata "sensors"). */
    SIM_ENV_FIELD_SENSORS = 1,
    /* float32 [n]: reward of the transition just taken. */
    SIM_ENV_FIELD_REWARD = 2,
    /* float32 [n]: 1 if the episode terminated on this transition, else 0. */
    SIM_ENV_FIELD_DONE = 3,
    /* float32 [n]: 1 if the episode succeeded on this transition, else 0. */
    SIM_ENV_FIELD_SUCCESS = 4,
    /* float32 [n, 18]: privileged state, for critics only (see metadata). */
    SIM_ENV_FIELD_PRIVILEGED = 5,
    /* float32 [n]: 1 if the drone crashed (collision or hard landing), else 0. */
    SIM_ENV_FIELD_COLLISION = 6,
    /* uint64 [n]: current episode index per environment (after autoreset). */
    SIM_ENV_FIELD_EPISODES = 7,
    /* float32 [n, 24]: diagnostics (see metadata "diagnostics"). */
    SIM_ENV_FIELD_DIAGNOSTICS = 8,
    /* int32 [n]: split-local target design index of the current episode. */
    SIM_ENV_FIELD_DESIGN = 9,
    /* float32 [K, 3, H, W]: rendered goal image per target design (static). */
    SIM_ENV_FIELD_GOAL = 10,
    /* float32 [3]: bucket sampling weights; writable by the caller. */
    SIM_ENV_FIELD_BUCKET_WEIGHTS = 11,
    /* float32 [n]: bucket of the episode ended on this transition (-1 if not done). */
    SIM_ENV_FIELD_BUCKET = 12,
    /* float32 [n]: 1 if the drone left the district or ceiling, else 0. */
    SIM_ENV_FIELD_OOB = 13,
    SIM_ENV_FIELD_COUNT = 14
};

/* Creates an environment batch of n drones.
 * width * 3 must equal height * 4 (fixed 4:3 visibility and camera projection).
 * style: 0 training, 1 validation, 2 test.
 * Returns NULL on failure. */
void *sim_env_create(uint32_t n, uint64_t seed, uint32_t width, uint32_t height,
                     uint32_t style, const char *city_path);

/* Null-terminated UTF-8 JSON describing layouts, dynamics and sign conventions.
 * Owned by the environment; valid until sim_env_destroy(). */
const char *sim_env_metadata(void *env);

/* Resets all environments with a new seed; episode counters restart at 0. */
int sim_env_reset(void *env, uint64_t seed);

/* Steps all environments. actions: float32 [n, 4] on the environment stream,
 * each component clamped to [-1, 1]. Layout per env: (thrust, roll-like, pitch-like, yaw-like).
 * Caller must order producers before sim_env_stream(). */
int sim_env_step(void *env, const float *actions);

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
