#ifndef SIM_ENV_H
#define SIM_ENV_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* Build in sim/: cargo oxide build --arch sm_120
 * --device-codegen-crate sim_physics,sim_env -- -p sim-env --release
 * Produces target/release/libsim_env.so, including compiled Rust CUDA bundles.
 * NVIDIA's generated image-anchor loader supports dlopen/ctypes; do not strip
 * the .oxart artifact sections from the shared library.
 *
 * All calls for a handle must be serialized. Device ordinal is zero. Create
 * synchronizes once at startup; reset and step enqueue work without a CPU wait.
 * Style 0 uses seeded training colors; styles 1 (validation) and 2 (final test)
 * use distinct held-out palettes/lighting and separate gap RNG domains.
 * Reset changes layout RNG, not static materials.
 *
 * Use torch.cuda.ExternalStream(sim_env_stream(handle)) for policy inference,
 * action production, native calls, and all accesses to the returned tensor views.
 * Other streams require explicit CUDA event dependencies. No returned pointer
 * is Vulkan-owned memory. Do not free any pointer; destroy invalidates all views.
 * Actions are n initialized GPU int64 elements in {0,1,2}: left/neutral/right.
 * Retain them until the borrowed CUDA stream has completed the step.
 *
 * Fields (all contiguous, CUDA-only):
 * 0 float32[n,3,height,width], NCHW sRGB values in [0,1], 12*n*w*h bytes
 * 1 float32[n,8], 32*n bytes: relative goal XYZ, velocity XYZ,
 *   remaining clock fraction, previous action (-1,0,+1). No gap information.
 * 2 float32[n], 4*n bytes: previous transition reward
 * 3 float32[n], 4*n bytes: previous transition terminal flag
 * 4 float32[n], 4*n bytes: previous transition success flag
 * 5 float32[n,4], 16*n bytes: relative gap X, relative wall Z,
 *   gap half-width, elapsed clock fraction. Privileged critic only.
 * 6 float32[n], 4*n bytes: previous transition collision flag
 * 7 uint64[n], 8*n bytes: independent completed episode counts
 *
 * One control step is .05s (six Rust CUDA physics steps at 120Hz). Radius .18m,
 * wall centre Z=-6m, thickness .3m, height 3m, gap half-width .8m, corridor X=+-4m.
 * A swept sphere collision tests the whole wall slab. Goal Z=-12m. The 150-step
 * finite horizon is TERMINAL, never a truncation requiring bootstrap. Rewards:
 * .2*forward progress -.01, plus +5 success or -5 collision. Autoreset preserves
 * reward/terminal flags, advances that environment's gap RNG, then publishes the
 * RESET state's RGB/proprio/privileged observations. Reset clears all flags and
 * episode counts. Destroy performs shutdown synchronization.
 *
 * Errors: create returns NULL; reset/step return -1; buffer/stream return zero.
 * Error string is thread-local and stable until the next failure on that thread.
 * A handle must be live and uniquely owned; stale/arbitrary handles are invalid.
 */
void *sim_env_create(uint32_t n, uint64_t seed, uint32_t width, uint32_t height, uint32_t style);
int32_t sim_env_reset(void *handle, uint64_t seed);
int32_t sim_env_step(void *handle, const int64_t *actions);
uint64_t sim_env_buffer(void *handle, int32_t field);
uint64_t sim_env_stream(void *handle);
void sim_env_destroy(void *handle);
const char *sim_env_error(void);

#ifdef __cplusplus
}
#endif
#endif
