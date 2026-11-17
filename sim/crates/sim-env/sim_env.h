#ifndef SIM_ENV_H
#define SIM_ENV_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* ABI v2: real-city altitude-stabilized navigation.
 *
 * Build in sim/: cargo oxide build --arch sm_120
 * --device-codegen-crate sim_physics,sim_env -- -p sim-env --release
 * Do not strip embedded .oxart CUDA artifact sections from libsim_env.so.
 *
 * All handle calls are serialized; GPU ordinal zero. city_path must point to a
 * NUL-terminated UTF-8 city directory (layout.json plus actual city GLBs).
 * Create imports one shared city, preprocesses exact collision-free routes and
 * uploads immutable BVH/catalogs; it may synchronize. Missing/unusable geometry
 * or insufficient meaningful routes is an error, never a toy/easy fallback.
 * style=0 train,1 validation,2 test selects disjoint64x64m districts centered
 * at(-.4h,-.4h),(+.4h,-.4h),(+.4h,+.4h), h=layout city half extent.
 * Each independently fixed seeded catalog contains128 distinct blocked-direct-
 * line routes, length12..38m, detour>=2m and>=8%, shortest8-connected2m grid
 * paths visibility-shortcut at planning sphere radius.85m. All splits use the
 * same obstacle rule; no easy curriculum. Reset seed selects catalog routes.
 *
 * Use torch.cuda.ExternalStream(sim_env_stream(handle)) for inference, actions,
 * native calls and tensor access. Other streams need explicit dependencies.
 * Reset/step enqueue GPU work without CPU waits or host state/image staging.
 * Returned device pointers are CUDA-owned, never Vulkan resources. Do not free.
 * Actions are n initialized GPU int64 values:0 hover,1 forward,2 forward-right,
 * 3 right,4 backward-right,5 backward,6 backward-left,7 left,8 forward-left.
 * Diagonals normalized; invalid actions deterministically hover. Keep actions
 * live until the borrowed stream completes. Immediate horizontal velocity3m/s,
 * fixed altitude6m; each20Hz action integrates six swept120Hz substeps.
 * The level camera auto-yaws toward final goal (80m far plane); commands are
 * goal-facing camera-frame directions held through the control step.
 * Actual collision uses sphere radius.35m and exact rendered triangle geometry.
 * Agents do not collide with or render one another. Success final radius1m.
 *
 * Fields (all contiguous, ordinary CUDA-only allocations):
 * 0 float32[n,3,height,width] NCHW sRGB[0,1]
 * 1 float32[n,8]: goal[0,altitude-y,-horizontal_distance] in camera view,
 *   velocity[right,up,forward], remaining clock fraction,previous_action/8
 * 2 float32[n] previous transition reward
 * 3 float32[n] previous transition terminal flag
 * 4 float32[n] previous transition success flag
 * 5 float32[n,4]: next waypoint right/32,forward/32,remaining path metres/48,
 *   elapsed clock fraction. Positive forward points toward camera forward.
 *   PRIVILEGED: only explicit state-ceiling actor/diagnostic oracle may use it.
 * 6 float32[n] previous transition actual geometry collision flag
 * 7 uint64[n] completed episode count
 * 8 float32[n,8]: world xyz,final-goal xyz,global route_id,clock steps.
 *   DIAGNOSTIC ONLY: never include in policy inputs.
 *
 * Reward=.2*(old remaining path-new remaining path)-.01, plus+5 final-goal
 * success or-5 collision (collision precedence). Remaining path is distance to
 * next waypoint plus its suffix path length; advance within.25m. Horizon320 is
 * TERMINAL, not bootstrap truncation; timeout adds no bonus. Autoreset preserves
 * reward/done/success/collision while publishing RESET RGB/proprio/privileged/
 * diagnostics. Explicit reset clears flags/counters. Destroy synchronizes.
 *
 * sim_env_metadata returns immutable UTF-8 JSON, stable until handle destroy;
 * caller copies it and must not free it. It includes ABI/task/action/control,
 * city content fingerprint/path, geometry counts, exact split/catalog rules,
 * reward, input conventions, altitude/radius/horizon and route counts.
 * Errors: create/metadata return NULL,reset/step -1,buffer/stream zero.
 * sim_env_error is thread-local, stable until next failure on that thread.
 * Stale/arbitrary handles or unreadable pointers violate the ABI contract.
 */
void *sim_env_create(uint32_t n, uint64_t seed, uint32_t width, uint32_t height,
                     uint32_t style, const char *city_path);
const char *sim_env_metadata(void *handle);
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
