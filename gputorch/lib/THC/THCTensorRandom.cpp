#include "THCTensorRandom.h"
#include "THCGeneral.h"

/*#include <thrust/functional.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_mtgp32_host.h>
#include <curand_mtgp32dc_p_11213.h>*/

#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

/* Sets up generator. Allocates but does not create the generator states. */
void initializeGenerator(Generator* gen)
{
/*  THGPUCheck(gpuMalloc((void**)&gen->gen_states, MAX_NUM_BLOCKS * sizeof(curandStateMtgp32)));
  THGPUCheck(gpuMalloc((void**)&gen->kernel_params, sizeof(mtgp32_kernel_params)));
  if (curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, gen->kernel_params) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP constants failed.");
  }*/
}

/* Frees memory allocated during setup. */
void destroyGenerator(Generator* gen)
{
/*  if (gen->gen_states)
  {
    THGPUCheck(gpuFree(gen->gen_states));
    gen->gen_states = NULL;
  }
  if (gen->kernel_params)
  {
    THGPUCheck(gpuFree(gen->kernel_params));
    gen->kernel_params = NULL;
  }*/
}

/* Creates a new generator state given the seed. */
void createGeneratorState(Generator* gen, unsigned long seed)
{
/*  if (curandMakeMTGP32KernelState(gen->gen_states, mtgp32dc_params_fast_11213,
                                  gen->kernel_params, MAX_NUM_BLOCKS, seed) != CURAND_STATUS_SUCCESS)
  {
    THError("Creating MTGP kernel state failed.");
  }
*/
}

/* Initialize generator array (must be called before any other function) */
void THCRandom_init(THGPURNGState* state, int devices, int current_device)
{
  state->num_devices = devices;
  state->gen = (Generator*)malloc(state->num_devices * sizeof(Generator));
  for (int i = 0; i < state->num_devices; ++i)
  {
    state->gen[i].initf = 0;
    state->gen[i].initial_seed = 0;
    state->gen[i].gen_states = NULL;
   // state->gen[i].kernel_params = NULL;
  }
  state->current_gen = &state->gen[current_device];
  // Initialize the generator for the current device. Other generators will be
  // initialized on-demand in THCRandom_setGenerator.
  initializeGenerator(state->current_gen);
  THCRandom_seed(state);
}

/* Destroy generators and free memory */
void THCRandom_shutdown(THGPURNGState* state)
{
  if (state->gen == NULL) return;
  for (int i = 0; i < state->num_devices; ++i)
  {
    destroyGenerator(&state->gen[i]);
  }
  free(state->gen);
  state->gen = NULL;
  state->current_gen = NULL;
}

/* Set the generator for the current device */
void THCRandom_setGenerator(THGPURNGState* state, int device)
{
  if (device >= state->num_devices) THError("Invalid device index.");
  state->current_gen = &state->gen[device];
  if (state->current_gen->initf == 0)
  {
    initializeGenerator(state->current_gen);
    THCRandom_seed(state);
  }
}

/* Reset the generator for the current device after a device reset */
void THCRandom_resetGenerator(THGPURNGState* state)
{
  initializeGenerator(state->current_gen);
  THCRandom_manualSeed(state, state->current_gen->initial_seed);
}

/* Random seed */
unsigned long THCRandom_seed(THGPURNGState* state)
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeed(state, s);
  return s;
}

unsigned long THCRandom_seedAll(THGPURNGState* state)
{
  unsigned long s = (unsigned long)time(0);
  THCRandom_manualSeedAll(state, s);
  return s;
}

/* Manually set the seed */
void THCRandom_manualSeed(THGPURNGState* state, unsigned long seed)
{
  if (state->current_gen == NULL)
  {
    THError("Random number generators have not been initialized.");
  }
  state->current_gen->initial_seed = seed;
  createGeneratorState(state->current_gen, seed);
  state->current_gen->initf = 1;
}

void THCRandom_manualSeedAll(THGPURNGState* state, unsigned long seed)
{
  int currentDevice;
 // THGPUCheck(gpuGetDevice(&currentDevice));
  /*for (int i = 0; i < state->num_devices; ++i) {
    THGPUCheck(gpuSetDevice(i));
    THCRandom_setGenerator(state, i);
    THCRandom_manualSeed(state, seed);
  }
  THGPUCheck(gpuSetDevice(currentDevice));
  THCRandom_setGenerator(state, currentDevice);*/
}

/* Get the initial seed */
unsigned long THCRandom_initialSeed(THGPURNGState* state)
{
  return state->current_gen->initial_seed;
}

void THCRandom_getRNGState(THGPURNGState* state, THByteTensor *rng_state)
{
  // The RNG state comprises the MTPG32 states and the seed.
 /* static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THByteTensor_resize1d(rng_state, total_size);
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
  THGPUCheck(gpuMemcpy(THByteTensor_data(rng_state), state->current_gen->gen_states,
                         states_size, gpuMemcpyDeviceToHost));
  memcpy(THByteTensor_data(rng_state) + states_size, &state->current_gen->initial_seed, seed_size);*/
}

void THCRandom_setRNGState(THGPURNGState* state, THByteTensor *rng_state)
{
  /*static const size_t states_size = MAX_NUM_BLOCKS * sizeof(curandStateMtgp32);
  static const size_t seed_size = sizeof(unsigned long);
  static const size_t total_size = states_size + seed_size;
  THArgCheck(THByteTensor_nElement(rng_state) == total_size, 1, "RNG state is wrong size");
  THArgCheck(THByteTensor_isContiguous(rng_state), 1, "RNG state must be contiguous");
   THGPUCheck(gpuMemcpy(state->current_gen->gen_states, THByteTensor_data(rng_state),
                         states_size, gpuMemcpyHostToDevice));
  memcpy(&state->current_gen->initial_seed, THByteTensor_data(rng_state) + states_size, seed_size);*/
}

#define GENERATE_KERNEL1(NAME, ARG1, CURAND_FUNC, TRANSFORM)                                             \
void NAME(int size, THGPUTensor *result, ARG1)                                                           \
{                                                                                                        \
  std::mt19937 gen;                                                                                      \
  Concurrency::array_view<float, 1> avResult(THGPUTensor_nElement(result), THGPUTensor_data(result));    \
  for (int i = 0; i < size; i++) {                                                                       \
    std::CURAND_FUNC<float> rand(0.0, 0.9);                                                              \
    float x = rand(gen);                                                                                 \
    x = TRANSFORM;                                                                                       \
    avResult[i] = x;                                                                                     \
  }                                                                                                      \
  avResult.synchronize();                                                                                \
}

#define GENERATE_KERNEL2(NAME, ARG1, ARG2, CURAND_FUNC, TRANSFORM)                                       \
void NAME(int size, THGPUTensor *result, ARG1, ARG2)                                                     \
{                                                                                                        \
  std::mt19937 gen;                                                                                      \
  Concurrency::array_view<float, 1> avResult(THGPUTensor_nElement(result), THGPUTensor_data(result));    \
  for (int i = 0; i < size; i++) {                                                                       \
    std::CURAND_FUNC<float> rand(0, 0.9);                                                                \
    float x = rand(gen);                                                                                 \
    x = TRANSFORM;                                                                                       \
    avResult[i] = x;                                                                                     \
  }                                                                                                      \
  avResult.synchronize();                                                                                \
}

GENERATE_KERNEL2(generate_uniform, double a, double b, uniform_real_distribution, x * (b-a) + a)
GENERATE_KERNEL1(generate_bernoulli, double p, uniform_real_distribution, (float)x <= p)
GENERATE_KERNEL2(generate_normal, double mean, double stdv, normal_distribution,(float)((x * stdv) + mean))
GENERATE_KERNEL1(generate_geometric, double p, uniform_real_distribution, (log(1-x) / log(p)) + 1)
GENERATE_KERNEL1(generate_exponential, double lambda, uniform_real_distribution, (float)(-1. / lambda * log(1-x)))
GENERATE_KERNEL2(generate_cauchy, double median, double sigma, uniform_real_distribution, (float)(median + sigma * tan(M_PI*(x-0.5))))

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2

/* Separate kernel because curand_log_normal gets extra parameters. */
void generate_log_normal(int size, THGPUTensor *result, float mean, float stddev)
{
  std::mt19937 gen;
  Concurrency::array_view<float, 1> avResult(THGPUTensor_nElement(result), THGPUTensor_data(result));
  for (int i = 0; i < size; i++)
  {
    std::lognormal_distribution<float> rand(mean, stddev);
    float x = rand(gen);
    avResult[i] = x;
  }
  avResult.synchronize();
}


#define NUM_BLOCKS min((int)DIVUP(size, BLOCK_SIZE), MAX_NUM_BLOCKS)
void THGPUTensor_uniform(THGPURNGState* state, THGPUTensor *self_, double a, double b)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  float *data = THGPUTensor_data(self);

  generate_uniform(size, self, a, b);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_bernoulli(THGPURNGState* state, THGPUTensor *self_, double p)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  float *data = THGPUTensor_data(self);

  generate_bernoulli(size, self, p);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_normal(THGPURNGState* state, THGPUTensor *self_, double mean, double stdv)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  float *data = THGPUTensor_data(self);

  generate_normal(size, self, mean, stdv);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_logNormal(THGPURNGState* state, THGPUTensor *self_, double mean, double stdv)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  float *data = THGPUTensor_data(self);

  generate_log_normal(size, self, mean, stdv);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_geometric(THGPURNGState* state, THGPUTensor *self_, double p)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  float *data = THGPUTensor_data(self);

  generate_geometric(size, self, p);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_exponential(THGPURNGState* state, THGPUTensor *self_, double lambda)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  float *data = THGPUTensor_data(self);

  generate_exponential(size, self, lambda);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_cauchy(THGPURNGState* state, THGPUTensor *self_, double median, double sigma)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  float *data = THGPUTensor_data(self);

  generate_cauchy(size, self, median, sigma);

  THGPUTensor_freeCopyTo(self, self_);
};
#undef NUM_BLOCKS

