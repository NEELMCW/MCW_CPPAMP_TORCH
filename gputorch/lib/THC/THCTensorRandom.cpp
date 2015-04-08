#include "THCTensorRandom.h"
#include "THCGeneral.h"
#include "copyHelpers.h"
#define MAX_NUM_BLOCKS 64
#define BLOCK_SIZE 256

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

/* Sets up generator. Allocates but does not create the generator states. */
void initializeGenerator(Generator* gen)
{
  //TODO : To be implemented.
}

/* Frees memory allocated during setup. */
void destroyGenerator(Generator* gen)
{
  //TODO : To be implemented.
}

/* Creates a new generator state given the seed. */
void createGeneratorState(Generator* gen, unsigned long seed)
{
  //TODO : To be implemented.
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
  //TODO : To be implemented.
}

/* Get the initial seed */
unsigned long THCRandom_initialSeed(THGPURNGState* state)
{
  return state->current_gen->initial_seed;
}

void THCRandom_getRNGState(THGPURNGState* state, THByteTensor *rng_state)
{
  //TODO : To be implemented.
}

void THCRandom_setRNGState(THGPURNGState* state, THByteTensor *rng_state)
{
  //TODO : To be implemented.
}

// TODO: currently can't use pfe since no kernel versions of all CURAND_FUNC from underlying AMP
// Just prepare data on host and then copy to device side of the array
#define GENERATE_KERNEL1(NAME, ARG1, CURAND_FUNC, TRANSFORM)                                               \
void NAME(int size, THGPUTensor *result, ARG1)                                                             \
{                                                                                                          \
  std::mt19937 gen;                                                                                        \
  float vec[size];                                                                                         \
  for (int i = 0; i < size; i++) {                                                                         \
    std::CURAND_FUNC<float> rand(0.0, 0.9);                                                                \
    float x = rand(gen);                                                                                   \
    x = TRANSFORM;                                                                                         \
    vec[i] = x;                                                                                            \
  }                                                                                                        \
  float* device_ptr = static_cast<float*>(Concurrency::getDevicePointer(result->storage->data));           \
  THGPUCheck(gpuMemcpy(device_ptr, result->storageOffset * sizeof(float), vec, 0, size * sizeof(float), gpuMemcpyHostToDevice));\
}

// TODO: currently can't use pfe since no kernel versions of all CURAND_FUNC from underlying AMP
// Just prepare data on host and then copy to device side of the array
#define GENERATE_KERNEL2(NAME, ARG1, ARG2, CURAND_FUNC, TRANSFORM)                                         \
void NAME(int size, THGPUTensor *result, ARG1, ARG2)                                                       \
{                                                                                                          \
  std::mt19937 gen;                                                                                        \
  float vec[size];                                                                                         \
  for (int i = 0; i < size; i++) {                                                                         \
    std::CURAND_FUNC<float> rand(0, 0.9);                                                                  \
    float x = rand(gen);                                                                                   \
    x = TRANSFORM;                                                                                         \
    vec[i] = x;                                                                                            \
  }                                                                                                        \
  float* device_ptr = static_cast<float*>(Concurrency::getDevicePointer(result->storage->data));           \
  THGPUCheck(gpuMemcpy(device_ptr, result->storageOffset * sizeof(float), vec, 0, size * sizeof(float), gpuMemcpyHostToDevice));\
}

GENERATE_KERNEL2(generate_uniform, double a, double b, uniform_real_distribution, x * (b - a) + a)
GENERATE_KERNEL1(generate_bernoulli, double p, uniform_real_distribution, (float)x <= p)
GENERATE_KERNEL2(generate_normal, double mean, double stdv, normal_distribution,(float)((x * stdv) + mean))
GENERATE_KERNEL1(generate_geometric, double p, uniform_real_distribution, (log(1 - x) / log(p)) + 1)
GENERATE_KERNEL1(generate_exponential, double lambda, uniform_real_distribution, (float)(-1. / lambda * log(1 - x)))
GENERATE_KERNEL2(generate_cauchy, double median, double sigma, uniform_real_distribution, (float)(median + sigma * tan(M_PI*(x-0.5))))

#undef GENERATE_KERNEL1
#undef GENERATE_KERNEL2

/* Separate kernel because curand_log_normal gets extra parameters. */
void generate_log_normal(int size, THGPUTensor *self_, float mean, float stddev)
{
  std::mt19937 gen;
  float vec[size];
  for (int i = 0; i < size; i++)
  {
    std::lognormal_distribution<float> rand(mean, stddev);
    float x = rand(gen);
    vec[i] = x;
  }
  float* device_ptr = static_cast<float*>(Concurrency::getDevicePointer(self_->storage->data));
  THGPUCheck(gpuMemcpy(device_ptr, self_->storageOffset * sizeof(float), vec, 0, size * sizeof(float), gpuMemcpyHostToDevice));
}

#define NUM_BLOCKS min((int)DIVUP(size, BLOCK_SIZE), MAX_NUM_BLOCKS)
void THGPUTensor_uniform(THGPURNGState* state, THGPUTensor *self_, double a, double b)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);

  generate_uniform(size, self, a, b);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_bernoulli(THGPURNGState* state, THGPUTensor *self_, double p)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);

  generate_bernoulli(size, self, p);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_normal(THGPURNGState* state, THGPUTensor *self_, double mean, double stdv)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);

  generate_normal(size, self, mean, stdv);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_logNormal(THGPURNGState* state, THGPUTensor *self_, double mean, double stdv)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);
  generate_log_normal(size, self_, mean, stdv);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_geometric(THGPURNGState* state, THGPUTensor *self_, double p)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);

  generate_geometric(size, self, p);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_exponential(THGPURNGState* state, THGPUTensor *self_, double lambda)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);

  generate_exponential(size, self, lambda);

  THGPUTensor_freeCopyTo(self, self_);
};

void THGPUTensor_cauchy(THGPURNGState* state, THGPUTensor *self_, double median, double sigma)
{
  THGPUTensor *self = THGPUTensor_newContiguous(self_);
  long size = THGPUTensor_nElement(self);

  generate_cauchy(size, self, median, sigma);

  THGPUTensor_freeCopyTo(self, self_);
};
#undef NUM_BLOCKS
