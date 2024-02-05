#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

thrust::device_ptr<glm::vec3> dev_thrust_pos_SortedGridCell; 
thrust::device_ptr<glm::vec3> dev_thrust_vel_SortedGridCell; 

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_pos_SortedGridCell; 
glm::vec3 *dev_vel_SortedGridCell; 

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos_SortedGridCell, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_SortedGridCell failed!");

  cudaMalloc((void**)&dev_vel_SortedGridCell, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel_SortedGridCell failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  glm::vec3 perceived_center, c, perceived_velocity, velocity_change;
  int num_neighbors1=0, num_neighbors3=0; 
  for (auto i =0; i<N; ++i){
    if (i!=iSelf) {
      auto dist_to_other = glm::length(pos[i]-pos[iSelf]); 
      if (dist_to_other < rule1Distance) {
        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        perceived_center += pos[i]; 
        num_neighbors1+= 1; 
      }
      if (dist_to_other < rule2Distance) {
        // Rule 2: boids try to stay a distance d away from each other
        c -= (pos[i] - pos[iSelf]);
      }
      if (dist_to_other < rule3Distance) {
        // Rule 3: boids try to match the speed of surrounding boids
        perceived_velocity += vel[i]; 
        num_neighbors3+=1; 
      }
    }
  }
  if (num_neighbors1>0) {
    perceived_center /= (float)num_neighbors1; 
    velocity_change += (perceived_center - pos[iSelf]) * rule1Scale; 
  }

  velocity_change += c*rule2Scale; 

  if (num_neighbors3>0) {
    perceived_velocity /= (float)num_neighbors3; 
    velocity_change += perceived_velocity*rule3Scale; 
  }
  return velocity_change; 
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
  int index = threadIdx.x + (blockIdx.x * blockDim.x); 
  if (index>=N) { return; }
  auto new_velo = vel1[index] + computeVelocityChange(N, index, pos, vel1); 
  vel2[index] = new_velo.length()>maxSpeed? maxSpeed*glm::normalize(new_velo):new_velo; 
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    //
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {return; }

    auto grid_xyz = floor(pos[index] - gridMin) * inverseCellWidth; 
    int gridIdx = gridIndex3Dto1D(int(grid_xyz.x), int(grid_xyz.y), int(grid_xyz.z), gridResolution); 

    gridIndices[index] = gridIdx; 
    indices[index] = index;     
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  int index = (blockIdx.x * blockDim.x) + threadIdx.x; // each index per particle 
  if (index < N) {
    int gridCellIdx = particleGridIndices[index];  // map each particle index to its bounding cell index 
    if (index ==0 || particleGridIndices[index] != particleGridIndices[index-1]) {
      gridCellStartIndices[gridCellIdx] = index; 
    }
    if (index == N-1 || particleGridIndices[index]!= particleGridIndices[index+1]) {
      gridCellEndIndices[gridCellIdx] = index;
    }   
  }
}

__global__ void reshufflePosVelIdx(int N, int *particleArrayIndices, glm::vec3 *pos, glm::vec3 *vel, glm::vec3 *sorted_pos, glm::vec3 *sorted_vel){
    
  // pointer chasing 
  // for (auto idx = gridCellStartIndices[gridIdx]; idx<=gridCellEndIndices[gridIdx]; ++idx){ // idx is the pointer in the sorted values 
  //   int i = particleArrayIndices[idx];
  // }
  int index = (blockIdx.x * blockDim.x) + threadIdx.x; // each index per particle 
  if (index >= N) {return; }
  int sorted_idx = particleArrayIndices[index]; 
  sorted_pos[index] = pos[sorted_idx]; 
  sorted_vel[index] = vel[sorted_idx]; 
}


__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int index = (blockIdx.x * blockDim.x) + threadIdx.x; 
  if (index < N) {

    auto curr_pos = pos[index]; 

    auto gridxyz_float = (curr_pos - gridMin) * inverseCellWidth; 
    auto gridxyz_int = floor(gridxyz_float); 

    gridxyz_int.x = int(gridxyz_int.x); 
    gridxyz_int.y = int(gridxyz_int.y); 
    gridxyz_int.z = int(gridxyz_int.z); 

    int x_from, x_to, y_from, y_to, z_from, z_to; 

    if((gridxyz_float.x - gridxyz_int.x)>=0.5){
      x_from = gridxyz_int.x; 
      x_to = imin(gridxyz_int.x+1, gridResolution-1); 
    }
    else {
      x_from = imax(gridxyz_int.x - 1, 0); 
      x_to = gridxyz_int.x;
    }
    if((gridxyz_float.y - gridxyz_int.y)>=0.5){
      y_from = gridxyz_int.y; 
      y_to = imin(gridxyz_int.y+1, gridResolution-1); 
    }
    else {
      y_from = imax(gridxyz_int.y - 1, 0); 
      y_to = gridxyz_int.y;
    }
    if((gridxyz_float.z - gridxyz_int.z)>=0.5){
      z_from = gridxyz_int.z; 
      z_to = imin(gridxyz_int.z+1, gridResolution-1); 
    }
    else {
      z_from = imax(gridxyz_int.z - 1, 0); 
      z_to = gridxyz_int.z;
    }
    
    glm::vec3 perceived_center, c, perceived_velocity, velocity_change;
    int num_neighbors1=0, num_neighbors3=0;
    for (auto x = x_from; x<=x_to; ++x) {
      for (auto y = y_from; y<=y_to; ++y){
        for (auto z = z_from; z<=z_to; ++z){
          int gridIdx = gridIndex3Dto1D(int(x), int(y), int(z), gridResolution); 
          int iSelf = index; 
          if(gridCellStartIndices[gridIdx]==-1) {continue; } 
          for (auto idx = gridCellStartIndices[gridIdx]; idx<=gridCellEndIndices[gridIdx]; ++idx){ // idx is the pointer in the sorted values 
            int i = particleArrayIndices[idx];  // i is the actual boid index 
            if (i!=iSelf) {
              auto pos_other = pos[i]; 
              auto dist_to_other = glm::distance(pos_other, curr_pos); 
              if (dist_to_other < rule1Distance) {
                // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                perceived_center += pos_other; 
                num_neighbors1+= 1; 
              }
              if (dist_to_other < rule2Distance) {
                // Rule 2: boids try to stay a distance d away from each other
                c -= (pos_other - curr_pos);
              }
              if (dist_to_other < rule3Distance) {
                // Rule 3: boids try to match the speed of surrounding boids
                perceived_velocity += vel1[i]; 
                num_neighbors3+=1; 
              }
            }
          }

        }
      }
    }
    
    if (num_neighbors1>0) {
      perceived_center /= (float)num_neighbors1; 
      velocity_change += (perceived_center - curr_pos) * rule1Scale; 
    }

    velocity_change += c*rule2Scale; 

    if (num_neighbors3>0) {
      perceived_velocity /= (float)num_neighbors3; 
      velocity_change += perceived_velocity*rule3Scale; 
    }

    glm::vec3 new_velo = vel1[index] + velocity_change; 
    vel2[index] = glm::length(new_velo)>maxSpeed? maxSpeed*glm::normalize(new_velo):new_velo; 
  }
}


__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  int index = (blockIdx.x * blockDim.x) + threadIdx.x; 
  if (index < N) {

    auto curr_pos = pos[index]; 

    auto gridxyz_float = (curr_pos - gridMin) * inverseCellWidth; 
    auto gridxyz_int = floor(gridxyz_float); 

    gridxyz_int.x = int(gridxyz_int.x); 
    gridxyz_int.y = int(gridxyz_int.y); 
    gridxyz_int.z = int(gridxyz_int.z); 

    int x_from, x_to, y_from, y_to, z_from, z_to; 

    if((gridxyz_float.x - gridxyz_int.x)>=0.5){
      x_from = gridxyz_int.x; 
      x_to = imin(gridxyz_int.x+1, gridResolution-1); 
    }
    else {
      x_from = imax(gridxyz_int.x - 1, 0); 
      x_to = gridxyz_int.x;
    }

    if((gridxyz_float.y - gridxyz_int.y)>=0.5){
      y_from = gridxyz_int.y; 
      y_to = imin(gridxyz_int.y+1, gridResolution-1); 
    }
    else {
      y_from = imax(gridxyz_int.y - 1, 0); 
      y_to = gridxyz_int.y;
    }
    if((gridxyz_float.z - gridxyz_int.z)>=0.5){
      z_from = gridxyz_int.z; 
      z_to = imin(gridxyz_int.z+1, gridResolution-1); 
    }
    else {
      z_from = imax(gridxyz_int.z - 1, 0); 
      z_to = gridxyz_int.z;
    }
    
    glm::vec3 perceived_center, c, perceived_velocity, velocity_change;
    int num_neighbors1=0, num_neighbors3=0;
    for (auto x = x_from; x<=x_to; ++x) {
      for (auto y = y_from; y<=y_to; ++y){
        for (auto z = z_from; z<=z_to; ++z){
          int gridIdx = gridIndex3Dto1D(x, y, z, gridResolution); 
          int iSelf = index; 
          if(gridCellStartIndices[gridIdx]==-1) {continue; } 
          for (auto idx = gridCellStartIndices[gridIdx]; idx<=gridCellEndIndices[gridIdx]; ++idx){ // idx is the pointer in the sorted values 
            int i = idx;  // i is the actual boid index 
            if (i!=iSelf) {
              auto pos_other = pos[i]; 
              auto dist_to_other = glm::distance(pos_other, curr_pos); 
              if (dist_to_other < rule1Distance) {
                // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                perceived_center += pos_other; 
                num_neighbors1+= 1; 
              }
              if (dist_to_other < rule2Distance) {
                // Rule 2: boids try to stay a distance d away from each other
                c -= (pos_other - curr_pos);
              }
              if (dist_to_other < rule3Distance) {
                // Rule 3: boids try to match the speed of surrounding boids
                perceived_velocity += vel1[i]; 
                num_neighbors3+=1; 
              }
            }
          }

        }
      }
    }
    

    if (num_neighbors1>0) {
      perceived_center /= (float)num_neighbors1; 
      velocity_change += (perceived_center - curr_pos) * rule1Scale; 
    }

    velocity_change += c*rule2Scale; 

    if (num_neighbors3>0) {
      perceived_velocity /= (float)num_neighbors3; 
      velocity_change += perceived_velocity*rule3Scale; 
    }

    glm::vec3 new_velo = vel1[index] + velocity_change; 
    vel2[index] = glm::length(new_velo)>maxSpeed? maxSpeed*glm::normalize(new_velo):new_velo; 
  }
}

__global__ void kernUpdateVelNeighborSearchCoherent_REF(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    glm::vec3 pos_self = pos[index];
    glm::vec3 grid_cell = inverseCellWidth * (pos_self - gridMin);
    glm::vec3 grid_cell_int = glm::floor(grid_cell);
    glm::vec3 grid_cell_frac = grid_cell - grid_cell_int;

    // The following vectors store whether neighbors in the positive and negative direction should be checked
    glm::ivec3 check_neg;
    glm::ivec3 check_pos;

    check_neg.x = (grid_cell_frac.x <= 0.5f && grid_cell_int.x > 0) ? 1.f : 0.f;
    check_neg.y = (grid_cell_frac.y <= 0.5f && grid_cell_int.y > 0) ? 1.f : 0.f;
    check_neg.z = (grid_cell_frac.z <= 0.5f && grid_cell_int.z > 0) ? 1.f : 0.f;
    check_pos.x = (grid_cell_frac.x > 0.5f  && grid_cell_int.x < gridResolution - 1) ? 1.f : 0.f;
    check_pos.y = (grid_cell_frac.y > 0.5f  && grid_cell_int.y < gridResolution - 1) ? 1.f : 0.f;
    check_pos.z = (grid_cell_frac.z > 0.5f  && grid_cell_int.z < gridResolution - 1) ? 1.f : 0.f;

    // Velocity change due to each rule
    // Boids try to fly towards center of mass of neighboring boids
    glm::vec3 velocity_change;
    glm::vec3 perceived_center;
    glm::vec3 c;
    glm::vec3 perceived_velocity;

    int num_neighbors_r1 = 0;
    int num_neighbors_r3 = 0;

    for (int z = grid_cell_int.z - check_neg.z; z <= grid_cell_int.z + check_pos.z; z++)
    {
        for (int y = grid_cell_int.y - check_neg.y; y <= grid_cell_int.y + check_pos.y; y++)
        {
            for (int x = grid_cell_int.x - check_neg.x; x <= grid_cell_int.x + check_pos.x; x++)
            {
                int neighbor_grid_cell_1D = gridIndex3Dto1D(x, y, z, gridResolution);

                if (gridCellStartIndices[neighbor_grid_cell_1D] == -1)
                {
                    continue;
                }

                for (int cell_index = gridCellStartIndices[neighbor_grid_cell_1D]; cell_index <= gridCellEndIndices[neighbor_grid_cell_1D]; cell_index++)
                {
                    glm::vec3 pos_other = pos[cell_index];

                    if (cell_index != index) {
                        float dist_to_other = glm::distance(pos_other, pos_self);

                        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
                        if (dist_to_other < rule1Distance)
                        {
                            perceived_center += pos_other;
                            num_neighbors_r1++;
                        }

                        // Rule 2: boids try to stay a distance d away from each other
                        if (dist_to_other < rule2Distance)
                        {
                            c -= (pos_other - pos_self);
                        }

                        // Rule 3: boids try to match the speed of surrounding boids
                        if (dist_to_other < rule3Distance) {
                            perceived_velocity += vel1[cell_index];
                            num_neighbors_r3++;
                        }
                    }
                }
            }
        }
    }

    // Calculate contributions for each rule
    // Rule 1
    if (num_neighbors_r1 > 0)
    {
        velocity_change += (perceived_center / (float)num_neighbors_r1 - pos_self) * rule1Scale;
    }

    // Rule 2
    velocity_change += c * rule2Scale;

    // Rule 3
    if (num_neighbors_r3 > 0)
    {
        velocity_change += (perceived_velocity / (float)num_neighbors_r3) * rule3Scale;
    }

    // - Clamp the speed change before putting the new speed in vel2
    glm::vec3 new_velocity = vel1[index] + velocity_change;
    if (glm::length(new_velocity) > maxSpeed)
    {
        new_velocity = maxSpeed * glm::normalize(new_velocity);
    }
    vel2[index] = new_velocity;
    }

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize); 
  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2); 
  checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2); 
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // TODO-1.2 ping-pong the velocity buffers
  glm::vec3* temp = dev_vel2;
  dev_vel2 = dev_vel1;
  dev_vel1 = temp;
}


void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed
  
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullBlocksPerGrid_gridCell((gridCellCount + blockSize - 1) / blockSize);
  
  // - label each particle with its array index as well as its grid index use 2x width grids
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, 
    gridMinimum, gridInverseCellWidth, 
    dev_pos, dev_particleArrayIndices, 
    dev_particleGridIndices);

  // - unstable key sort using Thrust 
    dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
    dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
   thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - naively unroll the loop for find the start and end indices of each cell's data pointers
  kernResetIntBuffer<<<fullBlocksPerGrid_gridCell, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1); 
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices,  
    dev_gridCellStartIndices, dev_gridCellEndIndices); 

  // - perform velocity update using neighbor search and then update positions
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, 
    gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, 
    dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2); 
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel1); 
  // - ping-pong buffers
  glm::vec3* temp = dev_vel2;
  dev_vel2 = dev_vel1;
  dev_vel1 = temp;
}

__global__ void kernReshuffleParticlePosVelData(
    int N, glm::vec3* pos1, glm::vec3* pos2, glm::vec3* vel1, glm::vec3* vel2,
    int* particleArrayIndices) {
    // Swaps the position and velocity data values to correspond to the cell
    // indices rather than the boid indices

    // Calculate index of Boid
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N)
    {
        return;
    }

    // Get sorted particle index value
    int particle_array_index = particleArrayIndices[index];
    pos2[index] = pos1[particle_array_index];
    vel2[index] = vel1[particle_array_index];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 fullBlocksPerGrid_gridCell((gridCellCount + blockSize - 1) / blockSize);
  
  // - label each particle with its array index as well as its grid index use 2x width grids
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, 
    gridMinimum, gridInverseCellWidth, 
    dev_pos, dev_particleArrayIndices, 
    dev_particleGridIndices);

  // - unstable key sort using Thrust 
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - naively unroll the loop for find the start and end indices of each cell's data pointers
  kernResetIntBuffer<<<fullBlocksPerGrid_gridCell, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1); 
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices,  
  dev_gridCellStartIndices, dev_gridCellEndIndices); 

  // TODO - 2.3: reshuffle the particle data in `dev_gridCellStartIndices` and `dev_gridCellEndIndices` to avoid pointer chasing 
  reshufflePosVelIdx<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_pos_SortedGridCell, dev_vel_SortedGridCell); 

  // - perform velocity update using neighbor search and then update positions
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, 
    gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, 
    dev_gridCellEndIndices, dev_pos_SortedGridCell, dev_vel_SortedGridCell, dev_vel2); 

  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos_SortedGridCell, dev_vel2); 
  
  // - ping-pong buffers
  glm::vec3* temp = dev_pos;
  dev_pos = dev_pos_SortedGridCell;
  dev_pos_SortedGridCell = temp;
  
  temp = dev_vel1; 
  dev_vel1 = dev_vel2; 
  dev_vel2 = temp; 
  

}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_gridCellEndIndices); 
  cudaFree(dev_gridCellStartIndices); 
  cudaFree(dev_particleArrayIndices); 
  cudaFree(dev_particleGridIndices); 

  cudaFree(dev_pos_SortedGridCell); 
  cudaFree(dev_vel_SortedGridCell); 
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");

  return;
}
