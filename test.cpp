#include <iostream>
#include <vector>
#include <algorithm>

// Define imin and imax functions (you may have these already defined somewhere)
inline int imin(int a, int b) { return a < b ? a : b; }
inline int imax(int a, int b) { return a > b ? a : b; }

int Ncell = 15; 

// CUDA kernel reset function (host-side)
void kernResetIntBuffer(int N, int *intBuffer, int value) {
  for (int i = 0; i < N; ++i) {
    intBuffer[i] = value;
  }
}

// CUDA kernel identify cell start end function (host-side)
void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
                               int *gridCellStartIndices, int *gridCellEndIndices) {
  std::fill_n(gridCellStartIndices, Ncell, -1);
  std::fill_n(gridCellEndIndices, Ncell, -1);

  for (int i = 0; i < N; ++i) {
    int gridCellIdx = particleGridIndices[i];

    if (gridCellStartIndices[gridCellIdx] == -1) {
      gridCellStartIndices[gridCellIdx] = i;
    } else {
      gridCellStartIndices[gridCellIdx] = imin(gridCellStartIndices[gridCellIdx], i);
    }
    gridCellEndIndices[gridCellIdx] = imax(gridCellEndIndices[gridCellIdx], i);
  }
}

void kernIdentifyCellStartEnd_ref(int N, int *particleGridIndices,
                               int *gridCellStartIndices, int *gridCellEndIndices) {
  std::fill_n(gridCellStartIndices, 15, -1);
  std::fill_n(gridCellEndIndices, 15, -1);

  for (int i = 0; i < N; ++i) {
    int gridCellIdx = particleGridIndices[i];

    if (i==0 || particleGridIndices[i] != particleGridIndices[i-1] )
    gridCellStartIndices[gridCellIdx] = i;
    
    if (i==N-1 || particleGridIndices[i-1] != particleGridIndices[i] )
    gridCellEndIndices[gridCellIdx] = i;
  }
}

// Unit test function
void testCUDA() {
  const int N = 10;
  int particleGridIndices[N] = {0, 5, 5,5,6,7, 8, 10,10, 13};
  int gridCellStartIndices[Ncell];
  int gridCellEndIndices[Ncell];

  kernIdentifyCellStartEnd(N, particleGridIndices, gridCellStartIndices, gridCellEndIndices);

  // Print results
  std::cout << "Grid Cell Start Indices: ";
  for (int i = 0; i < Ncell; ++i) {
    std::cout << gridCellStartIndices[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Grid Cell End   Indices: ";
  for (int i = 0; i < Ncell; ++i) {
    std::cout << gridCellEndIndices[i] << " ";
  }
  std::cout << std::endl;
}

int main() {
  testCUDA();
  return 0;
}
