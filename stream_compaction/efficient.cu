#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Kernal that does a up-sweep
        __global__ void kernUpSweep(int n, int levelPowerOne, int levelPower,  int *odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            int divide = index / levelPowerOne; 
            if (index - (divide * levelPowerOne) == 0) {
                odata[index + levelPowerOne - 1] += odata[index + levelPower - 1];
            }

        }

        // Kernal that does a down-sweep
        __global__ void kernDownSweep(int n, int levelPowerPlusOne, int levelPower, int *odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            int divide = index / levelPowerPlusOne;
            if (index - (divide * levelPowerPlusOne) == 0) {
                int temp = odata[index + levelPower - 1];
                odata[index + levelPower - 1] = odata[index + levelPowerPlusOne - 1];
                odata[index + levelPowerPlusOne - 1] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // Initialize blockSize and fullBlocksPerGrid
            int blockSize = 128;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Initialize variables and device arrays
            int totalLevels = ilog2ceil(n);
            int arraySize = pow(2, totalLevels); // To handle non-power of two lengths
            int *dev_array;

            // Allocate device array. 
            cudaMalloc((void**) &dev_array, arraySize * sizeof(int));
            checkCUDAError("cudaMalloc dev_array failed!");

            // Copy input data into dev_read
            cudaMemcpy(dev_array, idata, sizeof(int) * n, cudaMemcpyHostToDevice);


            timer().startGpuTimer();
            // TODO
            // Go through the levels for Up Sweep
            for (unsigned int level = 0; level <= totalLevels; level++) {
                int levelPowerOne = pow(2, level + 1);
                int levelPower = pow(2, level);

                // invoke kernel
                kernUpSweep << <fullBlocksPerGrid, blockSize >> >(n, levelPowerOne, levelPower, dev_array);

            }

            // Copy values to a temporary array
            int* temp_array = new int[arraySize];
            cudaMemcpy(temp_array, dev_array, sizeof(int) * arraySize, cudaMemcpyDeviceToHost);

            // Set the last element to zero
            temp_array[arraySize - 1] = 0;

            // Copy array back to GPU
            cudaMemcpy(dev_array, temp_array, sizeof(int) * arraySize, cudaMemcpyHostToDevice);

            // Go through the levels for Down Sweep
            for (int level = totalLevels - 1; level >= 0; level--) {
                int levelPowerPlusOne = pow(2, level + 1);
                int levelPower = pow(2, level);

                // invoke kernel
                kernDownSweep << <fullBlocksPerGrid, blockSize >> >(n, levelPowerPlusOne, levelPower, dev_array);

            }

            // Copy data from GPU to output array
            cudaMemcpy(odata, dev_array, sizeof(int) * n, cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            // Free memory
            cudaFree(dev_array);
            delete temp_array;
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
