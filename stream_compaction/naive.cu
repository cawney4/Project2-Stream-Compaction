#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"


namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        // Kernel that does a naive INCLUSIVE scan
        __global__ void kernNaiveScan(int n, int levelPower, int *odata, const int *idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) {
                return;
            }

            if (index >= levelPower) {
                odata[index] = idata[index - levelPower] + idata[index];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            // Initialize blockSize and fullBlocksPerGrid
            int blockSize = 256;
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            // Initialize variables and device arrays
            int totalLevels = ilog2ceil(n);
            int *dev_write;
            int *dev_read;

            // Allocate device arrays
            cudaMalloc((void**) &dev_write, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_write failed!");

            cudaMalloc((void**) &dev_read, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_read failed!");

            // Copy input data into dev_read
            cudaMemcpy(dev_read, idata, sizeof(int) * n, cudaMemcpyHostToDevice);


            timer().startGpuTimer();
            // TODO     
            
            // Go through the levels of Naive scan
            for (unsigned int level = 1; level <= totalLevels; level++) {
                int levelPower = pow(2, level - 1);

                // invoke kernel
                kernNaiveScan << <fullBlocksPerGrid, blockSize >> >(n, levelPower, dev_write, dev_read);

                // Ping-pong write and read arrays
                int* temp = dev_write;
                dev_write = dev_read;
                dev_read = temp;
            }

            // Copy final values into temporary array
            int* tempArray = new int[n];
            cudaMemcpy(tempArray, dev_read, sizeof(int) * n, cudaMemcpyDeviceToHost);

            // Copy values from tempArray while shifting values to convert inclusive scan to exclusive scan
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = tempArray[i - 1];
            }
            
            timer().endGpuTimer();

            // Free memory
            delete tempArray;
            cudaFree(dev_write);
            cudaFree(dev_read);

        }
    }
}
