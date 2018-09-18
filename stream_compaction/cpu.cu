#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
            int sum = 0;
            for (unsigned int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }

	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            // TODO
            int count = 0;
            for (unsigned int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }

	        timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        // TODO
            int* map01 = new int[n];
            int* scanned = new int[n];
            for (unsigned int i = 0; i < n; i++) {
                if (idata[i] == 0) {
                    map01[i] = 0;
                }
                else {
                    map01[i] = 1;
                }
            }

            int sum = 0;
            for (unsigned int i = 0; i < n; i++) {
                scanned[i] = sum;
                sum += map01[i];
            }

            unsigned int count = 0;
            for (unsigned int i = 0; i < n; i++) {
                //if (scanned[i] != scanned[i - 1]) {
                if (map01[i] == 1) {
                    odata[scanned[i]] = idata[i];
                    count++;
                }
            }
            
	        timer().endCpuTimer();

            delete map01;
            delete scanned;

            return count;
        }
    }
}
