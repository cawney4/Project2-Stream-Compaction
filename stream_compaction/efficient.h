#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        void scan(int n, int *odata, const int *idata, const bool time);

        int compact(int n, int *odata, const int *idata);
    }
}
