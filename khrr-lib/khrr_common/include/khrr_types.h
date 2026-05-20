#pragma once

namespace khrr_common {

using real = double;

struct alignas(8) real3 {
    double x;
    double y;
    double z;
};

}