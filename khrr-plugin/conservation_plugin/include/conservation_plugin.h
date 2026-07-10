#pragma once
#include <memory>
#include <cstdint>
#include <khrr_types.h>
#include <particle_data.h>
#include <spdlog/spdlog.h>

#include <range/v3/all.hpp>

namespace khrr_plugin::conservation {

using khrr_common::real;
using khrr_common::real3;

using vec_real = std::shared_ptr<std::vector<real>>;
using vec_real3 = std::shared_ptr<std::vector<real3>>;
struct Context{
    vec_real3 positions;
    vec_real3 velocities;
    vec_real masses;
    vec_real grav_potential;

    size_t size() const {
        assert(positions);
        return positions->size();
    }
    bool is_valid() const {
        static constexpr const char ERR_NOT_SET[] = "{} invalid: {} not set";
        static constexpr const char ERR_MISMATCH[] = "{} invalid: {} size mismatch: {} vs {} on positions";

        if (!positions) {
            spdlog::error(ERR_NOT_SET, NAME, "positions");
            return false;
        }

        if (!velocities) {
            spdlog::error(ERR_NOT_SET, NAME, "velocities");
            return false;
        }

        if (!masses) {
            spdlog::error(ERR_NOT_SET, NAME, "masses");
            return false;
        }

        if (!grav_potential) {
            spdlog::error(ERR_NOT_SET, NAME, "grav_potential");
            return false;
        }

        const size_t N = positions->size();

        if (velocities->size() != N) {
            spdlog::error(ERR_MISMATCH, NAME, "velocities", velocities->size(), N);
            return false;
        }

        if (masses->size() != N) {
            spdlog::error(ERR_MISMATCH, NAME, "masses", masses->size(), N);
            return false;
        }

        if (grav_potential->size() != N) {
            spdlog::error(ERR_MISMATCH, NAME, "grav_potential", grav_potential->size(), N);
            return false;
        }

        return true;
    }

private:
    static constexpr const char NAME[] = "Conservation plugin context";
};

struct ConservationInfo {
    real energy{
        std::numeric_limits<double>::quiet_NaN()
    };
    real3 angular_momentum{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    };
    real3 momentum{
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN()
    };
};

class ConservationPlugin {
public:
    void init_with_context(
        Context context
    ) {
        spdlog::info("Init conservation plugin with context");
        initialized = true;
    }

    void init_with_values(
        const ConservationInfo& info
    ) {
        spdlog::info("Init conservation plugin with values");
        spdlog::info("-- Initial enegy: {}", info.energy);
        spdlog::info(
            "-- Initial angular_momentum: ({}; {}; {})",
            info.angular_momentum.x,
            info.angular_momentum.y,
            info.angular_momentum.z
        );
        spdlog::info(
            "-- Initial momentum: ({}; {}; {})",
            info.momentum.x,
            info.momentum.y,
            info.momentum.z
        );

        this->initial_info.energy = info.energy;
        this->initial_info.angular_momentum = info.angular_momentum;
        this->initial_info.momentum = info.momentum;
        initialized = true;
    }

    ConservationInfo calc_conservation(
        Context context
    ) {
        spdlog::info("Calc conservation");
        if (!context.is_valid()) {
            return;
        }

        const size_t N = context.size();

        auto data = ranges::views::zip(
            *context.positions,
            *context.velocities,
            *context.masses,
            *context.grav_potential
        );

        for (auto&& [pos, vel, mass, grav] : data) {

        }
    }

    void check_conservation(
        Context context
    ) {
        spdlog::info("Check conservation");
        if (!context.is_valid()) {
            return;
        }

        const size_t N = context.size();

        auto data = ranges::views::zip(
            *context.positions,
            *context.velocities,
            *context.masses,
            *context.grav_potential
        );

        for (auto&& [pos, vel, mass, grav] : data) {

        }
    }

private:
    ConservationInfo initial_info;
    bool initialized = false;
};

} // khrr_plugin::conservation