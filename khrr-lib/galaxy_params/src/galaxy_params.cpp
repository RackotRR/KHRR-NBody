#include <galaxy_params.h>
#include <range/v3/all.hpp>

namespace khrr_galaxy_params {

size_t count_stars(const GalaxiesParams& galaxies_params) {
    return ranges::accumulate(
        galaxies_params.galaxies
            | ranges::views::transform(&GalaxyParams::N_s),
        0ull
    );
}

size_t count_dm(const GalaxiesParams& galaxies_params) {
    return ranges::accumulate(
        galaxies_params.galaxies
            | ranges::views::transform(&GalaxyParams::N_dm),
        0ull
    );
}

}