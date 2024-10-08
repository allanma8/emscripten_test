#pragma once

#include <cstddef>

namespace type {
    struct image_size_t {
        size_t width;
        size_t height;
        size_t channels;

        //! Get size
        //! \note size in bytes is `size * sizeof(array type)`
        [[nodiscard]] constexpr size_t size() const {
            return width * height * channels;
        }
    };
}
