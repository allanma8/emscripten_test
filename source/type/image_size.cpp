#include <type/image_size.hpp>

#include <emscripten/bind.h>

using namespace type;

EMSCRIPTEN_BINDINGS(type_module) {
    emscripten::value_object<image_size_t>("image_size_t")
        .field("width", &image_size_t::width)
        .field("height", &image_size_t::height)
        .field("channels", &image_size_t::channels);
}
