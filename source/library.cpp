#include <emscripten/bind.h>

#include <exception>

//
// DEBUG SHIT FOR EXCEPTIONS COZ EMSCRIPTEN DOESNT ADD THIS FOR SOME REASON ???
//
std::string getExceptionMessage(int exceptionPtr) {
    return {reinterpret_cast<std::exception *>(exceptionPtr)->what()};
}

EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::function("getExceptionMessage", &getExceptionMessage);
}