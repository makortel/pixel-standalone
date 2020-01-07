#ifndef cupla_check_h_
#define cupla_check_h_

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// Cuipla headers
#include <cupla/api/common.hpp>

namespace cupla {

  [[noreturn]] inline void abortOnCuplaError(
      const char* file, int line, const char* cmd, const char* message, const char* description = nullptr) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "CUPLA_CHECK(" << cmd << ");\n";
    out << message << "\n";
    if (description)
      out << description << "\n";
    throw std::runtime_error(out.str());
  }

  // check the result of a Cupla function
  inline bool check_(
      const char* file, int line, const char* cmd, cuplaError_t result, const char* description = nullptr) {
    if (__builtin_expect(result == cuplaSuccess, true))
      return true;

    const char* message = cuplaGetErrorString(result);
    abortOnCuplaError(file, line, cmd, message, description);
    return false;
  }

}  // namespace cupla

#define CUPLA_CHECK(ARG, ...) (cupla::check_(__FILE__, __LINE__, #ARG, (ARG), ##__VA_ARGS__))

#endif  // cupla_check_h_
