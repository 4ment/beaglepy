// Copyright 2023 Mathieu Fourment.
// beaglepy is free software under the GPLv3; see LICENSE file for details.

#pragma once

#include <string>

namespace beagle {
using std::string;

static const char *errors[] = {
    "Success",
    "Unspecified error",
    "Not enough memory could be allocated",
    "Unspecified exception",
    "The instance index is out of range, or the instance has not been created",
    "One of the indices specified exceeded the range of the array",
    "No resource matches requirements",
    "No implementation matches requirements",
    "Floating-point error (e.g., NaN)"};

class BeagleException : public std::exception {
 public:
  explicit BeagleException(const char *functionName, int errCode)
      : functionName_{functionName}, errCode_(errCode) {
    message_ = "BEAGLE function, " + functionName_ + ", returned error code " +
               std::to_string(errCode_);
    if (errCode_ > 0 || errCode_ < -8) {
      message_ += " (unrecognized error code)";
    } else {
      message_ += " (" + string(errors[-errCode_]) + ")";
    }
  }

  const char *what() const noexcept override { return message_.c_str(); }

  string GetFunctionName() const { return functionName_; }

  int GetErrorCode() const { return errCode_; }

 private:
  string functionName_;
  int errCode_;
  string message_;
};
}  // namespace beagle
