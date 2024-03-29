// Copyright 2023 Mathieu Fourment.
// beaglepy is free software under the GPLv3; see LICENSE file for details.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "exception.hpp"
#include "libhmsbeagle/beagle.h"

namespace py = pybind11;
using beagle::BeagleException;

using double_np =
    py::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;

PYBIND11_MODULE(beagle, m) {
  m.doc() = R"pbdoc(Python interface for BEAGLE)pbdoc";

  py::register_exception<BeagleException>(m, "BeagleException");

  py::enum_<BeagleReturnCodes>(m, "BeagleReturnCodes", py::arithmetic(),
                               R"pbdoc(
    Error return codes

    This enumerates all possible BEAGLE return codes.  Error codes are always negative.
    )pbdoc")
      .value("BEAGLE_SUCCESS", BEAGLE_SUCCESS, "Success")
      .value("BEAGLE_ERROR_GENERAL", BEAGLE_ERROR_GENERAL, "Unspecified error")
      .value("BEAGLE_ERROR_OUT_OF_MEMORY", BEAGLE_ERROR_OUT_OF_MEMORY,
             "Not enough memory could be allocated")
      .value("BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION",
             BEAGLE_ERROR_UNIDENTIFIED_EXCEPTION, "Unspecified exception")
      .value("BEAGLE_ERROR_UNINITIALIZED_INSTANCE",
             BEAGLE_ERROR_UNINITIALIZED_INSTANCE,
             "The instance index is out of range, or the instance has not been "
             "created")
      .value("BEAGLE_ERROR_OUT_OF_RANGE", BEAGLE_ERROR_OUT_OF_RANGE,
             "One of the indices specified exceeded the range of the array")
      .value("BEAGLE_ERROR_NO_RESOURCE", BEAGLE_ERROR_NO_RESOURCE,
             "No resource matches requirements")
      .value("BEAGLE_ERROR_NO_IMPLEMENTATION", BEAGLE_ERROR_NO_IMPLEMENTATION,
             "No implementation matches requirements")
      .value("BEAGLE_ERROR_FLOATING_POINT", BEAGLE_ERROR_FLOATING_POINT,
             "Floating-point error (e.g., NaN)")
      .export_values();

  py::enum_<BeagleFlags>(m, "BeagleFlags", py::arithmetic(), R"pbdoc(
    Hardware and implementation capability flags
    This enumerates all possible hardware and implementation capability flags.
    Each capability is a bit in a 'long'
    )pbdoc")
      .value("BEAGLE_FLAG_PRECISION_SINGLE", BEAGLE_FLAG_PRECISION_SINGLE,
             "Single precision computation")
      .value("BEAGLE_FLAG_PRECISION_DOUBLE", BEAGLE_FLAG_PRECISION_DOUBLE,
             "Double precision computation")

      .value("BEAGLE_FLAG_COMPUTATION_SYNCH", BEAGLE_FLAG_COMPUTATION_SYNCH,
             "Synchronous computation (blocking)")
      .value("BEAGLE_FLAG_COMPUTATION_ASYNCH", BEAGLE_FLAG_COMPUTATION_ASYNCH,
             "Asynchronous computation (non-blocking)")

      .value("BEAGLE_FLAG_EIGEN_REAL", BEAGLE_FLAG_EIGEN_REAL,
             "Real eigenvalue computation")
      .value("BEAGLE_FLAG_EIGEN_COMPLEX", BEAGLE_FLAG_EIGEN_COMPLEX,
             "Complex eigenvalue computation")

      .value("BEAGLE_FLAG_SCALING_MANUAL", BEAGLE_FLAG_SCALING_MANUAL,
             "Manual scaling")
      .value("BEAGLE_FLAG_SCALING_AUTO", BEAGLE_FLAG_SCALING_AUTO,
             "Auto-scaling on (deprecated, may not work correctly)")
      .value(
          "BEAGLE_FLAG_SCALING_ALWAYS", BEAGLE_FLAG_SCALING_ALWAYS,
          "Scale at every updatePartials (deprecated, may not work correctly)")
      .value("BEAGLE_FLAG_SCALING_DYNAMIC", BEAGLE_FLAG_SCALING_DYNAMIC,
             "Manual scaling with dynamic checking (deprecated, may not work "
             "correctly)")

      .value("BEAGLE_FLAG_SCALERS_RAW", BEAGLE_FLAG_SCALERS_RAW,
             "Save raw scalers")
      .value("BEAGLE_FLAG_SCALERS_LOG", BEAGLE_FLAG_SCALERS_LOG,
             "Save log scalers")

      .value("BEAGLE_FLAG_INVEVEC_STANDARD", BEAGLE_FLAG_INVEVEC_STANDARD,
             "Inverse eigen vectors passed to BEAGLE have not been transposed")
      .value("BEAGLE_FLAG_INVEVEC_TRANSPOSED", BEAGLE_FLAG_INVEVEC_TRANSPOSED,
             "Inverse eigen vectors passed to BEAGLE have been transposed")

      .value("BEAGLE_FLAG_VECTOR_SSE", BEAGLE_FLAG_VECTOR_SSE,
             "SSE computation")
      .value("BEAGLE_FLAG_VECTOR_AVX", BEAGLE_FLAG_VECTOR_AVX,
             "AVX computation")
      .value("BEAGLE_FLAG_VECTOR_NONE", BEAGLE_FLAG_VECTOR_NONE,
             "No vector computation")

      .value("BEAGLE_FLAG_THREADING_CPP", BEAGLE_FLAG_THREADING_CPP,
             "C++11 threading")
      .value("BEAGLE_FLAG_THREADING_OPENMP", BEAGLE_FLAG_THREADING_OPENMP,
             "OpenMP threading")
      .value("BEAGLE_FLAG_THREADING_NONE", BEAGLE_FLAG_THREADING_NONE,
             "No threading (default)")

      .value("BEAGLE_FLAG_PROCESSOR_CPU", BEAGLE_FLAG_PROCESSOR_CPU,
             "Use CPU as main processor")
      .value("BEAGLE_FLAG_PROCESSOR_GPU", BEAGLE_FLAG_PROCESSOR_GPU,
             "Use GPU as main processor")
      .value("BEAGLE_FLAG_PROCESSOR_FPGA", BEAGLE_FLAG_PROCESSOR_FPGA,
             "Use FPGA as main processor")
      .value("BEAGLE_FLAG_PROCESSOR_CELL", BEAGLE_FLAG_PROCESSOR_CELL,
             "Use Cell as main processor")
      .value("BEAGLE_FLAG_PROCESSOR_PHI", BEAGLE_FLAG_PROCESSOR_PHI,
             "Use Intel Phi as main processor")
      .value("BEAGLE_FLAG_PROCESSOR_OTHER", BEAGLE_FLAG_PROCESSOR_OTHER,
             "Use other type of processor")

      .value("BEAGLE_FLAG_FRAMEWORK_CUDA", BEAGLE_FLAG_FRAMEWORK_CUDA,
             "Use CUDA implementation with GPU resources")
      .value("BEAGLE_FLAG_FRAMEWORK_OPENCL", BEAGLE_FLAG_FRAMEWORK_OPENCL,
             "Use OpenCL implementation with GPU resources")
      .value("BEAGLE_FLAG_FRAMEWORK_CPU", BEAGLE_FLAG_FRAMEWORK_CPU,
             "Use CPU implementation")

      .value("BEAGLE_FLAG_PARALLELOPS_STREAMS", BEAGLE_FLAG_PARALLELOPS_STREAMS,
             "Operations in updatePartials may be assigned to separate device "
             "streams")
      .value("BEAGLE_FLAG_PARALLELOPS_GRID", BEAGLE_FLAG_PARALLELOPS_GRID,
             "Operations in updatePartials may be folded into single kernel "
             "launch (necessary for partitions; typically performs better for "
             "problems with fewer pattern sites)")
      .export_values();

  py::enum_<BeagleBenchmarkFlags>(m, "BeagleBenchmarkFlags", py::arithmetic(),
                                  R"pbdoc(
    Benchmarking mode flags for resource performance evaluation with beagleGetOrderedResourceList
    This enumerates all possible benchmarking mode flags.
    Each mode is a bit in a 'long'.
    )pbdoc")
      .value("BEAGLE_BENCHFLAG_SCALING_NONE", BEAGLE_BENCHFLAG_SCALING_NONE,
             "No scaling")
      .value("BEAGLE_BENCHFLAG_SCALING_ALWAYS", BEAGLE_BENCHFLAG_SCALING_ALWAYS,
             "Scale at every iteration")
      .value("BEAGLE_BENCHFLAG_SCALING_DYNAMIC",
             BEAGLE_BENCHFLAG_SCALING_DYNAMIC,
             "Scale every fixed number of iterations or when a numerical error "
             "occurs, and re-use scale factors for subsequent iterations")
      .export_values();

  py::enum_<BeagleOpCodes>(m, "BeagleOpCodes", py::arithmetic(), R"pbdoc(
    Operation codes

    This enumerates all possible BEAGLE operation codes.
    )pbdoc")
      .value("BEAGLE_OP_COUNT", BEAGLE_OP_COUNT,
             "Total number of integers per beagleUpdatePartials operation")
      .value("BEAGLE_PARTITION_OP_COUNT", BEAGLE_PARTITION_OP_COUNT,
             "Total number of integers per beagleUpdatePartialsByPartition "
             "operation")
      .value("BEAGLE_OP_NONE", BEAGLE_OP_NONE,
             "Specify no use for indexed buffer")
      .export_values();

  py::class_<BeagleInstanceDetails>(m, "BeagleInstanceDetails",
                                    "Information about a specific instance")
      .def(py::init<>([] { return BeagleInstanceDetails(); }))
      .def_readwrite("resource_number", &BeagleInstanceDetails::resourceNumber,
                     "Resource upon which instance is running")
      .def_readwrite("resource_name", &BeagleInstanceDetails::resourceName,
                     "Name of resource on which this instance is running")
      .def_readwrite("impl_name", &BeagleInstanceDetails::implName,
                     "Name of implementation on which this instance is running")
      .def_readwrite("impl_description",
                     &BeagleInstanceDetails::implDescription,
                     "Description of implementation with details such as how "
                     "auto-scaling is performed")
      .def_readwrite("flags", &BeagleInstanceDetails::flags,
                     "Bit-flags that characterize the activate capabilities of "
                     "the resource and implementation for this instance");

  py::class_<BeagleResource>(m, "BeagleResource",
                             "Description of a hardware resource")
      .def_readwrite("name", &BeagleResource::name, "Name of resource")
      .def_readwrite("description", &BeagleResource::description,
                     "Description of resource")
      .def_readwrite("support_flags", &BeagleResource::supportFlags,
                     "Bit-flags of supported capabilities on resource")
      .def_readwrite("required_flags", &BeagleResource::requiredFlags,
                     "Bit-flags that identify resource type");

  py::class_<BeagleResourceList>(m, "BeagleResourceList",
                                 "List of hardware resources")
      .def_readwrite("list", &BeagleResourceList::list,
                     "Pointer list of resources")
      .def_readwrite("length", &BeagleResourceList::length, "Length of list");

  py::class_<BeagleBenchmarkedResource>(
      m, "BeagleBenchmarkedResource",
      "Description of a benchmarked hardware resource")
      .def_readwrite("number", &BeagleBenchmarkedResource::number,
                     "Resource number")
      .def_readwrite("name", &BeagleBenchmarkedResource::name,
                     "Name of resource")
      .def_readwrite("description", &BeagleBenchmarkedResource::description,
                     "Description of resource")
      .def_readwrite("support_flags", &BeagleBenchmarkedResource::supportFlags,
                     "Bit-flags of supported capabilities on resource")
      .def_readwrite("required_flags",
                     &BeagleBenchmarkedResource::requiredFlags,
                     "Bit-flags that identify resource type")
      .def_readwrite(
          "return_code", &BeagleBenchmarkedResource::returnCode,
          "Return code of for benchmark attempt (see BeagleReturnCodes)")
      .def_readwrite("impl_name", &BeagleBenchmarkedResource::implName,
                     "Name of implementation used to benchmark resource")
      .def_readwrite("benched_flags", &BeagleBenchmarkedResource::benchedFlags,
                     "Bit-flags that characterize the activate capabilities of "
                     "the resource and implementation for this benchmark")
      .def_readwrite("benchmark_result",
                     &BeagleBenchmarkedResource::benchmarkResult,
                     "Benchmark result in milliseconds")
      .def_readwrite("performance_ratio",
                     &BeagleBenchmarkedResource::performanceRatio,
                     "Performance ratio relative to default CPU resource");

  m.def("get_version", &beagleGetVersion, R"pbdoc(
    Get version

    This function returns a string with the library version number.

    :return: A string with the version number
    )pbdoc");

  m.def("get_citation", &beagleGetCitation, R"pbdoc(
    Get citation

    This function returns a string describing the version of the library and how to cite it.

    :return: A string describing the version of the library and how to cite it
    )pbdoc");

  m.def("get_resource_list",
        []() {
          BeagleResourceList* resourceList = beagleGetResourceList();
          std::vector<BeagleResource> resources;
          for (int i = 0; i < resourceList->length; i++) {
            resources.push_back(resourceList->list[i]);
          }
          return resources;
        },
        R"pbdoc(
    Get list of hardware resources

    This function returns a list of BeagleResource describing the available hardware resources.

    :return: A list of hardware resources available to the library
    )pbdoc");

  m.def(
      "get_benchmarked_resource_list",
      [](int tipCount, int compactBufferCount, int stateCount, int patternCount,
         int categoryCount, std::vector<int> resourceList, int resourceCount,
         long preferenceFlags, long requirementFlags, int eigenModelCount,
         int partitionCount, int calculateDerivatives, long benchmarkFlags) {
        BeagleBenchmarkedResourceList* benchmarkedResourceList =
            beagleGetBenchmarkedResourceList(
                tipCount, compactBufferCount, stateCount, patternCount,
                categoryCount, resourceList.data(), resourceCount,
                preferenceFlags, requirementFlags, eigenModelCount,
                partitionCount, calculateDerivatives, benchmarkFlags);
        std::vector<BeagleBenchmarkedResource> resources;
        if (benchmarkedResourceList == NULL) {
          return resources;
        }
        for (int i = 0; i < benchmarkedResourceList->length; i++) {
          resources.push_back(benchmarkedResourceList->list[i]);
        }
        return resources;
      },
      R"pbdoc(
    Get a benchmarked list of hardware resources for the given analysis parameters

    This function returns a list of BeagleBenchmarkedResource describing the available hardware resources with
    benchmark times and CPU performance ratios for each resource. Resources are benchmarked
    with the given analysis parameters and the array is ordered from fastest to slowest.
    If there is an error the function returns an empty list.

    :param tipCount:              Number of tip data elements (input)
    :param compactBufferCount:    Number of compact state representation tips (input)
    :param stateCount:            Number of states in the continuous-time Markov chain (input)
    :param patternCount:          Number of site patterns (input)
    :param categoryCount:         Number of rate categories (input)
    :param resourceList:          List of resources to be benchmarked, None implies no restriction (input)
    :param resourceCount:         Length of resourceList list (input)
    :param preferenceFlags:       Bit-flags indicating preferred implementation characteristics, see BeagleFlags (input)
    :param requirementFlags:      Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
    :param eigenModelCount:       Number of full-alignment rate matrix eigen-decomposition models (input)
    :param partitionCount:        Number of partitions (input)
    :param calculateDerivatives:  Indicates if calculation of derivatives are required (input)
    :param benchmarkFlags:        Bit-flags indicating benchmarking preferences (input)

    :return: An ordered (fastest to slowest) list of hardware resources available to the library for the specified analysis parameters
    )pbdoc");

  m.def("create_instance",
        [](int tipCount, int partialsBufferCount, int compactBufferCount,
           int stateCount, int patternCount, int eigenBufferCount,
           int matrixBufferCount, int categoryCount, int scaleBufferCount,
           std::optional<std::vector<int>> resourceList, int resourceCount,
           long preferenceFlags, long requirementFlags,
           BeagleInstanceDetails& returnInfo) {
          int instance = beagleCreateInstance(
              tipCount, partialsBufferCount, compactBufferCount, stateCount,
              patternCount, eigenBufferCount, matrixBufferCount, categoryCount,
              scaleBufferCount,
              resourceList.has_value() ? resourceList->data() : nullptr,
              resourceCount, preferenceFlags, requirementFlags, &returnInfo);
          return instance;
        },
        py::arg("tipCount"), py::arg("partialsBufferCount"),
        py::arg("compactBufferCount"), py::arg("stateCount"),
        py::arg("patternCount"), py::arg("eigenBufferCount"),
        py::arg("matrixBufferCount"), py::arg("categoryCount"),
        py::arg("scaleBufferCount"), py::arg("resourceList") = py::none(),
        py::arg("resourceCount"), py::arg("preferenceFlags"),
        py::arg("requirementFlags"), py::arg("returnInfo"),
        R"pbdoc(
    Create a single instance

    This function creates a single instance of the BEAGLE library and can be called
    multiple times to create multiple data partition instances each returning a unique
    identifier.

    :param tipCount:              Number of tip data elements (input)
    :param partialsBufferCount:   Number of partials buffers to create (input)
    :param compactBufferCount:    Number of compact state representation buffers to create (input)
    :param stateCount:            Number of states in the continuous-time Markov chain (input)
    :param patternCount:          Number of site patterns to be handled by the instance (input)
    :param eigenBufferCount:      Number of rate matrix eigen-decomposition, category weight, category rates, and state frequency buffers to allocate (input)
    :param matrixBufferCount:     Number of transition probability matrix buffers (input)
    :param categoryCount:         Number of rate categories (input)
    :param scaleBufferCount:      Number of scale buffers to create, ignored for auto scale or always scale (input)
    :param resourceList:          List of potential resources on which this instance is allowed (input, None implies no restriction)
    :param resourceCount:         Length of resourceList list (input)
    :param preferenceFlags:       Bit-flags indicating preferred implementation characteristics, see BeagleFlags (input)
    :param requirementFlags:      Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
    :param returnInfo:            Object to return implementation and resource details (see :class:`BeagleInstanceDetails`)

    :returns: the unique instance identifier (<0 if failed, see BEAGLE_RETURN_CODES :class:`BeagleReturnCodes`)
    )pbdoc");

  m.def("finalize_instance",
        [](int instance) {
          int errCode = beagleFinalizeInstance(instance);
          if (errCode != 0) {
            throw BeagleException("finalize_instance", errCode);
          }
        },
        R"pbdoc(
    Finalize this instance

    This function finalizes the instance by releasing allocated memory.
    :param instance:  Instance number

    :raises BeagleException:
    )pbdoc");

  m.def("finalize",
        []() {
          int errCode = beagleFinalize();
          if (errCode != 0) {
            throw BeagleException("finalize", errCode);
          }
        },
        R"pbdoc(
    Finalize the library

    This function finalizes the library and releases all allocated memory.
    This function is automatically called under GNU C via __attribute__ ((destructor)).

    :raises BeagleException:
    )pbdoc");

  m.def("set_CPU_thread_count",
        [](int instance, int threadCount) {
          int errCode = beagleSetCPUThreadCount(instance, threadCount);
          if (errCode != 0) {
            throw BeagleException("set_CPU_thread_count", errCode);
          }
        },
        R"pbdoc(
    Set number of threads for native CPU implementation

    This function sets the max number of worker threads to be used with a native CPU
    implementation. It should only be called after beagleCreateInstance and requires the
    BEAGLE_FLAG_THREADING_CPP flag to be set. It has no effect on GPU-based
    implementations. It has no effect with the default BEAGLE_FLAG_THREADING_NONE setting.
    If BEAGLE_FLAG_THREADING_CPP is set and this function is not called BEAGLE will use 
    a heuristic to set an appropriate number of threads.

    :param instance:             Instance number (input)
    :param threadCount:          Number of threads (input)

    :raises BeagleException:
    )pbdoc");

  m.def("set_tip_states",
        [](int instance, int tipIndex, const std::vector<int>& inStates) {
          int errCode = beagleSetTipStates(instance, tipIndex, inStates.data());
          if (errCode != 0) {
            throw BeagleException("set_tip_states", errCode);
          }
        },
        R"pbdoc(
    Set the compact state representation for tip node

    This function copies a compact state representation into an instance buffer.
    Compact state representation is an array of states: 0 to stateCount - 1 (missing = stateCount).
    The inStates array should be patternCount in length (replication across categoryCount is not
    required).

    :param instance:  Instance number (input)
    :param tipIndex:  Index of destination compactBuffer (input)
    :param inStates:  Source of compact states (input)

    :raises BeagleException:
    )pbdoc");

  m.def("set_tip_partials",
        [](int instance, int tipIndex, const double_np& inPartials) {
          int errCode =
              beagleSetTipPartials(instance, tipIndex, inPartials.data());
          if (errCode != 0) {
            throw BeagleException("set_tip_partials", errCode);
          }
        },
        R"pbdoc(
    Set an instance partials buffer for tip node

    This function copies an array of partials into an instance buffer. The inPartials array should
    be stateCount * patternCount in length. For most applications this will be used
    to set the partial likelihoods for the observed states. Internally, the partials will be copied
    categoryCount times.

    :param instance:      Instance number in which to set a partialsBuffer (input)
    :param tipIndex:      Index of destination partialsBuffer (input)
    :param inPartials:    Source partials values to set (input)

    :raises BeagleException:
    )pbdoc");

  m.def(
      "set_partials",
      [](int instance, int bufferIndex, const std::vector<double>& inPartials) {
        int errCode =
            beagleSetPartials(instance, bufferIndex, inPartials.data());
        if (errCode != 0) {
          throw BeagleException("set_partials", errCode);
        }
      },
      R"pbdoc(
    Set an instance partials buffer

    This function copies an array of partials into an instance buffer. The inPartials array should
    be stateCount * patternCount * categoryCount in length.

    :param instance:      Instance number in which to set a partialsBuffer (input)
    :param bufferIndex:   Index of destination partialsBuffer (input)
    :param inPartials:    Source partials values to set (input)

    :raises BeagleException:
    )pbdoc");

  m.def(
      "get_partials",
      [](int instance, int bufferIndex, int scaleIndex, double_np& inPartials) {
        int errCode = beagleGetPartials(instance, bufferIndex, scaleIndex,
                                        inPartials.mutable_data());
        if (errCode != 0) {
          throw BeagleException("get_partials", errCode);
        }
      },
      R"pbdoc(
    Get partials from an instance buffer

    This function copies an instance buffer into the array outPartials. The outPartials array should
    be stateCount * patternCount * categoryCount in length.

    :param instance:      Instance number from which to get partialsBuffer (input)
    :param bufferIndex:   Index of source partialsBuffer (input)
    :param scaleIndex:    Index of scaleBuffer to apply to partialsBuffer (input)
    :param outPartials:   Destination partialsBuffer (output)

    :raises BeagleException:
    )pbdoc");

  m.def("set_eigen_decomposition",
        [](int instance, int eigenIndex, double_np inEigenVectors,
           double_np inInverseEigenVectors, double_np inEigenValues) {
          int errCode = beagleSetEigenDecomposition(
              instance, eigenIndex, inEigenVectors.data(),
              inInverseEigenVectors.data(), inEigenValues.data());
          if (errCode != 0) {
            throw BeagleException("set_eigen_decomposition", errCode);
          }
        },
        R"pbdoc(
    Set an eigen-decomposition buffer

    This function copies an eigen-decomposition into an instance buffer.

    :param instance:              Instance number (input)
    :param eigenIndex:            Index of eigen-decomposition buffer (input)
    :param inEigenVectors:        Flattened matrix (stateCount x stateCount) of eigen-vectors (input)
    :param inInverseEigenVectors: Flattened matrix (stateCount x stateCount) of inverse-eigen- vectors (input)
    :param inEigenValues:         Vector of eigenvalues

    :raises BeagleException:
    )pbdoc");

  m.def("set_state_frequencies",
        [](int instance, int stateFrequenciesIndex,
           const std::vector<double>& inStateFrequencies) {
          int errCode = beagleSetStateFrequencies(
              instance, stateFrequenciesIndex, inStateFrequencies.data());
          if (errCode != 0) {
            throw BeagleException("set_state_frequencies", errCode);
          }
        },
        R"pbdoc(
    Set a state frequency buffer

    This function copies a state frequency array into an instance buffer.

    :param instance:              Instance number (input)
    :param stateFrequenciesIndex: Index of state frequencies buffer (input)
    :param inStateFrequencies:    State frequencies array (stateCount) (input)

    :raises BeagleException:
    )pbdoc");

  m.def("set_category_weights",
        [](int instance, int categoryWeightsIndex,
           const std::vector<double>& inCategoryWeights) {
          int errCode = beagleSetCategoryWeights(instance, categoryWeightsIndex,
                                                 inCategoryWeights.data());
          if (errCode != 0) {
            throw BeagleException("set_category_weights", errCode);
          }
        },
        R"pbdoc(
    Set a category weights buffer

    This function copies a category weights array into an instance buffer.

    :param instance:              Instance number (input)
    :param categoryWeightsIndex:  Index of category weights buffer (input)
    :param inCategoryWeights:     Category weights array (categoryCount) (input)

    :raises BeagleException:
    )pbdoc");

  m.def("set_category_rates",
        [](int instance, const std::vector<double>& inCategoryRates) {
          int errCode =
              beagleSetCategoryRates(instance, inCategoryRates.data());
          if (errCode != 0) {
            throw BeagleException("set_category_rates", errCode);
          }
        },
        R"pbdoc(
    Set the default category rates buffer

    This function sets the default vector of category rates for an instance.

    :param instance:              Instance number (input)
    :param inCategoryRates:       Array containing categoryCount rate scalers (input)

    :raises BeagleException:
    )pbdoc");

  m.def("set_category_rates_with_index",
        [](int instance, int categoryRatesIndex,
           const std::vector<double>& inCategoryRates) {
          int errCode = beagleSetCategoryRatesWithIndex(
              instance, categoryRatesIndex, inCategoryRates.data());
          if (errCode != 0) {
            throw BeagleException("set_category_rates_with_index", errCode);
          }
        },
        R"pbdoc(
    Set a category rates buffer

    This function sets the vector of category rates for a given buffer in an instance.

    :param instance:              Instance number (input)
    :param categoryRatesIndex:    Index of category rates buffer (input)
    :param inCategoryRates:       Array containing categoryCount rate scalers (input)

    :raises BeagleException:
    )pbdoc");

  m.def("set_pattern_weights",
        [](int instance, const std::vector<double>& inPatternWeights) {
          int errCode =
              beagleSetPatternWeights(instance, inPatternWeights.data());
          if (errCode != 0) {
            throw BeagleException("set_pattern_weights", errCode);
          }
        },
        R"pbdoc(
    Set pattern weights

    This function sets the vector of pattern weights for an instance.

    :param instance:              Instance number (input)
    :param inPatternWeights:      Array containing patternCount weights (input)

    :raises BeagleException:
    )pbdoc");

  m.def("set_pattern_partitions",
        [](int instance, int partitionCount,
           const std::vector<int>& inPatternWeights) {
          int errCode = beagleSetPatternPartitions(instance, partitionCount,
                                                   inPatternWeights.data());
          if (errCode != 0) {
            throw BeagleException("set_pattern_partitions", errCode);
          }
        },
        R"pbdoc(
    Set pattern partition assignments

    This function sets the vector of pattern partition indices for an instance. It should
    only be called after beagleSetTipPartials and beagleSetPatternWeights.

    :param instance:             Instance number (input)
    :param partitionCount:       Number of partitions (input)
    :param inPatternPartitions:  Array containing partitionCount partition indices (input)

    :raises BeagleException:
    )pbdoc");

  m.def("convolve_transition_matrices",
        [](int instance, const std::vector<int>& firstIndices,
           const std::vector<int>& secondIndices,
           const std::vector<int>& resultIndices, int matrixCount) {
          int errCode = beagleConvolveTransitionMatrices(
              instance, firstIndices.data(), secondIndices.data(),
              resultIndices.data(), matrixCount);
          if (errCode != 0) {
            throw BeagleException("convolve_transition_matrices", errCode);
          }
        },
        R"pbdoc(
    Convolve lists of transition probability matrices

    This function convolves two lists of transition probability matrices.

    :param instance:                  Instance number (input)
    :param firstIndices:              List of indices of the first transition probability matrices to convolve (input)
    :param secondIndices:             List of indices of the second transition probability matrices to convolve (input)
    :param resultIndices:             List of indices of resulting transition probability matrices (input)
    :param matrixCount:               Length of lists

    :raises BeagleException:
    )pbdoc");

  m.def("update_transition_matrices",
        [](int instance, int eigenIndex,
           const std::vector<int>& probabilityIndices,
           std::optional<std::vector<int>> firstDerivativeIndices,
           std::optional<std::vector<int>> secondDerivativeIndices,
           const std::vector<double>& edgeLengths, int count) {
          int errCode = beagleUpdateTransitionMatrices(
              instance, eigenIndex, probabilityIndices.data(),
              firstDerivativeIndices.has_value()
                  ? firstDerivativeIndices->data()
                  : nullptr,
              secondDerivativeIndices.has_value()
                  ? secondDerivativeIndices->data()
                  : nullptr,
              edgeLengths.data(), count);
          if (errCode != 0) {
            throw BeagleException("update_transition_matrices", errCode);
          }
        },
        R"pbdoc(
    Calculate a list of transition probability matrices

    This function calculates a list of transition probabilities matrices and their first and
    second derivatives (if requested).

    :param instance:                  Instance number (input)
    :param eigenIndex:                Index of eigen-decomposition buffer (input)
    :param probabilityIndices:        List of indices of transition probability matrices to update (input)
    :param firstDerivativeIndices:    List of indices of first derivative matrices to update (input, None implies no calculation)
    :param secondDerivativeIndices:   List of indices of second derivative matrices to update (input, None implies no calculation)
    :param edgeLengths:               List of edge lengths with which to perform calculations (input)
    :param count:                     Length of lists

    :raises BeagleException:
    )pbdoc");

  m.def("update_transition_matrices_with_multiple_models",
        [](int instance, const std::vector<int>& eigenIndices,
           const std::vector<int>& categoryRateIndices,
           const std::vector<int>& probabilityIndices,
           std::optional<std::vector<int>> firstDerivativeIndices,
           std::optional<std::vector<int>> secondDerivativeIndices,
           const std::vector<double>& edgeLengths, int count) {
          int errCode = beagleUpdateTransitionMatricesWithMultipleModels(
              instance, eigenIndices.data(), categoryRateIndices.data(),
              probabilityIndices.data(),
              firstDerivativeIndices.has_value()
                  ? firstDerivativeIndices->data()
                  : nullptr,
              secondDerivativeIndices.has_value()
                  ? secondDerivativeIndices->data()
                  : nullptr,
              edgeLengths.data(), count);
          if (errCode != 0) {
            throw BeagleException(
                "update_transition_matrices_with_multiple_models", errCode);
          }
        },
        R"pbdoc(
    Calculate a list of transition probability matrices with multiple models

    This function calculates a list of transition probabilities matrices and their first and
    second derivatives (if requested).

    :param instance:                  Instance number (input)
    :param eigenIndices:              List of indices of eigen-decomposition buffers to use for updates (input)
    :param categoryRateIndices:       List of indices of category-rate buffers to use for updates (input)
    :param probabilityIndices:        List of indices of transition probability matrices to update (input)
    :param firstDerivativeIndices:    List of indices of first derivative matrices to update (input, None implies no calculation)
    :param secondDerivativeIndices:   List of indices of second derivative matrices to update (input, None implies no calculation)
    :param edgeLengths:               List of edge lengths with which to perform calculations (input)
    :param count:                     Length of lists

    :raises BeagleException:
    )pbdoc");

  m.def("set_transition_matrix",
        [](int instance, int matrixIndex, const std::vector<double>& inMatrix,
           double paddedValue) {
          int errCode = beagleSetTransitionMatrix(instance, matrixIndex,
                                                  inMatrix.data(), paddedValue);
          if (errCode != 0) {
            throw BeagleException("set_transition_matrix", errCode);
          }
        },
        R"pbdoc(
    Set a finite-time transition probability matrix

    This function copies a finite-time transition probability matrix into a matrix buffer. This function
    is used when the application wishes to explicitly set the transition probability matrix rather than
    using the beagleSetEigenDecomposition and beagleUpdateTransitionMatrices functions. The inMatrix array should be
    of size stateCount * stateCount * categoryCount and will contain one matrix for each rate category.

    :param matrixIndex:   Index of matrix buffer (input)
    :param inMatrix:      Source transition probability matrix (input)
    :param paddedValue:   Value to be used for padding for ambiguous states (e.g. 1 for probability matrices, 0 for derivative matrices) (input)

    :raises BeagleException:
    )pbdoc");

  m.def("get_transition_matrix",
        [](int instance, int matrixIndex, double_np& outMatrix) {
          int errCode = beagleGetTransitionMatrix(instance, matrixIndex,
                                                  outMatrix.mutable_data());
          if (errCode != 0) {
            throw BeagleException("get_transition_matrix", errCode);
          }
        },
        R"pbdoc(
    Get a finite-time transition probability matrix

    This function copies a finite-time transition matrix buffer into the array outMatrix. The
    outMatrix array should be of size stateCount * stateCount * categoryCount and will be filled
    with one matrix for each rate category.

    :param instance:     Instance number (input)
    :param matrixIndex:  Index of matrix buffer (input)
    :param outMatrix:    Destination transition probability matrix (output)

    :raises BeagleException:
    )pbdoc");

  m.def("set_transition_matrices",
        [](int instance, const std::vector<int>& matrixIndices,
           const std::vector<double>& inMatrices,
           const std::vector<double>& paddedValues, int count) {
          int errCode = beagleSetTransitionMatrices(
              instance, matrixIndices.data(), inMatrices.data(),
              paddedValues.data(), count);
          if (errCode != 0) {
            throw BeagleException("set_transition_matrices", errCode);
          }
        },
        R"pbdoc(
    Set multiple transition matrices

    This function copies multiple transition matrices into matrix buffers. This function
    is used when the application wishes to explicitly set the transition matrices rather than
    using the beagleSetEigenDecomposition and beagleUpdateTransitionMatrices functions. The inMatrices array should be
    of size stateCount * stateCount * categoryCount * count.

    :param instance:      Instance number (input)
    :param matrixIndices: Indices of matrix buffers (input)
    :param inMatrices:    Source transition matrices (input)
    :param paddedValues:  Values to be used for padding for ambiguous states (e.g. 1 for probability matrices, 0 for derivative matrices) (input)
    :param count:         Number of transition matrices (input)
    :raises BeagleException:
    )pbdoc");

  py::class_<BeagleOperation>(m, "BeagleOperation",
                              "A list of integer indices which specify a "
                              "partial likelihoods operation.")
      .def(py::init<int, int, int, int, int, int, int>())
      .def_readwrite("destination_partials",
                     &BeagleOperation::destinationPartials,
                     "index of destination, or parent, partials buffer")
      .def_readwrite(
          "destination_scale_write", &BeagleOperation::destinationScaleWrite,
          "index of scaling buffer to write to (if set to BEAGLE_OP_NONE then "
          "calculation of new scalers is disabled)")
      .def_readwrite(
          "destination_scale_read", &BeagleOperation::destinationScaleRead,
          "index of scaling buffer to read from (if set to BEAGLE_OP_NONE then "
          "use of existing scale factors is disabled)")
      .def_readwrite("child1_partials", &BeagleOperation::child1Partials,
                     "index of first child partials buffer")
      .def_readwrite(
          "child1_transition_matrix", &BeagleOperation::child1TransitionMatrix,
          "index of transition matrix of first partials child buffer")
      .def_readwrite("child2_partials", &BeagleOperation::child2Partials,
                     "index of second child partials buffer")
      .def_readwrite(
          "child2_transition_matrix", &BeagleOperation::child2TransitionMatrix,
          "index of transition matrix of second partials child buffer");

  m.def("update_partials",
        [](int instance, std::vector<BeagleOperation> operations,
           int operationCount, int cumulativeScaleIndex) {
          int errCode =
              beagleUpdatePartials(instance, operations.data(), operationCount,
                                   cumulativeScaleIndex);
          if (errCode != 0) {
            throw BeagleException("update_partials", errCode);
          }
        },
        R"pbdoc(
    Calculate or queue for calculation partials using a list of operations

    This function either calculates or queues for calculation a list partials. Implementations
    supporting ASYNCH may queue these calculations while other implementations perform these
    operations immediately and in order.

    :param instance:                  Instance number (input)
    :param operations:                BeagleOperation list specifying operations (input)
    :param operationCount:            Number of operations (input)
    :param cumulativeScaleIndex:      Index number of scaleBuffer to store accumulated factors (input)

    :raises BeagleException:
    )pbdoc");

  py::class_<BeagleOperationByPartition>(
      m, "BeagleOperationByPartition",
      "A list of integer indices which specify a partial likelihoods "
      "operation.")
      .def_readwrite("destination_partials",
                     &BeagleOperationByPartition::destinationPartials,
                     "index of destination, or parent, partials buffer")
      .def_readwrite(
          "destination_scale_write",
          &BeagleOperationByPartition::destinationScaleWrite,
          "index of scaling buffer to write to (if set to BEAGLE_OP_NONE then "
          "calculation of new scalers is disabled)")
      .def_readwrite(
          "destination_scale_read",
          &BeagleOperationByPartition::destinationScaleRead,
          "index of scaling buffer to read from (if set to BEAGLE_OP_NONE then "
          "use of existing scale factors is disabled)")
      .def_readwrite("child1_partials",
                     &BeagleOperationByPartition::child1Partials,
                     "index of first child partials buffer")
      .def_readwrite(
          "child1_transition_matrix",
          &BeagleOperationByPartition::child1TransitionMatrix,
          "index of transition matrix of first partials child buffer")
      .def_readwrite("child2_partials",
                     &BeagleOperationByPartition::child2Partials,
                     "index of second child partials buffer")
      .def_readwrite(
          "child2_transition_matrix",
          &BeagleOperationByPartition::child2TransitionMatrix,
          "index of transition matrix of second partials child buffer")
      .def_readwrite("partition", &BeagleOperationByPartition::partition,
                     "index of partition")
      .def_readwrite(
          "cumulative_scale_index",
          &BeagleOperationByPartition::cumulativeScaleIndex,
          "index number of scaleBuffer to store accumulated factors");

  m.def("update_partials_by_partition",
        [](int instance, std::vector<BeagleOperationByPartition> operations,
           int operationCount) {
          int errCode = beagleUpdatePartialsByPartition(
              instance, operations.data(), operationCount);
          if (errCode != 0) {
            throw BeagleException("update_partials_by_partition", errCode);
          }
        },
        R"pbdoc(
    Calculate or queue for calculation partials using a list of partition operations

    This function either calculates or queues for calculation a list partitioned partials. Implementations
    supporting ASYNCH may queue these calculations while other implementations perform these
    operations immediately and in order.

    :param instance:                  Instance number (input)
    :param operations:                BeagleOperation list specifying operations (input)
    :param operationCount:            Number of operations (input)

    :raises BeagleException:
    )pbdoc");

  m.def("wait_for_partials",
        [](int instance, const std::vector<int>& destinationPartials,
           int destinationPartialsCount) {
          int errCode = beagleWaitForPartials(
              instance, destinationPartials.data(), destinationPartialsCount);
          if (errCode != 0) {
            throw BeagleException("wait_for_partials", errCode);
          }
        },
        R"pbdoc(
    Block until all calculations that write to the specified partials have completed.

    This function is optional and only has to be called by clients that "recycle" partials.

    If used, this function must be called after a beagleUpdatePartials call and must refer to
    indices of "destinationPartials" that were used in a previous beagleUpdatePartials
    call.  The library will block until those partials have been calculated.

    :param instance:                  Instance number (input)
    :param destinationPartials:       List of the indices of destinationPartials that must be calculated before the function returns
    :param destinationPartialsCount:  Number of destinationPartials (input)

    :raises BeagleException:
    )pbdoc");

  m.def("accumulate_scale_factors",
        [](int instance, const std::vector<int>& scaleIndices, int count,
           int cumulativeScaleIndex) {
          int errCode = beagleAccumulateScaleFactors(
              instance, scaleIndices.data(), count, cumulativeScaleIndex);
          if (errCode != 0) {
            throw BeagleException("accumulate_scale_factors", errCode);
          }
        },
        R"pbdoc(
    Accumulate scale factors

    This function adds (log) scale factors from a list of scaleBuffers to a cumulative scale
    buffer. It is used to calculate the marginal scaling at a specific node for each site.

    :param instance:                  Instance number (input)
    :param scaleIndices:              List of scaleBuffers to add (input)
    :param count:                     Number of scaleBuffers in list (input)
    :param cumulativeScaleIndex:      Index number of scaleBuffer to accumulate factors into (input)

    :raises BeagleException:
    )pbdoc");

  m.def("accumulate_scale_factors_by_partition",
        [](int instance, const std::vector<int>& scaleIndices, int count,
           int cumulativeScaleIndex, int partitionIndex) {
          int errCode = beagleAccumulateScaleFactorsByPartition(
              instance, scaleIndices.data(), count, cumulativeScaleIndex,
              partitionIndex);
          if (errCode != 0) {
            throw BeagleException("accumulate_scale_factors_by_partition",
                                  errCode);
          }
        },
        R"pbdoc(
    Accumulate scale factors by partition

    This function adds (log) scale factors from a list of scaleBuffers to a cumulative scale
    buffer. It is used to calculate the marginal scaling at a specific node for each site.

    :param instance:                  Instance number (input)
    :param scaleIndices:              List of scaleBuffers to add (input)
    :param count:                     Number of scaleBuffers in list (input)
    :param cumulativeScaleIndex:      Index number of scaleBuffer to accumulate factors into (input)
    :param partitionIndex:            Index of partition to accumulate into (input)

    :raises BeagleException:
    )pbdoc");

  m.def("remove_scale_factors",
        [](int instance, const std::vector<int>& scaleIndices, int count,
           int cumulativeScaleIndex) {
          int errCode = beagleRemoveScaleFactors(instance, scaleIndices.data(),
                                                 count, cumulativeScaleIndex);
          if (errCode != 0) {
            throw BeagleException("remove_scale_factors", errCode);
          }
        },
        R"pbdoc(
    Remove scale factors

    This function removes (log) scale factors from a cumulative scale buffer. The
    scale factors to be removed are indicated in a list of scaleBuffers.

    :param instance:                  Instance number (input)
    :param scaleIndices:              List of scaleBuffers to remove (input)
    :param count:                     Number of scaleBuffers in list (input)
    :param cumulativeScaleIndex:      Index number of scaleBuffer containing accumulated factors (input)

    :raises BeagleException:
    )pbdoc");

  m.def("remove_scale_factors_by_partition",
        [](int instance, const std::vector<int>& scaleIndices, int count,
           int cumulativeScaleIndex, int partitionIndex) {
          int errCode = beagleRemoveScaleFactorsByPartition(
              instance, scaleIndices.data(), count, cumulativeScaleIndex,
              partitionIndex);
          if (errCode != 0) {
            throw BeagleException("remove_scale_factors_by_partition", errCode);
          }
        },
        R"pbdoc(
    Remove scale factors by partition

    This function removes (log) scale factors from a cumulative scale buffer. The
    scale factors to be removed are indicated in a list of scaleBuffers.

    :param instance:                  Instance number (input)
    :param scaleIndices:              List of scaleBuffers to remove (input)
    :param cumulativeScaleIndex:      Index number of scaleBuffer containing accumulated factors (input)
    :param partitionIndex:            Index of partition to remove from (input)

    :raises BeagleException:
    )pbdoc");

  m.def("reset_scale_factors",
        [](int instance, int cumulativeScaleIndex) {
          int errCode = beagleResetScaleFactors(instance, cumulativeScaleIndex);
          if (errCode != 0) {
            throw BeagleException("reset_scale_factors", errCode);
          }
        },
        R"pbdoc(
    Reset scalefactors

    This function resets a cumulative scale buffer.

    :param instance:                  Instance number (input)
    :param cumulativeScaleIndex:      Index number of cumulative scaleBuffer (input)

    :raises BeagleException:
    )pbdoc");

  m.def("reset_scale_factors_by_partition",
        [](int instance, int cumulativeScaleIndex, int partitionIndex) {
          int errCode = beagleResetScaleFactorsByPartition(
              instance, cumulativeScaleIndex, partitionIndex);
          if (errCode != 0) {
            throw BeagleException("reset_scale_factors_by_partition", errCode);
          }
        },
        R"pbdoc(
    Reset scalefactors by partition

    This function resets a cumulative scale buffer.

    :param instance:                  Instance number (input)
    :param cumulativeScaleIndex:      Index number of cumulative scaleBuffer (input)
    :param partitionIndex:            Index of partition to reset (input)

    :raises BeagleException:
    )pbdoc");

  m.def("copy_scale_factors",
        [](int instance, int destScalingIndex, int srcScalingIndex) {
          int errCode = beagleCopyScaleFactors(instance, destScalingIndex,
                                               srcScalingIndex);
          if (errCode != 0) {
            throw BeagleException("copy_scale_factors", errCode);
          }
        },
        R"pbdoc(
    Copy scale factors

    This function copies scale factors from one buffer to another.

    :param instance:                  Instance number (input)
    :param destScalingIndex:          Destination scaleBuffer (input)
    :param srcScalingIndex:           Source scaleBuffer (input)

    :raises BeagleException:
    )pbdoc");

  m.def("get_scale_factors",
        [](int instance, int srcScalingIndex, double_np& outScaleFactors) {
          int errCode = beagleGetScaleFactors(instance, srcScalingIndex,
                                              outScaleFactors.mutable_data());
          if (errCode != 0) {
            throw BeagleException("get_scale_factors", errCode);
          }
        },
        R"pbdoc(
    Get scale factors

    This function retrieves a buffer of scale factors.

    :param instance:                  Instance number (input)
    :param srcScalingIndex:           Source scaleBuffer (input)
    :param outScaleFactors:           Destination scaleFactors (output)

    :raises BeagleException:
    )pbdoc");

  m.def("calculate_root_log_likelihoods",
        [](int instance, const std::vector<int>& bufferIndices,
           const std::vector<int>& categoryWeightsIndices,
           const std::vector<int>& stateFrequenciesIndices,
           const std::vector<int>& cumulativeScaleIndices, int count,
           double_np& outSumLogLikelihood) {
          int errCode = beagleCalculateRootLogLikelihoods(
              instance, bufferIndices.data(), categoryWeightsIndices.data(),
              stateFrequenciesIndices.data(), cumulativeScaleIndices.data(),
              count, outSumLogLikelihood.mutable_data());
          if (errCode != 0) {
            throw BeagleException("calculate_root_log_likelihoods", errCode);
          }
        },
        R"pbdoc(
    Calculate site log likelihoods at a root node

    This function integrates a list of partials at a node with respect to a set of partials-weights
    and state frequencies to return the log likelihood sum.

    :param instance:                 Instance number (input)
    :param bufferIndices:            List of partialsBuffer indices to integrate (input)
    :param categoryWeightsIndices:   List of weights to apply to each partialsBuffer (input). There should be one categoryCount sized set for each of parentBufferIndices
    :param stateFrequenciesIndices:  List of state frequencies for each partialsBuffer (input). There should be one set for each of parentBufferIndices
    :param cumulativeScaleIndices:   List of scaleBuffers containing accumulated factors to apply to each partialsBuffer (input). There should be one index for each of parentBufferIndices
    :param count:                    Number of partialsBuffer to integrate (input)
    :param outSumLogLikelihood:      Destination for resulting log likelihood (output)

    :raises BeagleException:
    )pbdoc");

  m.def("calculate_root_log_likelihoods_by_partition",
        [](int instance, const std::vector<int>& bufferIndices,
           const std::vector<int>& categoryWeightsIndices,
           const std::vector<int>& stateFrequenciesIndices,
           const std::vector<int>& cumulativeScaleIndices,
           const std::vector<int>& partitionIndices, int partitionCount,
           int count, double_np& outSumLogLikelihoodByPartition,
           double_np& outSumLogLikelihood) {
          int errCode = beagleCalculateRootLogLikelihoodsByPartition(
              instance, bufferIndices.data(), categoryWeightsIndices.data(),
              stateFrequenciesIndices.data(), cumulativeScaleIndices.data(),
              partitionIndices.data(), partitionCount, count,
              outSumLogLikelihoodByPartition.mutable_data(),
              outSumLogLikelihood.mutable_data());
          if (errCode != 0) {
            throw BeagleException("calculate_root_log_likelihoods_by_partition",
                                  errCode);
          }
        },
        R"pbdoc(
    Calculate site log likelihoods at a root node with per partition buffers

    This function integrates lists of partials at a node with respect to a set of partials-weights
    and state frequencies to return the log likelihood sums.

    :param instance:                 Instance number (input)
    :param bufferIndices:            List of partialsBuffer indices to integrate (input)
    :param categoryWeightsIndices:   List of weights to apply to each partialsBuffer (input). There should be one categoryCount sized set for each of bufferIndices
    :param stateFrequenciesIndices:  List of state frequencies for each partialsBuffer (input). There should be one set for each of bufferIndices
    :param cumulativeScaleIndices:   List of scaleBuffers containing accumulated factors to apply to each partialsBuffer (input). There should be one index for each of bufferIndices
    :param partitionIndices:         List of partition indices indicating which sites in each partialsBuffer should be used (input). There should be one index for each of bufferIndices
    :param partitionCount:           Number of distinct partitionIndices (input)
    :param count:                    Number of sets of partitions to integrate across (input)
    :param outSumLogLikelihoodByPartition:      Destination for resulting log likelihoods for each partition (output)
    :param outSumLogLikelihood:      Destination for resulting log likelihood (output)

    :raises BeagleException:
    )pbdoc");

  m.def("calculate_edge_log_likelihoods",
        [](int instance, const std::vector<int>& parentBufferIndices,
           const std::vector<int>& childBufferIndices,
           const std::vector<int>& probabilityIndices,
           std::optional<std::vector<int>> firstDerivativeIndices,
           std::optional<std::vector<int>> secondDerivativeIndices,
           const std::vector<int>& categoryWeightsIndices,
           const std::vector<int>& stateFrequenciesIndices,
           const std::vector<int>& cumulativeScaleIndices, int count,
           double_np& outSumLogLikelihood,
           std::optional<double_np> outSumFirstDerivative,
           std::optional<double_np> outSumSecondDerivative) {
          int errCode = beagleCalculateEdgeLogLikelihoods(
              instance, parentBufferIndices.data(), childBufferIndices.data(),
              probabilityIndices.data(),
              firstDerivativeIndices.has_value()
                  ? firstDerivativeIndices->data()
                  : nullptr,
              secondDerivativeIndices.has_value()
                  ? secondDerivativeIndices->data()
                  : nullptr,
              categoryWeightsIndices.data(), stateFrequenciesIndices.data(),
              cumulativeScaleIndices.data(), count,
              outSumLogLikelihood.mutable_data(),
              outSumFirstDerivative.has_value()
                  ? outSumFirstDerivative->mutable_data()
                  : nullptr,
              outSumSecondDerivative.has_value()
                  ? outSumSecondDerivative->mutable_data()
                  : nullptr);
          if (errCode != 0) {
            throw BeagleException("calculate_edge_log_likelihoods", errCode);
          }
        },
        py::arg("instance"), py::arg("parentBufferIndices"),
        py::arg("childBufferIndices"), py::arg("probabilityIndices"),
        py::arg("firstDerivativeIndices") = py::none(),
        py::arg("secondDerivativeIndices") = py::none(),
        py::arg("categoryWeightsIndices"), py::arg("stateFrequenciesIndices"),
        py::arg("cumulativeScaleIndices"), py::arg("count"),
        py::arg("outSumLogLikelihood"),
        py::arg("outSumFirstDerivative") = py::none(),
        py::arg("outSumSecondDerivative") = py::none(), R"pbdoc(
    Calculate site log likelihoods and derivatives along an edge

    This function integrates a list of partials at a parent and child node with respect
    to a set of partials-weights and state frequencies to return the log likelihood
    and first and second derivative sums.

    :param instance:                  Instance number (input)
    :param parentBufferIndices:       List of indices of parent partialsBuffers (input)
    :param childBufferIndices:        List of indices of child partialsBuffers (input)
    :param probabilityIndices:        List indices of transition probability matrices for this edge (input)
    :param firstDerivativeIndices:    List indices of first derivative matrices (input)
    :param secondDerivativeIndices:   List indices of second derivative matrices (input)
    :param categoryWeightsIndices:    List of weights to apply to each partialsBuffer (input)
    :param stateFrequenciesIndices:   List of state frequencies for each partialsBuffer (input). There should be one set for each of parentBufferIndices
    :param cumulativeScaleIndices:    List of scaleBuffers containing accumulated factors to apply to each partialsBuffer (input). There should be one index for each of parentBufferIndices
    :param outSumLogLikelihood:       Destination for resulting log likelihood (output)
    :param outSumFirstDerivative:     Destination for resulting first derivative (output)
    :param outSumSecondDerivative:    Destination for resulting second derivative (output)

    :raises BeagleException:
    )pbdoc");

  m.def("calculate_edge_log_likelihoods_by_partition",
        [](int instance, const std::vector<int>& parentBufferIndices,
           const std::vector<int>& childBufferIndices,
           const std::vector<int>& probabilityIndices,
           std::optional<std::vector<int>> firstDerivativeIndices,
           std::optional<std::vector<int>> secondDerivativeIndices,
           const std::vector<int>& categoryWeightsIndices,
           const std::vector<int>& stateFrequenciesIndices,
           const std::vector<int>& cumulativeScaleIndices,
           const std::vector<int>& partitionIndices, int partitionCount,
           int count, double_np& outSumLogLikelihoodByPartition,
           double_np& outSumLogLikelihood,
           std::optional<double_np> outSumFirstDerivativeByPartition,
           std::optional<double_np> outSumFirstDerivative,
           std::optional<double_np> outSumSecondDerivativeByPartition,
           std::optional<double_np> outSumSecondDerivative) {
          int errCode = beagleCalculateEdgeLogLikelihoodsByPartition(
              instance, parentBufferIndices.data(), childBufferIndices.data(),
              probabilityIndices.data(),
              firstDerivativeIndices.has_value()
                  ? firstDerivativeIndices->data()
                  : nullptr,
              secondDerivativeIndices.has_value()
                  ? secondDerivativeIndices->data()
                  : nullptr,
              categoryWeightsIndices.data(), stateFrequenciesIndices.data(),
              cumulativeScaleIndices.data(), partitionIndices.data(),
              partitionCount, count,
              outSumLogLikelihoodByPartition.mutable_data(),
              outSumLogLikelihood.mutable_data(),
              outSumFirstDerivativeByPartition.has_value()
                  ? outSumFirstDerivativeByPartition->mutable_data()
                  : nullptr,
              outSumFirstDerivative.has_value()
                  ? outSumFirstDerivative->mutable_data()
                  : nullptr,
              outSumSecondDerivativeByPartition.has_value()
                  ? outSumSecondDerivativeByPartition->mutable_data()
                  : nullptr,
              outSumSecondDerivative.has_value()
                  ? outSumSecondDerivative->mutable_data()
                  : nullptr);
          if (errCode != 0) {
            throw BeagleException("calculate_edge_log_likelihoods_by_partition",
                                  errCode);
          }
        },
        R"pbdoc(
    Calculate multiple site log likelihoods and derivatives along an edge with per partition buffers

    This function integrates lists of partials at a parent and child node with respect
    to a set of partials-weights and state frequencies to return the log likelihood
    and first and second derivative sums.

    :param instance:                  Instance number (input)
    :param parentBufferIndices:       List of indices of parent partialsBuffers (input)
    :param childBufferIndices:        List of indices of child partialsBuffers (input)
    :param probabilityIndices:        List indices of transition probability matrices for this edge (input)
    :param firstDerivativeIndices:    List indices of first derivative matrices (input)
    :param secondDerivativeIndices:   List indices of second derivative matrices (input)
    :param categoryWeightsIndices:    List of weights to apply to each partialsBuffer (input)
    :param stateFrequenciesIndices:   List of state frequencies for each partialsBuffer (input). Thereshould be one set for each of parentBufferIndices
    :param cumulativeScaleIndices:    List of scaleBuffers containing accumulated factors to apply to each partialsBuffer (input). There should be one index for each of parentBufferIndices
    :param partitionIndices:          List of partition indices indicating which sites in each partialsBuffer should be used (input). There should be one index for each of parentBufferIndices
    :param partitionCount:            Number of distinct partitionIndices (input)
    :param count:                     Number of sets of partitions to integrate across (input)
    :param outSumLogLikelihoodByPartition:      Destination for resulting log likelihoods for each partition (output)
    :param outSumLogLikelihood:       Destination for resulting log likelihood (output)
    :param outSumFirstDerivativeByPartition:     Destination for resulting first derivative for each partition (output)
    :param outSumFirstDerivative:     Destination for resulting first derivative (output)
    :param outSumSecondDerivativeByPartition:    Destination for resulting second derivative for each partition (output)
    :param outSumSecondDerivative:    Destination for resulting second derivative (output)

    :raises BeagleException:
    )pbdoc");

  m.def("get_log_likelihood",
        [](int instance, double_np& outSumLogLikelihood) {
          int errCode = beagleGetLogLikelihood(
              instance, outSumLogLikelihood.mutable_data());
          if (errCode != 0) {
            throw BeagleException("get_log_likelihood", errCode);
          }
        },
        R"pbdoc(
    Returns log likelihood sum and subsequent to an asynchronous integration call

    This function is optional and only has to be called by clients that use the non-blocking
    asynchronous computation mode (BEAGLE_FLAG_COMPUTATION_ASYNCH).

    If used, this function must be called after a beagleCalculateRootLogLikelihoods or
    beagleCalculateEdgeLogLikelihoods call. The library will block until the likelihood
    has been calculated.
    :param instance:                  Instance number (input)
    :param outSumLogLikelihood:       Destination for resulting log likelihood (output)

    :raises BeagleException:
    )pbdoc");

  m.def("get_derivatives",
        [](int instance, double_np& outSumFirstDerivative,
           std::optional<double_np> outSumSecondDerivative) {
          int errCode = beagleGetDerivatives(
              instance, outSumFirstDerivative.mutable_data(),
              outSumSecondDerivative.has_value()
                  ? outSumSecondDerivative->mutable_data()
                  : nullptr);
          if (errCode != 0) {
            throw BeagleException("get_derivatives", errCode);
          }
        },
        R"pbdoc(
    Returns derivative sums subsequent to an asynchronous integration call

    This function is optional and only has to be called by clients that use the non-blocking
    asynchronous computation mode (BEAGLE_FLAG_COMPUTATION_ASYNCH).

    If used, this function must be called after a beagleCalculateEdgeLogLikelihoods call.
    The library will block until the derivatiives have been calculated.

    :param instance:                  Instance number (input)
    :param outSumFirstDerivative:     Destination for resulting first derivative (output)
    :param outSumSecondDerivative:    Destination for resulting second derivative (output)

    :raises BeagleException:
    )pbdoc");

  m.def("get_site_log_likelihoods",
        [](int instance, double_np& outLogLikelihoods) {
          int errCode = beagleGetSiteLogLikelihoods(
              instance, outLogLikelihoods.mutable_data());
          if (errCode != 0) {
            throw BeagleException("get_site_log_likelihoods", errCode);
          }
        },
        R"pbdoc(
    Get site log likelihoods for last beagleCalculateRootLogLikelihoods or beagleCalculateEdgeLogLikelihoods call

    This function returns the log likelihoods for each site.

    :param instance:               Instance number (input)
    :param outLogLikelihoods:      Destination for resulting log likelihoods (output)

    :raises BeagleException:
    )pbdoc");

  m.def("get_site_derivatives",
        [](int instance, double_np& outFirstDerivatives,
           std::optional<double_np> outSecondDerivatives) {
          int errCode = beagleGetSiteDerivatives(
              instance, outFirstDerivatives.mutable_data(),
              outSecondDerivatives.has_value()
                  ? outSecondDerivatives->mutable_data()
                  : nullptr);
          if (errCode != 0) {
            throw BeagleException("get_site_derivatives", errCode);
          }
        },
        R"pbdoc(
    Get site derivatives for last beagleCalculateEdgeLogLikelihoods call

    This function returns the derivatives for each site.

    :param instance:               Instance number (input)
    :param outFirstDerivatives:    Destination for resulting first derivatives (output)
    :param outSecondDerivatives:   Destination for resulting second derivatives (output)

    :raises BeagleException:
    )pbdoc");
}
