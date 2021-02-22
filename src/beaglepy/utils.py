import sys

import beaglepy


def print_flags(flags):
    if flags & beaglepy.BEAGLE_FLAG_PROCESSOR_CPU:
        sys.stdout.write(" PROCESSOR_CPU")
    if flags & beaglepy.BEAGLE_FLAG_PROCESSOR_GPU:
        sys.stdout.write(" PROCESSOR_GPU")
    if flags & beaglepy.BEAGLE_FLAG_PROCESSOR_FPGA:
        sys.stdout.write(" PROCESSOR_FPGA")
    if flags & beaglepy.BEAGLE_FLAG_PROCESSOR_CELL:
        sys.stdout.write(" PROCESSOR_CELL")
    if flags & beaglepy.BEAGLE_FLAG_PRECISION_DOUBLE:
        sys.stdout.write(" PRECISION_DOUBLE")
    if flags & beaglepy.BEAGLE_FLAG_PRECISION_SINGLE:
        sys.stdout.write(" PRECISION_SINGLE")
    if flags & beaglepy.BEAGLE_FLAG_COMPUTATION_ASYNCH:
        sys.stdout.write(" COMPUTATION_ASYNCH")
    if flags & beaglepy.BEAGLE_FLAG_COMPUTATION_SYNCH:
        sys.stdout.write(" COMPUTATION_SYNCH")
    if flags & beaglepy.BEAGLE_FLAG_EIGEN_REAL:
        sys.stdout.write(" EIGEN_REAL")
    if flags & beaglepy.BEAGLE_FLAG_EIGEN_COMPLEX:
        sys.stdout.write(" EIGEN_COMPLEX")
    if flags & beaglepy.BEAGLE_FLAG_SCALING_MANUAL:
        sys.stdout.write(" SCALING_MANUAL")
    if flags & beaglepy.BEAGLE_FLAG_SCALING_AUTO:
        sys.stdout.write(" SCALING_AUTO")
    if flags & beaglepy.BEAGLE_FLAG_SCALING_ALWAYS:
        sys.stdout.write(" SCALING_ALWAYS")
    if flags & beaglepy.BEAGLE_FLAG_SCALING_DYNAMIC:
        sys.stdout.write(" SCALING_DYNAMIC")
    if flags & beaglepy.BEAGLE_FLAG_SCALERS_RAW:
        sys.stdout.write(" SCALERS_RAW")
    if flags & beaglepy.BEAGLE_FLAG_SCALERS_LOG:
        sys.stdout.write(" SCALERS_LOG")
    if flags & beaglepy.BEAGLE_FLAG_VECTOR_NONE:
        sys.stdout.write(" VECTOR_NONE")
    if flags & beaglepy.BEAGLE_FLAG_VECTOR_SSE:
        sys.stdout.write(" VECTOR_SSE")
    if flags & beaglepy.BEAGLE_FLAG_VECTOR_AVX:
        sys.stdout.write(" VECTOR_AVX")
    if flags & beaglepy.BEAGLE_FLAG_THREADING_NONE:
        sys.stdout.write(" THREADING_NONE")
    if flags & beaglepy.BEAGLE_FLAG_THREADING_OPENMP:
        sys.stdout.write(" THREADING_OPENMP")
    if flags & beaglepy.BEAGLE_FLAG_FRAMEWORK_CPU:
        sys.stdout.write(" FRAMEWORK_CPU")
    if flags & beaglepy.BEAGLE_FLAG_FRAMEWORK_CUDA:
        sys.stdout.write(" FRAMEWORK_CUDA")
    if flags & beaglepy.BEAGLE_FLAG_FRAMEWORK_OPENCL:
        sys.stdout.write(" FRAMEWORK_OPENCL")

def print_resource_list():
	rList = beaglepy.get_resource_list()
	sys.stdout.write("Available resources:\n")
	for i in range(len(rList)):
		sys.stdout.write("\tResource {}:\n\t\tName : {}\n".format(i, rList[i].name))
		sys.stdout.write("\t\tDesc : {}\n".format(rList[i].description))
		sys.stdout.write("\t\tFlags:")
		print_flags(rList[i].support_flags)
		sys.stdout.write("\n")
	sys.stdout.write("\n")

