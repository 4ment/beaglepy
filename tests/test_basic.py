import pytest
from beagle import *
import numpy as np


@pytest.mark.parametrize(
    "p0,p1,rate_count",
    [
        ([0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], 1),
        ([0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0], 4),
    ],
)
def test_partials(p0, p1, rate_count):
    prefered_flags = (BEAGLE_FLAG_FRAMEWORK_CUDA
                      | BEAGLE_FLAG_PRECISION_SINGLE
                      | BEAGLE_FLAG_PROCESSOR_GPU)

    # create an instance of the BEAGLE library
    instance, instDetails = create_instance(
        0,  # Number of tip data elements (input)
        2,  # Number of partials buffers to create (input)
        0,  # Number of compact state representation buffers to create (input)
        2,  # Number of states in the continuous-time Markov chain (input)
        2,  # Number of site patterns to be handled by the instance (input)
        1,  # Number of rate matrix eigen-decomposition buffers to allocate (input)
        0,  # Number of rate matrix buffers (input)
        rate_count,  # Number of rate categories (input)
        0,  # Number of scaling buffers
        None,  # List of potential resource on which this instance is allowed (input, NULL implies no restriction
        prefered_flags,  # 	Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input)
        0  # Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
    )

    if instance < 0:
        sys.stderr.write("Failed to obtain BEAGLE instance\n\n")
        exit(1)

    set_partials(instance, 0, p0 * rate_count)
    set_partials(instance, 1, p1 * rate_count)
    p0out = np.empty(len(p0) * rate_count)
    p1out = np.empty(len(p1) * rate_count)
    get_partials(instance, 0, BEAGLE_OP_NONE, p0out)
    get_partials(instance, 1, BEAGLE_OP_NONE, p1out)

    np.testing.assert_array_equal(p0 * rate_count, p0out)
    np.testing.assert_array_equal(p1 * rate_count, p1out)

    finalize_instance(instance)


@pytest.mark.parametrize("s0,s1,rate_count", [([1.0], [2.0], 1),
                                              ([3.0], [4.0], 4)])
def test_matrix(s0, s1, rate_count):
    prefered_flags = (BEAGLE_FLAG_FRAMEWORK_CUDA
                      | BEAGLE_FLAG_PRECISION_SINGLE
                      | BEAGLE_FLAG_PROCESSOR_GPU)

    # create an instance of the BEAGLE library
    instance, instDetails = create_instance(
        0,  # Number of tip data elements (input)
        1,  # Number of partials buffers to create (input)
        0,  # Number of compact state representation buffers to create (input)
        2,  # Number of states in the continuous-time Markov chain (input)
        1,  # Number of site patterns to be handled by the instance (input)
        1,  # Number of rate matrix eigen-decomposition buffers to allocate (input)
        2,  # Number of rate matrix buffers (input)
        rate_count,  # Number of rate categories (input)
        0,  # Number of scaling buffers
        None,  # List of potential resource on which this instance is allowed (input, NULL implies no restriction
        prefered_flags,  # 	Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input)
        0  # Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
    )

    if instance < 0:
        sys.stderr.write("Failed to obtain BEAGLE instance\n\n")
        exit(1)

    m0 = s0 * 4 * rate_count
    m1 = s1 * 4 * rate_count
    set_transition_matrix(instance, 0, m0, 0)
    set_transition_matrix(instance, 1, m1, 0)
    m0out = np.empty(len(m0))
    m1out = np.empty(len(m1))
    get_transition_matrix(instance, 0, m0out)
    get_transition_matrix(instance, 1, m1out)

    np.testing.assert_array_equal(m0, m0out)
    np.testing.assert_array_equal(m1, m1out)

    m0 = np.array(m0) + 3.0
    m1 = np.array(m1) + 4.0
    m0m1 = np.concatenate((m0, m1))
    set_transition_matrices(instance, [0, 1], m0m1, [0, 0])
    get_transition_matrix(instance, 0, m0out)
    get_transition_matrix(instance, 1, m1out)
    np.testing.assert_array_equal(m0m1, np.concatenate((m0out, m1out)))

    finalize_instance(instance)
