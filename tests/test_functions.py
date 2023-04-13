import numpy as np
import pytest

from beaglepy import *


def create_instance_helper(
    taxon_count,
    state_count,
    pattern_count,
    use_tip_states,
    category_count,
    init_scale_buffer=False,
    prefered_flags=0,
    required_flag=0,
    resources=None,
):
    partials_buffer_count = 2 * taxon_count - 1
    compact_buffer_count = 0
    if not use_tip_states:
        partials_buffer_count += taxon_count
    else:
        compact_buffer_count = taxon_count
    scale_buffer_count = partials_buffer_count + 1 if init_scale_buffer else 0
    matrix_buffer_count = 2 * taxon_count - 1

    instance_details = BeagleInstanceDetails()
    instance = create_instance(
        taxon_count,  # Number of tip data elements (input)
        partials_buffer_count,  # Number of partials buffers to create (input)
        compact_buffer_count,  # Number of compact state representation buffers to create (input)
        state_count,  # Number of states in the continuous-time Markov chain (input)
        pattern_count,  # Number of site patterns to be handled by the instance (input)
        1,  # Number of rate matrix eigen-decomposition buffers to allocate (input)
        matrix_buffer_count,  # Number of rate matrix buffers (input)
        category_count,  # Number of rate categories (input)
        scale_buffer_count,  # Number of scaling buffers
        resources,  # List of potential resource on which this instance is allowed (input, NULL implies no restriction
        len(resources) if resources is not None else 0,
        prefered_flags,  # 	Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input)
        required_flag,  # Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
        instance_details,
    )
    return instance, instance_details


def test_get_version():
    version = get_version()
    assert isinstance(version, str)


def test_get_citation():
    citation = get_citation()
    assert isinstance(citation, str)


@pytest.mark.parametrize("use_tip_states", [False, True])
@pytest.mark.parametrize("category_count", [1, 4])
def test_create_instance(use_tip_states, category_count):
    instance, instance_details = create_instance_helper(
        4, 4, 2, use_tip_states, category_count
    )

    assert instance >= 0
    finalize_instance(instance)


def test_finalize_instance():
    with pytest.raises(
        BeagleException, match=rf"error code {int(BEAGLE_ERROR_UNINITIALIZED_INSTANCE)}"
    ):
        finalize_instance(0)

    instance, instance_details = create_instance_helper(2, 4, 2, True, 1)
    finalize_instance(instance)

    with pytest.raises(
        BeagleException, match=rf"error code {int(BEAGLE_ERROR_UNINITIALIZED_INSTANCE)}"
    ):
        finalize_instance(instance)


def test_set_CPU_thread_count():
    # set BEAGLE_FLAG_THREADING_CPP
    instance, instance_details = create_instance_helper(2, 4, 2, True, 1)
    set_CPU_thread_count(instance, 2)
    finalize_instance(instance)


def test_tip_states():
    instance, instance_details = create_instance_helper(2, 4, 2, True, 1)
    set_tip_states(instance, 0, np.array([1, 2]))

    with pytest.raises(BeagleException, match=rf"{int(BEAGLE_ERROR_OUT_OF_RANGE)}"):
        set_tip_states(instance, 10, [1, 2])

    finalize_instance(instance)


def test_tip_partials():
    instance, instance_details = create_instance_helper(2, 4, 2, False, 1)
    partials = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    set_tip_partials(instance, 0, partials)

    with pytest.raises(BeagleException, match=rf"{int(BEAGLE_ERROR_OUT_OF_RANGE)}"):
        set_tip_partials(instance, 10, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

    finalize_instance(instance)


def test_set_get_partials():
    instance, instance_details = create_instance_helper(2, 4, 2, True, 1)
    partials = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    set_partials(instance, 0, partials)

    with pytest.raises(BeagleException, match=rf"{int(BEAGLE_ERROR_OUT_OF_RANGE)}"):
        set_partials(instance, 10, partials)

    partials2 = np.empty_like(partials)
    get_partials(instance, 0, BEAGLE_OP_NONE, partials2)

    np.testing.assert_array_equal(partials, partials2)

    with pytest.raises(BeagleException, match=rf"{int(BEAGLE_ERROR_OUT_OF_RANGE)}"):
        get_partials(instance, 10, BEAGLE_OP_NONE, partials2)

    finalize_instance(instance)


def test_set_get_transition_matrix():
    instance, instance_details = create_instance_helper(2, 4, 2, True, 1)
    matrix = np.full(16, 2.0)
    set_transition_matrix(instance, 0, matrix, 0)
    matrix2 = np.empty_like(matrix)
    get_transition_matrix(instance, 0, matrix2)

    np.testing.assert_array_equal(matrix, matrix2)

    # beagleGetTransitionMatrix does not check indices (unlike beagleGetPartials)
    # with pytest.raises(BeagleException, match=rf"{int(BEAGLE_ERROR_OUT_OF_RANGE)}"):
    #     get_transition_matrix(instance, 10, matrix2)

    finalize_instance(instance)
