import itertools
import sys

import numpy as np
import pytest

from beaglepy import *
from beaglepy.utils import print_flags

dna_map = {
    'A': [1.0, 0.0, 0.0, 0.0],
    'C': [0.0, 1.0, 0.0, 0.0],
    'G': [0.0, 0.0, 1.0, 0.0],
    'T': [0.0, 0.0, 0.0, 1.0],
}

# an eigen decomposition for the JC69 model
# fmt: off
evec = np.array(
    [
        1.0, 2.0, 0.0, 0.5,
        1.0, -2.0, 0.5, 0.0,
        1.0, 2.0, 0.0, -0.5,
        1.0, -2.0, -0.5, 0.0,
    ]
)
ivec = np.array(
    [
        0.25, .25, 0.25, 0.25,
        0.125, -0.125, 0.125, -0.125,
        0.0, 1.0, 0.0, -1.0,
        1.0, 0.0, -1.0, 0.0,
    ]
)
evalues = np.array(
    [0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333]
)
# fmt: on


@pytest.mark.parametrize(
    "manualScaling,autoScaling", [(False, False), (True, False), (False, True)]
)
def test_tiny(manualScaling, autoScaling):
    human = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA"
    chimp = "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA"
    gorilla = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA"

    rList = get_resource_list()
    sys.stdout.write("Available resources:\n")
    for i in range(len(rList)):
        sys.stdout.write("\tResource {}:\n\t\tName : {}\n".format(i, rList[i].name))
        sys.stdout.write("\t\tDesc : {}\n".format(rList[i].description))
        sys.stdout.write("\t\tFlags:")
        print_flags(rList[i].support_flags)
        sys.stdout.write("\n")
    sys.stdout.write("\n")

    gRates = False
    # generalized rate categories, separate root buffers

    # is nucleotides...
    stateCount = 4

    # get the number of site patterns
    nPatterns = len(human)

    rateCategoryCount = 4

    nRateCats = 1 if gRates else rateCategoryCount
    nRootCount = 1 if not gRates else rateCategoryCount
    nPartBuffs = 4 + nRootCount
    scaleCount = 2 + nRootCount if manualScaling else 0

    prefered_flags = (
        BEAGLE_FLAG_FRAMEWORK_CUDA
        | BEAGLE_FLAG_PRECISION_SINGLE
        | BEAGLE_FLAG_PROCESSOR_GPU
        | BEAGLE_FLAG_SCALING_AUTO
        if autoScaling
        else 0
    )

    instDetails = BeagleInstanceDetails()
    # create an instance of the BEAGLE library
    instance = create_instance(
        3,  # Number of tip data elements (input)
        nPartBuffs,  # Number of partials buffers to create (input)
        0,  # Number of compact state representation buffers to create (input)
        stateCount,  # Number of states in the continuous-time Markov chain (input)
        nPatterns,  # Number of site patterns to be handled by the instance (input)
        1,  # Number of rate matrix eigen-decomposition buffers to allocate (input)
        4,  # Number of rate matrix buffers (input)
        nRateCats,  # Number of rate categories (input)
        scaleCount,  # Number of scaling buffers
        None,  # List of potential resource on which this instance is allowed (input, NULL implies no restriction
        0,
        prefered_flags,  # 	Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input)
        0,  # Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
        instDetails,
    )

    if instance < 0:
        sys.stderr.write("Failed to obtain BEAGLE instance\n\n")
        exit(1)

    rNumber = instDetails.resource_number
    sys.stdout.write("Using resource {}:\n".format(rNumber))
    sys.stdout.write("\tRsrc Name : {}\n".format(instDetails.resource_name))
    sys.stdout.write("\tImpl Name : {}\n".format(instDetails.impl_name))
    sys.stdout.write("\tImpl Desc : {}\n".format(instDetails.impl_description))
    sys.stdout.write("\tFlags:")
    print_flags(instDetails.flags)
    sys.stdout.write("\n\n")

    if not (instDetails.flags & BEAGLE_FLAG_SCALING_AUTO):
        autoScaling = False

    humanPartials = np.array(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1.0, 1.0, 1.0, 1.0]) for n in human]
            )
        )
    )
    chimpPartials = np.array(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1.0, 1.0, 1.0, 1.0]) for n in chimp]
            )
        )
    )
    gorillaPartials = np.array(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1.0, 1.0, 1.0, 1.0]) for n in gorilla]
            )
        )
    )

    set_tip_partials(instance, 0, humanPartials)
    set_tip_partials(instance, 1, chimpPartials)
    set_tip_partials(instance, 2, gorillaPartials)

    rates = np.array([0.03338775, 0.25191592, 0.82026848, 2.89442785])

    # create base frequency array
    freqs = np.array([0.25] * 16)

    # create an array containing site category weights
    weights = np.array([1.0 / rateCategoryCount] * rateCategoryCount)
    patternWeights = np.array([1.0] * nPatterns)

    # set the Eigen decomposition
    set_eigen_decomposition(instance, 0, evec, ivec, evalues)

    set_state_frequencies(instance, 0, freqs)

    set_category_weights(instance, 0, weights)

    set_pattern_weights(instance, patternWeights)

    # a list of indices and edge lengths
    nodeIndices = [0, 1, 2, 3]
    edgeLengths = np.array([0.1, 0.1, 0.2, 0.1])

    rootIndices = [None] * nRootCount
    categoryWeightsIndices = [None] * nRootCount
    stateFrequencyIndices = [None] * nRootCount
    cumulativeScalingIndices = [None] * nRootCount

    for i in range(nRootCount):
        rootIndices[i] = 4 + i
        categoryWeightsIndices[i] = 0
        stateFrequencyIndices[i] = 0
        cumulativeScalingIndices[i] = 2 + i if manualScaling else BEAGLE_OP_NONE

        set_category_rates(instance, rates[i:])

        # tell BEAGLE to populate the transition matrices for the above edge lengths
        update_transition_matrices(
            instance,  # instance
            0,  # eigenIndex
            nodeIndices,  # probabilityIndices
            None,  # firstDerivativeIndices
            None,  # secondDerivativeIndices
            edgeLengths,  # edgeLengths
            len(edgeLengths),
        )

        # create a list of partial likelihood update operations
        # the order is [dest, destScaling, source1, matrix1, source2, matrix2]
        operationss = [
            BeagleOperation(
                3, 0 if manualScaling else BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1
            ),
            BeagleOperation(
                rootIndices[i],
                1 if manualScaling else BEAGLE_OP_NONE,
                BEAGLE_OP_NONE,
                2,
                2,
                3,
                3,
            ),
        ]

        if manualScaling:
            reset_scale_factors(instance, cumulativeScalingIndices[i])

        # update the partials
        update_partials(
            instance,  # instance
            operationss,  # eigenIndex
            len(operationss),
            cumulativeScalingIndices[i],
        )  # cumulative scaling index

    if autoScaling:
        scaleIndices = [3, 4]
        accumulate_scale_factors(instance, scaleIndices, 2, BEAGLE_OP_NONE)

    patternLogLik = np.empty(nPatterns)
    logL = np.empty(1)

    # calculate the site likelihoods at the root node
    returnCode = calculate_root_log_likelihoods(
        instance,  # instance
        rootIndices,  # bufferIndices
        categoryWeightsIndices,  # weights
        stateFrequencyIndices,  # stateFrequencies
        cumulativeScalingIndices,  # cumulative scaling index
        nRootCount,  # count
        logL,
    )  # outLogLikelihoods

    get_site_log_likelihoods(instance, patternLogLik)
    sumLogL = 0.0
    for i in range(nPatterns):
        sumLogL += patternLogLik[i] * patternWeights[i]
    #             sys.stdout.write("site lnL[{}] = {}\n".format(i, patternLogLik[i]))

    sys.stdout.write("logL = {} (PAUP logL = -1498.89812)\n".format(logL))
    sys.stdout.write("sumLogL = {}f\n".format(sumLogL))

    assert -1498.89812 == pytest.approx(logL[0], 0.0001)
    assert -1498.89812 == pytest.approx(sumLogL, 0.0001)

    # no rate heterogeneity:

    #     sys.stdout.write("logL = {} (PAUP logL = -1574.63623)\n\n".format(logL))

    finalize_instance(instance)


@pytest.mark.parametrize("manualScaling,autoScaling", [(False, False)])
def test_calculate_edge_log_likelihoods(manualScaling, autoScaling):
    human = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA"
    chimp = "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA"
    gorilla = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA"

    rList = get_resource_list()
    sys.stdout.write("Available resources:\n")
    for i in range(len(rList)):
        sys.stdout.write("\tResource {}:\n\t\tName : {}\n".format(i, rList[i].name))
        sys.stdout.write("\t\tDesc : {}\n".format(rList[i].description))
        sys.stdout.write("\t\tFlags:")
        print_flags(rList[i].support_flags)
        sys.stdout.write("\n")
    sys.stdout.write("\n")

    gRates = False
    # generalized rate categories, separate root buffers

    # is nucleotides...
    stateCount = 4

    # get the number of site patterns
    nPatterns = len(human)

    rateCategoryCount = 4

    nRateCats = 1 if gRates else rateCategoryCount
    nRootCount = 1 if not gRates else rateCategoryCount
    nPartBuffs = 4 + nRootCount
    scaleCount = 2 + nRootCount if manualScaling else 0

    prefered_flags = (
        BEAGLE_FLAG_FRAMEWORK_CUDA
        | BEAGLE_FLAG_PRECISION_SINGLE
        | BEAGLE_FLAG_PROCESSOR_GPU
        | BEAGLE_FLAG_SCALING_AUTO
        if autoScaling
        else 0
    )

    # create an instance of the BEAGLE library
    instDetails = BeagleInstanceDetails()
    instance = create_instance(
        3,  # Number of tip data elements (input)
        nPartBuffs,  # Number of partials buffers to create (input)
        0,  # Number of compact state representation buffers to create (input)
        stateCount,  # Number of states in the continuous-time Markov chain (input)
        nPatterns,  # Number of site patterns to be handled by the instance (input)
        1,  # Number of rate matrix eigen-decomposition buffers to allocate (input)
        4,  # Number of rate matrix buffers (input)
        nRateCats,  # Number of rate categories (input)
        scaleCount,  # Number of scaling buffers
        None,  # List of potential resource on which this instance is allowed (input, NULL implies no restriction
        0,
        prefered_flags,  # 	Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input)
        0,  # Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
        instDetails,
    )

    if instance < 0:
        sys.stderr.write("Failed to obtain BEAGLE instance\n\n")
        exit(1)

    rNumber = instDetails.resource_number
    sys.stdout.write("Using resource {}:\n".format(rNumber))
    sys.stdout.write("\tRsrc Name : {}\n".format(instDetails.resource_name))
    sys.stdout.write("\tImpl Name : {}\n".format(instDetails.impl_name))
    sys.stdout.write("\tImpl Desc : {}\n".format(instDetails.impl_description))
    sys.stdout.write("\tFlags:")
    print_flags(instDetails.flags)
    sys.stdout.write("\n\n")

    if not (instDetails.flags & BEAGLE_FLAG_SCALING_AUTO):
        autoScaling = False

    humanPartials = np.array(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1.0, 1.0, 1.0, 1.0]) for n in human]
            )
        )
    )
    chimpPartials = np.array(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1.0, 1.0, 1.0, 1.0]) for n in chimp]
            )
        )
    )
    gorillaPartials = np.array(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1.0, 1.0, 1.0, 1.0]) for n in gorilla]
            )
        )
    )

    set_tip_partials(instance, 0, humanPartials)
    set_tip_partials(instance, 1, chimpPartials)
    set_tip_partials(instance, 2, gorillaPartials)

    rates = np.array([0.03338775, 0.25191592, 0.82026848, 2.89442785])

    # create base frequency array
    freqs = np.array([0.25] * 16)

    # create an array containing site category weights
    weights = np.array([1.0 / rateCategoryCount] * rateCategoryCount)
    patternWeights = np.array([1.0] * nPatterns)

    # set the Eigen decomposition
    set_eigen_decomposition(instance, 0, evec, ivec, evalues)

    set_state_frequencies(instance, 0, freqs)

    set_category_weights(instance, 0, weights)

    set_pattern_weights(instance, patternWeights)

    # a list of indices and edge lengths
    nodeIndices = [0, 1, 2, 3]
    edgeLengths = np.array([0.1, 0.1, 0.3, 0.0])

    rootIndices = [None] * nRootCount
    rightIndices = [None] * nRootCount
    leftIndices = [None] * nRootCount
    categoryWeightsIndices = [None] * nRootCount
    stateFrequencyIndices = [None] * nRootCount
    cumulativeScalingIndices = [None] * nRootCount

    for i in range(nRootCount):
        rootIndices[i] = 4 + i
        rightIndices[i] = 2 + i
        leftIndices[i] = 3 + i
        categoryWeightsIndices[i] = 0
        stateFrequencyIndices[i] = 0
        cumulativeScalingIndices[i] = 2 + i if manualScaling else BEAGLE_OP_NONE

        set_category_rates(instance, rates[i:])

        # tell BEAGLE to populate the transition matrices for the above edge lengths
        update_transition_matrices(
            instance,  # instance
            0,  # eigenIndex
            nodeIndices,  # probabilityIndices
            None,  # firstDerivativeIndices
            None,  # secondDerivativeIndices
            edgeLengths,  # edgeLengths
            len(edgeLengths),
        )

        # create a list of partial likelihood update operations
        # the order is [dest, destScaling, source1, matrix1, source2, matrix2]
        operationss = [
            BeagleOperation(
                3, 0 if manualScaling else BEAGLE_OP_NONE, BEAGLE_OP_NONE, 0, 0, 1, 1
            ),
            BeagleOperation(
                rootIndices[i],
                1 if manualScaling else BEAGLE_OP_NONE,
                BEAGLE_OP_NONE,
                2,
                2,
                3,
                3,
            ),
        ]

        if manualScaling:
            reset_scale_factors(instance, cumulativeScalingIndices[i])

        # update the partials
        update_partials(
            instance,  # instance
            operationss,  # eigenIndex
            len(operationss),
            cumulativeScalingIndices[i],
        )  # cumulative scaling index

    if autoScaling:
        scaleIndices = [3, 4]
        accumulate_scale_factors(instance, scaleIndices, 2, BEAGLE_OP_NONE)

    patternLogLik = np.empty(nPatterns)
    logL = np.empty(1)

    calculate_edge_log_likelihoods(
        instance,  # instance
        [3],  # bufferIndices
        [2],  # bufferIndices
        [2],
        None,
        None,
        categoryWeightsIndices,  # weights
        stateFrequencyIndices,  # stateFrequencies
        cumulativeScalingIndices,  # cumulative scaling index
        nRootCount,  # count
        logL,  # outLogLikelihoods
        None,
        None,
    )

    get_site_log_likelihoods(instance, patternLogLik)
    sumLogL = 0.0
    for i in range(nPatterns):
        sumLogL += patternLogLik[i] * patternWeights[i]
    #             sys.stdout.write("site lnL[{}] = {}\n".format(i, patternLogLik[i]))

    sys.stdout.write("logL = {} (PAUP logL = -1498.89812)\n".format(logL))
    sys.stdout.write("sumLogL = {}f\n".format(sumLogL))

    assert -1498.89812 == pytest.approx(logL[0], 0.0001)
    assert -1498.89812 == pytest.approx(sumLogL, 0.0001)

    # no rate heterogeneity:

    #     sys.stdout.write("logL = {} (PAUP logL = -1574.63623)\n\n".format(logL))

    finalize_instance(instance)
