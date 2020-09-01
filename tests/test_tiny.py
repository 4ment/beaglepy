import pytest
from beagle import *
import sys
import itertools


def printFlags(inFlags):
    if inFlags & BEAGLE_FLAG_PROCESSOR_CPU:
        sys.stdout.write(" PROCESSOR_CPU")
    if inFlags & BEAGLE_FLAG_PROCESSOR_GPU:
        sys.stdout.write(" PROCESSOR_GPU")
    if inFlags & BEAGLE_FLAG_PROCESSOR_FPGA:
        sys.stdout.write(" PROCESSOR_FPGA")
    if inFlags & BEAGLE_FLAG_PROCESSOR_CELL:
        sys.stdout.write(" PROCESSOR_CELL")
    if inFlags & BEAGLE_FLAG_PRECISION_DOUBLE:
        sys.stdout.write(" PRECISION_DOUBLE")
    if inFlags & BEAGLE_FLAG_PRECISION_SINGLE:
        sys.stdout.write(" PRECISION_SINGLE")
    if inFlags & BEAGLE_FLAG_COMPUTATION_ASYNCH:
        sys.stdout.write(" COMPUTATION_ASYNCH")
    if inFlags & BEAGLE_FLAG_COMPUTATION_SYNCH:
        sys.stdout.write(" COMPUTATION_SYNCH")
    if inFlags & BEAGLE_FLAG_EIGEN_REAL:
        sys.stdout.write(" EIGEN_REAL")
    if inFlags & BEAGLE_FLAG_EIGEN_COMPLEX:
        sys.stdout.write(" EIGEN_COMPLEX")
    if inFlags & BEAGLE_FLAG_SCALING_MANUAL:
        sys.stdout.write(" SCALING_MANUAL")
    if inFlags & BEAGLE_FLAG_SCALING_AUTO:
        sys.stdout.write(" SCALING_AUTO")
    if inFlags & BEAGLE_FLAG_SCALING_ALWAYS:
        sys.stdout.write(" SCALING_ALWAYS")
    if inFlags & BEAGLE_FLAG_SCALING_DYNAMIC:
        sys.stdout.write(" SCALING_DYNAMIC")
    if inFlags & BEAGLE_FLAG_SCALERS_RAW:
        sys.stdout.write(" SCALERS_RAW")
    if inFlags & BEAGLE_FLAG_SCALERS_LOG:
        sys.stdout.write(" SCALERS_LOG")
    if inFlags & BEAGLE_FLAG_VECTOR_NONE:
        sys.stdout.write(" VECTOR_NONE")
    if inFlags & BEAGLE_FLAG_VECTOR_SSE:
        sys.stdout.write(" VECTOR_SSE")
    if inFlags & BEAGLE_FLAG_VECTOR_AVX:
        sys.stdout.write(" VECTOR_AVX")
    if inFlags & BEAGLE_FLAG_THREADING_NONE:
        sys.stdout.write(" THREADING_NONE")
    if inFlags & BEAGLE_FLAG_THREADING_OPENMP:
        sys.stdout.write(" THREADING_OPENMP")
    if inFlags & BEAGLE_FLAG_FRAMEWORK_CPU:
        sys.stdout.write(" FRAMEWORK_CPU")
    if inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA:
        sys.stdout.write(" FRAMEWORK_CUDA")
    if inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL:
        sys.stdout.write(" FRAMEWORK_OPENCL")


dna_map = {
    'A': [1.0, 0.0, 0.0, 0.0],
    'C': [0.0, 1.0, 0.0, 0.0],
    'G': [0.0, 0.0, 1.0, 0.0],
    'T': [0.0, 0.0, 0.0, 1.0]
}


@pytest.mark.parametrize("manualScaling,autoScaling", [(False, False),
                                                       (True, False),
                                                       (False, True)])
def test_tiny(manualScaling, autoScaling):
    human = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGGAGCTTAAACCCCCTTATTTCTACTAGGACTATGAGAATCGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTATACCCTTCCCGTACTAAGAAATTTAGGTTAAATACAGACCAAGAGCCTTCAAAGCCCTCAGTAAGTTG-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGACCAATGGGACTTAAACCCACAAACACTTAGTTAACAGCTAAGCACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCGGAGCTTGGTAAAAAGAGGCCTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGGCCTCCATGACTTTTTCAAAAGGTATTAGAAAAACCATTTCATAACTTTGTCAAAGTTAAATTATAGGCT-AAATCCTATATATCTTA-CACTGTAAAGCTAACTTAGCATTAACCTTTTAAGTTAAAGATTAAGAGAACCAACACCTCTTTACAGTGA"
    chimp = "AGAAATATGTCTGATAAAAGAATTACTTTGATAGAGTAAATAATAGGAGTTCAAATCCCCTTATTTCTACTAGGACTATAAGAATCGAACTCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTATCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTTACACCCTTCCCGTACTAAGAAATTTAGGTTAAGCACAGACCAAGAGCCTTCAAAGCCCTCAGCAAGTTA-CAATACTTAATTTCTGTAAGGACTGCAAAACCCCACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATTAATGGGACTTAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAATCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAA-TCACCTCAGAGCTTGGTAAAAAGAGGCTTAACCCCTGTCTTTAGATTTACAGTCCAATGCTTCA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCTAAAGCTGGTTTCAAGCCAACCCCATGACCTCCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAAGTTAAATTACAGGTT-AACCCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCATTAACCTTTTAAGTTAAAGATTAAGAGGACCGACACCTCTTTACAGTGA"
    gorilla = "AGAAATATGTCTGATAAAAGAGTTACTTTGATAGAGTAAATAATAGAGGTTTAAACCCCCTTATTTCTACTAGGACTATGAGAATTGAACCCATCCCTGAGAATCCAAAATTCTCCGTGCCACCTGTCACACCCCATCCTAAGTAAGGTCAGCTAAATAAGCTATCGGGCCCATACCCCGAAAATGTTGGTCACATCCTTCCCGTACTAAGAAATTTAGGTTAAACATAGACCAAGAGCCTTCAAAGCCCTTAGTAAGTTA-CAACACTTAATTTCTGTAAGGACTGCAAAACCCTACTCTGCATCAACTGAACGCAAATCAGCCACTTTAATTAAGCTAAGCCCTTCTAGATCAATGGGACTCAAACCCACAAACATTTAGTTAACAGCTAAACACCCTAGTCAAC-TGGCTTCAATCTAAAGCCCCGGCAGG-TTTGAAGCTGCTTCTTCGAATTTGCAATTCAATATGAAAT-TCACCTCGGAGCTTGGTAAAAAGAGGCCCAGCCTCTGTCTTTAGATTTACAGTCCAATGCCTTA-CTCAGCCATTTTACCACAAAAAAGGAAGGAATCGAACCCCCCAAAGCTGGTTTCAAGCCAACCCCATGACCTTCATGACTTTTTCAAAAGATATTAGAAAAACTATTTCATAACTTTGTCAAGGTTAAATTACGGGTT-AAACCCCGTATATCTTA-CACTGTAAAGCTAACCTAGCGTTAACCTTTTAAGTTAAAGATTAAGAGTATCGGCACCTCTTTGCAGTGA"

    rList = get_resource_list()
    sys.stdout.write("Available resources:\n")
    for i in range(len(rList)):
        sys.stdout.write("\tResource {}:\n\t\tName : {}\n".format(
            i, rList[i].name))
        sys.stdout.write("\t\tDesc : {}\n".format(rList[i].description))
        sys.stdout.write("\t\tFlags:")
        printFlags(rList[i].support_flags)
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

    prefered_flags = BEAGLE_FLAG_FRAMEWORK_CUDA | BEAGLE_FLAG_PRECISION_SINGLE | BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_SCALING_AUTO if autoScaling else 0

    # create an instance of the BEAGLE library
    instance, instDetails = create_instance(
        3,  # Number of tip data elements (input)
        nPartBuffs,  # Number of partials buffers to create (input)
        0,  # Number of compact state representation buffers to create (input)
        stateCount,  # Number of states in the continuous-time Markov chain (input)
        nPatterns,  # Number of site patterns to be handled by the instance (input)
        1,  # Number of rate matrix eigen-decomposition buffers to allocate (input)
        4,  # Number of rate matrix buffers (input)
        nRateCats,  # Number of rate categories (input)
        scaleCount,  # Number of scaling buffers
        [],  # List of potential resource on which this instance is allowed (input, NULL implies no restriction
        0,  # Length of resourceList list (input)
        prefered_flags,  #	Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input)
        0  # Bit-flags indicating required implementation characteristics, see BeagleFlags (input)
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
    printFlags(instDetails.flags)
    sys.stdout.write("\n\n")

    if not (instDetails.flags & BEAGLE_FLAG_SCALING_AUTO):
        autoScaling = False

    humanPartials = VectorDouble(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1., 1., 1., 1.]) for n in human])))
    chimpPartials = VectorDouble(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1., 1., 1., 1.]) for n in chimp])))
    gorillaPartials = VectorDouble(
        list(
            itertools.chain.from_iterable(
                [dna_map.get(n, [1., 1., 1., 1.]) for n in gorilla])))

    set_tip_partials(instance, 0, humanPartials)
    set_tip_partials(instance, 1, chimpPartials)
    set_tip_partials(instance, 2, gorillaPartials)

    rates = VectorDouble([0.03338775, 0.25191592, 0.82026848, 2.89442785])

    # create base frequency array
    freqs = VectorDouble([0.25] * 16)

    # create an array containing site category weights
    weights = VectorDouble([1.0 / rateCategoryCount] * rateCategoryCount)
    patternWeights = VectorDouble([1.0] * nPatterns)

    # an eigen decomposition for the JC69 model
    evec = VectorDouble([
        1.0, 2.0, 0.0, 0.5, 1.0, -2.0, 0.5, 0.0, 1.0, 2.0, 0.0, -0.5, 1.0,
        -2.0, -0.5, 0.0
    ])

    ivec = VectorDouble([
        0.25, 0.25, 0.25, 0.25, 0.125, -0.125, 0.125, -0.125, 0.0, 1.0, 0.0,
        -1.0, 1.0, 0.0, -1.0, 0.0
    ])

    evalues = VectorDouble(
        [0.0, -1.3333333333333333, -1.3333333333333333, -1.3333333333333333])

    # set the Eigen decomposition
    set_eigen_decomposition(instance, 0, evec, ivec, evalues)

    set_state_frequencies(instance, 0, freqs)

    set_category_weights(instance, 0, weights)

    set_pattern_weights(instance, patternWeights)

    # a list of indices and edge lengths
    nodeIndices = [0, 1, 2, 3]
    edgeLengths = VectorDouble([0.1, 0.1, 0.2, 0.1])

    rootIndices = [None] * nRootCount
    categoryWeightsIndices = [None] * nRootCount
    stateFrequencyIndices = [None] * nRootCount
    cumulativeScalingIndices = [None] * nRootCount

    for i in range(nRootCount):
        rootIndices[i] = 4 + i
        categoryWeightsIndices[i] = 0
        stateFrequencyIndices[i] = 0
        cumulativeScalingIndices[
            i] = 2 + i if manualScaling else BEAGLE_OP_NONE

        set_category_rates(instance, rates[i:])

        # tell BEAGLE to populate the transition matrices for the above edge lengths
        update_transition_matrices(
            instance,  # instance
            0,  # eigenIndex
            nodeIndices,  # probabilityIndices
            [],  # firstDerivativeIndices
            [],  # secondDerivativeIndices
            edgeLengths)  # edgeLengths

        # create a list of partial likelihood update operations
        # the order is [dest, destScaling, source1, matrix1, source2, matrix2]
        operationss = [
            BeagleOperation(3, 0 if manualScaling else BEAGLE_OP_NONE,
                            BEAGLE_OP_NONE, 0, 0, 1, 1),
            BeagleOperation(rootIndices[i],
                            1 if manualScaling else BEAGLE_OP_NONE,
                            BEAGLE_OP_NONE, 2, 2, 3, 3)
        ]

        if manualScaling:
            reset_scale_factors(instance, cumulativeScalingIndices[i])

        # update the partials
        update_partials(
            instance,  # instance
            operationss,  # eigenIndex
            cumulativeScalingIndices[i])  # cumulative scaling index

    if autoScaling:
        scaleIndices = [3, 4]
        accumulate_scale_factors(instance, scaleIndices, BEAGLE_OP_NONE)

    patternLogLik = VectorDouble([0.] * nPatterns)
    logL = VectorDouble([0.0])
    returnCode = 0

    # calculate the site likelihoods at the root node
    returnCode = calculate_root_log_likelihoods(
        instance,  # instance
        rootIndices,  # bufferIndices
        categoryWeightsIndices,  # weights
        stateFrequencyIndices,  # stateFrequencies
        cumulativeScalingIndices,  # cumulative scaling index
        nRootCount,  # count
        logL)  # outLogLikelihoods

    if returnCode < 0:
        sys.stderr.write("Failed to calculate root likelihood\n\n")
    else:

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
