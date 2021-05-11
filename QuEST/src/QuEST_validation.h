// Distributed under MIT licence. See https://github.com/QuEST-Kit/QuEST/blob/master/LICENCE.txt for details

/** @file
 * Provides validation defined in QuEST_validation.c which is used exclusively by QuEST.c
 */
 
# ifndef QUEST_VALIDATION_H
# define QUEST_VALIDATION_H

# include "QuEST.h"

# ifdef __cplusplus
extern "C" {
# endif

typedef enum {
    E_SUCCESS=0,
    E_INVALID_NUM_QUBITS,
    E_INVALID_TARGET_QUBIT,
    E_INVALID_CONTROL_QUBIT,
    E_INVALID_STATE_INDEX,
    E_INVALID_NUM_AMPS,
    E_INVALID_OFFSET_NUM_AMPS,
    E_TARGET_IS_CONTROL,
    E_TARGET_IN_CONTROLS,
    E_TARGETS_NOT_UNIQUE,
    E_INVALID_NUM_CONTROLS,
    E_NON_UNITARY_MATRIX,
    E_NON_UNITARY_COMPLEX_PAIR,
    E_ZERO_VECTOR,
    E_SYS_TOO_BIG_TO_PRINT,
    E_COLLAPSE_STATE_ZERO_PROB,
    E_INVALID_QUBIT_OUTCOME,
    E_CANNOT_OPEN_FILE,
    E_SECOND_ARG_MUST_BE_STATEVEC,
    E_MISMATCHING_QUREG_DIMENSIONS,
    E_MISMATCHING_QUREG_TYPES,
    E_DEFINED_ONLY_FOR_STATEVECS,
    E_DEFINED_ONLY_FOR_DENSMATRS,
    E_INVALID_PROB,
    E_UNNORM_PROBS,
    E_INVALID_ONE_QUBIT_DEPHASE_PROB,
    E_INVALID_TWO_QUBIT_DEPHASE_PROB,
    E_INVALID_ONE_QUBIT_DEPOL_PROB,
    E_INVALID_TWO_QUBIT_DEPOL_PROB
} ErrorCode;

void validateCreateNumQubits(int numQubits, const char* caller);

void validateStateIndex(Qureg qureg, long long int stateInd, const char* caller);

void validateTarget(Qureg qureg, int targetQubit, const char* caller);

void validateControlTarget(Qureg qureg, int controlQubit, int targetQubit, const char* caller);

void validateUniqueTargets(Qureg qureg, int qubit1, int qubit2, const char* caller);

void validateMultiControls(Qureg qureg, int* controlQubits, const int numControlQubits, const char* caller);

void validateMultiControlsTarget(Qureg qureg, int* controlQubits, const int numControlQubits, const int targetQubit, const char* caller);

void validateUnitaryMatrix(ComplexMatrix2 u, const char* caller);

void validateUnitaryComplexPair(Complex alpha, Complex beta, const char* caller);

void validateVector(Vector vector, const char* caller);

void validateStateVecQureg(Qureg qureg, const char* caller);

void validateDensityMatrQureg(Qureg qureg, const char* caller);

void validateOutcome(int outcome, const char* caller);

void validateMeasurementProb(qreal prob, const char* caller);

void validateMatchingQuregDims(Qureg qureg1, Qureg qureg2, const char *caller);

void validateMatchingQuregTypes(Qureg qureg1, Qureg qureg2, const char *caller);

void validateSecondQuregStateVec(Qureg qureg2, const char *caller);

void validateNumAmps(Qureg qureg, long long int startInd, long long int numAmps, const char* caller);

void validateFileOpened(int opened, const char* caller);

void validateProb(qreal prob, const char* caller);

void validateNormProbs(qreal prob1, qreal prob2, const char* caller);

void validateOneQubitDephaseProb(qreal prob, const char* caller);

void validateTwoQubitDephaseProb(qreal prob, const char* caller);

void validateOneQubitDepolProb(qreal prob, const char* caller);

void validateTwoQubitDepolProb(qreal prob, const char* caller);

void validateOneQubitDampingProb(qreal prob, const char* caller);


# ifdef __cplusplus
}
# endif

# endif // QUEST_VALIDATION_H
