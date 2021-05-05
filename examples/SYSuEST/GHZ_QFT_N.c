#include "QuEST.h"
#include "mytimer.hpp"
#include "stdio.h"
#define N 29
int main(int narg, char *argv[]) {

  QuESTEnv Env = createQuESTEnv();
  double t1 = get_wall_time();

  Qureg q = createQureg(N, Env);
  /* GHZ quantum circuit */
  hadamard(q, 0);
  for (int i = 0; i < N - 1; ++i)
    controlledNot(q, i, i + 1);
  /* end of GHZ circuit */

  /* QFT starts */
  for (int i = 0; i < N - 1; ++i) {
    for (int j = 0; j < i; ++j)
      controlledRotateZ(q, j, i, M_PI * pow(0.5, i - j));
    hadamard(q, i);
  }
  /* end of QFT circuit */

  float q_measure[N];
  for (long long int i = 0; i < N; ++i) {
    q_measure[i] = calcProbOfOutcome(q, i, 1);
  }

  Complex amp_measure[(1LL << N) < 10 ? (1LL << N) : 10];
  for (int i = 0; i < sizeof(amp_measure) / sizeof(Complex); ++i) {
    amp_measure[i] = getAmp(q, i);
  }
  double t2 = get_wall_time();
  if (!Env.rank) {
    FILE *fp = fopen("probs.dat", "w");
    if (fp == NULL) {
      printf("    open probs.dat failed, Bye!");
      return 0;
    }

    FILE *fvec = fopen("stateVector.dat", "w");
    if (fvec == NULL) {
      printf("    open stateVector.dat failed, Bye!");
      return 0;
    }
    printf("\n");
    for (long long int i = 0; i < N; ++i) {
      // printf("  probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
      fprintf(fp, "Probability for q[%2lld]==1 : %lf    \n", i, q_measure[i]);
    }
    fprintf(fp, "\n");
    printf("\n");

    for (int i = 0; i < sizeof(amp_measure) / sizeof(Complex); ++i) {
      // printf("Amplitude of %dth state vector: %f\n", i, prob);
      fprintf(fvec, "Amplitude of %dth state vector: %12.6f,%12.6f\n", i,
              amp_measure[i].real, amp_measure[i].imag);
    }

    printf("Complete the simulation takes time %12.6f seconds.", t2 - t1);
    printf("\n");
  }
  destroyQureg(q, Env);
  destroyQuESTEnv(Env);

  return 0;
}
