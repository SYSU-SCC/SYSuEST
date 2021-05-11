
#include <stdio.h>
#include <math.h>
#include "QuEST.h"
#include "mytimer.hpp"
#include <assert.h>
#include <cstdlib>
#include "cpp_wrapper.cpp"

const double PI = 4 * atan(1.0);

void InvQFT(const Qureg &Q, int num=0){

    if(num==0) num = Q.numQubitsRepresented;
    assert(num>0);

    for(int ib=num-1; ib>=0; --ib){
        hadamard(Q, ib);
        for(int ia=ib-1; ia>=0; --ia){
            controlledRotateZ(Q, ia, ib, -PI/(1<<(ib-ia)));
        }
    }
    
}

void ShorCircuit(const Qureg &QReg, int num_counts){

    for(int ind=0; ind<num_counts; ++ind)
        hadamard(QReg, ind);

    int numQubits = QReg.numQubitsRepresented;
    for(int ind=0; ind<numQubits; ++ind){
        controlledRotateX(QReg, ind, (ind+num_counts)%numQubits, PI/(1<<ind));
        controlledRotateY(QReg, ind, (ind+num_counts)%numQubits, PI/(1<<ind));
    }

    for(int ind=0; ind<num_counts; ++ind)
        controlledNot(QReg, ind, ind+num_counts);
    
    InvQFT(QReg, num_counts);

    /* inverse the qubit order */
    for(int i=0; i<num_counts/2; ++i)
        swapGate(QReg, i, num_counts-i-1);
}

int main (int narg, char *varg[]) {

    FILE *fp=fopen("probs.dat", "w");
    if(fp==NULL){
        printf("    open probs.dat failed, Bye!");
        return 0;
    }

    FILE *fvec=fopen("stateVectors.dat", "w");
    if(fp==NULL){
        printf("    open stateVector.dat failed, Bye!");
        return 0;
    }

    /* initialize python interpreter */
    Py_Initialize();

    int numQubits = 35;
    int num_counts = numQubits/2;
    int64_t N = 68242957193; // for 36 qubits, use 68242957193

    QuESTEnv env = createQuESTEnv();
    Qureg QReg = createQureg(numQubits, env);

    /* start timing */
    double t0 = get_wall_time();

    int num_try = 0;
    bool success = true;
    while(success){

        printf("%2d th try ...\n", ++num_try);
        initZeroState(QReg);

        ShorCircuit(QReg, num_counts);

        qreal prob;
        for(int ind=0; ind<numQubits; ++ind){
            prob = calcProbOfOutcome(QReg, ind, 1);
            if(env.rank==0) printf("Prob of qubit %2d (outcome=1) is: %f.\n", ind, prob);
        }

        int outcome;
        int64_t out_num = 0;
        for (int i = 0; i < num_counts; i++)
        {
            outcome = measure(QReg, i);
            out_num += outcome * (1<<i);
        }

        double phase = 1.0 * out_num / (1<<num_counts);

        if(env.rank==0) printf("phase = %lf\n", phase);
        int64_t a = rand()%N;
        if(check_solved(phase, a, N)) success=false;
    }

    if(env.rank == 0){
        double t1 = get_wall_time();
        if(env.rank==0) printf("Elasped time: %f.\n", t1-t0);
    }

    /* finish timing */
    double t = (get_wall_time() - t0)/num_try;
    if(env.rank==0) printf("Average cost time is %lf\n", t);

    /* write outputs */
    initZeroState(QReg);
    ShorCircuit(QReg, num_counts);

    qreal prob;
    for(int ind=0; ind<numQubits; ++ind){
        prob = calcProbOfOutcome(QReg, ind, 1);
        // printf("Prob of qubit %2d (outcome=1) is: %12.6f\n", ind, prob);
        if(env.rank==0) fprintf(fp, "Prob of qubit %2d (outcome=1) is: %12.6f\n", ind, prob);
    }

    for(int i=0; i<10; ++i){
        Complex amp = getAmp(QReg, i);
        if(env.rank==0) fprintf(fvec, "Amplitude of %dth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    }

    destroyQureg(QReg, env);
    destroyQuESTEnv(env);

    /* close python interpreter */
    Py_Finalize();

    return 0;
}
