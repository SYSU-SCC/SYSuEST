#include <stdio.h>
#include <math.h>
#include "QuEST.h"
#include <assert.h>
#include "mytimer.hpp"

const double PI = 4 * atan(1.0);

void QFT(const Qureg &Q, int num=0){

    if(num==0) num = Q.numQubitsRepresented;
    assert(num>0);

    for(int ib=0; ib<num; ib++)
    {
        for(int ia=0; ia<ib; ia++)
            controlledRotateZ(Q, ia, ib, PI/(1<<(ib-ia)));
        hadamard(Q, ib);
    }

}

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

int main () {

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

    int numQubits = 35;
    int halfnum = numQubits/2;

    QuESTEnv env = createQuESTEnv();
    Qureg QReg = createQureg(numQubits, env);
    initZeroState(QReg);

    /* start timing */
    double t0 = get_wall_time();

    for(int ind=0; ind<numQubits; ++ind)
        hadamard(QReg, ind);

    for(int ind=0; ind<numQubits; ++ind){
        controlledRotateX(QReg, ind, (ind+halfnum)%numQubits, PI/(ind+1));
        controlledRotateY(QReg, ind, (ind+halfnum)%numQubits, PI/(ind+1));
    }

    InvQFT(QReg, numQubits);

    /* add QFT and InvQFT */
    QFT(QReg, numQubits);
    InvQFT(QReg, numQubits);

    qreal prob;
    for(int ind=0; ind<numQubits; ++ind){
        prob = calcProbOfOutcome(QReg, ind, 1);
        printf("Prob of qubit %2d (outcome=1) is: %12.6f\n", ind, prob);
	    fprintf(fp, "Prob of qubit %2d (outcome=1) is: %12.6f\n", ind, prob);
    }

    for(int i=0; i<10; ++i){
        Complex amp = getAmp(QReg, i);
	    fprintf(fvec, "Amplitude of %dth state vector: %12.6f,%12.6f\n", i, amp.real, amp.imag);
    }

    /* finish timing */
    double t = get_wall_time() - t0;
    printf("InvQFT(%d) cost time: %lf\n", numQubits, t);

    destroyQureg(QReg, env);
    destroyQuESTEnv(env);
    return 0;
}
