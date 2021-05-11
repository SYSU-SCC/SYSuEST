#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include "QuEST.h"
#include <assert.h>
#include <cmath>
using namespace std;

using Ham_term = pair<double, string>;

vector<string> str_split(
    const string &str,
    const string &pattern)
{
    char *str_ptr = new char[str.size()+1];
    strcpy(str_ptr, str.c_str());
    vector<string> res;
    char *token = strtok(str_ptr, pattern.c_str());
    while(token!=nullptr){
        res.push_back(string(token));
        token = strtok(nullptr, pattern.c_str());
    }
    delete [] token;

    return res;
}


vector<Ham_term> parse_Ham(const string &filename){

    ifstream f_in(filename);
    string tmp1, tmp2;
    Ham_term ham_pair;
    vector<Ham_term> Ham_terms;
    while(!f_in.eof()){
        f_in >> tmp1 >> tmp2;
        ham_pair.first = stod(tmp1);
        ham_pair.second = tmp2;
        Ham_terms.push_back(ham_pair);
    }

    return Ham_terms;
}


double InnerProduct(const Qureg &Q1, const Qureg &Q2){
    ComplexArray CA1 = Q1.stateVec;
    ComplexArray CA2 = Q2.stateVec;

    assert(Q1.numAmpsTotal==Q2.numAmpsTotal);
    int64_t numStateVec = Q1.numAmpsTotal;
    
    double real_part = 0.0, imag_part = 0.0;
#if defined(NO_USE_SYSUEST)
    for(int64_t ind=0; ind<numStateVec; ++ind){
        real_part += CA1.real[ind]*CA2.real[ind]+CA1.imag[ind]*CA2.imag[ind];
        imag_part += CA1.real[ind]*CA2.imag[ind]-CA1.imag[ind]*CA2.real[ind];
    }
#else
    do {
      Complex part = calcInnerProduct(Q1, Q2);
      real_part = part.real;
      imag_part = part.imag;
    } while (0);
#endif

    /* The expectation of a Hamiltonian should be real! */
    assert(abs(imag_part)<1E-3);
    return real_part;
}
