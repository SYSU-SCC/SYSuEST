#include<Python.h>
#include<cstdint>
#include<iostream>
#include<string>
#include<vector>

using namespace std;

bool check_solved(double phase, int64_t a, int64_t N)
{
    
    PyObject *pName, *pModule, *pFunc_check;
    PyObject *pArgs, *pValue;
        
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('./')"); 

    std::string module_name = "func_python";
    pName = PyUnicode_DecodeFSDefault(module_name.c_str());
    pModule = PyImport_Import(pName);
    if(pModule != NULL){
        pFunc_check = PyObject_GetAttrString(pModule, "check_solved");
        assert(pFunc_check && PyCallable_Check(pFunc_check));
    }
    else{
        PyErr_Print();
        fprintf(stderr, "Failed to load \"%s\"\n", module_name.c_str());
        return false;
    }

    pArgs = PyTuple_New(3);
    PyTuple_SetItem(pArgs, 0, PyFloat_FromDouble(phase));
    PyTuple_SetItem(pArgs, 1, PyLong_FromLong(a));
    PyTuple_SetItem(pArgs, 2, PyLong_FromLong(N));
    pValue = PyObject_CallObject(pFunc_check, pArgs);
    int64_t r = PyLong_AsLong(pValue);
    
    return r;
}
