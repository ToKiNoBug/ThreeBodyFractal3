#ifndef MULTIREC_HPP
#define MULTIREC_HPP

#include "Derivative.h"

template<class state_t>
class MultiRec
{
public:
    MultiRec() {
        idx=0;
    }

    ~MultiRec()=default;

    inline void push(double t,const state_t & s) {
        idx=!idx;
        times[idx]=t;
        states[idx]=s;
    }

    inline double & currentTime() {
        return times[idx];
    }

    inline state_t & currentState() {
        return states[idx];
    }

    inline double & previousTime() {
        return times[!idx];
    }

    inline state_t & previousState() {
        return states[!idx];
    }

private:
    int8_t idx;
    std::array<double,2> times;
    std::array<state_t,2> states;
};

#endif // MULTIREC_HPP
