#include "Derivative.h"
/*
bool computeDiff(const Array33d &x,const BodyVec_t & mass,
                 const Array33d & safeDistance,Array33d * dv,Array33d * unsafeDistanceDest) {
    dv->setZero();
    Array33d distMat;
    distMat.fill(std::numeric_limits<double>::infinity());
    __dispLine
    for(int32_t i=0;i<3;i++) {
        for(int32_t j=i+1;j<3;j++) {
            auto xj_sub_xi=x.col(j)-x.col(i);
            const double distanceSquare=xj_sub_xi.square().sum();
            const double distance=std::sqrt(distanceSquare);
            distMat(i,j)=distance;
            distMat(j,i)=distance;

            auto G_mult_diffXji_div_dist_pow_5_2=G*xj_sub_xi/(distanceSquare*distance);

            dv->col(i)+=mass(j)*G_mult_diffXji_div_dist_pow_5_2;
            dv->col(j)-=mass(i)*G_mult_diffXji_div_dist_pow_5_2;
        }
    }
    __dispLine
    if((distMat<safeDistance).any()) {
        *unsafeDistanceDest=distMat;
        return false;
    }

    return true;
}
*/
