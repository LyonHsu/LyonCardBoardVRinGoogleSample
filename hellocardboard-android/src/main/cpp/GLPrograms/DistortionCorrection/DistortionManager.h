//
// Created by Consti10 on 31/10/2019.
//

#ifndef RENDERINGX_DISTORTIONMANAGER_H
#define RENDERINGX_DISTORTIONMANAGER_H

#include <array>
#include <string>
#include <sstream>
#include <Helper/MDebug.hpp>
#include <vector>
#include <sys/stat.h>
#include "Helper/GLHelper.hpp"

#include <glm/glm.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Helper/NDKHelper.h>
#include "android/log.h"
#include "MLensDistortion.h"
//#include "polynomial_radial_distortion.h"

class DistortionManager {
public:
    //This is the inverse function to the kN distortion parameters from the headset
    //Even tough the distortion parameters are only up to 2 radial values,
    //We need more for the inverse for a good fit
    static constexpr const int N_RADIAL_UNDISTORTION_COEFICIENTS=6;
    struct RadialDistortionCoefficients{
        float maxRadSquared;
        std::array<float,N_RADIAL_UNDISTORTION_COEFICIENTS> kN;
    };
    RadialDistortionCoefficients radialDistortionCoefficients;

    struct UndistortionHandles{
        GLuint uMaxRadSq;
        GLuint uKN;
        //Only active if mode==2
        GLuint uScreenParams_w;
        GLuint uScreenParams_h;
        GLuint uScreenParams_x_off;
        GLuint uScreenParams_y_off;
        //
        GLuint uTextureParams_w;
        GLuint uTextureParams_h;
        GLuint uTextureParams_x_off;
        GLuint uTextureParams_y_off;
    };
    //Left and right eye each
    std::array<MLensDistortion::ViewportParams,2> screen_params;
    std::array<MLensDistortion::ViewportParams,2> texture_params;

    enum DISTORTION_MODE{RADIAL,RADIAL_2};
    const DISTORTION_MODE distortionMode;

    bool leftEye=true;

public:
    DistortionManager():distortionMode(DISTORTION_MODE::RADIAL){};
    DistortionManager(const DISTORTION_MODE& distortionMode1):distortionMode(distortionMode1){}

    UndistortionHandles getUndistortionUniformHandles(const GLuint program)const;
    void beforeDraw(const UndistortionHandles& undistortionHandles)const;
    void afterDraw()const;

    static std::string writeGLPosition(const DistortionManager* distortionManager,const std::string &positionAttribute="aPosition");
    static std::string writeGLPositionWithDistortion(const DistortionManager &distortionManager, const std::string &positionAttribute);
    static std::string writeDistortionParams(const DistortionManager *distortionManager);

    void updateDistortion(const MPolynomialRadialDistortion& distortion,float maxRadSq);
    void updateDistortion(const MPolynomialRadialDistortion& inverseDistortion,float maxRadSq,
                          const std::array<MLensDistortion::ViewportParams,2> screen_params,const std::array<MLensDistortion::ViewportParams,2> texture_params);

    void updateDistortion(JNIEnv *env,jfloatArray undistData);
};


#endif //RENDERINGX_DISTORTIONMANAGER_H
