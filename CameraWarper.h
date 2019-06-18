/*
 * CameraWarper.h
 *
 *  Created on: June 17, 2019
 *      Author: Paulo Rezeck
 */

#ifndef CAMERA_WARPER
#define CAMERA_WARPER

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

class CameraWarper
{
private:
    float r_kinv[9];
    float k_rinv[9];
    float scale;
    cv::Mat xmap, ymap;
    cv::Rect dst_roi;

public:
    CameraWarper(float _scale);
    ~CameraWarper();

    void setCameraParams(cv::InputArray _K, cv::InputArray _R);

    void mapForward(float x, float y, float &u, float &v);
    void mapBackward(float u, float v, float &x, float &y);

    void buildMaps(cv::Size src_size);
    void detectResultRoi(cv::Size src_size, cv::Point &dst_tl, cv::Point &dst_br);
    cv::Point warp(cv::InputArray src, int interp_mode, int border_mode, cv::OutputArray dst);

    cv::Mat simpleBlend(const cv::Mat &left, const  cv::Mat &right);

    cv::Mat eulerAnglesToRotationMatrix(cv::Vec3f &theta);
};

#endif