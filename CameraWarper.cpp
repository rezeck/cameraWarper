/*
 * CameraWarper.cpp
 *
 *  Created on: June 17, 2019
 *      Author: Paulo Rezeck
 */
#include "CameraWarper.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <limits>
#include <iostream>

using namespace std;
using namespace cv;

CameraWarper::CameraWarper(float _scale)
{
    this->scale = _scale;
}

void CameraWarper::setCameraParams(InputArray _K, InputArray _R)
{
    Mat K = _K.getMat(), R = _R.getMat();

    CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
    CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);

    Mat_<float> Rinv = R.t();
    Mat_<float> R_Kinv = R * K.inv();
    this->r_kinv[0] = R_Kinv(0, 0);
    this->r_kinv[1] = R_Kinv(0, 1);
    this->r_kinv[2] = R_Kinv(0, 2);
    this->r_kinv[3] = R_Kinv(1, 0);
    this->r_kinv[4] = R_Kinv(1, 1);
    this->r_kinv[5] = R_Kinv(1, 2);
    this->r_kinv[6] = R_Kinv(2, 0);
    this->r_kinv[7] = R_Kinv(2, 1);
    this->r_kinv[8] = R_Kinv(2, 2);

    Mat_<float> K_Rinv = K * Rinv;
    this->k_rinv[0] = K_Rinv(0, 0);
    this->k_rinv[1] = K_Rinv(0, 1);
    this->k_rinv[2] = K_Rinv(0, 2);
    this->k_rinv[3] = K_Rinv(1, 0);
    this->k_rinv[4] = K_Rinv(1, 1);
    this->k_rinv[5] = K_Rinv(1, 2);
    this->k_rinv[6] = K_Rinv(2, 0);
    this->k_rinv[7] = K_Rinv(2, 1);
    this->k_rinv[8] = K_Rinv(2, 2);
}

CameraWarper::~CameraWarper() {}

void CameraWarper::mapForward(float x, float y, float &u, float &v)
{
    float x_ = this->r_kinv[0] * x + this->r_kinv[1] * y + this->r_kinv[2];
    float y_ = this->r_kinv[3] * x + this->r_kinv[4] * y + this->r_kinv[5];
    float z_ = this->r_kinv[6] * x + this->r_kinv[7] * y + this->r_kinv[8];

    float u_ = atan2f(x_, z_);
    float v_ = (float)CV_PI - acosf(y_ / sqrtf(x_ * x_ + y_ * y_ + z_ * z_));

    float r = sinf(v_) / (1 - cosf(v_));

    u = this->scale * r * cos(u_);
    v = this->scale * r * sin(u_);
}

void CameraWarper::mapBackward(float u, float v, float &x, float &y)
{
    u /= this->scale;
    v /= this->scale;

    float u_ = atan2f(v, u);
    float r = sqrtf(u * u + v * v);
    float v_ = 2 * atanf(1.f / r);

    float sinv = sinf((float)CV_PI - v_);
    float x_ = sinv * sinf(u_);
    float y_ = cosf((float)CV_PI - v_);
    float z_ = sinv * cosf(u_);

    float z;
    x = this->k_rinv[0] * x_ + this->k_rinv[1] * y_ + this->k_rinv[2] * z_;
    y = this->k_rinv[3] * x_ + this->k_rinv[4] * y_ + this->k_rinv[5] * z_;
    z = this->k_rinv[6] * x_ + this->k_rinv[7] * y_ + this->k_rinv[8] * z_;

    if (z > 0)
    {
        x /= z;
        y /= z;
    }
    else
        x = y = -1;
}

void CameraWarper::buildMaps(Size src_size)
{

    Point dst_tl, dst_br;
    this->detectResultRoi(src_size, dst_tl, dst_br);

    this->xmap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);
    this->ymap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);

    float x, y;
    for (int v = dst_tl.y; v <= dst_br.y; ++v)
    {
        for (int u = dst_tl.x; u <= dst_br.x; ++u)
        {
            this->mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
            this->xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;
            this->ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;
        }
    }
    this->dst_roi = Rect(dst_tl, dst_br);
}

Point CameraWarper::warp(InputArray src, int interp_mode, int border_mode, OutputArray dst)
{
    dst.create(this->dst_roi.height + 1, this->dst_roi.width + 1, src.type());
    cv::remap(src, dst, this->xmap, this->ymap, interp_mode, border_mode);

    return this->dst_roi.tl();
}

void CameraWarper::detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
{
    float tl_uf = (std::numeric_limits<float>::max)();
    float tl_vf = (std::numeric_limits<float>::max)();
    float br_uf = -(std::numeric_limits<float>::max)();
    float br_vf = -(std::numeric_limits<float>::max)();

    float u, v;
    for (int y = 0; y < src_size.height; ++y)
    {
        for (int x = 0; x < src_size.width; ++x)
        {
            this->mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
            tl_uf = (std::min)(tl_uf, u);
            tl_vf = (std::min)(tl_vf, v);
            br_uf = (std::max)(br_uf, u);
            br_vf = (std::max)(br_vf, v);
        }
    }

    dst_tl.x = static_cast<int>(tl_uf);
    dst_tl.y = static_cast<int>(tl_vf);
    dst_br.x = static_cast<int>(br_uf);
    dst_br.y = static_cast<int>(br_vf);
}


Mat CameraWarper::simpleBlend(const cv::Mat &left, const  cv::Mat &right){
    int baseline = 85;
    Size s = left.size();

    Mat proj = cv::Mat::zeros(s.height, 2*s.width - baseline, CV_8UC3);

    for (int w = 0; w < 2 * (s.width) - baseline; w++){
        if (w < (s.width - baseline)){
            left.col(w).copyTo(proj.col(w));
        } 
        else if (w >= (s.width - baseline) && w < s.width){
            float p = (float)(w-(s.width-baseline))/baseline;
            Mat res = left.col(w) * (1.0-p) + right.col(w-(s.width-baseline)) * (p);
            res.col(0).copyTo(proj.col(w));
        }
        else if (w >= s.width){
            right.col(w - (s.width) + baseline).copyTo(proj.col(w));
        }
    }
    return proj;
/*
    Mat l_gray, r_gray;
    if (left.channels() == 3){
        cvtColor(left, l_gray, cv::COLOR_BGR2GRAY);
    }
    else{
        l_gray= left.clone();
    }
    if (right.channels() == 3){
        cvtColor(right, r_gray, cv::COLOR_BGR2GRAY);
    }
    else{
        r_gray= right.clone();
    }

    Mat maskFirst = l_gray > 0;
    Mat maskSecond = r_gray > 0;

    Mat intersect;
    cv::multiply(maskFirst, maskSecond, intersect);
    

    Mat result, seam;

    cv::add(left/2, right/2, seam, intersect>0);

    cv::add(left, right, result, intersect==0);

    cv::add(seam, result, result, intersect>0);
     
    return result;
     */
}

// Calculates rotation matrix given euler angles.
Mat CameraWarper::eulerAnglesToRotationMatrix(Vec3f &theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<float>(3, 3) << 1, 0, 0,
               0, cosf(theta[0]), -sinf(theta[0]),
               0, sinf(theta[0]), cosf(theta[0]));

    // Calculate rotation about y axis
    Mat R_y = (Mat_<float>(3, 3) << cosf(theta[1]), 0, sinf(theta[1]),
               0, 1, 0,
               -sinf(theta[1]), 0, cosf(theta[1]));

    // Calculate rotation about z axis
    Mat R_z = (Mat_<float>(3, 3) << cosf(theta[2]), -sinf(theta[2]), 0,
               sinf(theta[2]), cosf(theta[2]), 0,
               0, 0, 1);

    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;

    return R;
}