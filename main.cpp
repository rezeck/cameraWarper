#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include <ctime>
#include <iostream>

#include "CameraWarper.h"

//#define SHOW_IMAGES
//#define SHOW_RECT_IMAGES

//#define STEREOGRAPHIC_WARPER
#define REMAP_WARPER

#define LOAD_CALIBRATION_FILE

#define TRY_ROTATION_CONFIG
//#define KEYBOARD_TURN

using namespace std;
using namespace cv;

Mat eulerAnglesToRotationMatrix(Vec3f &theta);
void setCameraParams(InputArray _K, InputArray _R, InputArray _T);

int main(int argc, char *argv[])
{
#ifdef LOAD_CALIBRATION_FILE
    FileStorage fs("../config/Stereo_rectification_parameters.yaml", FileStorage::READ); // load calibration file
    int height, width;
    fs["LEFT.height"] >> height;
    fs["LEFT.width"] >> width;
    //cout << "Calibration file:" << height << "x" << width << endl;

    Mat D1, D2, K1, K2, R1, R2, P1, P2;
    fs["LEFT.D"] >> D1;
    fs["RIGHT.D"] >> D2;
    fs["LEFT.K"] >> K1;
    fs["RIGHT.K"] >> K2;
    fs["LEFT.R"] >> R1;
    fs["RIGHT.R"] >> R2;
    fs["LEFT.P"] >> P1;
    fs["RIGHT.P"] >> P2;
#endif
    vector<Mat> images;

    images.push_back(imread("../1.png"));
    images.push_back(imread("../2.png"));
    images.push_back(imread("../3.png"));
    images.push_back(imread("../4.png"));

    for (int i = 0; i < images.size(); i++)
    {
        resize(images[i], images[i], cv::Size(752, 480), 0, 0, INTER_LANCZOS4); // only to fit with the calibration file
#ifdef SHOW_IMAGES
        cout << "Resize image " << i << " to: " << images[i].size().height << "x" << images[i].size().width << endl; // is the size correct?
        imshow("Image", images[i]);
        waitKey(0);
#endif
    }

    cv::Mat lmapx, lmapy, rmapx, rmapy;
    cv::Mat imgU1, imgU2;

    clock_t time_req = clock();
    cv::initUndistortRectifyMap(K1, D1, R1, P1, images[0].size(), CV_32F, lmapx, lmapy);
    //cv::initUndistortRectifyMap(K2, D2, R2, P2, images[1].size(), CV_32F, rmapx, rmapy);
    cv::remap(images[0], images[0], lmapx, lmapy, cv::INTER_LINEAR);
    //cv::remap(images[1], images[1], rmapx, rmapy, cv::INTER_LINEAR);
    time_req = clock() - time_req;
    cout << "\33[96mStereo Rectification 1: " << 1000 * (float)time_req / CLOCKS_PER_SEC << " ms\33[0m" << endl;

    time_req = clock();
    cv::initUndistortRectifyMap(K1, D1, R1, P1, images[2].size(), CV_32F, lmapx, lmapy);
    //cv::initUndistortRectifyMap(K2, D2, R2, P2, images[3].size(), CV_32F, rmapx, rmapy);
    cv::remap(images[2], images[3], lmapx, lmapy, cv::INTER_LINEAR);
    //cv::remap(images[3], images[3], rmapx, rmapy, cv::INTER_LINEAR);
    time_req = clock() - time_req;
    cout << "\33[96mStereo Rectification 2: " << 1000 * (float)time_req / CLOCKS_PER_SEC << " ms\33[0m" << endl;

#ifdef SHOW_RECT_IMAGES
    imshow("Image", images[0]);
    waitKey(0);

    imshow("Image", images[1]);
    waitKey(0);
#endif

    float scale = 100.0;
    Vec3f rot_1 = {0.01, -3.8, -1.610};
    Vec3f rot_2 = {0.01, 0.65, -1.5808};

    K1.convertTo(K1, CV_32F);
    K2.convertTo(K2, CV_32F);

    bool fit_1 = true;
    while (true)
    {
        cout << "\33[96mScale: " << scale << "\33[0m" << endl;
        cout << "\33[95mAng1: " << rot_1 << "\33[0m" << endl;
        cout << "\33[94mAng2: " << rot_2 << "\33[0m" << endl;

        Mat R1_ = eulerAnglesToRotationMatrix(rot_1);
        cout << "\33[95mR1:\n"
             << R1_ << "\33[0m" << endl;

        Mat R2_ = eulerAnglesToRotationMatrix(rot_2);
        cout << "\33[95mR2:\n"
             << R2_ << "\33[0m" << endl;

#ifdef REMAP_WARPER // faster
        Mat dst1, dst2;
        CameraWarper cw1(scale);
        cw1.setCameraParams(K1, R1_);
        cw1.buildMaps(images[0].size());

        CameraWarper cw2(scale);
        cw2.setCameraParams(K2, R2_);
        cw2.buildMaps(images[3].size());

        for (int i = 0; i < 10; i++)
        {
            time_req = clock();
            cw1.warp(images[0], INTER_LINEAR, BORDER_CONSTANT, dst1);
            cw2.warp(images[3], INTER_LINEAR, BORDER_CONSTANT, dst2);
            time_req = clock() - time_req;
            cout << "\33[96mWarp images: " << 1000 * (float)time_req / CLOCKS_PER_SEC << " ms\33[0m" << endl;
        }

        cout << "Size Left:" << dst1.size() << "\33" << endl;
        cout << "Size Right:" << dst2.size() << "\33" << endl;

        resize(dst1, dst1, cv::Size(752, 480), 0, 0, INTER_LINEAR);
        resize(dst2, dst2, cv::Size(752, 480), 0, 0, INTER_LINEAR);

        //cv::Rect myROI(0, 200, dst1.size().width, dst1.size().height-400);
        //dst1 = dst1(myROI);
        //dst2 = dst2(myROI);

        imshow("Left", dst1);  // check this images
        imshow("Right", dst2); // check this images

        Mat output;
        for (int i = 0; i < 10; i++)
        {
            time_req = clock();
            output = cw1.simpleBlend(dst1, dst2);
            time_req = clock() - time_req;
            cout << "\33[96mBlending images: " << 1000 * (float)time_req / CLOCKS_PER_SEC << " ms" << endl;
        }

        imshow("Output", output); // check this images
        waitKey();
        break;
#else
        time_req = clock();
        detail::StereographicWarper wrap_1(scale);
        wrap_1.warp(images[0], K1, R1_, INTER_LINEAR, BORDER_CONSTANT, dst1);
        detail::StereographicWarper wrap_2(scale);
        wrap_2.warp(images[3], K2, R2_, INTER_LINEAR, BORDER_CONSTANT, dst2);
        time_req = clock() - time_req;
        cout << "\33[96mWarp images: " << 1000 * (float)time_req / CLOCKS_PER_SEC << " ms\33[0m" << endl;

        cout << "Size Left:" << dst1.size() << "\33" << endl;
        cout << "Size Right:" << dst2.size() << "\33" << endl;

        resize(dst1, dst1, cv::Size(752, 480), 0, 0, INTER_LINEAR);
        resize(dst2, dst2, cv::Size(752, 480), 0, 0, INTER_LINEAR);

        imshow("Left", dst1);  // check this images
        imshow("Right", dst2); // check this images

        char c = waitKey();
        if (c == '1')
        {
            fit_1 = true;
            continue;
        }
        else if (c == '2')
        {
            fit_1 = false;
            continue;
        }

        if (c == 'q')
            break;
        else if (c == 'd')
            scale += 10;
        else if (c == 'c')
            scale -= 10;

        else if (c == 'f')
        {
            if (fit_1)
                rot_1[0] += 0.01;
            else
                rot_2[0] += 0.01;
        }
        else if (c == 'v')
        {
            if (fit_1)
                rot_1[0] -= 0.01;
            else
                rot_2[0] -= 0.01;
        }
        else if (c == 'g')
        {
            if (fit_1)
                rot_1[1] += 0.01;
            else
                rot_2[1] += 0.01;
        }
        else if (c == 'b')
        {
            if (fit_1)
                rot_1[1] -= 0.01;
            else
                rot_2[1] -= 0.01;
        }
        else if (c == 'h')
        {
            if (fit_1)
                rot_1[2] += 0.01;
            else
                rot_2[2] += 0.01;
        }
        else if (c == 'n')
        {
            if (fit_1)
                rot_1[2] -= 0.01;
            else
                rot_2[2] -= 0.01;
        }
#endif
    }

    return EXIT_SUCCESS;
}

// Calculates rotation matrix given euler angles.
Mat eulerAnglesToRotationMatrix(Vec3f &theta)
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
