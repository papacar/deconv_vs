#pragma once
#include "common.h"
#include <tiffio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void vec2mat(std::vector<std::vector<double> > M2D, const cv::OutputArray out_im);