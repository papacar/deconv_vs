#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

/**************************************************************************************************
 * Fn:	void low_pass_filter(cv::InputArray _srcI, cv::OutputArray _dstI);
 *
 * Low pass filter. reject the high frequency information
 *
 * Author:	Lccur
 *
 * Date:	2017/8/13
 *
 * Parameters:
 * _srcI -    	Source i.
 * _dstI -    	Destination i.
 */

void low_pass_filter(cv::InputArray _srcI, cv::OutputArray _dstI);
