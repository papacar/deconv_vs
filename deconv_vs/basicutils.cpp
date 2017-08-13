#include "basicutils.hpp"

void low_pass_filter(cv::InputArray _srcI, cv::OutputArray _dstI)
{
	// padded the image for better fft processing
	cv::Mat padded;
	cv::Mat srcI = _srcI.getMat();
	int opt_width = cv::getOptimalDFTSize(srcI.rows);
	int opt_height = cv::getOptimalDFTSize(srcI.cols);
	cv::copyMakeBorder(srcI, padded, 0, opt_width - srcI.rows, 0, opt_height - srcI.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// DFT
	cv::Mat planes[] = { cv::Mat_<double>(padded), cv::Mat::zeros(padded.size(), CV_64F) };
	cv::Mat com_img, DFT_filtered;
	cv::merge(planes, 2, com_img);
	cv::dft(com_img, com_img);
	
	cv::Mat mask = cv::Mat::ones(com_img.rows, com_img.cols, CV_8UC1);
	int boundh = com_img.rows * 0.05;
	int boundv = com_img.cols * 0.05;
	cv::Point pts[4] = {
		cv::Point(boundh, boundv),
		cv::Point(com_img.cols - boundh, boundv),
		cv::Point(com_img.cols - boundh, com_img.rows - boundv),
		cv::Point(boundh, com_img.rows - boundv),
	};
	cv::fillConvexPoly(mask, pts, 4, cv::Scalar(0));
	com_img.copyTo(DFT_filtered, mask);

	// transform fourier format image to real image;
	cv::idft(DFT_filtered, com_img);
	cv::split(com_img, planes);
	cv::magnitude(planes[0], planes[1], planes[0]);
	cv::normalize(planes[0], planes[0], 0, 1, CV_MINMAX);
	cv::imshow("Mask", mask*255);
	cv::imshow("Low Filtered", planes[0]);
	planes[0].copyTo(_dstI);
}