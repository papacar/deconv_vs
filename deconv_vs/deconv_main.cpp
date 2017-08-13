#include "common.hpp"
#include "psf_boost.hpp"
#include "deconv.hpp"
#include "cvutils.hpp"
#include "vtkutils.hpp"
#include "config.hpp"
#include "basicutils.hpp"

#define M_2PI (6.283185307179586476925286766559)
int AiryRadius;

int main()
{
	double em_wavelen = 520.0;
	double NA = 1.2;
	double refr_index = 1.333;
	int psf_size = 16;
	std::vector<double> mse_stat;
	AiryRadius = 1000;
	cv::Mat PSF, image, out_img, de_img;
	std::vector<std::vector<double> > psf_vec;

	born_wolf_full(0, psf_vec, M_2PI / em_wavelen, NA, refr_index, psf_size);
	vec2mat(psf_vec, PSF);

	cv::Mat lowPassImage;
	image = cv::imread("imgs/gakki.png", 0);

	low_pass_filter(image, lowPassImage);

	multiply_fourier(lowPassImage, PSF, out_img);
	RichardLucy(out_img, PSF, de_img, mse_stat, 20);
	//divide_fourier(out_img, PSF, de_img);

	cv::imshow("GAKKI", image);
	cv::imshow("CONV", out_img);
	cv::imshow("RL", de_img);
	cv::imshow("PSF", PSF);
	cv::waitKey(-1);

	//plot1D(mse_stat, "MSE statistic");

	return 0;
}