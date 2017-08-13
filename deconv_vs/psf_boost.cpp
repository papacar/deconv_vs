#pragma once
#include "psf_boost.hpp"
#include <complex>

#include <boost/lambda/lambda.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/progress.hpp>

extern int AiryRadius;

typedef boost::multiprecision::cpp_dec_float_50 float_type;
typedef boost::math::policies::policy<
	boost::math::policies::domain_error<boost::math::policies::ignore_error>,
	boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
	boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
	boost::math::policies::denorm_error<boost::math::policies::ignore_error>,
	boost::math::policies::pole_error<boost::math::policies::ignore_error>,
	boost::math::policies::evaluation_error<boost::math::policies::ignore_error>
> ignore_all_policy;

int bessel_series_file()
{
	double v = 0;
	std::vector<float_type> bessel_table;
	float_type bessel_tmp;
	float_type bessel_inte;
	try
	{
		std::ofstream out("../data/bessel_table.txt", 'w');
		for (int i = 0; i < 2000; i++)
		{
			bessel_tmp = boost::math::sph_bessel(0, v);
			v += 0.01;
			bessel_inte += bessel_tmp*0.01;
			out << bessel_tmp << '\t' << bessel_inte << '\n';
		}
		out.close();
	}
	catch (std::exception ex)
	{
		std::cout << "Thrown exception " << ex.what() << std::endl;
	}
	return 0;
}

void bessel_serial_gen(int w, int h)
{
	double v = 0;
	std::vector<float_type> bessel_int_table;
	float_type base_unit, bessel_sum;
	double delta = 2 / w;
	bessel_sum = 0;
	int up_board, low_board;

	try
	{
		for (int i = 0; i < w; i++)
		{
			bessel_sum += base_unit * delta;
			base_unit = boost::math::sph_bessel(0, v);
			bessel_int_table.push_back(bessel_sum);
		}
	}
	catch (std::exception ex)
	{
		std::cout << "Theown exception " << ex.what() << std::endl;
	}
	up_board = h * 2 / 3;
	int up_padding = up_board / 2;
	for (std::vector<float_type>::iterator iter = bessel_int_table.begin();
		iter != bessel_int_table.end(); iter++)
	{

	}
}

double normalCFD(double value)
{
	return 0.5*std::erfc(-value * std::sqrt(0.5));
}

double born_wolf_point(double k, double NA, double n_i, int x, int y, int z)
{
	double const_ratio = k * NA / n_i * std::sqrt(x*x + y*y);
	double const_opd = -0.5 * k * z * NA*NA / n_i / n_i;
	std::complex<double>bess_sum(0.0, 0.0);
	double bess_tmp, v = 0.0;
	// 这是用于控制积分精度的，从0积到1这个数值越大，则积分精度越高
	// 如果要追求绝对的精度的话，要做一个trade off，就是数值精度的问题
	int num_p = 10000;
	double delta_v = 1.0 / num_p;
	std::complex<double>opd(0.0, v);

	// Always using try for unpredicted error;
	try
	{
		for (int i = 0; i <= num_p; i++)
		{
			bess_tmp = boost::math::sph_bessel(0, v*const_ratio);

			// TODO(peo):这里还会有大改动，可能计算过程出现失误
			opd.imag(v*v * const_opd);
			bess_sum += bess_tmp * std::exp(opd) * delta_v;
			v += delta_v;
		}
	}
	catch (std::exception ex)
	{
		std::cout << "Thrown exception " << ex.what() << std::endl;
	}
	return std::pow(1.0*std::abs(bess_sum), 2);
}

int born_wolf_full(int z, std::vector<std::vector<double> >& M2D,
	double k, double NA, double n_i, int num_p)
{
	std::vector<std::vector<double> >M2D_cp(num_p);
	double step = AiryRadius / num_p;
	double bessel_res = 0.0;
	z *= step;
	M2D.resize(num_p * 2);
#ifdef _OBSERVE_MAX_PIXEL
	std::cout << "Max_pixel : " << max_pixel << std::endl;
#endif
	try {
		for (int i = 0; i < num_p; i++) {
			M2D[i].resize(num_p * 2);
			M2D[i + num_p].resize(num_p * 2);
			M2D_cp[i].resize(num_p);
			for (int j = 0; j <= i; j++) {
				bessel_res = born_wolf_point(k, NA, n_i, j*step, i*step, z);
				M2D_cp[i][j] = bessel_res;
				M2D_cp[j][i] = bessel_res;
			}
		}
		for (int i = 0; i < num_p; ++i) {
			for (int j = 0; j < num_p; ++j) {
				M2D[i + num_p][j + num_p] = M2D_cp[i][j];
				M2D[num_p - i - 1][j + num_p] = M2D_cp[i][j];
				M2D[i + num_p][num_p - j - 1] = M2D_cp[i][j];
				M2D[num_p - i - 1][num_p - j - 1] = M2D_cp[i][j];
			}
		}
	}
	catch (std::exception ex)
	{
		std::cout << "Thrown exception : " << ex.what() << std::endl;
	}
	return 0;
}