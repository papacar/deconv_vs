#pragma once

#include "common.h"
#include <boost/lambda/lambda.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/special_functions/bessel.hpp>

typedef boost::multiprecision::cpp_dec_float_50 float_type;

typedef boost::math::policies::policy<
	boost::math::policies::domain_error<boost::math::policies::ignore_error>,
	boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
	boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
	boost::math::policies::denorm_error<boost::math::policies::ignore_error>,
	boost::math::policies::pole_error<boost::math::policies::ignore_error>,
	boost::math::policies::evaluation_error<boost::math::policies::ignore_error>
> ignore_all_policy;

/**************************************************************************************************
 * Fn:	int bessel_series_file();
 *
 * Bessel series file.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Returns:	An int.
 */

int bessel_series_file();

/**************************************************************************************************
 * Fn:	void bessel_serial_gen(int w, int h);
 *
 * Bessel serial generate.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Parameters:
 * w - 		  	The width.
 * h - 		  	The height.
 */

void bessel_serial_gen(int w, int h);

/**************************************************************************************************
 * Fn:	double normalCFD(double value);
 *
 * Normal CFD.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Parameters:
 * value - 	The value.
 *
 * Returns:	A double.
 */

double normalCFD(double value);

/**************************************************************************************************
 * Fn:	double born_wolf_point(double k, double NA, double n_i, int x, int y, int z);
 *
 * Born wolf point.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Parameters:
 * k - 		  	A double to process.
 * NA - 	  	The na.
 * n_i - 	  	The i.
 * x - 		  	The x coordinate.
 * y - 		  	The y coordinate.
 * z - 		  	The z coordinate.
 *
 * Returns:	A double.
 */

double born_wolf_point(double k, double NA, double n_i, int x, int y, int z);

/**************************************************************************************************
 * Fn:
 *  int born_wolf_full(int z, std::vector<std::vector<double> >& M2D, double k, double NA,
 *  double n_i, int num_p);
 *
 * Born wolf full.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Parameters:
 * z - 		  	The z coordinate.
 * M2D - 	  	[in,out] The 2D.
 * k - 		  	A double to process.
 * NA - 	  	The na.
 * n_i - 	  	The i.
 * num_p -    	Number of ps.
 *
 * Returns:	An int.
 */

int born_wolf_full(int z, std::vector<std::vector<double> >& M2D,
	double k, double NA, double n_i, int num_p);