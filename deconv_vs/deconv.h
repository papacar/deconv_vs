#pragma once
#include "common.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\imgproc.hpp>
#include <fftw3.h>

/**************************************************************************************************
 * Fn:
 *  int divSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags,
 *  bool conjB = true);
 *
 * Div spectrums. Inputs must be complex data structure.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Parameters:
 * _srcA -    	Source a.
 * _srcB -    	Source b.
 * _dst - 	  	Destination for the.
 * flags -    	The flags.
 * conjB -    	(Optional) True to conj b.
 *
 * Returns:	An int.
 */

int divSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB = true);

/**************************************************************************************************
 * Fn:
 *  int divSpectrumszeros(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst,
 *  int flags, bool conjB);
 *
 * Div spectrumszeros.once meet the divider is zero, the ans will be zero
 *
 * Author:	Lccur
 *
 * Date:	2017/7/30
 *
 * Parameters:
 * _srcA -    	Source a.
 * _srcB -    	Source b.
 * _dst - 	  	Destination for the.
 * flags -    	The flags.
 * conjB -    	True to conj b.
 *
 * Returns:	An int.
 */

int divSpectrumszeros(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB);

/**************************************************************************************************
 * Fn:	void divide_fourier(cv::InputArray A, cv::InputArray B, cv::OutputArray C);
 *
 * Divide fourier. Input will be OpenCV formatter.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Parameters:
 * A - 		  	A cv::InputArray to process.
 * B - 		  	A cv::InputArray to process.
 * C - 		  	A cv::OutputArray to process.
 */

void divide_fourier(cv::InputArray A, cv::InputArray B, cv::OutputArray C);

/**************************************************************************************************
 * Fn:	void multiply_fourier(cv::InputArray A, cv::InputArray B, cv::OutputArray C);
 *
 * Multiply fourier. Operation in frequency domain.
 *
 * Author:	Lccur
 *
 * Date:	2017/7/29
 *
 * Parameters:
 * A - 		  	A cv::InputArray to process.
 * B - 		  	A cv::InputArray to process.
 * C - 		  	A cv::OutputArray to process.
 */

void multiply_fourier(cv::InputArray A, cv::InputArray B, cv::OutputArray C);

/**************************************************************************************************
 * Fn:	void forwardbackwardtest(cv::InputArray A, cv::OutputArray B)
 *
 * Forwardbackwardtests.only for testing if the algorithm is right.
 *
 * Author:	Lccur
 *
 * Date:	2017/8/2
 *
 * Parameters:
 * A - 		  	A cv::InputArray to process.
 * B - 		  	A cv::OutputArray to process.
 */

void forwardbackwardtest(cv::InputArray A, cv::OutputArray B);

/**************************************************************************************************
 * Fn:	void shift_fre(cv::InputArray A);
 *
 * Shift frequency. make the high frequency be in the center of image.
 *
 * Author:	Lccur
 *
 * Date:	2017/8/3
 *
 * Parameters:
 * A - 	A cv::InputArray to process.
 */

void shift_fre(cv::InputArray A);

/**************************************************************************************************
 * Fn:
 *  void RichardLucy(cv::InputArray _srcI, cv::InputArray _coreI, cv::OutputArray _dst,
 *  int iteration = 5);
 *
 * Richard lucy. Impletementation of Richardson-Lucy algorithm.
 *
 * Author:	Lccur
 *
 * Date:	2017/8/5
 *
 * Parameters:
 * _srcI - 	   	Source i.
 * _coreI -    	The core i.
 * _dst - 	   	Destination for the.
 * iteration - 	(Optional) The iteration.
 */

void RichardLucy(cv::InputArray _srcI,
	cv::InputArray _coreI,
	cv::OutputArray _dst,
	int iteration);

/**************************************************************************************************
 * Fn:
 *  void RichardLucy(cv::InputArray _srcI, cv::InputArray _coreI, cv::OutputArray _dst,
 *  int iteration = 5);
 *
 * Richard lucy output with the vector
 *
 * Author:	Lccur
 *
 * Date:	2017/8/6
 *
 * Parameters:
 * _srcI - 	   	Source i.
 * _coreI -    	The core i.
 * _dst - 	   	Destination for the.
 * iteration - 	(Optional) The iteration.
 */

void RichardLucy(cv::InputArray _srcI,
	cv::InputArray _coreI,
	cv::OutputArray _dst,
	std::vector<double> &mse_stat,
	int iteration
	);
/**************************************************************************************************
 * Fn:	double MSEest(cv::InputArray _srcIA, cv::InputArray _srcIB);
 *
 * Milliseconds eest.
 *
 * Author:	Lccur
 *
 * Date:	2017/8/5
 *
 * Parameters:
 * _srcIA -   	Source ia.
 * _srcIB -   	Source ib.
 *
 * Returns:	A double.
 */

double MSEest(cv::InputArray _srcIA, cv::InputArray _srcIB);
/**************************************************************************************************
// End of deconv.h
 **************************************************************************************************/