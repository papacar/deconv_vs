#include "deconv.hpp"

int divSpectrums(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB)
{
	cv::Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
	int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
	int rows = srcA.rows, cols = srcA.cols;

	CV_Assert(type == srcB.type() && srcA.size() == srcB.size());
	CV_Assert(type == CV_64FC1 || type == CV_64FC2);

	_dst.create(srcA.rows, srcA.cols, type);
	cv::Mat dst = _dst.getMat();

	CV_Assert(dst.data != srcA.data); // non-inplace check
	CV_Assert(dst.data != srcB.data); // non-inplace check

	bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
		srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

	if (is_1d && !(flags & cv::DFT_ROWS)) {
		cols = cols + rows - 1, rows = 1;
	}
	int ncols = cols*cn;
	int j0 = cn == 1;
	int j1 = ncols - (cols % 2 == 0 && cn == 1);

	if (depth == CV_64F) {
		const double *dataA = srcA.ptr<double>();
		const double *dataB = srcB.ptr<double>();
		auto *dataC = dst.ptr<double>();
		auto eps = DBL_EPSILON; // prevent div0 problems

		size_t stepA = srcA.step / sizeof(dataA[0]);
		size_t stepB = srcB.step / sizeof(dataB[0]);
		size_t stepC = dst.step / sizeof(dataC[0]);

		if (!is_1d && cn == 1) {
			// two channels real and image
			for (int k = 0; k < (cols % 2 ? 1 : 2); k++) {
				if (k == 1) {
					dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
				}
				dataC[0] = dataA[0] / (dataB[0] + eps);
				if (rows % 2 == 0) {
					dataC[(rows - 1) * stepC] = dataA[(rows - 1) * stepA] / (dataB[(rows - 1) * stepB] + eps);
				}
				if (!conjB) {
					for (int j = 1; j <= rows - 2; j += 2) {
						double denom =
							dataB[j * stepB] * dataB[j * stepB] + dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] +
							eps;
						double re =
							dataA[j * stepA] * dataB[j * stepB] + dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];
						double im =
							dataA[(j + 1) * stepA] * dataB[j * stepB] - dataA[j * stepA] * dataB[(j + 1) * stepB];

						dataC[j * stepC] = (re / denom);
						dataC[(j + 1) * stepC] = (im / denom);
					}
				}
				else {
					for (int j = 1; j <= rows - 2; j += 2) {
						double denom =
							dataB[j * stepB] * dataB[j * stepB] + dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB] +
							eps;
						double re =
							dataA[j * stepA] * dataB[j * stepB] + dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];
						double im =
							dataA[(j + 1) * stepA] * dataB[j * stepB] - dataA[j * stepA] * dataB[(j + 1) * stepB];

						dataC[j * stepC] = re / denom;
						dataC[(j + 1) * stepC] = im / denom;
					}
				}
				if (k == 1) {
					dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
				}
			}
		}

		for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC) {
			if (is_1d && cn == 1) {
				dataC[0] = dataA[0] / (dataB[0] + eps);
				if (cols % 2 == 0) {
					dataC[j1] = dataA[j1] / (dataB[j1] + eps);
				}
			}
			if (!conjB) {
				for (int j = 0; j < j1; j += 2) {
					double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
					double re = dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1];
					double im = dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1];
					dataC[j] = re / denom;
					dataC[j + 1] = im / denom;
				}
			}
			else {
				for (int j = j0; j < j1; j += 2) {
					double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1] + eps;
					double re = dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1];
					double im = dataA[j + 1] * dataB[j] + dataA[j] * dataB[j + 1];
					dataC[j] = re / denom;
					dataC[j + 1] = im / denom;
				}
			}
		}
	}
	return 0;
}

int divSpectrumszeros(cv::InputArray _srcA, cv::InputArray _srcB, cv::OutputArray _dst, int flags, bool conjB)
{
	cv::Mat srcA = _srcA.getMat(), srcB = _srcB.getMat();
	int depth = srcA.depth(), cn = srcA.channels(), type = srcA.type();
	int rows = srcA.rows, cols = srcA.cols;

	CV_Assert(type == srcB.type() && srcA.size() == srcB.size());
	CV_Assert(type == CV_64FC1 || type == CV_64FC2);

	_dst.create(srcA.rows, srcA.cols, type);
	cv::Mat dst = _dst.getMat();

	CV_Assert(dst.data != srcA.data); // non-inplace check
	CV_Assert(dst.data != srcB.data); // non-inplace check

	bool is_1d = (flags & cv::DFT_ROWS) || (rows == 1 || (cols == 1 &&
		srcA.isContinuous() && srcB.isContinuous() && dst.isContinuous()));

	if (is_1d && !(flags & cv::DFT_ROWS)) {
		cols = cols + rows - 1, rows = 1;
	}
	int ncols = cols*cn;
	int j0 = cn == 1;
	int j1 = ncols - (cols % 2 == 0 && cn == 1);

	if (depth == CV_64F) {
		const double *dataA = srcA.ptr<double>();
		const double *dataB = srcB.ptr<double>();
		auto *dataC = dst.ptr<double>();
		auto eps = DBL_EPSILON;

		size_t stepA = srcA.step / sizeof(dataA[0]);
		size_t stepB = srcB.step / sizeof(dataB[0]);
		size_t stepC = dst.step / sizeof(dataC[0]);

		if (!is_1d && cn == 1) {
			// two channels real and image
			for (int k = 0; k < (cols % 2 ? 1 : 2); k++) {
				if (k == 1) {
					dataA += cols - 1, dataB += cols - 1, dataC += cols - 1;
				}
				dataC[0] = dataA[0] / (dataB[0] + eps);
				if (rows % 2 == 0) {
					dataC[(rows - 1) * stepC] = dataA[(rows - 1) * stepA] / (dataB[(rows - 1) * stepB] + eps);
				}
				if (!conjB) {
					for (int j = 1; j <= rows - 2; j += 2) {
						double denom =
							dataB[j * stepB] * dataB[j * stepB] + dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB];
						double re =
							dataA[j * stepA] * dataB[j * stepB] + dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];
						double im =
							dataA[(j + 1) * stepA] * dataB[j * stepB] - dataA[j * stepA] * dataB[(j + 1) * stepB];

						if (denom < eps && denom > -eps) {
							dataC[j * stepC] = 0;
							dataC[(j + 1) * stepC] = 0;
						}
						else
						{
							dataC[j * stepC] = re / denom;
							dataC[(j + 1) * stepC] = im / denom;
						}
					}
				}
				else {
					for (int j = 1; j <= rows - 2; j += 2) {
						double denom =
							dataB[j * stepB] * dataB[j * stepB] + dataB[(j + 1) * stepB] * dataB[(j + 1) * stepB];
						double re =
							dataA[j * stepA] * dataB[j * stepB] - dataA[(j + 1) * stepA] * dataB[(j + 1) * stepB];
						double im =
							dataA[(j + 1) * stepA] * dataB[j * stepB] + dataA[j * stepA] * dataB[(j + 1) * stepB];

						if (denom < eps && denom > -eps) {
							dataC[j * stepC] = 0;
							dataC[(j + 1) * stepC] = 0;
						}
						else
						{
							dataC[j * stepC] = re / denom;
							dataC[(j + 1) * stepC] = im / denom;
						}
					}
				}
				if (k == 1) {
					dataA -= cols - 1, dataB -= cols - 1, dataC -= cols - 1;
				}
			}
		}

		for (; rows--; dataA += stepA, dataB += stepB, dataC += stepC) {
			if (is_1d && cn == 1) {
				dataC[0] = dataA[0] / (dataB[0] + eps);
				if (cols % 2 == 0) {
					dataC[j1] = dataA[j1] / (dataB[j1] + eps);
				}
			}
			if (!conjB) {
				for (int j = 0; j < j1; j += 2) {
					double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1];
					double re = dataA[j] * dataB[j] + dataA[j + 1] * dataB[j + 1];
					double im = dataA[j + 1] * dataB[j] * dataB[j] - dataA[j] * dataB[j + 1];
					if (denom < eps && denom > -eps) {
						dataC[j * stepC] = 0;
						dataC[(j + 1) * stepC] = 0;
					}
					else
					{
						dataC[j * stepC] = re / denom;
						dataC[(j + 1) * stepC] = im / denom;
					}
				}
			}
			else {
				for (int j = j0; j < j1; j += 2) {
					double denom = dataB[j] * dataB[j] + dataB[j + 1] * dataB[j + 1];
					double re = dataA[j] * dataB[j] - dataA[j + 1] * dataB[j + 1];
					double im = dataA[j + 1] * dataB[j] - dataA[j] * dataB[j + 1];
					if (denom < eps && denom > -eps) {
						dataC[j * stepC] = eps;
						dataC[(j + 1) * stepC] = eps;
					}
					else
					{
						dataC[j * stepC] = re / denom;
						dataC[(j + 1) * stepC] = im / denom;
					}
				}
			}
		}
	}
	return 0;
}

void divide_fourier(cv::InputArray A, cv::InputArray B, cv::OutputArray C)
{
	cv::Mat paddedA, paddedB, spectrumC, paddedBs;
	cv::Mat matA, matB;
	matA = A.getMat(); matB = B.getMat();
	int width = matA.cols;//+matB.cols;
	int height = matA.rows;//+matB.cols;
	int opt_width = width;//cv::getOptimalDFTSize(width);
	int opt_height = height;//cv::getOptimalDFTSize(height);
	cv::Rect rect(0, 0, opt_width / 2, opt_height / 2);

	cv::copyMakeBorder(matA, paddedA, 0, opt_height - matA.rows, 0, opt_width - matA.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::copyMakeBorder(matB, paddedB, opt_height - matB.rows, 0, opt_width - matB.cols, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	// DFT
	cv::Mat planesA[] = { cv::Mat_<double>(paddedA), cv::Mat::zeros(paddedA.size(), CV_64F) };
	cv::Mat planesB[] = { cv::Mat_<double>(paddedB), cv::Mat::zeros(paddedB.size(), CV_64F) };
	cv::Mat planesC[] = { cv::Mat::zeros(paddedB.size(), CV_64F), cv::Mat::zeros(paddedB.size(), CV_64F) };

	cv::Mat com_A, com_B, com_C, icom_B;
	cv::merge(planesA, 2, com_A);
	cv::merge(planesB, 2, com_B);
	cv::dft(com_A, com_A);
	cv::dft(com_B, com_B);
	cv::split(com_B, planesB);

	divSpectrums(com_A, com_B, com_C, 0, true);

	cv::idft(com_C, com_C);
	cv::split(com_C, planesC);

	cv::magnitude(planesC[0], planesC[1], planesC[0]);
	cv::normalize(planesC[0], planesC[0], 0, 1, CV_MINMAX);
	planesC[0].copyTo(C);
}

void multiply_fourier(cv::InputArray A, cv::InputArray B, cv::OutputArray C)
{
	cv::Mat paddedA, paddedB, spectrumC;
	cv::Mat matA, matB;
	matA = A.getMat(); matB = B.getMat();
	int width = A.cols() + B.cols()*2;
	int height = A.rows() + B.rows()*2;
	int opt_width = cv::getOptimalDFTSize(width);
	int opt_height = cv::getOptimalDFTSize(height);
	cv::Rect rect(0, 0, matA.cols+matB.cols, matA.rows+matB.rows);

	cv::copyMakeBorder(matA, paddedA, matB.rows, opt_height - matA.rows- matB.rows, matB.cols, opt_width - matA.cols - matB.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::copyMakeBorder(matB, paddedB, opt_height - matB.rows, 0, opt_width - matB.cols, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	//cv::copyMakeBorder(matB, paddedB, 0, opt_height - matB.rows, 0, opt_width - matB.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));


	// DFT
	cv::Mat planesA[] = { cv::Mat_<double>(paddedA), cv::Mat::zeros(paddedA.size(), CV_64F) };
	cv::Mat planesB[] = { cv::Mat_<double>(paddedB), cv::Mat::zeros(paddedB.size(), CV_64F) };
	cv::Mat planesC[] = { cv::Mat::zeros(paddedB.size(), CV_64F), cv::Mat::zeros(paddedB.size(), CV_64F) };

	cv::Mat com_A, com_B, com_C;
	cv::merge(planesA, 2, com_A);
	cv::merge(planesB, 2, com_B);
	cv::dft(com_A, com_A);
	cv::dft(com_B, com_B);

	cv::mulSpectrums(com_A, com_B, com_C, 0, false);

	cv::dft(com_C, com_C, cv::DFT_INVERSE);
	cv::split(com_C, planesC);
	cv::magnitude(planesC[0], planesC[1], planesC[0]);
	cv::normalize(planesC[0], planesC[0], 0, 1, CV_MINMAX);
	planesC[0](rect).copyTo(C);
}

void forwardbackwardtest(cv::InputArray A, cv::OutputArray B)
{
	cv::Mat matA, paddedA, comA;
	matA = A.getMat();
	int opt_width = cv::getOptimalDFTSize(matA.cols);
	int opt_height = cv::getOptimalDFTSize(matA.rows);
	cv::Rect rect(0, 0, matA.cols, matA.rows);
	cv::copyMakeBorder(matA, paddedA, 0, opt_height - matA.rows, 0, opt_width - matA.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planeA[] = { cv::Mat_<double>(paddedA), cv::Mat::zeros(paddedA.size(), CV_64F) };
	cv::merge(planeA, 2, comA);
	cv::dft(comA, comA);
	cv::idft(comA, comA);
	cv::split(comA, planeA);
	cv::magnitude(planeA[0], planeA[1], planeA[0]);
	cv::normalize(planeA[0], planeA[0], 0, 1, CV_MINMAX);
	planeA[0](rect).copyTo(B);
}

void divideFFTW(cv::InputArray A, cv::OutputArray B)
{

}

void cv2fftw(cv::InputArray A)
{

}

void shift_fre(cv::InputArray A)
{
	cv::Mat matA, magI;
	matA = A.getMat();
	magI = matA(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void RichardLucy(cv::InputArray _srcI,
	cv::InputArray _coreI,
	cv::OutputArray _dst,
	int iteration = 5)
{
	cv::Mat srcI = _srcI.getMat();
	cv::Mat coreI = _coreI.getMat();
	cv::Mat coreIpadded, estIpadded, corrIpadded, img_complex, core_complex, denom_complex, corr_complex, img_denom, img_est, img_corr;

	// initial the est_img for first iteration
	img_est = srcI.clone();

	// make padded
	int width = srcI.cols + coreI.cols * 2;
	int height = srcI.rows + coreI.rows * 2;
	int opt_width = cv::getOptimalDFTSize(width);
	int opt_height = cv::getOptimalDFTSize(height);    denom_complex = img_complex.clone();

	// ensure the main information location
	cv::Rect rect(coreI.cols / 2, coreI.rows / 2, srcI.cols, srcI.rows);
	cv::Rect rect2(coreI.cols, coreI.rows, srcI.cols, srcI.rows);

	// padded the image make the main information in the center.
	// constant part
	cv::copyMakeBorder(coreI, coreIpadded, 0, opt_height - coreI.rows, 0, opt_width - coreI.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planeCore[] = { cv::Mat_<double>(coreIpadded), cv::Mat::zeros(coreIpadded.size(), CV_64F) };
	cv::merge(planeCore, 2, core_complex);
	// execute DFT
	cv::dft(core_complex, core_complex);

	for (int i = 0; i < iteration; i++) {

		// make image information in the center
		// padded the image with zero round for protecting the image from the deconvolutino pullution.
		cv::copyMakeBorder(img_est, estIpadded, coreI.rows, opt_height - srcI.rows-coreI.rows, coreI.cols, opt_width - srcI.cols-coreI.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		// form the complex Mat
		cv::Mat planeI[] = { cv::Mat_<double>(estIpadded), cv::Mat::zeros(estIpadded.size(), CV_64F) };
		cv::merge(planeI, 2, img_complex);
		// execute the DFT
		cv::dft(img_complex, img_complex);

		// form the RL denom
		cv::mulSpectrums(img_complex, core_complex, denom_complex, cv::DFT_COMPLEX_OUTPUT);
		cv::idft(denom_complex, denom_complex, cv::DFT_COMPLEX_OUTPUT);

		cv::split(denom_complex, planeI);
		cv::magnitude(planeI[0](rect), planeI[1](rect), img_denom);
		cv::divide(srcI, img_denom, img_corr);

		cv::copyMakeBorder(img_corr, corrIpadded, coreI.rows / 2, opt_height - srcI.rows - coreI.rows / 2, coreI.cols / 2, opt_width - srcI.cols - coreI.cols / 2, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		cv::Mat planeCorr[] = { cv::Mat_<double>(corrIpadded), cv::Mat::zeros(corrIpadded.size(), CV_64F) };
		cv::merge(planeCorr, 2, corr_complex);
		cv::dft(corr_complex, corr_complex);
		cv::mulSpectrums(corr_complex, core_complex, corr_complex, cv::DFT_COMPLEX_OUTPUT);
		cv::idft(corr_complex, corr_complex, cv::DFT_COMPLEX_OUTPUT);
		cv::split(corr_complex, planeCorr);
		cv::magnitude(planeCorr[0](rect2), planeCorr[1](rect2), img_corr);
		// subsitatute estimated image with newer estimated image
		cv::multiply(img_est, img_corr, img_est);
	}

	img_est.copyTo(_dst);
}

void RichardLucy(cv::InputArray _srcI,
	cv::InputArray _coreI,
	cv::OutputArray _dst,
	std::vector<double> &mse_stat,
	int iteration = 5)
{
	cv::Mat srcI = _srcI.getMat();
	cv::Mat coreI = _coreI.getMat();
	cv::Mat coreIpadded, estIpadded, corrIpadded, img_complex, core_complex, denom_complex, corr_complex, img_denom, img_est, img_est_newer, img_corr;

	// initial the est_img for first iteration
	img_est = srcI.clone();

	// make padded
	int width = srcI.cols + coreI.cols * 2;
	int height = srcI.rows + coreI.rows * 2;
	int opt_width = cv::getOptimalDFTSize(width);
	int opt_height = cv::getOptimalDFTSize(height);    denom_complex = img_complex.clone();

	// ensure the main information location
	cv::Rect rect(coreI.cols/2, coreI.rows/2, srcI.cols, srcI.rows);
	cv::Rect rect2(0, 0, srcI.cols, srcI.rows);

	// padded the image make the main information in the center.
	// constant part
	cv::copyMakeBorder(coreI, coreIpadded, opt_height - coreI.rows, 0, opt_width - coreI.cols, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planeCore[] = { cv::Mat_<double>(coreIpadded), cv::Mat::zeros(coreIpadded.size(), CV_64F) };
	cv::merge(planeCore, 2, core_complex);
	// execute DFT
	cv::dft(core_complex, core_complex);

	for (int i = 0; i < iteration; i++) {

		// make image information in the center
		cv::copyMakeBorder(img_est, estIpadded, coreI.rows, opt_height - srcI.rows-coreI.rows, coreI.cols, opt_width - srcI.cols-coreI.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
		// form the complex Mat
		cv::Mat planeI[] = { cv::Mat_<double>(estIpadded), cv::Mat::zeros(estIpadded.size(), CV_64F) };
		cv::merge(planeI, 2, img_complex);
		// execute the DFT
		cv::dft(img_complex, img_complex);

		// form the RL denom convolution
		cv::mulSpectrums(img_complex, core_complex, denom_complex, cv::DFT_COMPLEX_OUTPUT);
		cv::idft(denom_complex, denom_complex, cv::DFT_COMPLEX_OUTPUT);
		cv::split(denom_complex, planeI);
		cv::magnitude(planeI[0](rect), planeI[1](rect), img_denom);
		cv::divide(srcI, img_denom, img_corr);
		cv::copyMakeBorder(img_corr, corrIpadded, coreI.rows, opt_height - srcI.rows-coreI.rows, coreI.cols, opt_width - srcI.cols-coreI.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

		// deconvolution
		cv::Mat planeCorr[] = { cv::Mat_<double>(corrIpadded), cv::Mat::zeros(corrIpadded.size(), CV_64F) };
		cv::merge(planeCorr, 2, corr_complex);
		cv::dft(corr_complex, corr_complex);
		// TODO(peo):Here something wrong with the divation
		// Yes divition is wrong the multiply is true!
		cv::mulSpectrums(corr_complex.clone(), core_complex, corr_complex, 0);
		cv::idft(corr_complex, corr_complex, cv::DFT_COMPLEX_OUTPUT);
		cv::split(corr_complex, planeCorr);

		cv::magnitude(planeCorr[0](rect), planeCorr[1](rect), img_corr);
		// subsitatute estimated image with newer estimated image
		cv::multiply(img_est, img_corr, img_est_newer);

		// statistic the MSE during the RL processing
		std::cout << "iteration: " << i << "\tMSE: " << MSEest(img_est_newer, img_est) << std::endl;
		mse_stat.push_back(MSEest(img_est_newer, img_est));

		// update the img_est corresponding to f in the equation.
		img_est_newer.copyTo(img_est);
	}
	img_est.copyTo(_dst);
}

double MSEest(cv::InputArray _srcIA, cv::InputArray _srcIB)
{
	cv::Mat srcA = _srcIA.getMat();
	cv::Mat srcB = _srcIB.getMat();
	cv::Mat disI;
	cv::subtract(srcA, srcB, disI);
	cv::pow(disI, 2, disI);
	double mse = cv::sum(disI)[0];
	return mse;
}


void norm_show(cv::InputArray _srcI, const char *windowName)
{
	cv::Mat srcI = _srcI.getMat();
	cv::Mat tmp;
	cv::normalize(srcI, tmp, 0, 1, CV_MINMAX);
	cv::imshow(windowName, tmp);
}