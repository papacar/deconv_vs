#include "cvutils.hpp"

void vec2mat(std::vector<std::vector<double> > M2D, const cv::OutputArray out_im) {
	double *ptr;
	out_im.create(M2D.size(), M2D.size(), CV_64F);
	for (int i = 0; i < M2D.size(); ++i) {
		ptr = out_im.getMat().ptr<double>(i);
		for (int j = 0; j < M2D[0].size(); ++j) {
			ptr[j] = M2D[i][j];
		}
	}
}
