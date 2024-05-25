#pragma once
#ifdef NN_H
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<assert.h>

typedef struct {
	int rows; 
	int cols;
	int stride;
	float* es;
} Mat;

int* td;
void input_data(Mat x, float* ti) {
	x.es = ti;
}

void mat_copy(Mat dst, Mat src);
//float cost(Mat dout, Mat m);
float sigmoidf(float x);
void mat_sigmoid(Mat m);
Mat mat_malloc(int rows, int cols);
void mat_sum(Mat dst, Mat a);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_fill(Mat m, float n);
void mat_rand(Mat m, float st, float en);
Mat mat_row(Mat m, int row);
void mat_print(Mat m, const char name[], int padding);

typedef struct {
	int count;
	Mat* ws;
	Mat* bs;
	Mat* as;
} NN;

void nn_print(NN nn, const char* name);
NN nn_alloc(int* arch, int arch_count);
void nn_rand(NN nn, float low, float high);
void nn_forward(NN nn);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_backprop(NN nn, NN g, Mat ti, Mat to);
void nn_fdiff(NN nn, NN g, float eps, Mat ti, Mat to);
void nn_learn(NN nn, NN g, float rate);

#define MAT_IDX(m, i, j) m.es[(i)*(m).stride + (j)] 
#define PRINT_MAT(m) mat_print(m, #m, 0)

#define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs[0]))
#define NN_PRINT(m) nn_print(m,#m)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]


void nn_zero(NN nn) {
	for (int i = 0; i < nn.count; i++) {
		mat_fill(nn.bs[i], 0);
		mat_fill(nn.ws[i], 0);
		mat_fill(nn.as[i], 0);
	}
	mat_fill(nn.as[nn.count], 0);
}

NN nn_alloc(int* arch, int arch_count) {
	assert(arch_count > 0);
	NN nn;
	nn.count = arch_count - 1;

	nn.ws = (Mat*) calloc(nn.count, sizeof(*nn.ws));
	assert(nn.ws != NULL);
	nn.bs = (Mat*) calloc(nn.count, sizeof(*nn.bs));
	assert(nn.bs != NULL);
	nn.as = (Mat*) calloc(nn.count + 1, sizeof(*nn.as));
	assert(nn.as != NULL);

	nn.as[0] = mat_malloc(1, arch[0]);
	for (int i = 1; i < arch_count; i++) {
		nn.ws[i - 1] = mat_malloc(nn.as[i - 1].cols, arch[i]);
		nn.bs[i - 1] = mat_malloc(1, arch[i]);
		nn.as[i] = mat_malloc(1, arch[i]);
	}

	return nn;
}

void nn_rand(NN nn, float low, float high) {
	for (int i = 0; i < nn.count; i++) {
		mat_rand(nn.ws[i], low, high);
		mat_rand(nn.bs[i], low, high);
	}
}

void nn_forward(NN nn) {
	for (int i = 1; i <= nn.count; i++) {
		mat_dot(nn.as[i], nn.as[i - 1], nn.ws[i - 1]);
		mat_sum(nn.as[i], nn.bs[i - 1]);
		mat_sigmoid(nn.as[i]);
	}
}

float nn_cost(NN nn, Mat ti, Mat to) {
	assert(ti.rows == to.rows);
	assert(to.cols == NN_OUTPUT(nn).cols);
	int n = ti.rows;

	float c = 0;
	for (int i = 0; i < n; i++) {
		Mat x = mat_row(ti, i);
		Mat y = mat_row(to, i);

		mat_copy(NN_INPUT(nn), x);
		nn_forward(nn);
		for (int j = 0; j < y.cols; j++) {
			float d = MAT_IDX(NN_OUTPUT(nn), 0, j) - MAT_IDX(y, 0, j);
			c += d * d;
		}
	}
	return c / n;
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to) {
	assert(ti.rows == to.rows);
	assert(to.cols == NN_OUTPUT(nn).cols);
	int n = ti.rows;

	nn_zero(g);
	
	for (int i = 0; i < n; i++) {
		mat_copy(NN_INPUT(nn), mat_row(ti, i));
		nn_forward(nn);
		for (int j = 0; j <= nn.count; j++) {
			mat_fill(g.as[j], 0);
		}
		nn_zero(g);
		for (int j = 0; j < to.cols; j++) {
			MAT_IDX(NN_OUTPUT(g), 0, j) = MAT_IDX(NN_OUTPUT(nn), 0, j) - MAT_IDX(to, i, j);
		}
		// i - current sample, current layer
		// j - current activation
		// k - previous activation


		for (int l = nn.count; l > 0; l--) {
			for (int j = 0; j < nn.as[l].cols; j++) {
				// j - weight matrix col
				// k - weight matrix row
				float a = MAT_IDX(nn.as[l], 0, j);
				float da = MAT_IDX(g.as[l], 0, j);
				MAT_IDX(g.bs[l - 1], 0, j) += 2 * da * a * (1 - a);
				for (int k = 0; k < nn.as[l - 1].cols; k++) {
					float pa = MAT_IDX(nn.as[l - 1], 0, k);
					float w = MAT_IDX(nn.ws[l - 1], k, j);
					MAT_IDX(g.ws[l - 1], k, j) += 2 * da * a * (1 - a) * pa;
					MAT_IDX(g.as[l - 1], 0, k) += 2 * da * a * (1 - a) * w;
				}
			}
		}
	}

	for (int i = 0; i < g.count; i++) {
		for (int j = 0; j < g.ws[i].rows; j++) {
			for (int k = 0; k < g.ws[i].cols; k++) {
				MAT_IDX(g.ws[i], j, k) /= n;
			}
		}
	}

	for (int i = 0; i < g.count; i++) {
		for (int j = 0; j < g.bs[i].rows; j++) {
			for (int k = 0; k < g.bs[i].cols; k++) {
				MAT_IDX(g.bs[i], j, k) /= n;
			}
		}
	}

	for (int i = 0; i < g.count; i++) {
		for (int j = 0; j < g.as[i].rows; j++) {
			for (int k = 0; k < g.as[i].cols; k++) {
				MAT_IDX(g.as[i], j, k) /= n;
			}
		}
	}
}

void nn_fdiff(NN nn, NN g, float eps, Mat ti, Mat to) {
	float saved;
	for (int i = 0; i < nn.count; i++) {
		float c = nn_cost(nn, ti, to);
		for (int j = 0; j < nn.ws[i].rows; j++) {
			for (int l = 0; l < nn.ws[i].cols; l++) {
				saved = MAT_IDX(nn.ws[i], j, l);
				MAT_IDX(nn.ws[i], j, l) += eps;
				MAT_IDX(g.ws[i], j, l) = (nn_cost(nn, ti, to) - c) / eps;
				MAT_IDX(nn.ws[i], j, l) = saved;
			}
		}

		for (int j = 0; j < nn.bs[i].rows; j++) {
			for (int l = 0; l < nn.bs[i].cols; l++) {
				saved = MAT_IDX(nn.bs[i], j, l);
				MAT_IDX(nn.bs[i], j, l) += eps;
				MAT_IDX(g.bs[i], j, l) = (nn_cost(nn, ti, to) - c) / eps;
				MAT_IDX(nn.bs[i], j, l) = saved;
			}
		}
	}
}

void nn_learn(NN nn, NN g, float rate) {
	for (int i = 0; i < nn.count; i++) {
		for (int j = 0; j < nn.ws[i].rows; j++) {
			for (int k = 0; k < nn.ws[i].cols; k++) {
				MAT_IDX(nn.ws[i], j, k) -= rate * MAT_IDX(g.ws[i], j, k);
			}
		}

		for (int j = 0; j < nn.bs[i].rows; j++) {
			for (int k = 0; k < nn.bs[i].cols; k++) {
				MAT_IDX(nn.bs[i], j, k) -= rate * MAT_IDX(g.bs[i], j, k);
			}
		}
	}
}

void nn_print(NN nn, const char* name) {
	printf("%s = [\n", name);
	for (int i = 0; i < nn.count; i++) {
		mat_print(nn.ws[i], "ws", 4);
		mat_print(nn.bs[i], "bs", 4);
	}
	printf("]\n");
}

void mat_copy(Mat dst, Mat src)
{
	assert(dst.rows == src.rows);
	assert(dst.cols == src.cols);
	for (size_t i = 0; i < dst.rows; ++i) {
		for (size_t j = 0; j < dst.cols; ++j) {
			MAT_IDX(dst, i, j) = MAT_IDX(src, i, j);
		}
	}
}

float sigmoidf(float x) {
	return 1.f / (1.f + expf(-x));
}

void mat_sigmoid(Mat m) {
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++) {
			MAT_IDX(m, i, j) = sigmoidf(MAT_IDX(m, i, j));
		}
	}
}

Mat mat_malloc(int rows, int cols) {
	Mat m = {rows, cols, cols, (float*) calloc(cols * rows, sizeof(*m.es)) };
	return m;
}

void mat_sum(Mat dst, Mat a) {
	assert(dst.cols == a.cols);
	assert(dst.rows == a.rows);

	int rows = a.rows;
	int cols = a.cols;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			MAT_IDX(dst, i, j) = MAT_IDX(dst, i, j) + MAT_IDX(a, i, j);
		}
	}
}

void mat_dot(Mat dst, Mat a, Mat b) {
	assert(a.cols == b.rows);
	assert(dst.rows == a.rows);
	assert(dst.cols == b.cols);
	
	int n = a.cols;
	for (int i = 0; i < dst.rows; i++) {
		for (int j = 0; j < dst.cols; j++) {
			MAT_IDX(dst, i, j) = 0;
			for (int k = 0; k < n; k++) {
				MAT_IDX(dst, i, j) += MAT_IDX(a, i, k) * MAT_IDX(b, k, j);
			}
		}
	}
}

void mat_fill(Mat m, float n) {
	int rows = m.rows;
	int cols = m.cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			MAT_IDX(m, i, j) = n;
		}
	}
}

void mat_rand(Mat m, float st, float en) {
	int rows = m.rows;
	int cols = m.cols;

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			MAT_IDX(m, i, j) = st + (en - st) * ((float)rand() / (float)RAND_MAX);
		}
	}
}

Mat mat_row(Mat m, int row) {
	Mat r = { 1, m.cols, m.stride, &MAT_IDX(m, row, 0) };
	return r;
} 
 

void mat_print(Mat m, const char name[], int padding) {
	int rows = m.rows;
	int cols = m.cols;

	printf("%*s%s = [\n", (int) padding, "", name);
	for (int i = 0; i < rows; i++) {
		printf("%*s", (int)padding, "");
		for (int j = 0; j < cols; j++) {
			printf("%f ", MAT_IDX(m, i, j));
		}
		printf("\n");
	}
	printf("%*s]\n", (int) padding, "");
}

#endif
