#define NN_H
#include"nn.h"



int main(void) {

	srand(time(0));
	float td[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0
	};

	int stride = 3;
	int n = sizeof(td) / sizeof(td[0]) / stride;
	Mat ti = { n, 2, stride, td };
	Mat to = { n, 1, stride, td + 2 };

	int arch[] = {2,2,1};
	NN nn = nn_alloc(arch, ARRAY_LEN(arch));
	NN g = nn_alloc(arch, ARRAY_LEN(arch));
	nn_rand(nn, 0, 1);
	NN_PRINT(nn);
	/*for (int i = 0; i < n; i++) {
		mat_copy(NN_INPUT(nn), mat_row(ti, i));
		PRINT_MAT(NN_INPUT(nn));
		nn_forward(nn);
		PRINT_MAT(NN_OUTPUT(nn));
	}*/
	float rate = 1;
	int epoch = 20000;
	printf("cost: %f\n", nn_cost(nn, ti, to));
	for (int i = 0; i < epoch; i++) {
#if 0
		float eps = 1e-1;
		nn_fdiff(nn, g, eps, ti, to);
#else
		nn_backprop(nn, g, ti, to);
#endif
		//NN_PRINT(g);
		nn_learn(nn, g, rate);
		printf("cost: %f\n", nn_cost(nn, ti, to));
	}
	NN_PRINT(nn);

	printf("-----------------------------------\n");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			MAT_IDX(NN_INPUT(nn), 0, 0) = i;
			MAT_IDX(NN_INPUT(nn), 0, 1) = j;
			nn_forward(nn);
			float y = MAT_IDX(NN_OUTPUT(nn), 0, 0);
			printf("%d ^ %d = %f\n", i, j, y);
		}
	}

	return 0;
	/*
	float td[] = {
	0, 0, 0,
	0, 1, 1,
	1, 0, 1,
	1, 1, 0
	};

	int stride = 3;
	int n = sizeof(td) / sizeof(td[0]) / stride;
	Mat ti = { n, 2, stride, td };
	Mat to = { n, 1, stride, td + 2 };
	PRINT_MAT(ti);
	PRINT_MAT(to);
	srand(time(0));
	//input_data(x, td);
	Xor m = xor_alloc();
	Xor g = xor_alloc();
	

	mat_rand(m.w1, 0, 1);
	mat_rand(m.b1, 0, 1);
	mat_rand(m.w2, 0, 1);
	mat_rand(m.b2, 0, 1);

	PRINT_MAT(m.w1);
	PRINT_MAT(m.b1);
	PRINT_MAT(m.a1);
	PRINT_MAT(m.w2);
	PRINT_MAT(m.b2);
	PRINT_MAT(m.a2);

	int epoch = 1;
	// a1 = x * m.w1 + m.b1
	for (int i = 0; i < epoch; i++) {
		float eps = 1e-1;
		float rate = 1e-1;
		// printf("cost: %f\n", cost(m, ti, to));
		diff(m, g, eps, ti, to);
		learn(m, g, rate);
		printf("cost: %f\n", cost(m, ti, to));
	}
	printf("\n");
	PRINT_MAT(m.w1);
	PRINT_MAT(m.b1);
	PRINT_MAT(m.a1);
	PRINT_MAT(m.w2);
	PRINT_MAT(m.b2);
	PRINT_MAT(m.a2);
	
	printf("-----------------------------------\n");
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			MAT_IDX(m.a0, 0, 0) = i;
			MAT_IDX(m.a0, 0, 1) = j;
			forward(m);
			float y = *m.a2.es;
			printf("%d ^ %d = %f\n", i, j, y);
		}
	}
	return 0;*/
}