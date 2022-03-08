#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#define VECTOR_SIZE 3
#define TRUE 1
#define FALSE 0

void print_vector(double* x, int size) {
	for (int i=0; i<size; i++)
		printf("%.4f ", x[i]);
	printf("\n");
}

void vector_elementwise_add(double* result, double* a, double* b, int size) {
	for (int i=0; i<size; i++)
		result[i] = a[i] + b[i];
}

void vector_elementwise_subtract(double* result, double* a, double* b, int size) {
	for (int i=0; i<size; i++)
		result[i] = a[i] - b[i];
}

void vector_elementwise_multiply(double* result, double* a, double* b, int size) {
	for (int i=0; i<size; i++)
		result[i] = a[i] * b[i];
}

void vector_elementwise_scale(double* result, double* a, double scale, int size) {
	for (int i=0; i<size; i++)
		result[i] = a[i] * scale;
}

void vector_normalize(double* result, double* x, int size) {
	double magnitude = 0;
	for (int i=0; i<size; i++)
		magnitude += x[i]*x[i];
	magnitude = sqrt(magnitude);

	for (int i=0; i<size; i++)
		result[i] = x[i] / magnitude;
}

double vector_dot_product(double* a, double* b, int size) {
	double sum = 0;
	for (int i=0; i<size; i++)
		sum += a[i] * b[i];
	return sum;
}

double min(double a, double b) {
	return a < b? a: b;
}

double max(double a, double b) {
	return a > b? a: b;
}

double clip(double a, double min, double max) {
	if (a < min)
		return min;
	else if (a > max)
		return max;
	return a;
}

double intersect_sphere(double* O, double* D, double* S, double R) {
	double a = vector_dot_product(D, D, VECTOR_SIZE);
	double OS[VECTOR_SIZE] = {};
	vector_elementwise_subtract(OS, O, S, VECTOR_SIZE);
	double b = 2 * vector_dot_product(D, OS, VECTOR_SIZE);
	double c = vector_dot_product(OS, OS, VECTOR_SIZE) - R * R;
	double disc = b * b - 4 * a * c;
	if (disc > 0) {
		double distSqrt = sqrt(disc);
		double q = b < 0? (-b - distSqrt) / 2.0: (-b + distSqrt) / 2.0;
		double t0 = q / a;
        double t1 = c / q;
		double t2 = min(t0, t1);
		double t3 = max(t0, t1);
		if (t3 >= 0) {
			return t2 < 0? t3: t2;
		}
	}
	return DBL_MAX;
}

int trace_ray(double* col, double* O, double* D, double* L, double* position, double* color, double* color_light, double radius, double ambient, double diffuse, double specular_c, double specular_k) {
	// Find first point of intersection with the scene.
	double t = intersect_sphere(O, D, position, radius);
	if (t == DBL_MAX)
		return FALSE;
	// Find the point of intersection on the object.
	double M[VECTOR_SIZE] = {};
	vector_elementwise_scale(M, D, t, VECTOR_SIZE);
	vector_elementwise_add(M, M, O, VECTOR_SIZE);
	double N[VECTOR_SIZE] = {};
	vector_elementwise_subtract(N, M, position, VECTOR_SIZE);
	vector_normalize(N, N, VECTOR_SIZE);
	double toL[VECTOR_SIZE] = {};
	vector_elementwise_subtract(toL, L, M, VECTOR_SIZE);
	vector_normalize(toL, toL, VECTOR_SIZE);
	double toO[VECTOR_SIZE] = {};
	vector_elementwise_subtract(toO, O, M, VECTOR_SIZE);
	vector_normalize(toO, toO, VECTOR_SIZE);
	// Ambient light.
	for (int i=0; i<VECTOR_SIZE; i++)
		col[i] = ambient;
	// Lambert shading (diffuse).
	double temp[VECTOR_SIZE] = {};
	double temp2[VECTOR_SIZE] = {};
	vector_elementwise_scale(temp, color, diffuse * max(vector_dot_product(N, toL, VECTOR_SIZE), 0), VECTOR_SIZE);
	vector_elementwise_add(col, col, temp, VECTOR_SIZE);
	// Blinn-Phong shading (specular).
	vector_elementwise_add(temp, toL, toO, VECTOR_SIZE);
	vector_normalize(temp, temp, VECTOR_SIZE);
	vector_elementwise_scale(temp2, color_light, pow(max(vector_dot_product(N, temp, VECTOR_SIZE), 0), specular_k) * specular_c, VECTOR_SIZE);
	vector_elementwise_add(col, col, temp2, VECTOR_SIZE);
	return TRUE;
}

void run(double* img, int w, int h, double* O, double* L, double* position, double* color, double* color_light, double radius, double ambient, double diffuse, double specular_c, double specular_k) {
	double x = -1;
	double y = -1;
	double x_inc = 2.0 / (w - 1);
	double y_inc = 2.0 / (h - 1);
	double Q[VECTOR_SIZE] = {0, 0, 0};
	double D[VECTOR_SIZE] = {};
	double trace_ray_result[VECTOR_SIZE] = {};
	int is_intersect;
	for (int i=0; i<w; i++) {
		y = -1;
		for (int j=0; j<h; j++) {
			// Position of the pixel.
			Q[0] = x;
			Q[1] = y;
			// Direction of the ray going through
			// the optical center.
			vector_elementwise_subtract(D, Q, O, VECTOR_SIZE);
			vector_normalize(D, D, VECTOR_SIZE);
			// Launch the ray and get the color
			// of the pixel.
			is_intersect = trace_ray(trace_ray_result, O, D, L, position, color, color_light, radius, ambient, diffuse, specular_c, specular_k);
			if (is_intersect == TRUE) {
				for (int k=0; k<VECTOR_SIZE; k++) {
					img[(h - j - 1) * w * VECTOR_SIZE + i * VECTOR_SIZE + k] = clip(trace_ray_result[k], 0, 1);
				}
			}
			y += y_inc;
		}
		x += x_inc;
	}
}
