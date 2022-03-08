#ifndef __C_RAYTRACING_H__
#define __C_RAYTRACING_H__
void print_vector(double* x, int size);
void vector_elementwise_add(double* result, double* a, double* b, int size);
void vector_elementwise_subtract(double* result, double* a, double* b, int size);
void vector_elementwise_multiply(double* result, double* a, double* b, int size);
void vector_elementwise_scale(double* result, double* a, double scale, int size);
void vector_normalize(double* result, double* x, int size);
double vector_dot_product(double* a, double* b, int size);
double min(double a, double b);
double max(double a, double b);
double clip(double a, double min, double max);
double intersect_sphere(double* O, double* D, double* S, double R);
int trace_ray(double* col, double* O, double* D, double* L, double* position, double* color, double* color_light, double radius, double ambient, double diffuse, double specular_c, double specular_k);
void run(double* img, int w, int h, double* O, double* L, double* position, double* color, double* color_light, double radius, double ambient, double diffuse, double specular_c, double specular_k);
#endif