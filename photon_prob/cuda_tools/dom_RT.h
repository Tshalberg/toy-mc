#ifndef DOMH
#define DOMH

#include "vec3.h"
#include "ray.h"

class dom {
	public:
	__host__ __device__ dom() {}
	__host__ __device__ dom(vec3 cen, float r) : center(cen), radius(r)  {};
	__host__ __device__ virtual bool hit(const ray& r, float t_min, float t_max, vec3 *pHit, float *tHit) const;
	// __host__ __device__ inline int get_hits() const { return hits; }

	vec3 center;
	float radius;
	// int hits;
};


__device__ 
bool dom::hit(const ray& r, float t_min, float t_max, vec3 *pHit, float *tHit) const  {
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = dot(oc, r.direction());
	float c = dot(oc, oc) - radius*radius;
	float discriminant = b*b - a*c;
	if (discriminant > 0) {
		float temp = (-b - sqrt(b*b - a*c))/a;
		if (temp < t_max && temp > t_min) {
			*pHit = r.point_at_parameter(temp);
			*tHit = temp;
			return true;
		}
		temp = (-b + sqrt(b*b - a*c))/a;
		if (temp < t_max && temp > t_min) {
			*pHit = r.point_at_parameter(temp);
			*tHit = temp;
			return true;
		}
	}
	return false;
}


#endif