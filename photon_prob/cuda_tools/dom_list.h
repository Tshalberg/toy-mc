#ifndef DOMLISTH
#define DOMLISTH

#include "dom.h"

class dom_list: public dom {
	public:
		__host__ __device__ dom_list() {}
		__host__ __device__ dom_list(dom **l, int n) {list = l; list_size = n; }
		__host__ __device__ virtual bool hit(const vec3 position) const;

		dom **list;
		int list_size;
};

__host__ __device__ bool dom_list::hit(vec3 position) const {
	bool hit_anything = false;
	for (int i = 0; i < list_size; i++) {
		if (list[i]->hit(position)) {
			list[i]->hits += 1;
			hit_anything = true;
		}
	}
	return hit_anything;
}




#endif