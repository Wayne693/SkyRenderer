#include "PBR.cuh"
#include "thrust/extrema.h"

//#define PI = acos(-1);


float NDFGGXTR(Eigen::Vector3f n, Eigen::Vector3f h, float roughness)
{
	const float PI = acos(-1);
	float nDoth = n.dot(h);
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float nDoth2 = nDoth * nDoth;
	float factor = nDoth2 * (alpha2 - 1) + 1;
	return alpha2 / (PI * factor * factor);
}

float SchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness)
{
	float nDotv = n.dot(v);
	float r = (1 + roughness);
	float k = r * r / 8.0;

	return nDotv / (nDotv * (1 - k) + k);
}

float IBLSchlickGGX(Eigen::Vector3f n, Eigen::Vector3f v, float roughness)
{
	float nDotv = n.dot(v);
	float k = roughness * roughness / 2.0;

	return nDotv / (nDotv * (1 - k) + k);
}

float GeometrySmith(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f l, float roughness)
{
	float g1 = SchlickGGX(n, v, roughness);
	float g2 = SchlickGGX(n, l, roughness);

	return g1 * g2;
}

Eigen::Vector3f FresnelSchlick(Eigen::Vector3f h, Eigen::Vector3f v, Eigen::Vector3f F0)
{
	float hDotv = h.dot(v);

	return F0 + (Eigen::Vector3f(1.f, 1.f, 1.f) - F0) * pow(1 - hDotv, 5.f);
}

Eigen::Vector3f FresnelSchlickRoughness(Eigen::Vector3f n, Eigen::Vector3f v, Eigen::Vector3f F0, float roughness)
{
	float nDotv = thrust::max(0.f, n.dot(v));
	float r1 = thrust::max(1.f - roughness, F0.x());
	float r2 = thrust::max(1.f - roughness, F0.y());
	float r3 = thrust::max(1.f - roughness, F0.z());
	return F0 + (Eigen::Vector3f(r1, r2, r3) - F0) * pow(1 - nDotv, 5.f);
}