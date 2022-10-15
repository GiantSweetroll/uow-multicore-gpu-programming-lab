__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void task2(
	read_only image2d_t src_image,
	write_only image2d_t dst_image
) {
	// get pixel coordinate
	int2 coord = (int2) (get_global_id(0), get_global_id(1));

	// read pixel value
	float4 pixel = read_imagef(src_image, sampler, coord);

	// calculate luminance
	float lum = 0.299 * pixel.x + 0.587 * pixel.y + 0.114 * pixel.z;

	// replace RGB values with luminance
	pixel.xyz = (float3)(lum, lum, lum);

	// write new pixel value to output
	write_imagef(dst_image, coord, pixel);
}