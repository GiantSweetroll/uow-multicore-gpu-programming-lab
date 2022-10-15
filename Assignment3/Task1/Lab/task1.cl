__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
      CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void flip_horizontal(
	read_only image2d_t src_image,
	write_only image2d_t dst_image
) {
	// get pixel coordinate
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	// calculate coordinate offset from mirror
	float mx = get_global_size(0)/2.0f;
	float xOff = coord.x - mx;

	// calculate new position after mirrored
	int newPosX = mx - xOff;
	int2 newPos = (int2)(newPosX, coord.y);

	// read pixel value
	float4 pixel = read_imagef(src_image, sampler, newPos);

	// write new pixel value to output
	write_imagef(dst_image, coord, pixel);
}

__kernel void flip_vertical(
	read_only image2d_t src_image,
	write_only image2d_t dst_image
) {
	// get pixel coordinate
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	// calculate coordinate offset from mirror
	float my = get_global_size(1) / 2.0f;
	float yOff = coord.y - my;

	// calculate new position after mirrored
	int newPosY = my - yOff;
	int2 newPos = (int2)(coord.x, newPosY);

	// read pixel value
	float4 pixel = read_imagef(src_image, sampler, newPos);

	// write new pixel value to output
	write_imagef(dst_image, coord, pixel);
}

__kernel void task1(
	read_only image2d_t src_image,
	write_only image2d_t dst_image_horz,
	write_only image2d_t dst_image_vert,
	write_only image2d_t dst_image_both
) {
	// get pixel coordinate
	int2 coord = (int2) (get_global_id(0), get_global_id(1));

	// calculate coordinate offset from mirror
	float mx = get_global_size(0) / 2.0f;
	float my = get_global_size(1) / 2.0f;
	float xOff = coord.x - mx;
	float yOff = coord.y - my;

	// calculate new position after mirrored
	int newPosX = mx - xOff;
	int newPosY = my - yOff;
	int2 newPosHorz = (int2) (newPosX, coord.y);		// new pos for horizontal flip
	int2 newPosVert = (int2) (coord.x, newPosY);		// new pos for vertical flip
	int2 newPosBoth = (int2) (newPosX, newPosY);		// new pos for both flips

	// read pixel value
	float4 pixelHorz = read_imagef(src_image, sampler, newPosHorz);
	float4 pixelVert = read_imagef(src_image, sampler, newPosVert);
	float4 pixelBoth = read_imagef(src_image, sampler, newPosBoth);

	// write new pixel value to output
	write_imagef(dst_image_horz, coord, pixelHorz);		// Write to dst_image for horizontal flip
	write_imagef(dst_image_vert, coord, pixelVert);		// Write to dst_image for vertical flip
	write_imagef(dst_image_both, coord, pixelBoth);		// Write to dst_image for both flips
}