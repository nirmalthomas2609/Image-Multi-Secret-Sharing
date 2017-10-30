from Polynomial import Polynomial
import numpy as np

def get_inverse(c, field_value=257):
	inv  = None
	for i in range(1, field_value):
		if (c * i) % field_value == 1:
			inv = i
			break
	if inv is None:
		raise ValueError("The value of inv is None")
	return inv

def get_Lagrange_Polynomials(e, field_value=257):
	k = len(e)
	Lagrange_polynomials = []
	for i in range(k):
		temp_prod = 1
		temp_poly = Polynomial([1])
		for j in range(k):
			if (i != j):
				temp_poly = temp_poly.multiply(Polynomial([1, (field_value - e[j]) % field_value]))
				temp_val = ((e[i] - e[j] + field_value) % field_value)
				inverse_temp_val = get_inverse(temp_val)
				temp_poly = temp_poly.multiply(Polynomial([inverse_temp_val]))
		Lagrange_polynomials += [temp_poly]
	return Lagrange_polynomials

def get_prod_funs(e, field_value=257):
	k = len(e)
	temp_poly = Polynomial([1])
	for i in range(k):
		temp_poly = temp_poly.multiply(Polynomial([1, (field_value-e[i])%field_value]))
	return temp_poly

def reconstruct_polynomial(alpha, poly_alpha, field_value = 257):
	temp_poly = Polynomial([0])
	t = len(alpha)
	for i in range(t):
		poly = Polynomial([1])
		temp_prod = 1
		alpha_i = alpha[i]
		for j in range(t):
			if (i != j):
				poly = poly.multiply(Polynomial([1, (field_value - alpha[j])%field_value]))
				temp_prod = temp_prod * ((alpha_i - alpha[j] + field_value) % field_value)
				temp_prod %= field_value
		poly = poly.divide_by_constant(temp_prod)
		poly = poly.multiply(Polynomial([poly_alpha[i]]))
		temp_poly = temp_poly.add(poly)
	return temp_poly

def get_random_t_images(shadow_images, alpha, t):
	temp_list = [i for i in range(len(alpha))]
	shadows = []
	invalid_pos = []
	final_alpha =[]
	o = []
	for i in range(t):
		choose = np.random.choice(temp_list)
		o += [choose]
		temp_list.remove(choose)
		temp_img, temp_invalid_pos = shadow_images[choose]
		shape_img = temp_img.shape[0]
		temp_img = temp_img.reshape(shape_img * shape_img)
		ip = []
		for j in temp_invalid_pos:
			x_pos, y_pos = j
			ip += [(x_pos * shape_img) + y_pos]
		invalid_pos += [ip]
		shadows += [temp_img]
		final_alpha += [alpha[choose]]
	return shadows, invalid_pos, final_alpha

def get_content_from_file(filename):
	f = open(filename, 'r')
	content_ = f.readlines()
	f.close()
	content = []
	for i in content_:
		i = i[:len(i) - 1]
		pixel_value, num_pos = list(map(int, i.split(':')))
		content += [(pixel_value, num_pos)]
	return content

def get_original_image_back(new_img_info, size_original_img):
	original_img = [0] * (size_original_img**2)
	for i in new_img_info.keys():
		for j in new_img_info[i]:
			x_pos, y_pos = j
			if (x_pos >= 256 or y_pos >= 256):
				raise ValueError("Required Quantity of Shadow Images Not Available")
			original_img[int(x_pos*size_original_img+y_pos)] = i
	return original_img

def debug_reconstructed_polynomial(poly, alpha, fun_values):
	for i in range(len(alpha)):
		if poly.eval(alpha[i]) != fun_values[i]:
			raise ValueError("Recreated Polynomial is wrong")