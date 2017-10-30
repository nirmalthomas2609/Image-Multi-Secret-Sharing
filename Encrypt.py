import matplotlib.pyplot as plt
import numpy as np
import cv2
from Polynomial import Polynomial
from util_functions import *
from math import ceil, floor, sqrt

class Image_Encryption(object):

	def __init__(self, n, t, k, img, plot_histogram=True, show_image=True, self_debug=True):
#		img = cv2.imread(img_destination)
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		print ("Shape of the image is {}".format(img.shape))
		if show_image == True:
			plt.imshow(img, cmap='gray')
			plt.xlabel('INPUT IMAGE')
			plt.show()
		cv2.imwrite("Original_Image/Image.jpg", img)

		#Initialising the Class variables
		self.img_info = {}
		if (n <= t or t < k):
			raise ValueError("Does not satisfy parameter constraints")
			quit()
		self.n, self.t, self.k = n, t, k
		self.debug = self_debug
		
		#Initialising alpha
		self.alpha = []
		temp11 = [i for i in range(257)]
		for i in range(n):
			temp_variable = np.random.choice(temp11)
			temp11.remove(temp_variable)
			self.alpha += [temp_variable]
		if self_debug == True:
			print ("Alpha initialised to --> ",self.alpha) 
		
		#Initialising e
		temp11 = [i for i in range(257)]
		self.e = []
		for i in range (k):
			temp_variable = np.random.choice(temp11)
			temp11.remove(temp_variable)
			self.e += [temp_variable]
		if self_debug == True:
			print ("e initialised to --> ",self.e) 
		
		#Initialising q(x)
		temp11 = [i for i in range(257)]
		self.poly_q = []
		for i in range(t-k):
			temp_variable = np.random.choice(temp11)
			temp11.remove(temp_variable)
			self.poly_q += [temp_variable]
		self.poly_q = [np.random.randint(1, 257)] + self.poly_q
		if self_debug == True:
			print ("q(x) initialised to --> ",self.poly_q) 
		
		if plot_histogram == True:
			s=[]
		
		for x in range(img.shape[0]):
			for y in range(img.shape[1]):
				if img[x, y] in self.img_info:
					self.img_info[img[x, y]][0] += 1 
					if (self.img_info[img[x, y]][0] - 1) % k == 0:
						self.img_info[img[x, y]][1] += [[(x, y)]]
					else:
						self.img_info[img[x, y]][1][(self.img_info[img[x, y]][0] - 1) // k] += [(x, y)]
				else:
					self.img_info[img[x, y]] = [1, [[(x,y)]]]
				if plot_histogram == True:
					s += [img[x, y]]

		if self_debug == True:			
		#Debugging generation of self.img_info
			for i in self.img_info.keys():
				list_of_pos = self.img_info[i][1]
				for j in range(len(list_of_pos)):
					if (len(list_of_pos[j]) != k):
						if (len(list_of_pos[j]) > k):
							raise ValueError("Error cutting the bins of the histogram")
						elif (len(list_of_pos[j]) < k):
							if (j != len(list_of_pos) - 1):
								raise ValueError("Error cutting the bars of the histogram")
					for p in list_of_pos[j]:
						x_pos_var, y_pos_var = p
						if img[x_pos_var, y_pos_var] != i:
							raise ValueError("Incorrect encoding of image into image info")
			print ("Perfectly encoded the image information into self.img_info")



		info = open("info.txt", 'w')
		img_info_file = open("img_info.txt", 'w')

		for i in self.img_info.keys():
			img_info_file.write("{}:{}\n".format(i, self.img_info[i][0]))
			for j in range(k - len(self.img_info[i][1][-1])):
				self.img_info[i][1][-1] += [(np.random.randint(0, 256), np.random.randint(0, 256))]
			if self_debug == True:
				#Debugging padding self.img_info
				if (len(self.img_info[i][1][-1]) != k):
					raise ValueError("Incorrect padding of the bars of self.img_info")
			info.write("Pixel_value -> {}, number of positions -> {}, locations by bars -> {}\n".format(i, self.img_info[i][0], self.img_info[i][1]))
		info.close()
		img_info_file.close()
		if self_debug == True:
			print ("Perfectly padded self.img_info")

		if plot_histogram == True:
			s = np.array(s)
			plt.hist(s, bins = 255)
			plt.xlabel('Pixel Values')
			plt.ylabel('Number of occurrences')
			plt.show()

	def generate_shadow_images(self, store_shadows = True):
		print ("Encryption begins!")
		img_info = self.img_info
		n, t, k = self.n, self.t, self.k

		alpha = self.alpha
		e = self.e
		poly_q = self.poly_q

		shadow_image_size = None
		shadow_images = []

		for i in range(n):
			print ("Shadow Image no :- {}".format(i))
			x_secrets = []
			y_secrets = []
			lagrange = get_Lagrange_Polynomials(e)
			prod_fun = get_prod_funs(e)

			temp_poly = Polynomial(poly_q)
			temp_poly = temp_poly.multiply(prod_fun)


			if self.debug == True:
				#For debugging whether the first term of the generated polynomial is correct
				for mmm in range(len(e)):
					value = temp_poly.eval(e[mmm])
					if value != 0:
						raise ValueError("First term of the encrypting polynomial is wrong")
						quit()

				#For debugging whether the generated Lagrange Polynomial is correct
				for nnn in range(len(e)):
					for j in range(len(lagrange)):
						if nnn != j:
							if (lagrange[j].eval(e[nnn]) != 0):
								raise ValueError("Lagrange Polynmial is wrong")
						else:
							if (lagrange[j].eval(e[nnn]) != 1):
								raise ValueError("Lagrange Polynomial is wrong")

			for x in img_info.keys():
				for y in range(len(img_info[x][1])):
					bar = img_info[x][1][y]
					sum_polyx = Polynomial([0])
					sum_polyy = Polynomial([0])
					for u in range(k):
						s_x = img_info[x][1][y][u][0]
						s_y = img_info[x][1][y][u][1]
						tempx = (lagrange[u]).multiply(Polynomial([s_x]))
						tempy = (lagrange[u]).multiply(Polynomial([s_y]))
						sum_polyx = sum_polyx.add(tempx)
						sum_polyy = sum_polyy.add(tempy)
					x_poly = temp_poly.add(sum_polyx)
					y_poly = temp_poly.add(sum_polyy)
					x_secrets += [x_poly.eval(alpha[i])]
					y_secrets += [y_poly.eval(alpha[i])]
			temp_shadow = x_secrets + y_secrets
			if (shadow_image_size is None):
				temp_size = len(temp_shadow)
				while(1):
					if (floor(sqrt(temp_size)) == ceil(sqrt(temp_size))):
						break
					else:
						temp_size += 1
				shadow_image_size = temp_size
			temp_shadow = temp_shadow + [np.random.randint(0, 256) for u in range(len(x_secrets + y_secrets), shadow_image_size)]
			temp_shadow = np.array(temp_shadow).astype(int)
			temp_shadow = temp_shadow.reshape((int(sqrt(shadow_image_size)), int(sqrt(shadow_image_size))))
			invalid_positions = []
			for x in range(temp_shadow.shape[0]):
				for y in range(temp_shadow.shape[1]):
					if temp_shadow[x, y] == 256:
						invalid_positions += [(x, y)]
						temp_shadow[x, y] = 255
			if store_shadows == True:
				cv2.imwrite("Shadows/{}.jpg".format(i), temp_shadow)
			shadow_images += [(temp_shadow, invalid_positions)]
		print ("Encryption ends!")

		return shadow_images

	def decrypt_image(self, shadow_images, num_shadow_images_available, show_decrypted_image=True):
		print ("Decryption begins!")
		n, k, alpha, e, img_info = self.n, self.k, self.alpha, self.e, self.img_info
		t = num_shadow_images_available

		content_ = get_content_from_file("img_info.txt")

		new_img_info = {}
		total_num_bars = 0

		#To get the actual number of information encrypted pixels in the original image
		for i in range(len(content_)):
			if (content_[i][1] % k == 0):
				total_num_bars += (content_[i][1] // k) 
			else:
				total_num_bars += (content_[i][1] // k) + 1

		#Obtain required number of shadow images
		shadows, final_invalid_pos, final_alpha = get_random_t_images(shadow_images, alpha, t)
		
		count_x = 0
		count_y = total_num_bars
		file_open = open("reconstructed_img_info.txt", 'w')
		for i in range(len(content_)):
			new_img_info[content_[i][0]] = []
			for j in range(0, content_[i][1], k):
				alpha_poly_x = []
				alpha_poly_y = []
				for p in range(t):
					poly_alpha_x = shadows[p][count_x]
					poly_alpha_y = shadows[p][count_y]
					if (count_x in final_invalid_pos[p]):
						poly_alpha_x = 256
					if (count_y in final_invalid_pos[p]):
						poly_alpha_y = 256
					alpha_poly_x += [poly_alpha_x]
					alpha_poly_y += [poly_alpha_y]
				x_reconstructed_polynomial = reconstruct_polynomial(final_alpha, alpha_poly_x)
				y_reconstructed_polynomial = reconstruct_polynomial(final_alpha, alpha_poly_y)
				if self.debug == True:
					if (count_x == 0):
						debug_reconstructed_polynomial(x_reconstructed_polynomial, final_alpha, alpha_poly_x)
						print ("Debugged Reconstructed x polynomial")
						debug_reconstructed_polynomial(y_reconstructed_polynomial, final_alpha, alpha_poly_y)
						print ("Debugged Reconstructed y polynomial")
				iterator = k if (content_[i][1] - j >= k) else (content_[i][1] - j)
				for p in range(iterator):
					new_img_info[content_[i][0]] += [(x_reconstructed_polynomial.eval(e[p]), y_reconstructed_polynomial.eval(e[p]))]
				count_x += 1
				count_y += 1

		for i in new_img_info.keys():
			file_open.write("Pixel value --> {} at positions --> {}\n".format(i, new_img_info[i]))
		file_open.close()

		original_image = np.array(get_original_image_back(new_img_info, 256)).reshape((256, 256))
		print ("Decryption ends!")

		cv2.imwrite("Decrypted_Image/decrypted.jpg", original_image)

		if(show_decrypted_image == True):
			plt.imshow(original_image, cmap='gray')
			plt.xlabel("DECRYPTED IMAGE")
			plt.show()

		return original_image

