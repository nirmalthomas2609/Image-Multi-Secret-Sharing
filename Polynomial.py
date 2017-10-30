from operator import add
class Polynomial(object):
	field_value = 257
	def __init__(self, poly_coeff):
		self.poly = poly_coeff

	def divide_by_constant(self, c):
		'''
		#using numpy
		p1 = np.array(self.poly).astype(int)
		for i in range(self.field_value):
			if (((c * i) % self.field_value) == 1):
				break
		res = ((p1 * i) % self.field_value).astype(int)
		return res
		'''
		inv=None
		p1 = self.poly
		for j in range(self.field_value):
			if ((c * j) % self.field_value == 1):
				inv = j
				break
		p1 = (Polynomial(p1)).multiply(Polynomial([inv]))
		return p1


	def add(self, poly2):
		'''
		#using numpy
		p1 = np.array(self.poly).astype(int)
		p2 = np.array(poly2.poly).astype(int)
		temp1 = np.zeros(abs(p1.shape[0] - p2.shape[0]))
		if (temp1.shape[0] != 0):
			if (p1.shape[0] > p2.shape[0]):
				p2 = np.hstack([temp1, p2]).astype(int)
			else:
				p1 = np.hstack([temp1, p1]).astype(int)
		if (self.field_value is not None):
			res = np.trim_zeros(((p1 + p2) % (self.field_value)), 'b')
		else:
			res = np.trim_zeros((p1 + p2), 'b')
		return res
		'''
		p1 = self.poly
		p2 = poly2.poly
		temp1 = [0] * (abs(len(p1) - len(p2)))
		if (len(temp1) != 0):
			if (len(p1) > len(p2)):
				p2 = temp1 + p2
			else:
				p1 = temp1 + p1
		res = list(map(add, p1, p2))
		for i in range(len(res)):
			res[i] %= self.field_value
		c = Polynomial(res)
		return c


	def multiply(self, poly2):
		'''
		#using numpy
		p1 = np.array(self.poly).astype(int)
		p2 = np.array(poly2.poly).astype(int)
		l = [0] * (p1.shape[0] + p2.shape[0] - 1)
		degree_res = len(l) - 1
		degree_p1 = p1.shape[0] - 1
		degree_p2 = p2.shape[0] - 1
		for i in range(p1.shape[0]):
			for j in range(p2.shape[0]):
				pos = degree_p1 + degree_p2 - i - j
				index_l = degree_res - pos
				l[index_l] += p1[i] * p2[j]
		if (self.field_value is not None):
			res = (np.array(l) % (self.field_value)).astype(int)
		else:
			res = np.array(l).astype(int)
		res = np.trim_zeros(res, 'b')
		return res
		'''
		p1 = self.poly
		p2 = poly2.poly
		l = [0] * (len(p1) + len(p2) - 1)
		degree_res = len(l) - 1
		degree_p1 = len(p1) - 1
		degree_p2 = len(p2) - 1
		for i in range(len(p1)):
			for j in range(len(p2)):
				pos = degree_p1 + degree_p2 - i - j
				index_l = degree_res - pos
				l[index_l] += p1[i] * p2[j]
		for i in range(len(l)):
			l[i] %= self.field_value
		res = Polynomial(l)
		return res
 
	def eval(self, x):
		p1 = self.poly
		s = 0
		length = len(p1)
		for i in range(length):
			uu = 1
			for j in range(length - i - 1):
				uu = ((uu * x) %(self.field_value))
			s += (((p1[i]) * (uu)) % self.field_value)
		return (s % self.field_value)