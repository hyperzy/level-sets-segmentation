import numpy as np
import cv2
import time
import sys
np.set_printoptions(threshold = np.inf)
##################### NOTE: I did not check boundary overflow ###################

zero_set = ()
datatype = 'int'
# narrow band width
band_width = 10
band_range = ()
differential_range = ()
inner_range = ()
img_shape = ()
dist_mask = np.array([])
dist_list = np.array([])
cross_mask = (np.array([-1, 0, 1, 0], dtype='int'), np.array([0, 1, 0, -1], dtype='int'))
# lookup_mask_list = []
thresh_hold = 0.05
check_range = ()


class Initial_img:
	def __init__(self):
		self.window_name = ""
		self.up_left = [0, 0]
		self.bot_right = [0, 0]
		# self.center = ()
		# self.border = ()
		self.in_img = np.array([])

def onMouse(event, x, y, flags, param):
	dst = param.in_img.copy()

	if event == cv2.EVENT_LBUTTONDOWN:

		cur_point = (x, y)
		param.up_left = list(cur_point)
		cv2.circle(dst, cur_point, 2, (0, 255, 0), cv2.FILLED)
		cv2.imshow(param.window_name, dst)
	elif (event == cv2.EVENT_MOUSEMOVE) and (flags & cv2.EVENT_FLAG_LBUTTON):
		cur_point = (x, y)
		cv2.rectangle(dst, tuple(param.up_left), cur_point, (0, 255, 0), 2)
		cv2.imshow(param.window_name, dst)
	elif event == cv2.EVENT_LBUTTONUP:
		param.up_left[0], param.bot_right[0] = np.sort([param.up_left[0], x])
		param.up_left[1], param.bot_right[1] = np.sort([param.up_left[1], y])
	# if event == cv2.EVENT_LBUTTONDOWN:
	# 	param.center = (x, y)
	# if event == cv2.EVENT_LBUTTONUP:
	# 	param.border = (x, y)
	# 	dist = np.round(np.linalg.norm(np.array(param.center) - np.array(param.border)))
	# 	shape = (int(dist) * 2 + 1, int(dist) * 2 + 1)
	# 	temp = np.zeros(shape)
	# 	index_diff_boundary =  = np.where(temp == 0)
	# 	temp = np.round(np.sqrt((index[0] - dist)**2 + (index[1] - dist)**2)).reshape(shape)
	# 	index = np.where(temp == dist)
	# 	array1 = index[0]
	# 	array2 = index[1]
	# 	array1 += param.center[1] - int(dist)
	# 	array2 += param.center[0] - int(dist)
	# 	global zero_set
	# 	zero_set = (array1, array2)
	# 	dst[zero_set] = [255, 0, 0]
	# 	cv2.imshow(param.window_name, dst)
	# if event == cv2.EVENT_RBUTTONDOWN:
	# 	dst = param.in_img.copy()
	# 	cv2.imshow(param.window_name, dst)


'''
Initial distance small mask
'''
def DistMaskMatrix(band_width):
	mask = np.zeros([band_width * 2 + 1, band_width * 2 + 1])
	for i in range(band_width * 2 + 1):
		for j in range(band_width * 2 + 1):
			mask[i][j] = np.sqrt(pow(i - band_width, 2) + pow(j - band_width, 2))
	mask[np.where(mask > band_width)] = band_width + 1
	dist_list = np.unique(mask)[::-1]
	return mask, dist_list

# def LookupMaskList(band_width):
# 	mask_list = []
# 	for l in range(1, band_width + 1):
# 		mask = np.zeros((l * 2 + 1, l * 2 + 1), dtype = 'uint8')
# 		for i in range(band_width * 2 + 1):
# 			for j in range(band_width * 2 + 1):
# 				mask[i][j] = np.round(np.sqrt(np.power(i - band_width, 2) + pow(j - band_width, 2)))
# 		mask
# 		mask_list.append(mask)
# 	return mask_list

'''
Initial narrow band
first overload
@param rect_ul means the up left of rectangular
@param rect_br means the bottom right of rectangular
@pamra psi is unmasked
'''
def InitNarrowBand(band_width, rect_ul, rect_br):
	psi = (band_width + 1) * np.ones(img_shape)
	# height and width of rectangle
	height = rect_br[1] - rect_ul[1]
	width = rect_br[0] - rect_ul[0]
	# set interface to 0
	psi[rect_ul[1], rect_ul[0] + 1 : rect_br[0]] = 0
	psi[rect_br[1], rect_ul[0] + 1 : rect_br[0]] = 0
	psi[(rect_ul[1] + 1) : (rect_br[1]), rect_ul[0]] = 0
	psi[(rect_ul[1] + 1) : (rect_br[1]), rect_br[0]] = 0
	global zero_set
	zero_set = np.where(psi == 0)
	# print(rect_ul, rect_br)

	for w in dist_list:
		# ignore the first element
		if w == band_width + 1:
			continue
		# for i, j in zip(zero_set[0], zero_set[1]):
		index = np.where(dist_mask == w)
		for i, j in zip(index[0], index[1]):
			psi[(zero_set[0] - band_width + i, zero_set[1] - band_width + j)] = w
			# temp = psi[i - band_width : i + band_width + 1, j - band_width : j + band_width + 1]
			# index = np.where(temp >= dist_mask)
			# temp[index] = dist_mask[index]
			# print(psi[i - band_width : i + band_width + 1, j - band_width : j + band_width + 1])
	psi[zero_set] = 0
	global check_range
	check_range = np.where((psi <= band_width - 1) & (psi > 0))
	global inner_range
	psi[rect_ul[1] + 1: rect_br[1], rect_ul[0] + 1: rect_br[0]] *= -1
	inner_range = np.where(psi < 0)
	psi[rect_ul[1] + 1: rect_br[1], rect_ul[0] + 1: rect_br[0]] *= -1
	global band_range
	band_range = np.where(psi <= band_width)
	global differential_range
	differential_range = np.where(psi <= band_width - 1)
	return psi

'''
Initial narrow band
Second overload
'''
def InitNarrowBand1(band_width, zero_set):
	psi = (band_width + 1) * np.ones(img_shape)

	for w in dist_list:
		# ignore the first element
		if w == band_width + 1:
			continue
		# for i, j in zip(zero_set[0], zero_set[1]):
		index = np.where(dist_mask == w)
		for i, j in zip(index[0], index[1]):
			psi[(zero_set[0] - band_width + i, zero_set[1] - band_width + j)] = w
			# temp = psi[i - band_width : i + band_width + 1, j - band_width : j + band_width + 1]
			# index = np.where(temp >= dist_mask)
			# temp[index] = dist_mask[index]
			# print(psi[i - band_width : i + band_width + 1, j - band_width : j + band_width + 1])
	psi[zero_set] = 0
	global band_range
	band_range = np.where(psi <= band_width)
	global differential_range
	differential_range = np.where(psi < band_width)
	global check_range
	check_range = np.where((psi <= band_width - 1) & (psi > 0))
	return psi


def InitSDF(unmasked_psi):
	sdf_mask = np.ones(unmasked_psi.shape, dtype = datatype)
	# index of unmasked psi whose distance to front is band width
	max_dist_index = np.where(unmasked_psi > band_width - 1)
	row_min = np.min(max_dist_index[0])
	row_max = np.max(max_dist_index[0])
	sdf_mask[zero_set] = 0
	sdf_mask[inner_range] = -1
	# for i in range(row_min, row_max + 1):
	# 	last_sign = 1
	# 	sign_change = False
	# 	inside_band = True
	# 	inside_front = False
	# 	# next grid will be outside the narrow band
	# 	flag_out = False
	#
	# 	up = False
	# 	down = False
	# 	sub_col_min = np.min(max_dist_index[1][np.where(max_dist_index[0] == i)])
	# 	sub_col_max = np.max(max_dist_index[1][np.where(max_dist_index[0] == i)])
	# 	for j in range(sub_col_min, sub_col_max + 1):
	# 		# if boundary point
	# 		if unmasked_psi[i][j] == 0:
	# 			sdf_mask[i][j] = 0
	# 			# if not inside_front:
	# 			# 	inside_front = True
	# 			if np.count_nonzero(unmasked_psi[i - 1, j - 1: j + 2]) != 3:
	# 				up = True
	# 			if  np.count_nonzero(unmasked_psi[i + 1, j - 1: j + 2]) != 3:
	# 				down = True
	# 			# 	if unmasked_psi[i][j + 1] == 0:
	# 			# 		continue
	# 			# 	else:
	# 			# 		inside_front = False
	# 			# 		sign_change = not sign_change
	# 			# else:
	# 			# 	if unmasked_psi[i][j + 1] == 0:
	# 			# 		continue
	# 			# 	else:
	# 			# 		inside_front = False
	# 			if up and down:
	# 				sign_change = not sign_change
	# 				up = False
	# 				down = False
	# 		# not boundary point
	# 		# elif unmasked_psi[i][j] == band_width:
	# 		# 	if not inside_band:
	# 		# 		inside_band = True
	# 		# 	# since values outside the narrow band are maxsize
	# 		# 	if inside_band and unmasked_psi[i][j + 1] > band_width:
	# 		# 		inside_band = False
	# 		#
	# 		#
	# 		# 	if not sign_change:
	# 		# 		sdf_mask[i][j] = last_sign
	# 		# 	else:
	# 		# 		last_sign = -last_sign
	# 		# 		sdf_mask[i][j] = last_sign
	# 		else:
	# 			up = False
	# 			down =False
	# 			# if inside_band:
	# 			if not sign_change:
	# 				sdf_mask[i][j] = last_sign
	# 			else:
	# 				last_sign = -last_sign
	# 				sdf_mask[i][j] = last_sign
	# 				sign_change = False
	# 			# else:
	# 			# 	continue
	return sdf_mask

'''
Zero level set initialization

'''
def InitLevelSet(unmasked_psi, sdf_mask):
	return unmasked_psi * sdf_mask


def ExtensionPhi(psi, phi, phi_gradient):
	ext_phi = np.zeros(img_shape)
	ext_phi_gradient = [np.zeros(img_shape), np.zeros(img_shape)]
	for i, j in zip(differential_range[0], differential_range[1]):
		# if it is boundary
		max_dist = 0
		if psi[i][j] == 0:
			ext_phi[i][j] = phi[i][j]
		# if it is not boundary
		else:
			distance = np.fabs(psi[i][j])
			# find the close front
			index = np.where(dist_mask == distance)
			shift_i = int(i - band_width)
			shift_j = int(j - band_width)
			#index[0] -= band_width
			# np.where(psi[index] == 0)
			for ii, jj in zip(index[0], index[1]):
				if psi[shift_i + ii][shift_j + jj] == 0:
					# temp = np.sqrt(pow(band_width - ii, 2) + pow(band_width - jj, 2))
					# if (max_dist <= temp):
					# 	record_i, record_j = shift_i + ii, shift_j + jj
					# 	max_dist = temp
					ext_phi[i][j] = phi[shift_i + ii][shift_j + jj]
					ext_phi_gradient[0][i][j] = phi_gradient[0][shift_i + ii][shift_j + jj]
					ext_phi_gradient[1][i][j] = phi_gradient[1][shift_i + ii][shift_j + jj]
					break
	return ext_phi, ext_phi_gradient

def ExtensionPhi1(psi, phi, phi_gradient):
	new_psi = InitNarrowBand1(band_width, zero_set)

	ext_phi = np.zeros(img_shape)
	ext_phi_gradient = [np.zeros(img_shape), np.zeros(img_shape)]
	for i, j in zip(differential_range[0], differential_range[1]):
		# if it is boundary
		max_dist = 0
		if psi[i][j] == 0:
			ext_phi[i][j] = phi[i][j]
		# if it is not boundary
		else:
			distance = new_psi[i][j]
			# find the close front
			index = np.where(dist_mask == distance)
			shift_i = int(i - band_width)
			shift_j = int(j - band_width)
			#index[0] -= band_width
			# np.where(psi[index] == 0)
			for ii, jj in zip(index[0], index[1]):
				if new_psi[shift_i + ii][shift_j + jj] == 0:
					# temp = np.sqrt(pow(band_width - ii, 2) + pow(band_width - jj, 2))
					# if (max_dist <= temp):
					# 	record_i, record_j = shift_i + ii, shift_j + jj
					# 	max_dist = temp
					ext_phi[i][j] = phi[shift_i + ii][shift_j + jj]
					ext_phi_gradient[0][i][j] = phi_gradient[0][shift_i + ii][shift_j + jj]
					ext_phi_gradient[1][i][j] = phi_gradient[1][shift_i + ii][shift_j + jj]
					break
	return ext_phi, ext_phi_gradient



def Evolve(old_psi, phi, phi_gradient):
	psi = old_psi.copy()
	psi_next = old_psi.copy()
	# psi_next = (band_width + 1) * np.ones(img_shape)
	# begin = time.time()
	ext_phi, ext_phi_gradient = ExtensionPhi(old_psi, phi, phi_gradient)
	# end = time.time()
	# print("extension cost", end - begin)
	hit_border = False
	# delta that satisfies CFL condition
	delta_t = 1 / (np.fabs(phi_gradient[1]).max() + np.fabs(phi_gradient[0]).max()) - 0.9
	iteration = 0

	# check if the sign of differentiable boudary change
	index_diff_boundary = np.where((np.abs(old_psi) <= band_width - 1) & (np.abs(old_psi) > band_width -2))
	global zero_set
	while not hit_border:
		# begin = time.time()
		print("inside iteration", iteration)

		delta = np.zeros(img_shape)
		# rename variable for convenience
		index0 = differential_range[0]
		index1 = differential_range[1]

		psi_dx = (psi[(index0, index1 + 1)] - psi[(index0, index1 - 1)]) / 2
		psi_dy = (psi[(index0 + 1, index1)] - psi[(index0 - 1, index1)]) / 2
		psi_dxy = np.zeros(psi_dx.shape)
		psi_dxx = np.zeros(psi_dx.shape)
		psi_dyy = np.zeros(psi_dx.shape)
		ghe_item = np.zeros(psi_dx.shape)
		tpde_item = np.zeros(psi_dx.shape)
		# we only consider where psi_dx != 0 and psi_dy !=0
		inside_index_range = np.where((psi_dx != 0) & (psi_dy != 0))
		valid_index0 = index0[inside_index_range]
		valid_index1 = index1[inside_index_range]
		psi_dxy = (psi[(valid_index0 + 1, valid_index1 + 1)]\
					- psi[(valid_index0 - 1, valid_index1 + 1)]\
					- psi[(valid_index0 + 1, valid_index1 - 1)]\
					+ psi[(valid_index0 - 1, valid_index1 - 1)]) / 4
		psi_dxx = psi[(valid_index0, valid_index1 + 1)]\
					+ psi[(valid_index0, valid_index1 - 1)]\
					- 2 * psi[(valid_index0, valid_index1)]
		psi_dyy = psi[(valid_index0 + 1, valid_index1)]\
					+ psi[(valid_index0 - 1, valid_index1)]\
					- 2 * psi[(valid_index0, valid_index1)]
		valid_range = (valid_index0, valid_index1)
		ghe_item = ext_phi[valid_range] * (psi_dx[inside_index_range] ** 2 * psi_dyy\
					- 2 * psi_dx[inside_index_range] * psi_dy[inside_index_range] * psi_dxy\
					+ psi_dy[inside_index_range] ** 2 * psi_dxx)\
					/ (psi_dx[inside_index_range] ** 2 + psi_dy[inside_index_range] ** 2)

		delta[valid_range] += ghe_item

		ext_phi_dx = ext_phi_gradient[1]
		ext_phi_dy = ext_phi_gradient[0]
		# where forward diffrence of x and backward difference of y
		fx_by_index = np.where((ext_phi_dx > 0) & (ext_phi_dy < 0))
		x0_by_index = np.where((ext_phi_dy < 0) & (ext_phi_dx == 0))
		# where forward diffrence of x and forward difference of y
		fx_fy_index = np.where((ext_phi_dx > 0) & (ext_phi_dy > 0))
		x0_fy_index = np.where((ext_phi_dy > 0) & (ext_phi_dx == 0))
		# where backward diffrence of x and backward difference of y
		bx_by_index = np.where((ext_phi_dx < 0) & (ext_phi_dy < 0))
		bx_y0_index = np.where((ext_phi_dx < 0) & (ext_phi_dy == 0))
		# where backward diffrence of x and forward difference of y
		bx_fy_index = np.where((ext_phi_dx < 0) & (ext_phi_dy > 0))
		fx_y0_index = np.where((ext_phi_dx > 0) & (ext_phi_dy == 0))
		normal_force_fac = 1
		if fx_by_index[0].size != 0:
			delta[fx_by_index] += normal_force_fac * (
						ext_phi_dx[fx_by_index] * (psi[(fx_by_index[0], fx_by_index[1] + 1)] - psi[fx_by_index]) \
						+ ext_phi_dy[fx_by_index] * (psi[fx_by_index] - psi[(fx_by_index[0] - 1, fx_by_index[1])]))
		if fx_fy_index[0].size != 0:
			delta[fx_fy_index] += normal_force_fac * (
						ext_phi_dx[fx_fy_index] * (psi[(fx_fy_index[0], fx_fy_index[1] + 1)] - psi[fx_fy_index]) \
						+ ext_phi_dy[fx_fy_index] * (psi[(fx_fy_index[0] + 1, fx_fy_index[1])] - psi[fx_fy_index]))
		if bx_by_index[0].size != 0:
			delta[bx_by_index] += normal_force_fac * (
						ext_phi_dx[bx_by_index] * (psi[bx_by_index] - psi[(bx_by_index[0], bx_by_index[1] - 1)]) \
						+ ext_phi_dy[bx_by_index] * (psi[bx_by_index] - psi[(bx_by_index[0] - 1, bx_by_index[1])]))
		if bx_fy_index[0].size != 0:
			delta[bx_fy_index] += normal_force_fac * (
						ext_phi_dx[bx_fy_index] * (psi[bx_fy_index] - psi[(bx_fy_index[0], bx_fy_index[1] - 1)]) \
						+ ext_phi_dy[bx_fy_index] * (psi[(bx_fy_index[0] + 1, bx_fy_index[1])] - psi[bx_fy_index]))

		if x0_by_index[0].size != 0:
			delta[x0_by_index] += normal_force_fac * ext_phi_dy[x0_by_index] * (psi[x0_by_index] - psi[(x0_by_index[0] - 1, x0_by_index[1])])
		if x0_fy_index[0].size != 0:
			delta[x0_fy_index] += normal_force_fac * ext_phi_dy[x0_fy_index] * (psi[(x0_fy_index[0] + 1, x0_fy_index[1])] - psi[x0_fy_index])
		if bx_y0_index[0].size != 0:
			delta[bx_y0_index] += normal_force_fac * ext_phi_dx[bx_y0_index] * (psi[bx_y0_index] - psi[(bx_y0_index[0], bx_y0_index[1] - 1)])
		if fx_y0_index[0].size != 0:
			delta[fx_y0_index] += normal_force_fac * ext_phi_dx[fx_y0_index] * (psi[(fx_y0_index[0], fx_y0_index[1] + 1)] - psi[fx_y0_index])

		psi_next = psi + delta_t * delta

		if (old_psi[index_diff_boundary] * psi_next[index_diff_boundary] <= 0).any():
			hit_border = True
			break
		if (psi[check_range] * psi_next[check_range] <= 0).any():
			# break
			probable_front_index = np.where((psi_next >= 0) & (psi_next < band_width + 1))
			# for writing convinience
			a = probable_front_index
			index_of_a = np.where((psi_next[(a[0] + cross_mask[0][0], a[1] + cross_mask[1][0])] < 0) | \
						(psi_next[(a[0] + cross_mask[0][1], a[1] + cross_mask[1][1])] < 0) | (psi_next[(a[0] + cross_mask[0][2], a[1] + cross_mask[1][2])] < 0) | \
								  (psi_next[(a[0] + cross_mask[0][3], a[1] + cross_mask[1][3])] < 0))

			zero_set = (a[0][index_of_a], a[1][index_of_a])
			ext_phi, ext_phi_gradient = ExtensionPhi1(psi_next, phi, phi_gradient)

		if hit_border:
			break
		if iteration >= 500:
			break
		psi_next, psi = psi, psi_next
		iteration += 1
		# end = time.time()
		# print("Iteration cost", end - begin)

	probable_front_index = np.where((psi_next >= 0) & (psi_next < band_width + 1))
	# for writing convinience
	a = probable_front_index
	index_of_a = np.where((psi_next[(a[0] + cross_mask[0][0], a[1] + cross_mask[1][0])] < 0) | (psi_next[(a[0] + cross_mask[0][1], a[1] + cross_mask[1][1])] < 0) \
						  | (psi_next[(a[0] + cross_mask[0][2], a[1] + cross_mask[1][2])] < 0) | (psi_next[(a[0] + cross_mask[0][3], a[1] + cross_mask[1][3])] < 0))
	# global zero_set
	zero_set = (a[0][index_of_a], a[1][index_of_a])
	# psi_next = np.round(psi_next)
	# for i, j in zip(differential_range[0], differential_range[1]):
	# 	if psi_next[i][j] == 0:
	# 		zero_set_i.append(i)
	# 		zero_set_j.append(j)

	# global zero_set
	# zero_set = (np.array(zero_set_i), np.array(zero_set_j))
	global inner_range
	inner_range = np.where(psi_next < 0)
	return psi_next






########### initialize interface ###########
initial_img = Initial_img()
initial_img.in_img = cv2.imread("test.png")
gray_img = cv2.cvtColor(initial_img.in_img, cv2.COLOR_BGR2GRAY)
# gray_img = cv2.GaussianBlur(gray_img, (0, 0), 10)
initial_img.window_name = "Input Image"
cv2.namedWindow(initial_img.window_name, 0)
cv2.imshow(initial_img.window_name, initial_img.in_img)
cv2.setMouseCallback(initial_img.window_name, onMouse, initial_img)
while True:
	if cv2.waitKey(0) == 13:
		break
cv2.destroyAllWindows()
img_shape = gray_img.shape
# print(initial_img.up_left, initial_img.bot_right)

dist_mask, dist_list = DistMaskMatrix(band_width)
# lookup_mask_list = LookupMaskList(band_width)

# initial_img.up_left = [391, 33]
# initial_img.bot_right = [897, 887]
# initial_img.up_left = [14, 15]
# initial_img.bot_right = [83, 81]
# initial_img.up_left = [20, 12]
# initial_img.bot_right = [76, 82]

# \psi
begin = time.time()
unmasked_psi = InitNarrowBand(band_width, initial_img.up_left, initial_img.bot_right)
end = time.time()
print("cost", end - begin)
# unmasked_psi = InitNarrowBand1(band_width, zero_set)
iteration = 0

############# get phi ##############
def Gradient(img):
	shape = img.shape
	gradient_col = np.concatenate((np.diff(img),np.zeros((1, shape[0])).T), axis = 1)
	gradient_row = np.concatenate((np.zeros((1, shape[1])), np.diff(img, axis = 0)), axis = 0)
	return (gradient_row, gradient_col)
img_gradient = np.gradient(gray_img)
# img_gradient = Gradient(gray_img)
phi = 1 / (1 + (img_gradient[0] ** 2 + img_gradient[1] ** 2))
phi_gradient = np.gradient(phi)
# phi_gradient = Gradient(phi)
# # phi_dx = the second ite
# print(np.max(phi_gradient[1]), np.max(phi_gradient[0]))

test_show = np.zeros(img_shape + (3,), dtype='uint8')
test_show[:, :, 0] = gray_img.copy()
test_show[:, :, 1] = gray_img.copy()
test_show[:, :, 2] = gray_img.copy()
test_show[np.where(np.fabs(unmasked_psi) == 0)] = [0, 0, 255]
cv2.namedWindow("test", 0)
cv2.imshow("test", test_show)
cv2.waitKey(0)
while (iteration < 1000):
	print(iteration)
	# interface_start = initial_img.up_left
	# store signs info
	sdf_mask = InitSDF(unmasked_psi)

	masked_psi = InitLevelSet(unmasked_psi, sdf_mask)


	############# evolution #############
	# psi_next = Evolve(masked_psi, phi, phi_gradient)

	Evolve(masked_psi, phi, phi_gradient)
	unmasked_psi = InitNarrowBand1(band_width, zero_set)

	shape = gray_img.shape
	test_show = np.zeros(img_shape + (3,), dtype='uint8')
	test_show[:, :, 0] = gray_img.copy()
	test_show[:, :, 1] = gray_img.copy()
	test_show[:, :, 2] = gray_img.copy()
	# # a = np.where(masked_psi != 0)
	# a = np.where((masked_psi > 0) & (masked_psi <=band_width) )
	# test_show[a] = 200
	# test_show[np.where(masked_psi == 0)] = 0
	# test_show[np.where(masked_psi > band_width)] = 0
	# test_show[np.where((masked_psi >= -band_width) & (masked_psi < 0))] = 100

	# for i in range(band_width + 1):

	# for i in range(band_width + 1):
	test_show[np.where(np.fabs(unmasked_psi) == 0)] = [0, 0, 255]
	# test_show[np.where(psi_next == 0)] = 0
	# test_show[np.where((psi_next > 0) & (psi_next <= band_width))] = 200
	# test_show[np.where((psi_next < 0) & (psi_next >= -band_width))] = 100
	# test_show[np.where(sdf_mask < 0)] = 125
	cv2.namedWindow("test", 0)
	cv2.imshow("test", test_show)
	cv2.waitKey(1)

	iteration += 1

