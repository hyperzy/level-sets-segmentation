import numpy as np
import cv2
import time
import sys
np.set_printoptions(threshold = np.inf)
##################### NOTE: I did not check boundary overflow ###################

zero_set = ()
datatype = 'int'
# narrow band width
band_width = 5
band_range = ()
differential_range = ()
img_shape = ()
dist_mask = np.array([])
cross_mask = (np.array([-1, 0, 1, 0], dtype='int'), np.array([0, 1, 0, -1], dtype='int'))
# lookup_mask_list = []
thresh_hold = 0.05


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
	# 	index = np.where(temp == 0)
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
	mask = np.zeros([band_width * 2 + 1, band_width * 2 + 1], dtype = datatype)
	for i in range(band_width * 2 + 1):
		for j in range(band_width * 2 + 1):
			mask[i][j] = np.round(np.sqrt(np.power(i - band_width, 2) + pow(j - band_width, 2)))
	mask[np.where(mask > band_width)] = band_width + 1
	return mask

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

	for w in range(band_width, 0, -1):
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
	return psi

'''
Initial narrow band
Second overload
'''
def InitNarrowBand1(band_width, zero_set):
	psi = (band_width + 1) * np.ones(img_shape)

	for w in range(band_width, 0, -1):
		# for i, j in zip(zero_set[0], zero_set[1]):
		index = np.where(dist_mask == w)
		for i, j in zip(index[0], index[1]):
			psi[(zero_set[0] - band_width + i, zero_set[1] - band_width + j)] = w
		# 	temp = psi[i - band_width : i + band_width + 1, j - band_width : j + band_width + 1]
		# 	index = np.where(temp >= dist_mask)
		# 	temp[index] = dist_mask[index]
			# print(psi[i - band_width : i + band_width + 1, j - band_width : j + band_width + 1])
	psi[zero_set] = 0
	global band_range
	band_range = np.where(psi <= band_width)
	global differential_range
	differential_range = np.where((psi < band_width) & (psi > -band_width))
	return psi


def InitSDF(unmasked_psi):
	sdf_mask = np.ones(unmasked_psi.shape, dtype = datatype)
	band_set = np.where(unmasked_psi <= band_width)
	# index of unmasked psi whose distance to front is band width
	max_dist_index = np.where(unmasked_psi == band_width)
	row_min = np.min(max_dist_index[0])
	row_max = np.max(max_dist_index[0])

	for i in range(row_min, row_max + 1):
		last_sign = 1
		sign_change = False
		inside_band = True
		inside_front = False
		# next grid will be outside the narrow band
		flag_out = False

		up = False
		down = False
		sub_col_min = np.min(max_dist_index[1][np.where(max_dist_index[0] == i)])
		sub_col_max = np.max(max_dist_index[1][np.where(max_dist_index[0] == i)])
		for j in range(sub_col_min, sub_col_max + 1):
			# if boundary point
			if unmasked_psi[i][j] == 0:
				sdf_mask[i][j] = 0
				# if not inside_front:
				# 	inside_front = True
				if np.count_nonzero(unmasked_psi[i - 1, j - 1: j + 2]) != 3:
					up = True
				if  np.count_nonzero(unmasked_psi[i + 1, j - 1: j + 2]) != 3:
					down = True
				# 	if unmasked_psi[i][j + 1] == 0:
				# 		continue
				# 	else:
				# 		inside_front = False
				# 		sign_change = not sign_change
				# else:
				# 	if unmasked_psi[i][j + 1] == 0:
				# 		continue
				# 	else:
				# 		inside_front = False
				if up and down:
					sign_change = not sign_change
					up = False
					down = False
			# not boundary point
			# elif unmasked_psi[i][j] == band_width:
			# 	if not inside_band:
			# 		inside_band = True
			# 	# since values outside the narrow band are maxsize
			# 	if inside_band and unmasked_psi[i][j + 1] > band_width:
			# 		inside_band = False
			#
			#
			# 	if not sign_change:
			# 		sdf_mask[i][j] = last_sign
			# 	else:
			# 		last_sign = -last_sign
			# 		sdf_mask[i][j] = last_sign
			else:
				# if inside_band:
				if not sign_change:
					sdf_mask[i][j] = last_sign
				else:
					last_sign = -last_sign
					sdf_mask[i][j] = last_sign
					sign_change = False
				# else:
				# 	continue
	return sdf_mask

'''
Zero level set initialization

'''
def InitLevelSet(unmasked_psi, sdf_mask):
	return unmasked_psi * sdf_mask


def ExtensionPhi(psi, phi, phi_gradient):
	ext_phi = np.zeros(img_shape)
	ext_phi_gradient = [np.zeros(img_shape), np.zeros(img_shape)]
	for i, j in zip(band_range[0], band_range[1]):
		# if it is boundary
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
					ext_phi[i][j] = phi[shift_i + ii][shift_j + jj]
					ext_phi_gradient[0][i][j] = phi_gradient[0][shift_i + ii][shift_j + jj]
					ext_phi_gradient[1][i][j] = phi_gradient[1][shift_i + ii][shift_j + jj]
					break;
	return ext_phi, ext_phi_gradient



def Evolve(old_psi, phi, phi_gradient):
	psi = old_psi.copy()
	psi_next = old_psi.copy()
	# psi_next = (band_width + 1) * np.ones(img_shape)
	begin = time.time()
	ext_phi, ext_phi_gradient = ExtensionPhi(old_psi, phi, phi_gradient)
	end = time.time()
	print ("extension cost", end - begin)
	hit_border = False
	# delta that satisfies CFL condition
	delta_t = 1 / (np.fabs(phi_gradient[1]).max() + np.fabs(phi_gradient[0]).max()) - 0.9
	iteration = 0
	while not hit_border:
		begin = time.time()
		for i, j in zip(differential_range[0], differential_range[1]):
	#!!!!!!!!! problem to be fiex: what if the next evolution exceeds the narrow band boundary
			# geomatric heat equation item
			ghe_item = 0
			psi_dx = (psi[i][j + 1] - psi[i][j - 1]) / 2
			psi_dy = (psi[i + 1][j] - psi[i - 1][j]) / 2
			if psi_dx != 0 and psi_dy != 0:
				psi_dxy = (psi[i + 1][j + 1] - psi[i - 1][j + 1] - psi[i + 1][j - 1] + psi[i - 1][j - 1]) / 4
				psi_dxx = psi[i][j + 1] + psi[i][j - 1] - 2 * psi[i][j]
				psi_dyy = psi[i + 1][j] + psi[i - 1][j] - 2 * psi[i][j]
				ghe_item = ext_phi[i][j] * (psi_dx * psi_dx * psi_dyy - 2 * psi_dx * psi_dy * psi_dxy + psi_dy * psi_dy * psi_dxx) / (psi_dx * psi_dx + psi_dy * psi_dy)

			# transport pde item
			tpde_item = 0
			ext_phi_dx = ext_phi_gradient[1][i][j]
			ext_phi_dy = ext_phi_gradient[0][i][j]
			# determin the sign of phi_dx
			if (ext_phi_dx > 0):
				psi_dx = psi[i][j + 1] - psi[i][j]
				tpde_item += ext_phi_dx * psi_dx
			elif ext_phi_dx < 0:
				psi_dx = psi[i][j] - psi[i][j - 1]
				tpde_item += ext_phi_dx * psi_dx
			# determin sign of phi_dy
			if ext_phi_dy > 0:
				psi_dy = psi[i + 1][j] - psi[i][j]
				tpde_item += ext_phi_dy * psi_dy
			elif ext_phi_dy < 0:
				psi_dy = psi[i][j] - psi[i - 1][j]
				tpde_item += ext_phi_dy * psi_dy

			delta = (ghe_item + tpde_item)
			psi_next[i][j] = (psi[i][j] + delta_t * delta)
			if (abs(old_psi[i][j]) == band_width - 1 and psi_next[i][j] * old_psi[i][j] <= 0):
				hit_border = True
		# test_show = np.zeros((100, 100), dtype = 'uint8')
		# for i in range(band_width + 1):
		# 	test_show[np.where((np.fabs(psi_next) >= i - 0.5) & (np.fabs(psi_next) < i + 0.5))] = 255 - i * 40
		# cv2.namedWindow("test", 0)
		# cv2.imshow("test", test_show)
		# cv2.waitKey(0)
		if hit_border:
			break
		psi_next, psi = psi, psi_next
		# iteration += 1
		end = time.time()
		print("Iteration cost", end - begin)


	probable_front_index = np.where((psi_next >= 0) & (psi_next < band_width))
	# for writing convinience
	a = probable_front_index
	index_of_a = np.where((psi_next[(a[0] + cross_mask[0][0], a[1] + cross_mask[1][0])] < 0) | (psi_next[(a[0] + cross_mask[0][1], a[1] + cross_mask[1][1])] < 0)\
			 | (psi_next[(a[0] + cross_mask[0][2], a[1] + cross_mask[1][2])] < 0) | (psi_next[(a[0] + cross_mask[0][3], a[1] + cross_mask[1][3])] < 0))
	global zero_set
	zero_set = (a[0][index_of_a], a[1][index_of_a])
	# psi_next = np.round(psi_next)
	# for i, j in zip(differential_range[0], differential_range[1]):
	# 	if psi_next[i][j] == 0:
	# 		zero_set_i.append(i)
	# 		zero_set_j.append(j)

	# global zero_set
	# zero_set = (np.array(zero_set_i), np.array(zero_set_j))
	return psi_next






########### initialize interface ###########
initial_img = Initial_img()
initial_img.in_img = cv2.imread("test.png")
gray_img = cv2.cvtColor(initial_img.in_img, cv2.COLOR_BGR2GRAY)
initial_img.window_name = "Input Image"
# cv2.namedWindow(initial_img.window_name, 0)
# cv2.imshow(initial_img.window_name, initial_img.in_img)
# cv2.setMouseCallback(initial_img.window_name, onMouse, initial_img)
# while True:
# 	if cv2.waitKey(0) == 13:
# 		break
# cv2.destroyAllWindows()
img_shape = gray_img.shape
# print(initial_img.up_left, initial_img.bot_right)

dist_mask = DistMaskMatrix(band_width)
# lookup_mask_list = LookupMaskList(band_width)

# initial_img.up_left = [391, 33]
# initial_img.bot_right = [897, 887]
initial_img.up_left = [14, 15]
initial_img.bot_right = [83, 81]
# \psi

begin = time.time()
unmasked_psi = InitNarrowBand(band_width, initial_img.up_left, initial_img.bot_right)
end = time.time()
print("cost", end - begin)
# unmasked_psi = InitNarrowBand1(band_width, zero_set)
iteration = 0

############# get phi ##############
img_gradient = np.gradient(gray_img)
phi = 1 / (1 + img_gradient[0] ** 2 + img_gradient[1] ** 2)
phi_gradient = np.gradient(phi)
# # phi_dx = the second ite
# print(np.max(phi_gradient[1]), np.max(phi_gradient[0]))

while (iteration < 500):
	# interface_start = initial_img.up_left
	# store signs info
	sdf_mask = InitSDF(unmasked_psi)

	masked_psi = InitLevelSet(unmasked_psi, sdf_mask)


	############# evolution #############
	# psi_next = Evolve(masked_psi, phi, phi_gradient)

	Evolve(masked_psi, phi, phi_gradient)
	unmasked_psi = InitNarrowBand1(band_width, zero_set)

	shape = gray_img.shape
	test_show = gray_img.copy()
	# # a = np.where(masked_psi != 0)
	# a = np.where((masked_psi > 0) & (masked_psi <=band_width) )
	# test_show[a] = 200
	# test_show[np.where(masked_psi == 0)] = 0
	# test_show[np.where(masked_psi > band_width)] = 0
	# test_show[np.where((masked_psi >= -band_width) & (masked_psi < 0))] = 100

	# for i in range(band_width + 1):
	test_show[np.where(np.fabs(masked_psi) == 0)] = 125
	cv2.namedWindow("test", 0)
	cv2.imshow("test", test_show)
	cv2.waitKey(1)

	# for i in range(band_width + 1):
	test_show[np.where(np.fabs(unmasked_psi) == 0)] = 125
	# test_show[np.where(psi_next == 0)] = 0
	# test_show[np.where((psi_next > 0) & (psi_next <= band_width))] = 200
	# test_show[np.where((psi_next < 0) & (psi_next >= -band_width))] = 100
	# test_show[np.where(sdf_mask < 0)] = 125
	cv2.namedWindow("test", 0)
	cv2.imshow("test", test_show)
	cv2.waitKey(1)

	iteration += 1
	print(iteration)

