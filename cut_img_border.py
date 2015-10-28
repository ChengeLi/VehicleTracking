def cut_borders():
	# image_listing = sorted(glob.glob('./tempFigs/roi2Result/*.jpg'))
	image_listing = sorted(glob.glob('./canalResult/*.jpg'))

	# cut out the borders
	for ii in range(len(image_listing)):
	img = cv2.imread(image_listing[ii])
	# filename = image_listing[ii][22:] # for roi2 result
	# filename = str('./tempFigs/roi2ResultFinal/'+ filename)
	# realimg = img[128:502, 343:480,:]

	filename = image_listing[ii][14:]
	filename = str('./canalResultFinal/'+ filename)

	realimg = img[126:488, 170:655,:]
	cv2.imwrite(filename, realimg)