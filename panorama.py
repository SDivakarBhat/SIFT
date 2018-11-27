# import the necessary packages
import numpy as np
import imutils
import cv2
from pyimagesearch import sift 
class Stitcher:
	
	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = sift.detect(imageA,5)
		#print(kpsA)
		(kpsB, featuresB) = sift.detect(imageB,5)
		w1,h1 = imageB.shape[:2]
		w2,h2 = imageA.shape[:2]
		#kpsA = kpsA*255
		#img = cv2.drawKeypoints(imageA, kpsA, outImage=np.array([]), color=(0, 0, 255),)
		#cv2.imshow(img)
		#cv2.waitKey(0)
	
		#caculate canvas dimensions
		imgB_dim = np.float32([[0,0], [0,w1], [h1,w1], [h1,0]]).reshape(-1,1,2)
		imgA_dim_tmp = np.float32([[0,0], [0,w2], [h2,w2], [h2,0]]).reshape(-1,1,2)

		
		# match features between the two images
		M = self.matcher(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh, imageA, imageB)
		#print (M)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		print(H) 
		"""#relative perspective
		imgA_dim = cv2.perspectiveTransform(imgA_dim_tmp, H)

		#result dimensions
		result_dims = np.concatenate((imgB_dim, imgA_dim), axis = 0 )

		# Calculate dimensions of match points
		[x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
		[x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)
	
		# Create output array after affine transformation 
		transform_dist = [-x_min,-y_min]
		transform_array = np.array([[1, 0, transform_dist[0]], 
								[0, 1, transform_dist[1]], 
								[0,0,1]]) 

		result = cv2.warpPerspective(imageA, H,(x_max-x_min, y_max-y_min))
		cv2.imshow("warped Image",result)
		result[transform_dist[1]:w1+transform_dist[1],transform_dist[0]:h1+transform_dist[0]] = imageB"""

#

		"""sift1 = cv2.xfeatures2d.SIFT_create()
		kp1, des1 = sift1.detectAndCompute(imageA, None)
		kp2, des2 = sift1.detectAndCompute(imageB, None)
		print (des1)
		bf = cv2.BFMatcher()
		matches = bf.knnMatch(des1, des2, k=2)
		
		good = []
		for m in matches:
			if m[0].distance < 0.5*m[1].distance:
				good.append(m)
		matches = np.asarray(good)
		
		if len(matches[:,0]) >= 4:
			src = np.float32([kp1[m.queryIdx].pt for m in matches[:, 0]]).reshape(-1,1,2)
			dst = np.float32([kp2[m.trainIdx].pt for m in matches[:, 0]]).reshape(-1,1,2)
		
			H1, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
			print(H)
		else:
			raise Assertionerror("can't find enough keypoints.")"""

		result = cv2.warpPerspective(imageA, H,(imageB.shape[1] + imageA.shape[1], imageB.shape[0]))
		cv2.imshow("warped Image",result)
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		"""print ("Homography is : ", H)
		xh = np.linalg.inv(H)
		print ("Inverse Homography :", xh)
		ds = np.dot(xh, np.array([imageB.shape[1], imageB.shape[0], 1]));
		ds = ds/ds[-1]
		print ("final ds=>", ds)
		f1 = np.dot(xh, np.array([0,0,1]))
		f1 = f1/f1[-1]
		xh[0][-1] += abs(f1[0])
		xh[1][-1] += abs(f1[1])
		ds = np.dot(xh, np.array([imageB.shape[1], imageB.shape[0], 1]))
		offsety = abs(int(f1[1]))
		offsetx = abs(int(f1[0]))
		dsize = (int(ds[0])+offsetx, int(ds[1]) + offsety)
		print ("image dsize =>", dsize)
		tmp = cv2.warpPerspective(imageA, xh, dsize)
		cv2.imshow("warped", tmp)
		# cv2.waitKey()
		tmp[offsety:imageA.shape[0]+offsety, offsetx:imageA.shape[1]+offsetx] = imageA
		#a = tmp
		#H = self.matcher_obj.match(self.leftImage, each, 'right')
		#print "Homography :", H"""
		"""txyz = np.dot(H, np.array([imageA.shape[1], imageA.shape[0], 1]))
		txyz = txyz/txyz[-1]
		dsize = (int(txyz[0])+imageA.shape[1], int(txyz[1])+imageA.shape[0])
		tmp = cv2.warpPerspective(imageB, H, dsize)
		cv2.imshow("tp", tmp)
		cv2.waitKey()
		# tmp[:self.leftImage.shape[0], :self.leftImage.shape[1]]=self.leftImage
		#tmp = self.mix_and_match(self.leftImage, tmp)
		#print "tmp shape",tmp.shape
		#print "self.leftimage shape=", imageA.shape"""
				

		#result = tmp

		"""result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB"""

		# check to see if the keypoint matches should be visualized
		
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)

		# return the stitched image
		return result

	"""def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)"""

	def matcher(self, kpsA, kpsB, des1, des2,
		ratio, reprojThresh, imageA, imageB):
		# compute the raw matches and initialize the list of actual
		# matches
		# FLANN parameters
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50)   # or pass empty dictionary

		flann = cv2.FlannBasedMatcher(index_params,search_params)

		matches = flann.knnMatch(np.asarray(des1, np.float32),np.asarray(des2, np.float32),k=2)
		#raw_matches=matcher.knnMatch(np.asarray(desc1,np.float32),np.asarray(desc2,np.float32), 2)
		# Need to draw only good matches, so create a mask
		matchesMask = [[0,0] for i in range(len(matches))]
		print(len(matches))
		# ratio test as per Lowe's paper
		for i,(m,n) in enumerate(matches):
		    if m.distance < 0.7*n.distance:
		        matchesMask[i]=[1,0]

		"""draw_params = dict(matchColor = (0,255,0), singlePointColor = None,matches, flags = 2) # draw matches in green color
				
		kpA = []
		kpB = []
		kpA = [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size) for kp in kpsA]
		kpB = [cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size) for kp in kpsB]	
		img3 = cv2.drawMatches(imageA,(kpA),imageB,(kpB), matches,None)

		plt.imshow(img3, 'gray'),plt.show()"""

		"""matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))"""
		MIN_MATCH_COUNT = 4
		if len(matches)>MIN_MATCH_COUNT:
			src_pts = np.float32([ kpsA[m[0].queryIdx] for m in matches ]).reshape(-1,1,2)
			dst_pts = np.float32([ kpsB[m[0].trainIdx] for m in matches ]).reshape(-1,1,2)

			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
			matchesMask = mask.ravel().tolist()

			#h,w = img1.shape
			#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			#dst = cv2.perspectiveTransform(pts,M)
	
			#img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
			return (matches, M, matchesMask)
		else:
			print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
			matchesMask = None


		# computing a homography requires at least 4 matches
		"""if len(matches) > 4:
			# construct the two sets of points
			pointsCurrent = 
			ptsA = np.float32([int(kpsA[i]) for (_, i) in matches])
			ptsB = np.float32([int(kpsB[i]) for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)"""

			# return the matches along with the homograpy matrix
			# and status of each matched point
			#return (matches, h, status)

		# otherwise, no homography could be computed
	#return None"""
	# This draws matches and optionally a set of inliers in a different color
	# Note: I lifted this drawing portion from stackoverflow and adjusted it to my needs because OpenCV 2.4.11 does not
	# include the drawMatches function
	"""def drawMatches(self,img1, img2, kp1, kp2, matches, inliers = None):
    	# Create a new output image that concatenates the two images together
		(rows1, cols1) = img1.shape[:2]
		(rows2, cols2) = img2.shape[:2]		
		rows1 = img1.shape[0]
		cols1 = img1.shape[1]
		rows2 = img2.shape[0]
		cols2 = img2.shape[1]

		out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
		out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
		out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
		for mat in matches:
		# Get the matching keypoints for each of the images
			img1_idx = mat.queryIdx
			img2_idx = mat.trainIdx

		# x - columns, y - rows
		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		inlier = False

		if inliers is not None:
			for i in inliers:
				if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
					inlier = True

		# Draw a small circle at both co-ordinates
		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		# Draw a line in between the two points, draw inliers if we have them
		if inliers is not None and inlier:
			cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
		elif inliers is not None:
			cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

		if inliers is None:
			cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

		return out"""

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageB
		vis[0:hB, wA:] = imageA

		"""for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)"""
			
		hdif = (hB-hA) / 2
		for i in range(min(len(kpsB), len(kpsA))):
			pt_a = (int(kpsA[i,1]), int(kpsA[i,0] + hdif))
			pt_b = (int(kpsB[i,1] + wA), int(kpsB[i,0]))
			cv2.line(vis, pt_a, pt_b, (255, 0, 0))
		
			#ptA = (kpsA[i,1], kpsA[i,0]+hdif)
			#ptB = (kpsB[i,1] + wA, kpsB[i,0])
			#cv2.line(vis, (ptA), (ptB), (255, 0, 0))
			#ptA = (kpsA[m[0].queryIdx][0], kpsA[m[1].queryIdx][1])
			#ptB = (kpsB[m[0].trainIdx][0] + wA, kpsB[m[1].trainIdx][1])
			#cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis
