# same blob score (SBS) for two trajectories
def sameBlobScore(trj1,trj2,blobColorImgList,common_idx):
	x1 = trj1[0]
	y1 = trj1[1]
	x2 = trj2[0]
	y2 = trj2[1]

	for frame_idx in common_idx:
		blobImg = blobColorImgList[frame_idx]
		color1 = blobImg[y1,x1]
		color2 = blobImg[y2,x2]

		if color1 == color2:
			SBS = SBS+1

	return SBS







if __name__ == '__main__':
	blobColorImgList  = sorted(glob.glob('/media/TOSHIBA/DoTdata/CanalSt@BaxterSt-96.106/incPCP/Canal_blobImage/*.jpg'))

	trj1 = [x_re[i,idx] , y_re[i,idx]]

	trj1 = x_re[j,idx]  y_re[j,idx]














