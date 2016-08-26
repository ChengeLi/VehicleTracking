%shadow removal

NormalizedImg2 = zeros(oriImg.shape)
oriImg2 =double(oriImg)
epsilon = pow(10,-6)
% epsilon = 0
NormalizedImg2(:,:,0) = oriImg2(:,:,0)/(oriImg2(:,:,0)+oriImg2(:,:,1)+oriImg2(:,:,2)+epsilon)
NormalizedImg2(:,:,1) = oriImg2(:,:,1)/(oriImg2(:,:,0)+oriImg2(:,:,1)+oriImg2(:,:,2)+epsilon)
NormalizedImg2(:,:,2) = oriImg2(:,:,2)/(oriImg2(:,:,0)+oriImg2(:,:,1)+oriImg2(:,:,2)+epsilon)


% NormalizedImg(:,:,:)(isnan(NormalizedImg(:,:,:))) = 0
% NormalizedImg(:,:,:)(NormalizedImg(:,:,:)==inf) = 0

NormalizedImg = uint8(NormalizedImg*255)
figure()
imshow(NormalizedImg(:,:,::-1))
figure()
imshow(oriImg(:,:,::-1))


NormalizedImg(:,:,0) = uint8(NormalizedImg(:,:,0)/(nanmax(NormalizedImg(:,:,0)(:))-min(NormalizedImg(:,:,0)(:)))*255)
NormalizedImg(:,:,1) = uint8(NormalizedImg(:,:,1)/(nanmax(NormalizedImg(:,:,1)(:))-min(NormalizedImg(:,:,1)(:)))*255)
NormalizedImg(:,:,2) = uint8(NormalizedImg(:,:,2)/(nanmax(NormalizedImg(:,:,2)(:))-min(NormalizedImg(:,:,2)(:)))*255)


function newImg2 = cutOutBorder(Img):
% """if img has white borders"""
height,width = Img.shape(:2)
ImgRGBSum = zeros(Img.shape(0:2))
ImgRGBSum(:,:) = int32(Img(:,:,0))+int32(Img(:,:,1))+int32(Img(:,:,2))

borderIndicator_hori = where(ImgRGBSum(height/2,:)==255*3)(0)
borderIndicator_vert = where(ImgRGBSum(:,width/2)==255*3)(0)
borderIndicatorDiff_hori = diff(borderIndicator_hori)
borderIndicatorDiff_vert = diff(borderIndicator_vert)

boarder_hori = where(borderIndicatorDiff_hori>=0.5*width)(0)(0)+1
boarder_vert = where(borderIndicatorDiff_vert>=0.5*height)(0)(0)+1

% % oriImg = Img(boarder_vert:(height-boarder_vert),boarder_hori:(width-boarder_hori+50),:) 
newImg2 = Img(borderIndicator_vert(boarder_vert-1):borderIndicator_vert(boarder_vert),\
borderIndicator_hori(boarder_hori-1):borderIndicator_hori(boarder_hori),:)
end
	


% % method 1: decrease illuminance
imageList = sorted(glob.glob('../DoT/CanalSt@BaxterSt-96.106/imgs/*.jpg'))
oriImg = cv2.imread(imageList(0))
height, width = oriImg.shape(0:2)
NumImg = len(imageList)
NormalizedImgList = zeros((NumImg,oriImg.shape(0),oriImg.shape(1),oriImg.shape(2)))

for kk = range(len(imageList))
	oriImg = cv2.imread(imageList(kk))
	NormalizedImg = NormalizeImg(oriImg)
	NormalizedImgList(kk,:,:,:) = NormalizedImg
	% imshow(NormalizedImg(:,:,::-1))
	% pause(0.001)
end




% % method 2: 

oriImg = cv2.imread(imageList(20))

aveImg = cv2.imread('../DoT/CanalSt@BaxterSt-96.106/ave3.png')
aveImg = cutOutBorder(aveImg)
aveImg = cv2.resize(aveImg, oriImg.shape(:2)(::-1)) % the resize function is dimension reversed...==


diffImg = abs(oriImg - aveImg)
diffImg = cv2.cvtColor(diffImg, cv2.COLOR_BGR2GRAY)
th,diffImg = cv2.threshold(diffImg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


% % NormalizedImg = zeros(oriImg.shape)
% % NormalizedImg = uint16(oriImg/(aveImg+exp(-8)))
NormalizeImg = NormalizeImg(oriImg)

A = 1
gamma = 1.01
NormalizedImg = A*NormalizedImg**gamma


NormalizedImg = array(NormalizedImg, dtype=uint16)
figure()
imshow(NormalizedImg)

fr_shadow = cv2.cvtColor(NormalizedImg, cv2.COLOR_BGR2GRAY)
fr_shadow(:,:) = uint8(fr_shadow(:,:)/(nanmax(fr_shadow(:,:))-min(fr_shadow(:,:)))*255)


% % fr_shadow = array(fr_shadow, dtype=uint8)

thresh,fr_shadow_bin = cv2.threshold(fr_shadow,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
figure()
imshow(fr_shadow_bin)

finalImg = zeros(oriImg.shape(:2))
for hh = range(height)
	for ww = range(width)
		if (diffImg(hh,ww)>=0) and (fr_shadow(hh,ww)>thresh):
			finalImg(hh,ww) = fr_shadow(hh,ww);
        else
			finalImg(hh,ww) = 0;
        end
    end
end


figure()
imshow(uint8(finalImg))



























