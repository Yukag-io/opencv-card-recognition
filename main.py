#导入工具包
from imutils import contours
import numpy as np
import argparse 
import imutils
import cv2 
import myutils

#设置参数
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,
                help = "path to input image")
ap.add_argument("-t","--template",required=True,
                help = "path to template OCR-A image")
args = vars(ap.parse_args())

#指定信用卡类型
FIRST_NUMBER = {
"3": "American Express",
"4": "visa",
"5": "MasterCard",
"6": "Discover Card"
}


#绘图展示
def cv_show (name,image):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


###处理模板图
image = cv2.imread(args["template"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv_show("gray",gray)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv_show("thresh",thresh)
cnts, hierarchy = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
print(cnts)
i_cnts = image.copy()
cv2.drawContours(i_cnts, cnts, -1, (0,0,255), 3)
cv_show("i_cnts",i_cnts)
refCnts = myutils.sort_contours(cnts, method="left-to-right")[0]
#初始化模板字典，用来存放数字图像
digits = {}

#遍历排序后的轮廓
for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = thresh[y:y+h, x:x+w]
#尺寸归一化
    roi = cv2.resize(roi, (57,88))

    digits[i] = roi



###初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


###处理信用卡原图
card_image = cv2.imread(args["image"])
cv_show("card_original", card_image)
card_image = imutils.resize(card_image, width=300)
cv_show("card_image", card_image)
gray = cv2.cvtColor(card_image,cv2.COLOR_BGR2GRAY)
#顶帽操作，突出比背景更亮的区域
tophat = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv_show("tophat",tophat)
#计算x方向的sobel梯度
gradx = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradx = np.absolute(gradx)
(minval, maxval) = (np.min(gradx), np.max(gradx))
gradx = (255 * ((gradx - minval) / (maxval - minval))).astype("uint8")
cv_show("gradx",gradx)
#闭运算加二值化
gradx = cv2.morphologyEx(gradx, cv2.MORPH_CLOSE, rectKernel)
cv_show("gradx",gradx)
thresh = cv2.threshold(gradx,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show("thresh", thresh)
thresh2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,sqKernel)
cv_show("thresh2",thresh2)



###定位卡号区域并分区
cnts, hierarchy = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
locs = []
for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))
locs = sorted(locs, key = lambda x: x[0])



###模板匹配
output = []
for (i, (gx, gy, gw, gh)) in enumerate(locs):
    groupOutput = []
    group = gray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]
    group = cv2.threshold(group, 0, 255,  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show("group", group)
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = myutils.sort_contours(digitCnts, method = "left-to-right")[0]

    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi,(57,88))
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(card_image, (gx - 5,gy - 5), (gx + gw +5, gy + gh + 5),(0, 0, 255),1)
    cv2.putText(card_image, "".join(groupOutput), (gx, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(groupOutput)

    ###后处理与输出

    print("识别出的卡号: {}".format(" ".join(output)))

if len(output) > 0:
    first_digit = output[0]
    if first_digit in FIRST_NUMBER:
        print("信用卡类型: {}".format(FIRST_NUMBER[first_digit]))
else:
    print("⚠️ 未识别到任何卡号，请检查图片或调整参数！")

cv_show("Result", card_image)


#python main.py -i imge/card.png -t imge/template.png
