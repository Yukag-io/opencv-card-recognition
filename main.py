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
    