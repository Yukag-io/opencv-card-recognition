import cv2

def sort_contours(cnts, method="left-to-right"):
    # 初始化排序方向标志和坐标索引
    reverse = False
    i = 0
    
    # 处理排序方向
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
        
    # 用外接矩形来获取每个轮廓的 (x, y, w, h)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    
    # 将轮廓和边界框打包，根据第 i 个维度（x 或 y）进行排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    
    # 返回排序后的轮廓和边界框
    return cnts, boundingBoxes