import cv2
test = cv2.imread("/home/robo/下载/2.png")
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
cv2.imwrite('/home/robo/下载/2.png', test)
