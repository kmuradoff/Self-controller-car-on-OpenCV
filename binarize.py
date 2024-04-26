import cv2 as cv

print("Your OpenCV version is: " + cv.__version__)
cap = cv.VideoCapture(0)

noDrive = cv.imread("images/RU_road_sign_5.19.2.svg.png")
pedistrain = cv.imread("images/RU_road_sign_3.2.svg.png")

pedistrain = cv.resize(pedistrain, (64, 64))
noDrive = cv.resize(noDrive, (64, 64))

pedistrain = cv.inRange(pedistrain, (45, 116, 180), (255, 255, 255))
noDrive = cv.inRange(noDrive, (45, 116, 180), (255, 255, 255))

while (True):
    ret, frame = cap.read()
    cv.imshow("Frame", frame)
    frameCopy = frame.copy()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv = cv.blur(hsv, (5, 5))

    mask = cv.inRange(hsv, (45, 116, 180), (255, 255, 255))

    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    contour, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if contour:
        contour = sorted(contour, key=cv.contourArea, reverse=True)
        cv.drawContours(frame, contour, -1, (255, 0, 255), 3)

        (x, y, w, h) = cv.boundingRect(contour[0])
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.imshow("rectangle", frame)

        roImg = frameCopy[y:y+h, x: x+w]
        cv.imshow("detected", roImg)
        roImg = cv.resize(roImg, (64, 64))
        roImg = cv.inRange(roImg, (45, 116, 180), (255, 255, 255)) #Настроить маску

        pedistrain_val = 0
        noDrive_val = 0

        for i in range(64):
            for j in range(64):
                if roImg[i][j] == noDrive[i][j]:
                    noDrive_val += 1
                if roImg[i][j] == pedistrain[i][j]:
                    pedistrain_val += 1

        if pedistrain_val > 3000:
            print("pedistrain")
        if noDrive_val > 3000:
            print("noDrive")
        else:
            print("nothing")

    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
