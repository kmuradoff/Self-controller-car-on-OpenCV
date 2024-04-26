import cv2
import numpy as np

def detect_traffic_lights(image):
    # Преобразование изображения в цветовое пространство HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Определение диапазонов цветовых областей для красного, желтого и зеленого
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([60, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Создание масок для каждого цвета
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = mask_red1 + mask_red2
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Нахождение контуров цветовых областей
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Определение цвета и формы сигнала светофора
    color = "Unknown"
    shape = "Unknown"
    
    # Функция для определения формы сигнала светофора
    def detect_shape(contours):
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Отсечение маленьких областей
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if len(approx) == 3:
                    return "Triangle"
                elif len(approx) == 4 and aspect_ratio >= 0.8 and aspect_ratio <= 1.2:
                    return "Square"
                else:
                    return "Unknown"
        return "Unknown"

    # Определение цвета сигнала светофора
    color = "Unknown"
    shape = "Unknown"
    red_detected = False
    yellow_detected = False
    green_detected = False

    if len(contours_red) > 0:
        shape = detect_shape(contours_red)
        red_detected = True
    elif len(contours_yellow) > 0:
        shape = detect_shape(contours_yellow)
        yellow_detected = True
    elif len(contours_green) > 0:
        shape = detect_shape(contours_green)
        green_detected = True

    # Определение цвета сигнала светофора
    if red_detected and np.mean(mask_red[:int(mask_red.shape[0]*0.2), :]) > 0:
        color = "Red"
    elif yellow_detected and np.mean(mask_yellow[int(mask_yellow.shape[0]*0.4):int(mask_yellow.shape[0]*0.6), :]) > 0:
        color = "Yellow"
    elif green_detected and np.mean(mask_green[int(mask_green.shape[0]*0.8):, :]) > 0:
        color = "Green"

    return color, shape

# Запуск видеопотока с камеры
cap = cv2.VideoCapture(0)

while True:
    # Получение кадра с камеры
    ret, frame = cap.read()
    
    # Если кадр получен успешно
    if ret:
        # Обнаружение цвета сигнала светофора на кадре
        color, shape = detect_traffic_lights(frame)
        
        # Отображение текста с определенным цветом и формой
        cv2.putText(frame, f"{color}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Отображение обработанного кадра
        cv2.imshow('Traffic Lights Detection', frame)
        
        # Выход из цикла при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
