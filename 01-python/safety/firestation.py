import cv2
def draw_firestation():
    img = cv2.imread('sample.jpg')
    print("Brookline Fire Station")
    cv2.imshow('Brookline Fire Station', img)
    cv2.eaitKey(0)
    cv2.destroyAllWindows() 
    return
