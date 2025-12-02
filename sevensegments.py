import numpy as np
import cv2
import easyocr
import matplotlib.pyplot as plt

reader = easyocr.Reader(['en'])

#Warp the selected region into a rectangular frame
def four_point_crop(frame, pts):
    pts = np.array(pts, dtype='float32')

    #compute new widths, heights of image
    widthA = np.linalg.norm(pts[2]-pts[3])
    widthB = np.linalg.norm(pts[1]-pts[0])
    heightA = np.linalg.norm(pts[1]-pts[2])
    heightB = np.linalg.norm(pts[0]-pts[3])
    maxheight = int(max(heightA, heightB))
    maxwidth = int(max(widthA, widthB))
    dst = np.array([
        [0, 0],
        [maxwidth - 1, 0],
        [maxwidth - 1, maxheight - 1],
        [0, maxheight - 1]
    ], dtype="float32")

    #warp cropped region to fit new rectangle with heights defined above
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(frame, M, (maxwidth, maxheight))
    return warped


def Video_analysis(video_path, crop_points=None, start_frame=0, end_frame=None):
    

    cap = cv2.VideoCapture(video_path) #Open video file
    
    #Check if file opened or not
    if not cap.isOpened():
        print('Error: Could not open video')
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) #Set start frame

    #If end_frame is not provided, set it to the last frame
    if end_frame is None: 
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #Print video qualities
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video Properties \nFrame Width, Frame Height, FPS: ")
    print(frame_width, frame_height, fps)

    #Set crop region
    '''
    if crop_region == None:
        x, y, w, h = 0, 0, -1, -1 #Use whole image if image is note cropped
    else: 
        x, y, w, h = crop_region #
    '''
    
    data = []
    while True: 
        ret, frame = cap.read()

        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
            break #Break if at last frame or if exceeded end frame

        if crop_points is not None:
            cropped_frame = four_point_crop(frame, crop_points)
        
        cv2.imshow('Original Frame', cropped_frame)
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        gray = cv2.GaussianBlur(gray, (3, 3), 0) #Blur out imperfections
        _, thresh = cv2.threshold(gray, 0, 1500, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.dilate(thresh, np.ones((7,3), np.uint8), iterations=1) #make edges wider, filling in gaps b/w segments
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        cv2.imshow('Cropped and Black and White Frame', thresh)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.waitKey(1)
            break

        #Extract text 
        
        result = reader.readtext(thresh)
        text = result[0][-2] if result else ''
        #text = pytesseract.image_to_string(thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789.")
        # Extract number safely
        try:
            text = float(text.strip().replace('O', '0').replace(' ', '').replace('i', '1').replace('/','1'.replace('I','1')))
        except ValueError:
            text = None
        
        print(text)
        input()


        #Record time
        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        data.append((time_sec, text))

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    return data

