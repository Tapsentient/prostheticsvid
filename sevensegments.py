import numpy as np
import cv2
import easyocr
import matplotlib.pyplot as plt
import cropregion

reader = easyocr.Reader(['en'])


def four_point_crop(frame, pts):
    '''
    Warp the selected region into a rectangular frame
    '''
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

def text_ocr(frame):
    '''
    Extract text on Newtonmeter via easy_ocr module 
    '''
    result = reader.readtext(frame)
    text = result[0][-2] if result else ''
    try:
        text = float(text.strip().replace('O', '0').replace(' ', '').replace('i', '1').replace('/','1'.replace('I','1')))
    except ValueError:
        text = None
    return text 

def segment_ocr(frame):
    text = "xyz"
    return text

segments = []
current_points = []

def click_event(event, x, y, flags, param):
    global current_points, segments, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

        if len(current_points) == 2:
            p1, p2 = current_points
            segments.append((p1, p2))
            cv2.line(clone, p1, p2, (0, 255, 0), 2)
            cv2.imshow("Define Segments", clone)
            current_points = []

def define_segments(video_path, crop_points):
    global clone, segments, current_points
    segments = []
    current_points = []

    image = cropregion.halfway_frame(video_path)
    cropped_image = four_point_crop(image, crop_points)
    clone = cropped_image.copy()

    cv2.imshow("Define Segments", clone)
    cv2.setMouseCallback("Define Segments", click_event)
    print("Click 2 points per segment. Press ESC when done.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

    return segments

def is_segment_on(frame, p1, p2, threshold=0.5):
    x1, y1 = p1
    x2, y2 = p2
    num = 30 # Sample 30 points along the line
    xs = np.linspace(*p1, num).astype(int)
    ys= np.linspace(y1, y2, num).astype(int)
    
    values = frame[ys, xs]
    mean_val = np.mean(values) #White = ON, black = OFF

    return mean_val >threshold * 255

def Video_analysis(video_path, crop_points=None, start_frame=0, end_frame=None, segments=None):
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

    data = []
    while True: 
        ret, frame = cap.read()

        if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
            break #Break if at last frame or if exceeded end frame

        if crop_points is not None:
            cropped_frame = four_point_crop(frame, crop_points)
        
        threshold = 1500
        cv2.imshow('Original Frame', cropped_frame)
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        gray = cv2.GaussianBlur(gray, (3, 3), 0) #Blur out imperfections
        _, thresh = cv2.threshold(gray, 0, threshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.dilate(thresh, np.ones((7,3), np.uint8), iterations=1) #make edges wider, filling in gaps b/w segments
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

        cv2.imshow('Cropped and Black and White Frame', thresh)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.waitKey(1)
            break

        #Extract text 
        text = text_ocr(thresh)
        
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

