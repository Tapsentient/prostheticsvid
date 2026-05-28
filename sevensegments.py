import numpy as np
import cv2
import matplotlib.pyplot as plt
import cropregion

#import easyocr
#reader = easyocr.Reader(['en'])


"""
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
"""

segment_map = {
    (1,1,1,1,1,1,0): "0",
    (0,0,0,0,0,0,0): "0",
    (0,1,1,0,0,0,0): "1",
    (1,1,0,1,1,0,1): "2",
    (1,1,1,1,0,0,1): "3",
    (0,1,1,0,0,1,1): "4",
    (1,0,1,1,0,1,1): "5",
    (1,0,1,1,1,1,1): "6",
    (1,1,1,0,0,0,0): "7",
    (1,1,1,1,1,1,1): "8",
    (1,1,1,1,0,1,1): "9"
}

def segment_expander(segments):
    ndigits = len(segments)
    for digit in range(ndigits):
        for line in range(7):
            p1, p2 = segments[digit][line]
            yield p1, p2

def segment_ocr(frame, segments):
    '''
    Given the coordinates of the segments in a frame, returns
    the numeric digits displayed by the segments.  
    '''
    ndigits = len(segments)
    digits = []
    current_digit = []

    for segment in segment_expander(segments):
        on = is_segment_on(frame, *segment)
        current_digit.append(int(on))
        if len(current_digit) == 7:
            current_digit_char = segment_map.get(tuple(current_digit), None)
            if current_digit_char is None:
                return np.nan
            digits.append(current_digit_char)
            current_digit = []

    text = int("".join(digits))    
    return text

segments = []
current_points = []
current_segment = []

def click_event(event, x, y, flags, param):
    global current_points, segments, current_segment, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))

        if len(current_points) == 2:
            p1, p2 = current_points
            current_segment.append((p1, p2))
            cv2.line(clone, p1, p2, (0, 255, 0), 2)
            cv2.imshow("Define Segments", clone)
            current_points = []
        
        if len(current_segment) == 7:
            segments.append(current_segment)
            current_segment = []
            print(f"Digit {len(segments)} completed.")    
    

def define_segments(video_path, crop_points):
    '''
    Given a video of the newtonmeter and the region it is to be cropped to,
    displays the newtonmeter and allows you to draw lines where the segments are
    '''
    global clone, segments, current_points
    segments = []
    current_points = []
    current_segment = []

    image = cropregion.nth_frame(video_path, 0.5)
    cropped_image = cropregion.four_point_crop(image, crop_points)
    clone = cropped_image.copy()

    cv2.namedWindow("Define Segments", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Define Segments", 400, 200)
    cv2.imshow("Define Segments", clone)
    cv2.setMouseCallback("Define Segments", click_event)
    print("Click 2 points per segment. Select each digit in order," \
    "starting from the top and going clockwise. End with the middle. Press ESC when done.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

    if len(current_segment) != 0 or len(current_points) != 0:
        raise ValueError("Error: Please select seven segments per digit.")

    print("Segment coordinates:", segments)
    return segments

def is_segment_on(frame, p1, p2, threshold=0.5):
    '''
    Checks if the segment between points p1 and p2 in the given frame is on. 
    Returns a boolean - True if on, False if off. 
    '''
    x1, y1 = p1
    x2, y2 = p2
    num = 10 # Sample 30 points along the line
    xs = np.linspace(x1, x2, num).astype(int)
    ys= np.linspace(y1, y2, num).astype(int)

    values = frame[ys, xs]
    mean_val = np.mean(values) #White = ON, black = OFF
    return mean_val > threshold * 255

def newtonmeter_analysis(video_path, crop_points=None, start_frame=0, end_frame=None, segments=None):
    '''
    Returns the newtonmeter reading times 10 for each frame in an array. For intermediate values, 
    returns nan
    '''
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
            cropped_frame = cropregion.four_point_crop(frame, crop_points)
        
        threshold = 1500
        cv2.imshow('Original Frame', cropped_frame)
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY) #Convert to grayscale
        gray = cv2.GaussianBlur(gray, (3, 3), 0) #Blur out imperfections
        _, thresh = cv2.threshold(gray, 0, threshold, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.dilate(thresh, np.ones((7,3), np.uint8), iterations=1) #make edges wider, filling in gaps b/w segments
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.waitKey(1)
            cv2.waitKey(1)
            break

        #Extract text 
        text = segment_ocr(thresh, segments)
        
        print(text)
        for segment in segment_expander(segments):
            cv2.line(thresh, *segment, (0, 255, 0), 2)

        cv2.imshow('Cropped and Black and White Frame', thresh)
        #Record time
        input()
        time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        data.append((time_sec, text))

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    return data

