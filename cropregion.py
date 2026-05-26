import cv2
import numpy as np

clicked_points = []
clone = None
segments = []
current_points = []

def nth_frame(video_path, portion_through_video = 0):
    '''
    Extract a frame of the video for use in selecting crop region
    portion_through_video: Fraction of the video that has elapsed. 0 selects
    the first frame, 0.5 selects the halfway frame etc
    '''
    if portion_through_video>=1 or portion_through_video<0:
        raise ValueError("portion_through_video must be in [0, 1]")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = int(total_frames*portion_through_video)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret, frame = cap.read()
    cap.release()

    return frame

def select_rectangle(video_path):
    img = nth_frame(video_path)
    print("Drag a rectangle over the crop region. Use 'space' or " \
    "'enter' to finish selection. ")
    region = cv2.selectROI("Select Region", img)
    cv2.destroyWindow("Select Region")
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    return region

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