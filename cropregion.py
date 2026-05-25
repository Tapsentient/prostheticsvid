import cv2
import numpy as np

clicked_points = []
clone = None
segments = []
current_points = []

def halfway_frame(video_path):
    '''
    Extract the halfway frame of the video for use in selecting corners
    '''
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    
    ret, frame = cap.read()
    cap.release()

    return frame

def click_event(event, x, y, flags, param):
    global clicked_points, clone

    # Left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        # add point
        clicked_points.append((x, y))
        # draw the point on image
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 corners", clone)

        print(f"Point {len(clicked_points)}: {x, y}")

def select_four_corners(video_path):
    global clicked_points, clone
    clicked_points = []

    img = halfway_frame(video_path)
    if img is None:
        raise ValueError("Could not load image for corner selection")

    clone = img.copy()
    cv2.imshow("Select 4 corners", clone)
    cv2.setMouseCallback("Select 4 corners", click_event)

    print("INSTRUCTIONS:")
    print("Click the FOUR corners of the display in order:")
    print(" 1) Top-Left")
    print(" 2) Top-Right")
    print(" 3) Bottom-Right")
    print(" 4) Bottom-Left")
    print("Press 'q' when done.")

    # Wait until 4 points selected or user quits
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(clicked_points) >= 4:
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

    if len(clicked_points) != 4:
        raise ValueError("Error: You must click exactly 4 points.")

    print("Selected points:", clicked_points)
    return clicked_points

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