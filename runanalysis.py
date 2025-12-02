import cropregion
import sevensegments
import length
import matplotlib.pyplot as plt

#%% Define file name
file_name = "Test2.mp4"
video_path = r"/Users/aral/Desktop/Imperial Y2/Prosthetics" + '/' + file_name


#%% Find crop region
corners = select_four_corners("frame.jpg")
print("Use these points in your crop:", corners)

#%%
crop_region = (150, 83, 184, 98) #x, y, width, height; (0,0) is top left
#crop_points = [(174, 87), (325, 87), (322, 171), (166, 170)] #Top left, top right, bottom-right, bottom-left
crop_points = [(189, 98), (310, 100), (306, 186), (179, 181)]
start_frame = 30*75
end_frame = None #41*30

#Edit the video (crop/grayscale) to make it easier to analyse and extract data
data = Video_analysis(video_path, crop_points, start_frame, end_frame)
print(data)

times = [t for (t, v) in data if v is not None]
values = [v/10 for (t, v) in data if v is not None]
plt.scatter(times, values)
plt.ylabel('Force (N)')
plt.xlabel('Time (s)')
plt.show()

#%%
