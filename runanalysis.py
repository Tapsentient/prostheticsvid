#%%
%load_ext autoreload
%autoreload 2
#%%
import cropregion
import sevensegments
import length
import matplotlib.pyplot as plt

#%% Define file name
file_name = "Newtonmeter.mp4"
video_path = file_name

#%% Find crop region
corners = cropregion.select_four_corners(video_path)
print("Use these points in your crop:", corners)

#%%
crop_points = [(189, 98), (310, 100), (306, 186), (179, 181)]
start_frame = 30*75
end_frame = None #41*30

#Edit the video (crop/grayscale) to make it easier to analyse and extract data
data = sevensegments.Video_analysis(video_path, crop_points, start_frame, end_frame)
print(data)

times = [t for (t, v) in data if v is not None]
values = [v/10 for (t, v) in data if v is not None]
plt.scatter(times, values)
plt.ylabel('Force (N)')
plt.xlabel('Time (s)')
plt.show()

#%%
