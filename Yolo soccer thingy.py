import cv2
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import time
import matplotlib.pyplot as plt
from scipy import stats
import keyboard
# Global variables
points = []
lengthofsoccerfield = 55
widthofsoccerfield = 70
cameradistancefromfield = 10
cameraheight = 2
targetresolution = (1280, 720)
smallresolution = (640, 640)
playercolors = []
fieldbounds = []


def select_field(event, x, y, flags, param):
    image = param
    global fieldbounds
    if event == cv2.EVENT_LBUTTONDOWN:
        #if click is on the black part of the image, exit
        if x > targetresolution[0]:
            cv2.waitKey(100)
            cv2.destroyAllWindows()
        else:
            #if click is inside the field, add the color to the fieldbounds list
            if len(fieldbounds) == 2:
                lower = fieldbounds[0]
                upper = fieldbounds[1]
            else:
                lower = [255, 255, 255]
                upper = [0, 0, 0]
            for i in range(-5, 5):
                for j in range(-5, 5):
                    if image[y + i][x + j][0] < lower[0]:
                        lower[0] = image[y + i][x + j][0]
                    if image[y + i][x + j][1] < lower[1]:
                        lower[1] = image[y + i][x + j][1]
                    if image[y + i][x + j][2] < lower[2]:
                        lower[2] = image[y + i][x + j][2]
                    if image[y + i][x + j][0] > upper[0]:
                        upper[0] = image[y + i][x + j][0]
                    if image[y + i][x + j][1] > upper[1]:
                        upper[1] = image[y + i][x + j][1]
                    if image[y + i][x + j][2] > upper[2]:
                        upper[2] = image[y + i][x + j][2]
            if len(fieldbounds) == 0:
                fieldbounds.append(lower)
                fieldbounds.append(upper)
            else:
                fieldbounds[0] = lower
                fieldbounds[1] = upper
            cv2.rectangle(image, (x-5, y-5), (x+5, y+5), (0, 255, 0), 1)
            cv2.imshow('Select Field', image)


def select_players(event, x, y, flags, param):
    global playercolors
    image = param
    if event == cv2.EVENT_LBUTTONDOWN:
        # #find which detection the user clicked on
        # color = [0, 0, 0]
        # for i in range(len(locations)):
        #     loc = locations[i]
        #     if loc[0] < x < loc[2] and loc[1] < y < loc[3]:
        #         withoutfield = cv2.bitwise_and(image, image, mask=inverted_field_mask)
        #         # take mean color of the detection without (0,0,0) pixels
        #         color = withoutfield[loc[1]:(loc[1] + loc[3]) // 2, (loc[0] + loc[2]) // 4:(3 * (loc[0] + loc[2])) // 4,:]
        #         # delete the (0,0,0) pixels
        #         color = color.reshape(-1, 3)
        #         color = color[np.all(color!=[0, 0, 0], axis=1)]
        #         color = np.mean(color, axis=0)
        #         break
        # if color[0] == 0 and color[1] == 0 and color[2] == 0:
        #     print("Please click on a player.")
        #     return
        color = np.mean(image[y - 5:y + 5, x - 5:x + 5, :], axis=(0, 1))
        playercolors.append(color)
        cv2.rectangle(image, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
        cv2.imshow('Select Jerseys', image)
        if len(playercolors) == 2:
            cv2.waitKey(100)
            cv2.destroyAllWindows()

def select_points(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        points.append((x, y))
        cv2.imshow('Select Field Boundary', param)
        if len(points) == 4:
            cv2.waitKey(100)
            cv2.destroyAllWindows()

def warp_to_birdseye_view(image, src_points):
    dst_points = np.array([
        [0, 0],
        [targetresolution[0], 0],
        [targetresolution[0], targetresolution[1]],
        [0, targetresolution[1]]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(image, M, (targetresolution[0], targetresolution[1]))
    return warped_img, M

def createsections(image, resolution):
    sections = []
    sectionlocation = []
    for i in range(max(min([points[k][1] for k in range(len(points))]) - 100, 0), min(max([points[k][1] for k in range(len(points))]) + 100, image.shape[0]), resolution[1]): #y
        for j in range(max(min([points[k][0] for k in range(len(points))]) - 100, 0), min(max([points[k][0] for k in range(len(points))]) + 100, image.shape[1]), resolution[0]): #x
            sections.append(image[i:i+resolution[1], j:j+resolution[0]])
            sectionlocation.append((i, j))
    return sections, sectionlocation

if __name__ == "__main__":
    cam_port = 0
    field = int(input("0 for full field, 1 for left half, 2 for right half: "))
    # filename = r"MVI_6868-002.MP4"
    filename = input("Enter the filename of the video: ")
    cam = cv2.VideoCapture(filename)
    # cam = cv2.VideoCapture(1)
    result, image = cam.read()
    image = cv2.resize(image, targetresolution)
    calibration_image = image.copy()
    #train yolo model on "C:\Users\taof\Downloads\ymmm4.v3i.yolov8"
    model = YOLO(r"yolov8m.pt", task='detect')
    model.to('cuda')
    canvas = np.zeros((targetresolution[1], 400, 3), dtype=np.uint8)
    displaytext = "Select corners of the playing field. Start with the top left, then top right, bottom right, and finally bottom left. Left click to select. The program will automatically continue after all four corners have been clicked. "
    #for loop to put text on canvas
    i = 0
    count = 0
    while i < len(displaytext):
    #find space closest to 40 characters
        for j in range(min(45, len(displaytext) - i - 1), 0, -1):
            if displaytext[i + j] == " ":
                break
        cv2.putText(canvas, displaytext[i:i+j+1], (20, 25 + 25 * count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        i = i + j + 1
        count += 1
    cv2.namedWindow('Select Field Boundary')
    cv2.setMouseCallback('Select Field Boundary', select_points, np.hstack((calibration_image, canvas)))
    cv2.imshow("Select Field Boundary", np.hstack((calibration_image, canvas)))
    cv2.waitKey(0)
    if len(points)!=4:
        print("Please select all 4 corners of the soccer field.")
        exit(0)
    else:
        warped_calibration, M = warp_to_birdseye_view(calibration_image, np.array(points, dtype=np.float32))
    if field == 1:
        calibration_points = [(points[0][0] - 25, points[0][1] - 50), (points[1][0] + 25, points[1][1] - 50), (points[2][0] + 25, points[2][1]), (points[3][0] - 25, points[3][1])] #add 25 pixel buffer
    elif field == 2:
        calibration_points = [(points[0][0] - 25, points[0][1] - 50), (points[1][0] + 25, points[1][1] - 50), (points[2][0] + 25, points[2][1]), (points[3][0] - 25, points[3][1] + 50)]
    else:
        calibration_points = [(points[0][0] - 50, points[0][1] - 50), (points[1][0] + 50, points[1][1] - 50), (points[2][0] + 50, points[2][1] + 50), (points[3][0] - 50, points[3][1] + 50)]
    calibration_contour = np.array(calibration_points, dtype=np.int32)
    calibration_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(calibration_mask, [calibration_contour], (255, 255, 255))
    image = cv2.bitwise_and(image, calibration_mask)
    # cv2.imshow("Image", image)
    sections, sectionlocation = createsections(image, smallresolution)  # create a list of sections to run yolo on
    locations = []
    for i in range(len(sections)):  # run yolo on the sections and save the detection locations
        section = sections[i]
        for thing in model.predict(section, classes=[0]):
            for detection in thing.boxes.xyxy:
                detection = detection.squeeze().tolist()
                x1 = detection[0] + sectionlocation[i][1]
                y1 = detection[1] + sectionlocation[i][0]
                x2 = detection[2] + sectionlocation[i][1]
                y2 = detection[3] + sectionlocation[i][0]
                locations.append([int(x1), int(y1), int(x2), int(y2)])
    colors = []
    #have the user click on the different teams to classify them and then use the average color of the team to classify the rest of the players
    for loc in locations:  # display bounding boxes for each player and then prompt the user to click on two players from different teams to classify them
        cv2.rectangle(image, (int(loc[0]), int(loc[1])), (int(loc[2]), int(loc[3])), (0, 0, 255), 1)
    #show window with instructions and image for user to click on players
    # cv2.namedWindow('Select Field (left click to select, right click to continue)')
    # cv2.setMouseCallback('Select Field (left click to select, right click to continue)', select_field, image)
    # cv2.imshow('Select Field (left click to select, right click to continue)', image)
    # cv2.namedWindow("Select Field Instructions")
    canvas = np.zeros((targetresolution[1], 400, 3), dtype=np.uint8)
    displaytext = "Select the green areas of the field. Left click to select. Click this text in order to continue once green field has been selected. "
    # for loop to put text on canvas
    i = 0
    count = 0
    while i < len(displaytext):
        # find space closest to 40 characters
        for j in range(min(45, len(displaytext) - i - 1), 0, -1):
            if displaytext[i + j]==" ":
                break
        cv2.putText(canvas, displaytext[i:i + j + 1], (20, 25 + 25 * count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
        255, 255, 255), 1)
        i = i + j + 1
        count += 1
    cv2.namedWindow('Select Field')
    cv2.setMouseCallback('Select Field', select_field, np.hstack((image, canvas)))
    keyboard.on_press_key("enter", lambda _: cv2.destroyAllWindows())
    cv2.imshow('Select Field', np.hstack((image, canvas)))
    cv2.waitKey(0)
    print(fieldbounds)
    field_mask = cv2.inRange(image, np.array(fieldbounds[0]), np.array(fieldbounds[1]))
    inverted_field_mask = cv2.bitwise_not(field_mask)
    image = cv2.bitwise_and(image, image, mask=inverted_field_mask)
    cv2.namedWindow("Select Jerseys")
    canvas = np.zeros((targetresolution[1], 400, 3), dtype=np.uint8)
    # for loop to put text on canvas
    displaytext = "Select two different team's jerseys. Bounding boxes will appear, selecting the highlighted areas for the jersey colors. Left click to select. The program will automatically continue when two jerseys are selected."
    i = 0
    count = 0
    while i < len(displaytext):
        # find space closest to 45 characters
        for j in range(min(45, len(displaytext) - i - 1), 0, -1):
            if displaytext[i + j]==" ":
                break
        cv2.putText(canvas, displaytext[i:i + j + 1], (20, 25 + 25 * count), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
        255, 255, 255), 1)
        i = i + j + 1
        count += 1
    cv2.namedWindow('Select Jerseys')
    cv2.setMouseCallback('Select Jerseys', select_players, np.hstack((image, canvas)))
    cv2.imshow('Select Jerseys', np.hstack((image, canvas)))
    cv2.waitKey(0)
    #use the average color of the two teams to classify the rest of the players using the kmeans algorithm
    colors = np.array(playercolors)
    colors = colors.reshape(-1, 3)
    team1 = colors[0]
    team2 = colors[1]
    # ref = colors[2]
    print(team1, team2)
    ax = plt.axes(projection='3d')
    heatmapscale = 20

    heatmap = np.zeros((image.shape[0] // heatmapscale, image.shape[1] // heatmapscale), dtype=np.uint8)


    birds = cv2.imread(r"Soccer_field.png")
    birds = cv2.resize(birds, (targetresolution[0], targetresolution[1]), interpolation=cv2.INTER_LINEAR)
    video = cv2.VideoWriter(r"output.mp4", cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 24, (targetresolution[0] * 2, targetresolution[1]))
    frame_id = 0
    start_time = time.time()
    YOLO_VERBOSE = False
    while result and image is not None:
        t = time.time()
        #crop image into 480x640 sections that overlap, run yolo on each section, then stitch them back together
        result, image = cam.read()
        image = cv2.resize(image, targetresolution)
        orgimage = image.copy()
        image = cv2.bitwise_and(image, calibration_mask)
        if image is None:
            print("Image is None")
            break
        sections, sectionlocation = createsections(image, smallresolution) #create a list of sections to run yolo on
        locations = []
        for i in range(len(sections)): #run yolo on the sections and save the detection locations
            section = sections[i]
            for thing in model.predict(section):
                for detection in thing.boxes.xyxy:
                    detection = detection.squeeze().tolist()
                    x1 = detection[0] + sectionlocation[i][1]
                    y1 = detection[1] + sectionlocation[i][0]
                    x2 = detection[2] + sectionlocation[i][1]
                    y2 = detection[3] + sectionlocation[i][0]
                    locations.append([int(x1), int(y1), int(x2), int(y2)])
        for i in range(len(points)): #draw lines between the points that defines the field
            cv2.line(orgimage, points[i], points[(i + 1) % len(points)], (0, 0, 255), 1)
        birdseyeview = birds.copy()
        for loc in locations: # Warp the detections and display them on the bird's-eye view
            bottom_left = np.array([loc[0], loc[3], 1]).reshape(-1, 1)
            bottom_right = np.array([loc[2], loc[3], 1]).reshape(-1, 1)
            center = (bottom_left + bottom_right) // 2
            warped_center = M @ center
            warped_center /= warped_center[2]
            #take average color along vertical center line of the detection and classify the detection into two teams based on the average color
            # colors = image[loc[1] + ((loc[3] - loc[1]) // 10):(loc[1] + loc[3])//2, (loc[0] + loc[2])//4: 3*(loc[0] + loc[2])//4, :]
            #find the mode color of the detection
            field_mask = cv2.inRange(image, np.array(fieldbounds[0]), np.array(fieldbounds[1]))
            inverted_field_mask = cv2.bitwise_not(field_mask)
            withoutfield = cv2.bitwise_and(image, image, mask=inverted_field_mask)
            #take mean color of the detection without (0,0,0) pixels
            color = withoutfield[loc[1]:loc[3], loc[0]:loc[2], :]
            #delete the (0,0,0) pixels
            color = color.reshape(-1, 3)
            color = color[np.all(color != [0, 0, 0], axis=1)]
            color = np.mean(color, axis=0)
            ax.scatter(color[0], color[1], color[2], color='green')
            #calculate the distance between the color and the two team colors
            distance1 = np.linalg.norm(color - team1)
            distance2 = np.linalg.norm(color - team2)
            #classify the player based on which team color they are closer to
            if distance1 < distance2:
                circlecolor = team1
            else:
                circlecolor = team2
            cv2.rectangle(orgimage, (int(loc[0]), int(loc[1])), (int(loc[2]), int(loc[3])), circlecolor, 1)
            if field == 0: #whole field
                cv2.circle(birdseyeview, (int(warped_center[0]), int(warped_center[1])), 5, circlecolor, -1)
                heatmap[int(warped_center[0] // heatmapscale), int(warped_center[1] // heatmapscale)] += 1
            elif field == 1: #left half of the field
                cv2.circle(birdseyeview, (int(warped_center[0] // 2), int(warped_center[1])), 5, circlecolor, -1)
                heatmap[int(warped_center[0] // (heatmapscale * 2)), int(warped_center[1] // heatmapscale)] += 1
            elif field == 2: #right half of the field
                cv2.circle(birdseyeview, (int((warped_center[0] // 2) + (targetresolution[0] // 2)), int(warped_center[1])), 5, circlecolor, -1)
                heatmap[int((warped_center[0] // (heatmapscale * 2)) + (targetresolution[0] // (heatmapscale * 2))), int(warped_center[1] // heatmapscale)] += 1
        if cv2.waitKey(1) & 0xFF == ord('s'): #if the user presses s, save the video
            video.release()
            input("Video has been saved to output.mp4, press enter to continue")
            video = cv2.VideoWriter(r"output.mp4", cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 24, (targetresolution[0] * 2, targetresolution[1]))
        if cv2.waitKey(1) & 0xFF == ord('q'): #if the user presses q, save the video and exit
            break
        # Calculate FPS
        ax.scatter(team1[0], team1[1], team1[2], color='red')
        ax.scatter(team2[0], team2[1], team2[2], color='blue')
        plt.show()
        frame_id += 1
        fps = 1 / (time.time() - t)

        # Convert the fps to string
        fps_text = 'FPS: {:.2f}'.format(fps)

        # Set the position for the text (x, y) coordinates
        text_position = (birdseyeview.shape[1] - 250, 50)  # Adjust the values accordingly for your image size

        # Choose font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White
        line_type = 2

        # Put the fps text on the top right corner of the frame
        cv2.putText(birdseyeview, fps_text, text_position, font, font_scale, font_color, line_type)
        cv2.imshow("Bird's-eye View", birdseyeview)
        cv2.imshow("Input Image", orgimage)
        # heatmap = cv2.resize(heatmap, (640, 480))
        if fps > 1 and frame_id % int(fps) == 0:
            maximum = max(max(heatmap, key=lambda x: max(x)))
            normal_heatmap = heatmap / max(255, maximum)
            normal_heatmap = cv2.applyColorMap(np.uint8(normal_heatmap * 255), cv2.COLORMAP_JET)

            cv2.imshow("Heatmap", cv2.resize(normal_heatmap, targetresolution, interpolation=cv2.INTER_LINEAR))
        # cv2.imshow("stack", np.hstack((cv2.resize(image, (640, 480)), cv2.resize(birdseyeview, (640, 480)))))
        video.write(np.hstack((orgimage, birdseyeview)))
        cv2.waitKey(1)
    video.release()
    print("Video has been saved to C:\\Users\\taof\\Downloads\\output.mp4")
    #save the heatmap
    maximum = max(heatmap, key=lambda x: max(x))
    normal_heatmap = heatmap / maximum
    normal_heatmap = cv2.applyColorMap(np.uint8(normal_heatmap * 255), cv2.COLORMAP_JET)
    cv2.imwrite(r"C:\Users\taof\Downloads\heatmap.png", cv2.resize(normal_heatmap, targetresolution))
    end_time = time.time()
    elapsed_time = end_time - start_time
    actual_fps = frame_count / elapsed_time
    print("Average processing FPS over the entire video:", actual_fps)
    cv2.destroyAllWindows()