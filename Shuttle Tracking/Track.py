import argparse
import Models
import queue
import cv2
import numpy as np
import os
import natsort
from PIL import Image, ImageDraw
video_dir='./badminton_example/' 
def last_chars(x):
    return(x[2:])

pdir=sorted(os.listdir(video_dir), key = last_chars)
print(pdir)
 
for ii in range(len(pdir)):

   cdir=natsort.natsorted(os.listdir(os.path.join(video_dir,pdir[ii])))
   dirname=pdir[ii]
   print(cdir)
   for jj in range(len(cdir)):
    print(cdir)
    print(jj)
    print(str(video_dir)+str(pdir[ii])+'/'+str(cdir[jj]))
    #print(a)
    filename=cdir[jj]
#parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video_path", type=str,default=str(video_dir)+str(pdir[ii])+'/'+str(cdir[jj])+'/1.mp4')
    parser.add_argument("--output_video_path", type=str, default = './output/'+str(pdir[ii])+'/'+str(cdir[jj])+'/1.mp4')
    parser.add_argument("--save_weights_path", type = str  )
    parser.add_argument("--n_classes", type=int )
    parser.add_argument("--csv_path", type=str,default = './outputcsv/'+str(pdir[ii])+'/'+str(cdir[jj])+'/Label.csv' )
    args = parser.parse_args()
    input_video_path =  args.input_video_path
    output_video_path =  args.output_video_path
    save_weights_path = args.save_weights_path
    n_classes =  args.n_classes
    csv_path =  args.csv_path

    PATH1 = './output/'+str(pdir[ii])+'/'+str(cdir[jj])
    if not os.path.exists(PATH1):
        os.makedirs(PATH1)
    PATH2 = './outputcsv/'+str(pdir[ii])+'/'+str(cdir[jj])
    if not os.path.exists(PATH2):
        os.makedirs(PATH2)
#get video fps&video size
    video = cv2.VideoCapture(input_video_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

#start from first frame
    currentFrame = 0

#width and height in TrackNet
    width , height = 640, 360
#width , height = 3840, 2160
    img, img1, img2 = None, None, None

#load TrackNet model
    modelFN = Models.TrackNet.TrackNet
    m = modelFN( n_classes , input_height=height, input_width=width   )
    m.compile(loss='categorical_crossentropy', optimizer= 'adadelta' , metrics=['accuracy'])
    m.load_weights(  save_weights_path  )

# In order to draw the trajectory of tennis, we need to save the coordinate of preious 7 frames 
    q = queue.deque()
    for i in range(0,8):
        q.appendleft(None)

#save prediction images as vidoe
#Tutorial: https://stackoverflow.com/questions/33631489/error-during-saving-a-video-using-python-and-opencv
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video_path,fourcc, fps, (output_width,output_height))


#both first and second frames cant be predict, so we directly write the frames to output video
#capture frame-by-frame
    video.set(1,currentFrame); 
    ret, img1 = video.read()
#write image to video
    output_video.write(img1)
    currentFrame +=1
#resize it 
    img1 = cv2.resize(img1, ( width , height ))
#input must be float type
    img1 = img1.astype(np.float32)

#capture frame-by-frame
    video.set(1,currentFrame);
    ret, img = video.read()
#write image to video
    output_video.write(img)
    currentFrame +=1
#resize it 
    img = cv2.resize(img, ( width , height ))
#input must be float type
    img = img.astype(np.float32)
    with open(csv_path, 'a+') as file:
     file.write("file name,visibility,x-coordinate,y-coordinate,status")
     file.write('\n')
     data = "{},{},{},{},{}".format('0000.jpg', 1,-1,-1,0)
     file.write(data+'\n')
     data = "{},{},{},{},{}".format('0001.jpg', 1,-1,-1,0)
     file.write(data+'\n')



    while(True):
        img2 = img1
        img1 = img
        video.set(1,currentFrame)
        ret, img = video.read()
        if not ret: 
            break
        
        output_img = img
        img = cv2.resize(img, ( width , height ))
        img = img.astype(np.float32)
        X =  np.concatenate((img, img1, img2),axis=2)
        X = np.rollaxis(X, 2, 0)
        pr = m.predict( np.array([X]) )[0]
        pr = pr.reshape(( height ,  width , n_classes ) ).argmax( axis=2 )
        pr = pr.astype(np.uint8) 
        heatmap = cv2.resize(pr  , (output_width, output_height ))
        ret,heatmap = cv2.threshold(heatmap,127,255,cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT,dp=1,minDist=1,param1=50,param2=2,minRadius=2,maxRadius=7)
        PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)   
        PIL_image = Image.fromarray(PIL_image)
        with open(csv_path,'a+') as file:
            if circles is not None:
                if len(circles) == 1:
                    x = int(circles[0][0][0])
                    y = int(circles[0][0][1])
                    data = "{},{},{},{},{}".format(str(currentFrame).zfill(4)+'.jpg',1, x,y,0)
                    file.write(data+'\n')
                    print(currentFrame, x,y)
                    q.appendleft([x,y])   
                    q.pop()    
                else:
                    q.appendleft(None)
                    q.pop()
            else:
                print (currentFrame, -1,-1)
                data = "{},{},{},{},{}".format(str(currentFrame).zfill(4)+'.jpg',1, -1,-1,0)
                file.write(data+'\n')	 
                q.appendleft(None)
                q.pop()
        
        
        for i in range(0,8):
            if q[i] is not None:
                draw_x = q[i][0]
                draw_y = q[i][1]
                bbox =  (draw_x - 10, draw_y - 10, draw_x + 10, draw_y + 10)
                draw = ImageDraw.Draw(PIL_image)
                draw.ellipse(bbox, outline ='red',fill=128)
                del draw
        
        opencvImage =  cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
        output_video.write(opencvImage)
        currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()
print("finish")
