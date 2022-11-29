# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import natsort
import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def detect(opt,sign):
    
    flag=0
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    print('source',source)
    F_name=source.split('/')[2]
    MD_name=source.split('/')[3]
    R_name=source.split('/')[4]
   
    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            #pass
            print('Start')
            #shutil.rmtree(out)  # delete output folder
       # os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    #print('AAAAAAAAAAA',exp_name)
    #print(a)
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    #exp_name = exp_name 
    #print('EEEEEEEEEEEEEE',exp_name)
    #print(a)
    save_dir = increment_path(Path(project) / exp_name, exist_ok=True)  # increment run if project name exists
    #save_dir=str(Path(project) / str(exp_name))
    (save_dir / 'tracks_csv' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    #print(save_dir)
    
    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources
   
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        
        dt[2] += time_sync() - t3

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            #global sign
            #sign=0
            if sign==1:
                
                det=det[det[:, 0].sort()[1]]
                
            
            
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    
                    save_dir1=save_dir / 'Video_detected'/ F_name / MD_name / R_name
                    save_dir2=save_dir / 'Crop'/ F_name / MD_name / R_name
                    #print('ss',MD_name)
                    #print(a)
                   # print(p.name)
                   # print(a)
                    #save_path_v =str(save_dir1)
                    save_path_v = str(save_dir1 / p.name)  # im.jpg, vid.mp4, ...
                    #print(save_path)
                    #print(a)
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            #print('txt_file_name',txt_file_name)
           
            txt_path = str(save_dir / 'tracks_csv' / F_name / MD_name / R_name/ txt_file_name)  # im.txt
            PATH = str(save_dir / 'tracks_csv' / F_name / MD_name / R_name)
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                
                xywhs = xyxy2xywh(det[:, 0:4])
                
                
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                  with open(txt_path + '.csv', 'a') as f:
                    if flag==0:
                        f.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} ".format('frame', 'player1x','player1y','p1x_output0','p1y_output0','p1x_output1','p1y_output1','p1x_output2','p1y_output2','p1x_output3','p1y_output3', 'player2x','player2y','p2x_output0','p2y_output0','p2x_output1','p2y_output1','p2x_output2','p2y_output2','p2x_output3','p2y_output3',
'player3x','player3y','p3x_output0','p3y_output0','p3x_output1','p3y_output1','p3x_output2','p3y_output2','p3x_output3','p3y_output3',
'player4x','player4y','p4x_output0','p4y_output0','p4x_output1','p4y_output1','p4x_output2','p4y_output2','p4x_output3','p4y_output3'
))
                        flag=1
                        f.write('\n')
                    count=1
                   #for count in range(cc,5):
                    #print('count',count)
                    f.write(('%g ' * 1) % (frame_idx))
                    for j, (output) in enumerate(outputs[i]):
                        
                        bboxes = output[0:4]
                        id = output[4]
                        
                        cls = output[5]
                        conf = output[6]
                        #print('ID',id)
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_right = output[2]
                            bbox_down=output[3]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            #x_c=bbox_left+bbox_w/2.0
                            #y_c=bbox_top+bbox_h/2.0
                            x_c=(output[0]+output[2])/2
                            y_c=(output[1]+output[3])/2
                            # Write MOT compliant results to file
                            #print('j',j)
                            #print('id',id) 
                            #print('count',count)

                            while count<8:
                                if count==id:  
                                
                                    f.write(('%g ' * 10) % (x_c,y_c,bbox_left,bbox_top,bbox_left,bbox_down,bbox_right,bbox_down,bbox_right,bbox_top))
                                    #f.write('\n')
                                    #print('1count',count) 
                                    count+=1
                                    break
                                      
                            

                                else:
                               
                                    #print('None')
                                    f.write("{} {} {} {} {} {} {} {} {} {} ".format(None,None,None,None,None,None,None,None,None,None))
                                    #f.write('\n')
                                    #print('2count_None',count) 
                                    count+=1

                            
                                
                        


                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = f'{id:0.0f} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                #id1=int(id)
                                save_one_box(bboxes, imc, file= save_dir2 / f'{id}' / f'{frame_idx}.jpg', BGR=True)



               
                                  
        
                    f.write('\n')
                   
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')
                
            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')
            
            # Stream results
            im0 = annotator.result()
            
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_vid:
               
                if vid_path[i] != save_path_v:  # new video
                    vid_path[i] = save_path_v
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                       
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                       
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                   
                    PATH = "/".join(list(save_path_v.split('/')[0:-1])) 
                    if not os.path.exists(PATH):
                        os.makedirs(PATH)
                    save_path_v = str(Path(save_path_v).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    
                    #print(a)
                    vid_writer[i] = cv2.VideoWriter(save_path_v, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
   
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
       # per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks_csv/*.txt')))} tracks_csv saved to {save_dir / 'tracks_csv'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)

    return sign






def last_chars(x):
    return(x[2:])
if __name__ == '__main__':

 video_dir='./badminton_example/' 
 
 previous_dirname='M0'
 pdir=natsort.natsorted(os.listdir(video_dir))
 print(pdir)
 
 for i in range(len(pdir)):

  cdir=natsort.natsorted(os.listdir(os.path.join(video_dir,pdir[i])))
  dirname=pdir[i]

  for j in range(len(cdir)):
    
   
   rdir=natsort.natsorted(os.listdir(str(video_dir)+str(pdir[i])+'/'+str(cdir[j])))
   for k in range(len(rdir)):
    #print(a)
    rname=rdir[k]
    print(str(video_dir)+str(pdir[i])+'/'+str(cdir[j])+'/'+str(rname))
    if dirname!=previous_dirname:
      sign=1
    else:
      sign=0
    previous_dirname=dirname    
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
      # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default=str(video_dir)+str(pdir[i])+'/'+str(cdir[j])+'/'+str(rdir[k])+'/1.mp4', help='source')    
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    

    with torch.no_grad():
        
        detect(opt,sign)
