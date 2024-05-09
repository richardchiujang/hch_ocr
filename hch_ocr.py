# -*- coding: UTF-8 -*-
# !/usr/bin/python3
import time, tqdm, os
timeinit = time.time()
import socket
import numpy as np
import anyconfig
import cv2
from PIL import Image
import yaml
from easydict import EasyDict as edict
import torch
from utils.build import Logger, Dbn_model
import subprocess, sys

import argparse
def init_args():
    parser = argparse.ArgumentParser(description='HCH_OCR.pytorch')
    parser.add_argument('--drc_config_file', default='config/resnet18_FPN_Classhead.yaml', type=str, help='drcmodel config file')
    parser.add_argument('--drc_checkpoint_path', default=r'weights/model_validation_Loss_0p036458_Accuracy_0p9927.pth', type=str, help='drcmodel wehight path')
    parser.add_argument('--drc_flag', default=False, help='drcmodel work or skip, False is skip, True is work, default is False')
    parser.add_argument('--drc_input_folder', default='work/input', type=str, help='img folder path for inference')
    # parser.add_argument('--drc_output_folder', default='work/output/inference', type=str, help='img path for output')
    parser.add_argument('--dbn_model_path', default=r'weights/model_best_recall0940255_precision0967293_hmean0953583_train_loss0673593_best_model_epoch23.pth',
                        type=str, help='dbnmodel wehight path')
    # parser.add_argument('--dbn_input_folder', default='work/input', type=str, help='img path for predict')
    parser.add_argument('--dbn_output_folder', default='work/output/inference', type=str, help='img path for text recongnition output')
    parser.add_argument('--thre', default=0.5, type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', default=False, action='store_true', help='output polygon or box')
    parser.add_argument('--show', default=False, action='store_true', help='show result')
    # parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')  
    parser.add_argument('--crnn_cfg', type=str, default='lib/config/360CC_config.yaml', help='crnn config file')
    # parser.add_argument('--crnn_image_path', type=str, default='work/images/', help='the path to your image')
    # parser.add_argument('--crnn_checkpoint', type=str, default=r'weights/checkpoint_102_acc_0.9867_0.000081000.pth',    
    parser.add_argument('--crnn_checkpoint', type=str, default=r'weights/checkpoint_88_acc_0.9387_0.000006887.pth',
    # parser.add_argument('--crnn_checkpoint', type=str, default=r'C:\develop\CRNN_Chinese_Characters_Rec\output\360CC\checkpoints\checkpoint_88_acc_0.9387_0.000006887.pth',
                        help='the path to your crnn checkpoints')
    parser.add_argument('--crnn_output_folder', type=str, default='work/output/ocr_result', help='redirect the path for crnn output, this is ocr result.')
    parser.add_argument('--crnn_mode', type=str, default="inference", help='None or inference(default)')
    parser.add_argument('--device', type=str, default='cpu', help='cuda:0 or cuda or cpu(default)')  
    parser.add_argument('--ocrdebug', default=False, help='debug mode True or False(default)')
    parser.add_argument('--crnn_return', default=False, help='return result mode True or False(default)')
    parser.add_argument('--log_level', default='info', help='log level debug or info(default)')
    parser.add_argument('--while_mode', default=False, help='while mode for loop program for waiting data True or False(default)')
    args = parser.parse_args()
    return args

args = init_args()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
device = args.device
hostname = socket.gethostname()

import config.ocrdebug as ocrdebug
# print('ocrdebug.ocrdebug =', ocrdebug.ocrdebug)
if args.ocrdebug == 'True':
    ocrdebug.ocrdebug = True
    print('ocrdebug.ocrdebug =', ocrdebug.ocrdebug)

# start build drcmodel
drc_config = anyconfig.load(open(args.drc_config_file, 'rb'))
# from utils import strbase64to224array
from drcmodels import build_model as build_drcmodel
drc_config['arch']['backbone']['in_channels'] = 3
drcmodel = build_drcmodel(drc_config['arch'])
drc_checkpoint = torch.load(args.drc_checkpoint_path, map_location=device)
drcmodel.load_state_dict(drc_checkpoint)
# print('load drcmodel checkpoint')
drcmodel = drcmodel.to(device)
drcmodel.eval()

# start build dbnmodel if cpu then gpu_id=None
dbnmodel = Dbn_model(args.dbn_model_path, post_p_thre=args.thre, gpu_id=None)


with open(args.crnn_cfg, 'r', encoding='utf-8') as f:
    crnn_config = yaml.load(f, Loader=yaml.FullLoader)
    crnn_config = edict(crnn_config)

import lib.config.alphabets as alphabets
crnn_config.DATASET.ALPHABETS = alphabets.alphabet
crnn_config.MODEL.NUM_CLASSES = len(crnn_config.DATASET.ALPHABETS)

import lib.models.crnn as crnn
from lib.utils.utils import strLabelConverter
crnn_model = crnn.get_crnn(crnn_config).to(device)
if ocrdebug.ocrdebug:
    print('loading pretrained model from {0}'.format(args.crnn_checkpoint))
crnn_checkpoint = torch.load(args.crnn_checkpoint, map_location=device)
crnn_model.load_state_dict(crnn_checkpoint['state_dict'])
crnn_model.eval()
converter = strLabelConverter(crnn_config.DATASET.ALPHABETS)

from utils.ocr_util import crnn_batch_recognition #, crnn_recognition

from utils.ocr_util import cv2imread, rotate_image_up, cut_boxes, rotate_cut_boxes
from utils.ocr_util import padding_boxes, purge_debug_folder, combine_box, sort_boxes
from utils.util import show_img, draw_bbox, save_result, get_file_list, new_draw_bbox

import matplotlib.pyplot as plt
import pathlib

def main():
    tStart = time.time() # 計時開始
    count = 0
    for img_path in (get_file_list(args.drc_input_folder, p_postfix=['.jpg','.JPG'])):  # args.drc_input_folder
        img_path = pathlib.Path(img_path)
        log.logger.debug('now processing image: {}'.format(img_path))
        # log.logger.info('now processing image: {}'.format(img_path))
        # print('............... now processing image {}'.format(img_path))
        img = cv2imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgrotate = img.copy()

        if args.drc_flag == 'True':
            # print('drc_falg is True, so run drcmodel.')
            
            img224 = cv2.resize(img, (224, 224))
            if ocrdebug.ocrdebug:
                im = Image.fromarray(img224)
                im.save('work/output/post_img/{}_224.jpg'.format(img_path.stem), quality=100, format='JPEG')
            # with torch.no_grad():
            # img224arr = torch.from_numpy(img224arr/255).permute(2, 0, 1).unsqueeze(0).float().to(device)
            img224 = torch.from_numpy(img224/255).permute(2, 0, 1).unsqueeze(0).float().to(device)
            # 做文件方向的預測與轉向處理
            drcmodel.eval()  
            drc_label  = drcmodel(img224).argmax(dim=1).cpu().numpy()[0]
            log.logger.debug('file: {}, label: {}'.format(img_path, drc_label))
            imgrotate = rotate_image_up(img, drc_label)
            if ocrdebug.ocrdebug:
                im = Image.fromarray(imgrotate)
                im.save('work/output/post_img/{}_drc.jpg'.format(img_path.stem), quality=100, format='JPEG')
            # count += 1

    # for img_path in (get_file_list(args.dbn_input_folder, p_postfix=['.jpg','.JPG'])):
    # for img_path in (get_file_list(args.input_folder, p_postfix=['.jpg'])): 
        # 做文字框的預測與後處理   
        preds, boxes_list, score_list, t = dbnmodel.predict(imgrotate, is_output_polygon=args.polygon, short_size=726)
        boxes_list, score_list = np.flip(boxes_list, axis=0), np.flip(score_list, axis=0)  # 原輸出是由後到前，轉成由前到後
        

        if ocrdebug.ocrdebug:
            img = draw_bbox(imgrotate, boxes_list)
            if args.show:
                show_img(preds)
                show_img(img, title=os.path.basename(img_path))
                plt.show()
            # 保存结果到路径
            os.makedirs(args.dbn_output_folder, exist_ok=True)
            img_path = pathlib.Path(img_path)
            output_path = os.path.join(args.dbn_output_folder, img_path.stem + '_result.jpg')     # _result
            pred_path = os.path.join(args.dbn_output_folder, img_path.stem + '_pred.jpg')
            im = Image.fromarray(img)
            # im.save(output_path, quality=100, format='JPEG')  # output_path  儲存 dbn 預測的文字框彩圖
            # cv2.imwrite(output_path, img[:, :, ::-1])
            im = Image.fromarray(preds * 255).convert("RGB")
            # im.save(pred_path, quality=100, format='JPEG')   # pred_path 儲存 dbn 預測的黑白圖(二元化預測結果)           
            # cv2.imwrite(pred_path, preds * 255)
            save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)  # 儲存文字框座標 .txt

        # 合併文字框
        new_boxes_list = combine_box(boxes_list)
        for i in range(2): # 共做 3 次
            new_boxes_list = combine_box(new_boxes_list)

        if ocrdebug.ocrdebug:
            # new_boxes_list = combine_box(boxes_list)
            img = draw_bbox(imgrotate, new_boxes_list)
            img_path = pathlib.Path(img_path)
            output_path = os.path.join(args.dbn_output_folder, img_path.stem + '_result_comb.jpg')
            im = Image.fromarray(img)
            # im.save(output_path, quality=100, format='JPEG')  # 儲存合併文字框的彩圖
            pass

        # 剪下文字框
        # new_boxes_list = sort_boxes(new_boxes_list)
        transformed_boxes, src_pts, box_directions = cut_boxes(new_boxes_list, imgrotate, img_path.stem)
        
        if ocrdebug.ocrdebug:
            # img = new_draw_bbox(imgrotate, src_pts, box_directions) # new 垂直文字框 畫藍色框 水平紅框 
            img = new_draw_bbox(img, src_pts, box_directions) # new 把畫過沒有extention的圖再畫一次extention 垂直文字框 畫藍色框 水平綠框 
            img_path = pathlib.Path(img_path)
            output_path = os.path.join(args.dbn_output_folder, img_path.stem + '_result_exten.jpg') # 畫兩次文字框 一個是 combind 一個是 extention
            im = Image.fromarray(img)
            im.save(output_path, quality=100, format='JPEG')
        # 轉向處理
        transformed_boxes = rotate_cut_boxes(transformed_boxes, img_path.stem)
        log.logger.debug('rotate transformed_boxes len: {} {}'.format(img_path.stem, len(transformed_boxes)))
        # 將所有文字框轉成 H=32, W=max(w) 的圖片，高度等比例縮放，寬度補白邊，使用函數 padding_boxes()
        transformed_boxes, width = padding_boxes(transformed_boxes, img_path.stem)
        log.logger.debug('padding transformed_boxes len: {} {} {}'.format(img_path.stem, len(transformed_boxes), width))

        if len(transformed_boxes) == 0:
            log.logger.info('no text box found in image: {}'.format(img_path))
            continue
        else:   # args.crnn_output_folder
            pure_data = crnn_batch_recognition(crnn_config, transformed_boxes, crnn_model, converter, device, args.crnn_output_folder, width, new_boxes_list, box_directions, img_path.stem, mode=args.crnn_mode)
        if ocrdebug.ocrdebug:
            # print('pure_data: ', pure_data)
            pass
        if args.crnn_return == 'True':
            print(img_path.stem, pure_data)


        count += 1
    tEnd = time.time() # 計時結束
    log.logger.info('job finished. page count {}, cost time {:.2f} sec, tot runtime {:.2f}'.format(count, (tEnd - tStart), (tEnd - timeinit)))
    pass

if __name__ == '__main__':
    log = Logger('work/log/pythonOCR.log', level=args.log_level)
    log.logger.info('HCH OCR server start on {}'.format(hostname))
    log.logger.info('pytorch device: {}'.format(device))
    log.logger.info('ocrdebug.ocrdebug: {}'.format(ocrdebug.ocrdebug))
    log.logger.info('crnn_mode: {}'.format(args.crnn_mode))
    log.logger.debug('all args: {}'.format(args))
    if args.crnn_mode != "inference":
        purge_debug_folder()
        log.logger.info('purge_debug_folder: purge and mkdir work/output/cut_boxes, rotate_boxes, padding_boxes, post_img, inference, ocr_result')
    
    os.makedirs('work/wait', exist_ok=True)
    os.makedirs('work/wait_hist', exist_ok=True)
    os.makedirs('work/wait_data', exist_ok=True)


    # licenseVerify = "VerifyKey.pyc"
    # if not os.path.exists(licenseVerify):
    #     print("LICENSE NOT FOUND!")
    #     sys.exit()
    # output = subprocess.call(['python', licenseVerify], shell=True, text=True)
    # if output == 1:
    #     print("LICENSE INVALID!")
    #     sys.exit(1)
    # print("LICENSE VALID!")

    try:
        if args.while_mode == False:
            try:
                main()
            except:
                log.logger.exception("Catch an main() exception.", exc_info=True)
        else:
            loop_count = 0
            while True:
                path = r'.\work\wait'
                path_to = r'.\work\wait_hist'
                path_data = r'.\work\wait_data' 
                files, sorted_files = [], []
                files = os.listdir(path)

                files = [f for f in files if f.endswith('.txt')]

                if len(files) == 0:
                    # log.logger.debug('no file found in folder: {}'.format(r'.\work\wait'))
                    time.sleep(1)
                    loop_count += 1
                    if loop_count > 10:
                        log.logger.debug('wait folder is empty, break loop.')
                        loop_count = 0
                    continue
                else:
                    sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)))
                    log.logger.debug('sorted_files: {}'.format(sorted_files))
                    # print('sorted_files: ', sorted_files)
                    os.replace(os.path.join(path, sorted_files[0]), os.path.join(path_to, sorted_files[0]))

                    if os.path.isdir(os.path.join(path_data, sorted_files[0].split('.txt')[0])):  
                        # target file and folder exist check ok, chang args.drc_input_folder and args.crnn_output_folder
                        args.drc_input_folder = os.path.join(path_data, sorted_files[0].split('.txt')[0])
                        args.crnn_output_folder = args.drc_input_folder
                        log.logger.info('change processing folder: {}'.format(args.drc_input_folder))

                        try:
                            main()
                        except:
                            log.logger.exception("Catch an main() exception.", exc_info=True)
                    else:
                        continue

    except:
            log.logger.exception("Catch an exception.", exc_info=True)
            # print('Catch an exception.')





                    # for img_path in (get_file_list(args.drc_input_folder, p_postfix=['.jpg','.JPG'])):  # args.drc_input_folder
                    #     img_path = pathlib.Path(img_path)
                    #     log.logger.debug('now processing image: {}'.format(img_path))
                    #     # log.logger.info('now processing image: {}'.format(img_path))
                    #     # print('............... now processing image {}'.format(img_path))
                    #     img = cv2imread(img_path)
                    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #     imgrotate = img.copy()

                    #     if args.drc_flag == 'True':
                    #         # print('drc_falg is True, so run drcmodel.')
                            
                    #         img224 = cv2.resize(img, (224, 224))
                    #         if ocrdebug.ocrdebug:
                    #             im = Image.fromarray(img224)
                    #             im.save('work/output/post_img/{}_224.jpg'.format(img_path.stem), quality=100, format='JPEG')
                    #         # with torch.no_grad():
                    #         # img224arr = torch.from_numpy(img224arr/255).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    #         img224 = torch.from_numpy(img224/255).permute(2, 0, 1).unsqueeze(0).float().to(device)
                    #         # 做文件方向的預測與轉向處理
                    #         drcmodel.eval()  
                    #         drc_label  = drcmodel(img224).argmax(dim=1).cpu().numpy()[0]
                    #         log.logger.debug('file: {}, label: {}'.format(img_path, drc_label))
                    #         imgrotate = rotate_image_up(img, drc_label)
                    #         if ocrdebug.ocrdebug:
                    #             im = Image.fromarray(imgrotate)
                    #             im.save('work/output/post_img/{}_drc.jpg'.format(img_path.stem), quality=100, format='JPEG')
                    #         # count += 1

                    # # for img_path in (get_file_list(args.dbn_input_folder, p_postfix=['.jpg','.JPG'])):
                    # # for img_path in (get_file_list(args.input_folder, p_postfix=['.jpg'])): 
                    #     # 做文字框的預測與後處理   
                    #     preds, boxes_list, score_list, t = dbnmodel.predict(imgrotate, is_output_polygon=args.polygon, short_size=726)
                    #     boxes_list, score_list = np.flip(boxes_list, axis=0), np.flip(score_list, axis=0)  # 原輸出是由後到前，轉成由前到後
                        

                    #     if ocrdebug.ocrdebug:
                    #         img = draw_bbox(imgrotate, boxes_list)
                    #         if args.show:
                    #             show_img(preds)
                    #             show_img(img, title=os.path.basename(img_path))
                    #             plt.show()
                    #         # 保存结果到路径
                    #         os.makedirs(args.dbn_output_folder, exist_ok=True)
                    #         img_path = pathlib.Path(img_path)
                    #         output_path = os.path.join(args.dbn_output_folder, img_path.stem + '_result.jpg')     # _result
                    #         pred_path = os.path.join(args.dbn_output_folder, img_path.stem + '_pred.jpg')
                    #         im = Image.fromarray(img)
                    #         # im.save(output_path, quality=100, format='JPEG')  # output_path  儲存 dbn 預測的文字框彩圖
                    #         # cv2.imwrite(output_path, img[:, :, ::-1])
                    #         im = Image.fromarray(preds * 255).convert("RGB")
                    #         # im.save(pred_path, quality=100, format='JPEG')   # pred_path 儲存 dbn 預測的黑白圖(二元化預測結果)           
                    #         # cv2.imwrite(pred_path, preds * 255)
                    #         save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)  # 儲存文字框座標 .txt

                    #     # 合併文字框
                    #     new_boxes_list = combine_box(boxes_list)
                    #     for i in range(2): # 共做 3 次
                    #         new_boxes_list = combine_box(new_boxes_list)

                    #     if ocrdebug.ocrdebug:
                    #         # new_boxes_list = combine_box(boxes_list)
                    #         img = draw_bbox(imgrotate, new_boxes_list)
                    #         img_path = pathlib.Path(img_path)
                    #         output_path = os.path.join(args.dbn_output_folder, img_path.stem + '_result_comb.jpg')
                    #         im = Image.fromarray(img)
                    #         # im.save(output_path, quality=100, format='JPEG')  # 儲存合併文字框的彩圖
                    #         pass

                    #     # 剪下文字框
                    #     # new_boxes_list = sort_boxes(new_boxes_list)
                    #     transformed_boxes, src_pts, box_directions = cut_boxes(new_boxes_list, imgrotate, img_path.stem)
                        
                    #     if ocrdebug.ocrdebug:
                    #         # img = new_draw_bbox(imgrotate, src_pts, box_directions) # new 垂直文字框 畫藍色框 水平紅框 
                    #         img = new_draw_bbox(img, src_pts, box_directions) # new 把畫過沒有extention的圖再畫一次extention 垂直文字框 畫藍色框 水平綠框 
                    #         img_path = pathlib.Path(img_path)
                    #         output_path = os.path.join(args.dbn_output_folder, img_path.stem + '_result_exten.jpg') # 畫兩次文字框 一個是 combind 一個是 extention
                    #         im = Image.fromarray(img)
                    #         im.save(output_path, quality=100, format='JPEG')
                    #     # 轉向處理
                    #     transformed_boxes = rotate_cut_boxes(transformed_boxes, img_path.stem)
                    #     log.logger.debug('rotate transformed_boxes len: {} {}'.format(img_path.stem, len(transformed_boxes)))
                    #     # 將所有文字框轉成 H=32, W=max(w) 的圖片，高度等比例縮放，寬度補白邊，使用函數 padding_boxes()
                    #     transformed_boxes, width = padding_boxes(transformed_boxes, img_path.stem)
                    #     log.logger.debug('padding transformed_boxes len: {} {} {}'.format(img_path.stem, len(transformed_boxes), width))

                    #     if len(transformed_boxes) == 0:
                    #         log.logger.info('no text box found in image: {}'.format(img_path))
                    #         continue
                    #     else:   # args.crnn_output_folder
                    #         pure_data = crnn_batch_recognition(crnn_config, transformed_boxes, crnn_model, converter, device, args.crnn_output_folder, width, new_boxes_list, box_directions, img_path.stem, mode=args.crnn_mode)
                    #     if ocrdebug.ocrdebug:
                    #         # print('pure_data: ', pure_data)
                    #         pass
                    #     if args.crnn_return == 'True':
                    #         print(img_path.stem, pure_data)


                    #     count += 1
                    # tEnd = time.time() # 計時結束
                    # log.logger.info('job finished. page count {}, cost time {:.2f} sec, tot runtime {:.2f}'.format(count, (tEnd - tStart), (tEnd - timeinit)))
