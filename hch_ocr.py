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
    parser = argparse.ArgumentParser(description='OCR.pytorch')
    parser.add_argument('--drc_config_file', default='config/resnet18_FPN_Classhead.yaml', type=str, help='drcmodel config file path')
    parser.add_argument('--drc_checkpoint_path', default=r'weights/drc_model_testphaseLoss0p066Accuracy0p983.pth', type=str, help='drcmodel wehight file path')
    # parser.add_argument('--drc_checkpoint_path', default=r'C:\develop\doc_rotate_classification\output\model.pth', type=str, help='drcmodel wehight file path')
    parser.add_argument('--drc_flag', default=False, action=argparse.BooleanOptionalAction, help='document direction drcmodel enable or disable')
    parser.add_argument('--drc_input_folder', default='work/input', type=str, help='assign img folder path for input path default is work/input')
    parser.add_argument('--dbn_model_path', default=r'weights/dbn_model_best_e4_h0p684507037728362_l1p439903035575961.pth',
    # parser.add_argument('--dbn_model_path', default=r'C:\develop\DBNet_pytorch_Wenmu\output\DBNet_resnet18_FPN_DBHead\checkpoint\dbn_model_best_e12_h0.7420886028473502_l0.9898050118375707.pth',                        
                        type=str, help='dbnmodel wehight path')
    parser.add_argument('--dbn_output_folder', default='work/output/inference', type=str, help='img path for ocr result output default is work/output/inference')
    parser.add_argument('--thre', default=0.5, type=float, help='the threshould of dbn post_processing')
    parser.add_argument('--polygon', default=False, action=argparse.BooleanOptionalAction, help='output polygon(not work this version)')
    parser.add_argument('--show', default=False, action=argparse.BooleanOptionalAction, help='show result on screen')
    parser.add_argument('--crnn_cfg', type=str, default='lib/config/360CC_config.yaml', help='crnn config file path')
    parser.add_argument('--crnn_checkpoint', type=str, default=r'weights/crnn_checkpoint_123_acc_0.9812_0.000004900.pth',
    # parser.add_argument('--crnn_checkpoint', type=str, default=r'C:\develop\CRNN_Chinese_Characters_Rec\output\360CC\checkpoints\checkpoint_123_acc_0.9812_0.000004900.pth',                       
                        help='crnn weight path')
    parser.add_argument('--crnn_output_folder', type=str, default='work/output/ocr_result', 
                        help='redirect the path for crnn output, this is ocr result. default is work/output/ocr_result')
    parser.add_argument('--crnn_mode', type=str, default=None, 
                        help='None or \'inference\'(default) if inference mode then enable crnn_output_folder parameter and disable purge_debug_folder')
    parser.add_argument('--device', type=str, default='cpu', help='\'cuda:0\' or \'cuda\' or \'cpu\'(default)')  
    parser.add_argument('--ocrdebug', default=False, action=argparse.BooleanOptionalAction, help='debug mode True, this will enable purge image and log file in work/output')
    parser.add_argument('--crnn_return', default=False, action=argparse.BooleanOptionalAction, help='return result mode True')
    parser.add_argument('--log_level', default='info', help='log level \'debug\' or \'info\'(default)')
    parser.add_argument('--while_mode', default=False, action=argparse.BooleanOptionalAction, help='batch mode waiting data in a loop')
    ### 2024-06-27 因整合windows服務console log會發生錯誤, 透過參數控制是否整合服務 
    parser.add_argument('--integrate_svc', default=False, action=argparse.BooleanOptionalAction, help='integrate windows service')
    
    args = parser.parse_args()
    return args

args = init_args()
device = args.device
hostname = socket.gethostname()

import config.ocrdebug as ocrdebug
if args.ocrdebug:
    ocrdebug.ocrdebug = True
    print('please use -h to see help message.')
    print('ocrdebug.ocrdebug =', ocrdebug.ocrdebug)

# start build drcmodel
drc_config = anyconfig.load(open(args.drc_config_file, 'rb'))
from drcmodels import build_model as build_drcmodel
drc_config['arch']['backbone']['in_channels'] = 3
drcmodel = build_drcmodel(drc_config['arch'])
drc_checkpoint = torch.load(args.drc_checkpoint_path, map_location=device)
if ocrdebug.ocrdebug:
    print('loading drcmodel pretrained model from {0}'.format(args.drc_checkpoint_path))
    # log.logger.debug('loading drcmodel pretrained model from {0}'.format(args.drc_checkpoint_path))
drcmodel.load_state_dict(drc_checkpoint)
drcmodel = drcmodel.to(device)
drcmodel.eval()

# start build dbnmodel if cpu then gpu_id=None
dbnmodel = Dbn_model(args.dbn_model_path, post_p_thre=args.thre, gpu_id=None)
if ocrdebug.ocrdebug:
    print('loading dbnmodel pretrained model from {0}'.format(args.dbn_model_path))

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
    print('loading crnn_model pretrained model from {0}'.format(args.crnn_checkpoint))
crnn_checkpoint = torch.load(args.crnn_checkpoint, map_location=device)
crnn_model.load_state_dict(crnn_checkpoint['state_dict'])
crnn_model.eval()
converter = strLabelConverter(crnn_config.DATASET.ALPHABETS)

from utils.ocr_util import crnn_batch_recognition #, crnn_recognition

from utils.ocr_util import cv2imread, rotate_image_up, cut_boxes, rotate_cut_boxes, img_padding
from utils.ocr_util import padding_boxes, purge_debug_folder, combine_box, sort_boxes
from utils.util import show_img, draw_bbox, save_result, get_file_list, new_draw_bbox

import matplotlib.pyplot as plt
import pathlib

### 2024-06-27 因整合windows服務console log會發生錯誤, 透過參數控制是否整合服務 
integrate_svc = False
if args.integrate_svc:
   integrate_svc = args.integrate_svc


def main():
    tStart = time.time() # 計時開始
    count = 0
    for img_path in (get_file_list(args.drc_input_folder, p_postfix=['.jpg','.JPG'])):  # args.drc_input_folder
        img_path = pathlib.Path(img_path)
        log.logger.debug('now processing image: {}'.format(img_path))
        img = cv2imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgrotate = img.copy()

        if args.drc_flag:
            log.logger.debug('drc_falg is True, drcmodel enable.')
            img224 = img_padding(img)
            img224 = cv2.resize(img224, (224, 224))
            if ocrdebug.ocrdebug:
                im = Image.fromarray(img224)
                im.save('work/output/post_img/{}_224.jpg'.format(img_path.stem), quality=100, format='JPEG')

            img224 = torch.from_numpy(img224/255).permute(2, 0, 1).unsqueeze(0).float().to(device)
            # 做文件方向的預測與轉向處理
            drcmodel.eval()  
            drc_label  = drcmodel(img224).argmax(dim=1).cpu().numpy()[0]
            log.logger.debug('file: {}, label: {}'.format(img_path, drc_label))
            imgrotate = rotate_image_up(img, drc_label)
            if ocrdebug.ocrdebug:
                # print('rotate image: ', drc_label)
                if drc_label != 0:
                    # print('rotate image: ', drc_label)
                    log.logger.debug('rotate image by drc_label: {}'.format(drc_label))
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
            if args.polygon:
                args.polygon=False # disable polygon or error

            # save_result(output_path.replace('_result.jpg', '.txt'), boxes_list, score_list, args.polygon)  # 儲存文字框座標 .txt 暫關掉

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
            log.logger.debug('ocr result: {} len {}'.format(img_path, len(pure_data)))
            pass
        if args.crnn_return:
            print(img_path.stem, pure_data)
        count += 1
    tEnd = time.time() # 計時結束
    log.logger.info('job finished. page count {}, cost time {:.2f} sec, tot runtime {:.2f}'.format(count, (tEnd - tStart), (tEnd - timeinit)))
    pass

if __name__ == '__main__':
    ### 2024-06-27 因整合windows服務console log會發生錯誤, 透過參數控制是否整合服務 
    log = Logger(filename='work/log/pythonOCR.log', integrate_svc=integrate_svc, level=args.log_level)
    log.logger.info('HCH OCR server start on {}'.format(hostname))
    log.logger.info('pytorch device: {}'.format(device))
    log.logger.info('ocrdebug.ocrdebug: {}'.format(ocrdebug.ocrdebug))
    log.logger.info('crnn_mode: {}'.format(args.crnn_mode))
    log.logger.info('integrate_svc: {}'.format(args.integrate_svc))
    log.logger.debug('all args: {}'.format(args))
    if args.crnn_mode != 'inference' and args.ocrdebug:
        purge_debug_folder()
        log.logger.info('purge_debug_folder: purge and mkdir work/output/cut_boxes, rotate_boxes, padding_boxes, post_img, inference, ocr_result')
    if ocrdebug.ocrdebug:
        log.logger.debug('loading drcmodel pretrained model from {0}'.format(args.drc_checkpoint_path))
        log.logger.debug('loading dbnmodel pretrained model from {0}'.format(args.dbn_model_path))
        log.logger.debug('loading crnn_model pretrained model from {0}'.format(args.crnn_checkpoint))
    
    os.makedirs('work/wait', exist_ok=True)
    os.makedirs('work/wait_hist', exist_ok=True)
    os.makedirs('work/wait_data', exist_ok=True)
    path_txt = r'.\work\wait'
    path_to = r'.\work\wait_hist'
    path_data = r'.\work\wait_data'
    init_drc_flag = args.drc_flag

    licenseVerify = "VerifyKey.pyc"
    if not os.path.exists(licenseVerify):
        print("LICENSE NOT FOUND!")
        sys.exit()
    output = subprocess.call(['python', licenseVerify], shell=True, text=True)
    if output == 1:
         print("LICENSE INVALID!")
         sys.exit(1)
    print("LICENSE VALID!")
    

    try:
        if not args.while_mode:
            try:
                main()
            except:
                log.logger.exception("Catch an main() exception.", exc_info=True)
        else:
            loop_count = 0
            while True:
                if args.drc_flag != init_drc_flag:
                    args.drc_flag = init_drc_flag # reset drc_flag
                    log.logger.debug('reset drc_flag args.drc_flag={}'.format(args.drc_flag))
                else:
                    pass
                files, sorted_files = [], []
                files = os.listdir(path_txt)
                files = [f for f in files if f.endswith('.txt')]

                if len(files) == 0:
                    log.logger.debug('no file found in folder: {}'.format(r'.\work\wait'))
                    time.sleep(1)
                    loop_count += 1
                    if loop_count >= 59:
                        log.logger.info('wait folder is empty, still wait reset count.')
                        loop_count = 0
                    continue
                else:
                    sorted_files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path_txt, x)))
                    log.logger.info('now process file: {}'.format(sorted_files[0]))
                    # os.replace(os.path.join(path, sorted_files[0]), os.path.join(path_to, sorted_files[0]))  # move behind main() process
                    check_txt = open(os.path.join(path_txt, sorted_files[0]), 'r')
                    lines = check_txt.readlines()
                    for line in lines:
                        log.logger.debug('check {} line: {}'.format(os.path.join(path_txt, sorted_files[0]), line))
                        if len(line.strip().split("=")[0]) == 0: # nothing in txt file
                            pass
                        elif line.strip().split("=")[0] == 'drc_flag':
                            args.drc_flag = line.strip().split("=")[1]
                            log.logger.debug('change drc_flag: {}'.format(args.drc_flag))
                        else:
                            pass
                    check_txt.close()

                    if os.path.isdir(os.path.join(path_data, sorted_files[0].split('.txt')[0])):  
                        # target file and folder exist check ok, chang args.drc_input_folder and args.crnn_output_folder
                        args.drc_input_folder = os.path.join(path_data, sorted_files[0].split('.txt')[0])
                        args.crnn_output_folder = args.drc_input_folder
                        log.logger.info('change processing folder: {}'.format(args.drc_input_folder))
                        try:
                            log.logger.info('start main process ... ')
                            main()
                            log.logger.info('start main process DONE!')
                        except:
                            log.logger.exception("Catch an main() exception.", exc_info=True)    
                    
                    else:
                        pass
                    
                    os.replace(os.path.join(path_txt, sorted_files[0]), os.path.join(path_to, sorted_files[0]))  # process done, move to hist folder
    except:
            log.logger.exception("Catch an exception.", exc_info=True)
