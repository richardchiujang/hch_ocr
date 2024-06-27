# basic usage
please run command " python hch_ocr.py -h " for agrs help.

example:
python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result"
(說明) 指定輸入資料夾 輸出資料夾 就會輸出結果(txt)到指定輸出資料夾(會建立資料夾,但不會清空資料夾)
(說明) 這是建議的使用法，給一個資料夾路徑提供要處理的圖檔(JPG)，並給一個相同資料夾下的輸出資料夾名稱輸出結果，結果為圖檔相同名稱的文字檔(.txt)

(說明) --drc_input_folder 指定輸入資料夾
(說明) --crnn_output_folder 需同時搭配 --crnn_mode "inference"(預設) 指定輸出資料夾會自動建立最後一層資料夾
(說明) --crnn_mode "inference"(預設) 會從指定輸出路徑建立最後一層資料夾，但只有 --crnn_mode 不等於 "inference"(預設)時，才會清空每個work/output/... 資料夾內容
(說明) --ocrdebug, (--no-ocrdebug) 會產生相關結果到work/output/...各資料夾中(會先清空每個資料夾內容，--crnn_mode 不等於 "inference"時)
(說明) --crnn_return, (--no-crnn_return) "True" 回傳 OCR 結果在 console 中顯示
(說明) --drc_flag, (--no-drc_flag) "預設 flase 不做文件方向辨識"

release version : v0.5 2024/05/09
(說明) --while_mode, (--no-while_mode) 時會loop執行，並且變更輸入輸出資料夾位置與邏輯
(說明) 輸入 請先產生放資料夾的名稱於 ./work/wait_data 下，如 ./work/wait_data/data_folder_001
(說明) 請再產生一個　.txt 文件於 ./work/wait 資料夾下，名稱為 data_folder_001.txt 不需要內容
(說明) 系統會將他搬到 ./work/wait_hist 下，然後執行 ./work/wait_data/data_folder_001 下的內容並輸出到此路徑下

example:
python hch_ocr.py 
(說明) 什麼參數都沒設定 就只會產生測試OCR結果(文字檔)於預設的 work/output/ocr_result 沒有 debug

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result" --ocrdebug "True"
(說明) 指定輸入資料夾 輸出資料夾 就會輸出結果(txt)到指定輸出資料夾(會建立資料夾,但不會清空資料夾) 
(說明) 同時其他debug輸出會到work/output/...各資料夾中(會先清空每個資料夾內容)

python hch_ocr.py --drc_input_folder "work/input_test" --ocrdebug
python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result" --ocrdebug
(說明) 指定輸入資料夾 同時 debug 就會讀取指定輸入資料夾內容 並將所有結果輸出到 work/output/...各資料夾中(會先清空每個資料夾內容)

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_mode None
(說明) 指定輸入資料夾 只會輸出ocr結果到 work/output/ocr_result 資料夾中(會先清空每個資料夾內容)
(說明) 沒有 debug
(說明) --crnn_mode 預設為 inference 要變更請給 None
(說明) --crnn_mode 為 inference (預設)時，才會建立指定輸出資料夾(--crnn_output_folder) 否則一律輸出在 work/output/ocr_result

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result" --crnn_mode None
(說明) 指定輸入資料夾與輸出資料夾但是--crnn_mode None 只會輸出ocr結果到 work/output/ocr_result 資料夾中(會先清空每個資料夾內容)
(說明) --crnn_mode 為 inference (預設)時，才會建立指定輸出資料夾(--crnn_output_folder) 否則一律輸出在 work/output/ocr_result

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_mode None --ocrdebug --crnn_return
(說明) 指定輸入資料夾，沒指定輸出資料夾，crnn_mode=None，輸出在 work/output/ocr_result
(說明) 同時其他debug輸出在work/output/...各資料夾中
(說明) --crnn_return, --no-crnn_return 回傳 OCR 結果在 console 中顯示

usage: hch_ocr.py [-h] [--drc_config_file DRC_CONFIG_FILE] [--drc_checkpoint_path DRC_CHECKPOINT_PATH] [--drc_flag | --no-drc_flag] [--drc_input_folder DRC_INPUT_FOLDER] [--dbn_model_path DBN_MODEL_PATH]
                  [--dbn_output_folder DBN_OUTPUT_FOLDER] [--thre THRE] [--polygon | --no-polygon] [--show | --no-show] [--crnn_cfg CRNN_CFG] [--crnn_checkpoint CRNN_CHECKPOINT]
                  [--crnn_output_folder CRNN_OUTPUT_FOLDER] [--crnn_mode CRNN_MODE] [--device DEVICE] [--ocrdebug | --no-ocrdebug] [--crnn_return | --no-crnn_return] [--log_level LOG_LEVEL]
                  [--while_mode | --no-while_mode]

demo example: python hch_ocr.py --drc_flag --ocrdebug --crnn_return --log_level debug 

demo example: python hch_ocr.py --no-drc_flag --ocrdebug --crnn_return --log_level debug 

options:
  -h, --help            show this help message and exit
  --drc_config_file DRC_CONFIG_FILE
                        drcmodel config file path
  --drc_checkpoint_path DRC_CHECKPOINT_PATH
                        drcmodel wehight file path
  --drc_flag, --no-drc_flag
                        document direction drcmodel enable or disable (default: False)
  --drc_input_folder DRC_INPUT_FOLDER
                        assign img folder path for input path default is work/input
  --dbn_model_path DBN_MODEL_PATH
                        dbnmodel wehight path
  --dbn_output_folder DBN_OUTPUT_FOLDER
                        img path for ocr result output default is work/output/inference
  --thre THRE           the threshould of dbn post_processing
  --polygon, --no-polygon
                        output polygon(not work this version) (default: False)
  --show, --no-show     show result on screen (default: False)
  --crnn_cfg CRNN_CFG   crnn config file path
  --crnn_checkpoint CRNN_CHECKPOINT
                        crnn weight path
  --crnn_output_folder CRNN_OUTPUT_FOLDER
                        redirect the path for crnn output, this is ocr result. default is work/output/ocr_result
  --crnn_mode CRNN_MODE
                        None or 'inference'(default) if inference mode then enable crnn_output_folder parameter and disable purge_debug_folder
  --device DEVICE       'cuda:0' or 'cuda' or 'cpu'(default)
  --ocrdebug, --no-ocrdebug
                        debug mode True, this will enable purge image and log file in work/output (default: True)
  --crnn_return, --no-crnn_return
                        return result mode True (default: True)
  --log_level LOG_LEVEL
                        log level 'debug' or 'info'(default)
  --while_mode, --no-while_mode
                        batch mode waiting data in a loop (default: False)

Q&A:
drc_input_folder 
parser.add_argument('--drc_input_folder', default='work/input', type=str, help='img folder path for inference')
如果你執行時下 --while_mode 程式邏輯會將 drc_input_folder & crnn_output_folder 替換掉，所以你可以不用加(除非，你是使用 --no-while_mode 的舊邏輯(舊邏輯也是可以用的))

crnn_mode 參數有需要加嗎? 答案: 要加
--crnn_mode 的作用有二: 1."抑制"程式啟動時去處理(刪除) ./work/output 裡面的內容 2.使 args.crnn_output_folder 參數生效，所以要加


release note:
release version : v0.2 2023/09/12 
finetune dbnet, crnn
add arg "--drc_flag" for disable drcmodel
adjust crnn threshold to <= -0.5 

release version : v0.3 2023/09/14
debug ocr_util.py for reslove one single bbox only doc_image ocr issue, string be process as list because function expect a list

release version : v0.4 2024/04/24
debug hch_ocr.py for # -*- coding: UTF-8 -*- 造成console中文輸出亂碼(改英文)
debug 修改文字框後調參數(加大範圍變小)
update crnn weight file to checkpoint_88_acc_0.9387_0.000006887.pth

release version : v0.5 2024/05/09
add add_argument('--while_mode', default=False, help='while mode for loop program for waiting data True or False(default)')
add os.makedirs for work/wait wait_hist wait_data
add while loop
add main() function for main ocr process to prevent duplicate codes 
add target file and folder check logic

release version : v5.1 2024/05/17
change move txt step from pre-process to folder images post-process
add inside txt indicate wait/file drc_flag=True/False chang flag for temponary chang drc mode 

release version : v5.2 2024/06/13
fix bug boolean argument like --while_mode, --no-while_mode
adj bbox margin value ( ocr_util.py )

