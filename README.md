# basic usage
please run command " python hch_ocr.py -h " for agrs help.

example:
python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result"
(說明) 指定輸入資料夾 輸出資料夾 就會輸出結果(txt)到指定輸出資料夾(會建立資料夾,但不會清空資料夾)
(說明) 這是建議的使用法，給一個資料夾路徑提供要處理的圖檔(JPG)，並給一個相同資料夾下的輸出資料夾名稱輸出結果，結果為圖檔相同名稱的文字檔(.txt)

(說明) --drc_input_folder 指定輸入資料夾
(說明) --crnn_output_folder 需同時搭配 --crnn_mode "inference"(預設) 指定輸出資料夾會自動建立最後一層資料夾
(說明) --crnn_mode "inference"(預設) 會從指定輸出路徑建立最後一層資料夾，但只有 --crnn_mode 不等於 "inference"(預設)時，才會清空每個work/output/... 資料夾內容
(說明) --ocrdebug "True" 會產生相關結果到work/output/...各資料夾中(會先清空每個資料夾內容，--crnn_mode 不等於 "inference"時)
(說明) --crnn_return "True" 回傳 OCR 結果在 console 中顯示
(說明) --drc_flag "預設 flase 不做文件方向辨識"

release version : v0.5 2024/05/07
(說明) --while_mode "True" 時會loop執行，並且變更輸入輸出資料夾位置與邏輯
(說明) 輸入 請先產生放資料夾的名稱於 ./work/wait_data 下，如 ./work/wait_data/data_folder_001
(說明) 請再產生一個　.txt 文件於 ./work/wait 資料夾下，名稱為 data_folder_001.txt 不需要內容
(說明) 系統會將他搬到 ./work/wait_hist 下，然後執行 ./work/wait_data/data_folder_001 下的內容並輸出到此路徑下

example:
python hch_ocr.py 
(說明) 什麼參數都沒設定 就只會產生測試OCR結果(文字檔)於預設的 work/output/ocr_result 沒有 debug

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result" --ocrdebug "True"
(說明) 指定輸入資料夾 輸出資料夾 就會輸出結果(txt)到指定輸出資料夾(會建立資料夾,但不會清空資料夾) 
(說明) 同時其他debug輸出會到work/output/...各資料夾中(會先清空每個資料夾內容)

python hch_ocr.py --drc_input_folder "work/input_test" --ocrdebug "True"
python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result" --ocrdebug "True"
(說明) 指定輸入資料夾 同時 debug 就會讀取指定輸入資料夾內容 並將所有結果輸出到 work/output/...各資料夾中(會先清空每個資料夾內容)

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_mode None
(說明) 指定輸入資料夾 只會輸出ocr結果到 work/output/ocr_result 資料夾中(會先清空每個資料夾內容)
(說明) 沒有 debug
(說明) --crnn_mode 預設為 inference 要變更請給 None
(說明) --crnn_mode 為 inference (預設)時，才會建立指定輸出資料夾(--crnn_output_folder) 否則一律輸出在 work/output/ocr_result

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_output_folder "work/input_test/ocr_result" --crnn_mode None
(說明) 指定輸入資料夾與輸出資料夾但是--crnn_mode None 只會輸出ocr結果到 work/output/ocr_result 資料夾中(會先清空每個資料夾內容)
(說明) --crnn_mode 為 inference (預設)時，才會建立指定輸出資料夾(--crnn_output_folder) 否則一律輸出在 work/output/ocr_result

python hch_ocr.py --drc_input_folder "work/input_test" --crnn_mode None --ocrdebug "True" --crnn_return "True"
(說明) 指定輸入資料夾，沒指定輸出資料夾，crnn_mode=None，輸出在 work/output/ocr_result
(說明) 同時其他debug輸出在work/output/...各資料夾中
(說明) --crnn_return 回傳 OCR 結果在 console 中顯示


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

release version : v0.5 2024/05/08
add add_argument('--while_mode', default=True, help='while mode for loop program for waiting data True or False(default)')
add os.makedirs for work/wait wait_hist wait_data
add while loop
add main() function for main ocr process to prevent duplicate codes 
add target file and folder check logic


