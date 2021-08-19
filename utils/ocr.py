r'''
Author       : PiKaChu_wcg
Date         : 2021-08-19 05:59:55
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-19 06:20:42
FilePath     : \ifly_pic\utils\ocr.py
'''
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False,
                rec_model_dir='./inference/ch_ppocr_server_v2.0_rec_infer/',
                cls_model_dir='./inference/ch_ppocr_mobile_v2.0_cls_infer/',
                det_model_dir='./inference/ch_ppocr_server_v2.0_det_infer/')  # need to run only once to download and load model into memory
img_path = './images/9.png'
result = ocr.ocr(img_path, cls=True)
result="".jion(result)
