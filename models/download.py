from paddleocr import PaddleOCR

# 下载检测模型
det_model = PaddleOCR(det_model_dir='./en_PP-OCRv4_det_infer', lang='en', use_angle_cls=False, ocr_version='PP-OCRv4',rec_algorithm='SVTR_LCNet',    det_db_box_thresh=0.5,
    drop_score=0.2)
# 下载识别模型
rec_model = PaddleOCR(rec_model_dir='./en_PP-OCRv4_rec_infer', lang='en', use_angle_cls=False, ocr_version='PP-OCRv4',rec_algorithm='SVTR_LCNet'    det_db_box_thresh=0.5,
    drop_score=0.2)