# # doom_env/features/detectors.py
# from __future__ import annotations
# from typing import Any, Dict, List, Optional
# import numpy as np
# from doom_env.features.common import norm_name, is_ignored, YOLO_ID_TO_NAME

# class SimpleYOLO:
#     def __init__(self, model_path: str, device: Optional[int] = None, conf: float = 0.25):
#         from ultralytics import YOLO
#         self.model = YOLO(model_path)
#         self.device = device
#         self.conf = conf

#     def predict(self, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
#         results = self.model.predict(source=img_rgb, verbose=False, conf=self.conf, device=self.device)
#         r = results[0]
#         dets: List[Dict[str, Any]] = []
#         if r.boxes is None:
#             return dets
#         xyxy = r.boxes.xyxy.cpu().numpy()
#         cls  = r.boxes.cls.cpu().numpy().astype(int)
#         conf = r.boxes.conf.cpu().numpy()
#         for (x1,y1,x2,y2), c, p in zip(xyxy, cls, conf):
#             dets.append({
#                 "x": float(x1), "y": float(y1),
#                 "w": float(x2-x1), "h": float(y2-y1),
#                 "cls_id": int(c),
#                 "name": norm_name(YOLO_ID_TO_NAME.get(int(c), f"class_{int(c)}")),
#                 "conf": float(p),
#             })
#         return dets

# def get_detections_from_state(st: Any, screen_rgb: np.ndarray, use_labels: bool, yolo: Optional[SimpleYOLO]):
#     dets: List[Dict[str, Any]] = []
#     if use_labels:
#         if getattr(st, "labels", None) is None:
#             return dets
#         for lab in st.labels:
#             name = norm_name(getattr(lab, "object_name", "Unknown"))
#             if is_ignored(name):
#                 continue
#             dets.append({
#                 "x": float(lab.x), "y": float(lab.y),
#                 "w": float(lab.width), "h": float(lab.height),
#                 "name": name, "conf": 1.0,
#             })
#         return dets

#     # YOLO
#     assert yolo is not None
#     dets = yolo.predict(screen_rgb)
#     # ya viene normalizado
#     return dets


# doom_env/features/detectors.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from doom_env.features.common import norm_name, is_ignored, YOLO_ID_TO_NAME

class SimpleYOLO:
    def __init__(self, model_path: str, device: Optional[int] = None, conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def predict(self, img_rgb: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(source=img_rgb, verbose=False, conf=self.conf, device=self.device)
        r = results[0]
        dets: List[Dict[str, Any]] = []
        if r.boxes is None:
            return dets
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls  = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        for (x1,y1,x2,y2), c, p in zip(xyxy, cls, conf):
            dets.append({
                "x": float(x1), "y": float(y1),
                "w": float(x2-x1), "h": float(y2-y1),
                "cls_id": int(c),
                "name": norm_name(YOLO_ID_TO_NAME.get(int(c), f"class_{int(c)}")),
                "conf": float(p),
            })
        return dets

def get_detections_from_state(st: Any, screen_rgb: np.ndarray, use_labels: bool, yolo: Optional[SimpleYOLO]):
    dets: List[Dict[str, Any]] = []
    if use_labels:
        if getattr(st, "labels", None) is None:
            return dets
        for lab in st.labels:
            name = norm_name(getattr(lab, "object_name", "Unknown"))
            if is_ignored(name):
                continue
            dets.append({
                "x": float(lab.x), "y": float(lab.y),
                "w": float(lab.width), "h": float(lab.height),
                "name": name, "conf": 1.0,
            })
        return dets

    # YOLO
    assert yolo is not None
    dets = yolo.predict(screen_rgb)
    # ya viene normalizado
    return dets