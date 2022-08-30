import argparse, cv2, torch
import numpy as np

from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel
from http.server import BaseHTTPRequestHandler, HTTPServer
from traceback import print_exc
from typing import Tuple

# python detect_server.py --weights yolov7.pt --conf 0.25 --img-size 640

class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        if (size <= 0):
            self.send_response(400, "Empty Request")
        try:
            data = self.rfile.read(size)
            data = inference(data)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except:
            print_exc(limit=1)
            self.send_error(500)

class LoadSingleImages:  # for inference
    def __init__(self, buffer, img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        self.nf = 1 # number of files
        self.buffer = np.asarray(bytearray(buffer), dtype="uint8")

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration

        # Read image
        self.count += 1
        img0 = cv2.imdecode(self.buffer, cv2.IMREAD_COLOR)  # BGR
        assert img0 is not None, 'Not Image'

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return "", img, img0, None

    def __len__(self):
        return self.nf  # number of files

model = None
device = None
half = False
stride = 0
imgsz = 0
names = []
colors = []
old_img_w = 0
old_img_h = 0
old_img_b = 0

def init(opt):
    global model, device, half, stride, imgsz, names, colors
    global old_img_b, old_img_w, old_img_h
    weights, imgsz, trace = opt.weights, opt.img_size, not opt.no_trace

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

def inference(image_buffer) -> bytes:
    """ return .png format image data """
    global old_img_b, old_img_w, old_img_h

    # Set Dataloader
    # dataset = LoadImages(image_path, img_size=imgsz, stride=stride)
    dataset = LoadSingleImages(image_buffer, img_size=imgsz, stride=stride)
    if len(dataset) <= 0:
        return b''

    iter(dataset)
    path, img, im0, vid_cap = next(iter(dataset))
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=opt.augment)[0]

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t3 = time_synchronized()

    # Print time (inference + NMS)
    print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

    # Process detections
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
 
        return cv2.imencode(".png", im0)[1].tobytes()
    return b''

def bind_address(address: str) -> Tuple[str, int]:
    address_and_port = address.split(":")
    if len(address_and_port) <= 0:
        return "127.0.0.1", 18999
    elif len(address_and_port) == 1:
        return address_and_port[0], 18999
    else:
        return address_and_port[0], int(address_and_port[1])

def run_server_forever(server_address=("127.0.0.1", 18999), server_class=HTTPServer, handler_class=RequestHandler):
    httpd = server_class(server_address, handler_class)
    local_host = "127.0.0.1" if server_address[0] == "0.0.0.0" else server_address[0]
    print(f"Server started at http://{local_host}:{server_address[1]}")
    httpd.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--bind', type=bind_address, default=("127.0.0.1", 18999), help='http bind address, default to 127.0.0.1:18999')
    opt = parser.parse_args()
    print("Arguments:")
    print(opt)

    with torch.no_grad():
        init(opt)

        # start server
        run_server_forever(opt.bind)
