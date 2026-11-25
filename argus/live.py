"""Live CCTV/IP stream utilities. Stream URLs are supplied by the operator."""
import cv2

def open_camera_source(source):
    """Open an integer webcam index or RTSP/HTTP stream URL."""
    if isinstance(source,str) and source.strip().isdigit(): source=int(source.strip())
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera source: {source}")
    return cap
