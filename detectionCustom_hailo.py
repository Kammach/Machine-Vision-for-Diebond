import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"

import gi, hailo
gi.require_version("Gst", "1.0")

from hailo_apps.python.pipeline_apps.detection.detection_pipeline import GStreamerDetectionApp
from hailo_apps.python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class


def app_callback(element, buffer, user_data):
    if not buffer or not user_data.use_frame:
        return

    fmt, w, h = get_caps_from_pad(element.get_static_pad("src"))

    if all((fmt, w, h)):
        frame = get_numpy_from_buffer(buffer, fmt, w, h)


def main():
    GStreamerDetectionApp(
        app_callback,
        app_callback_class()
    ).run()


if __name__ == "__main__":
    main()
