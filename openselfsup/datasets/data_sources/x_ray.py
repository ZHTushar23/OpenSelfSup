from ..registry import DATASOURCES
from .image_list_multi_label import ImageListMultiLabel


@DATASOURCES.register_module
class XRay(ImageListMultiLabel):

    def __init__(self, root, list_file, memcached, mclient_path, return_label=True, *args, **kwargs):
        super(XRay, self).__init__(
            root, list_file, memcached, mclient_path, return_label)
