import re


class CatalPhoto(object):
    """Represents a photo in the Catalhoyuk archive."""

    # Regex for extracting record ID from a URL
    record_id_re = re.compile(r'(original|preview)=(\d+)', flags=re.IGNORECASE)

    def __init__(self, url, annotation=None):
        self.url = str(url)
        self.record_id = int(self.record_id_re.search(url).group(2))

        if annotation is not None:
            self.is_labeled = True
            self.has_whiteboard = annotation.lower().startswith('y')
            self.is_difficult = annotation != 'y' and annotation != 'n'
        else:
            self.is_labeled = False
