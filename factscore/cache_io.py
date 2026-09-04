"""Atomic file writes for the on-disk caches.

Every cache in factscore (the prompt pickles, the retrieval json and the
embedding pickle) is one file that is rewritten in full on each save. Writing
it in place with open(path, "w") leaves an unreadable file in two situations:
a run killed mid-dump truncates it, and two runs saving at once interleave
their bytes - the shorter write overwrites the longer one's prefix and leaves
its tail dangling, which json rejects as "Extra data". Writing to a temporary
sibling and renaming it into place makes a save all-or-nothing: a reader sees
either the previous file or the complete new one, never a mix.
"""

import os
import threading


def atomic_write(path, write, mode="wb"):
    """Call write(file) on a temporary file next to `path`, then rename it into place.

    `mode` is the mode the temporary file is opened with ("wb" for pickle, "w"
    for json). The temporary file is removed again if write() raises.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # unique per process and thread, so concurrent savers never share a temp file
    tmp_path = f"{path}.tmp{os.getpid()}-{threading.get_ident()}"
    try:
        with open(tmp_path, mode) as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
