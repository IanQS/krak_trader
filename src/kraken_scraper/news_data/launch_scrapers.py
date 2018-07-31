import time

from .registered import _registered
import multiprocessing

if __name__ == '__main__':
    multiprocessing.
    for news_src in _registered:
        p = multiprocessing.Process(news_src)
        p.daemon = True  # Daemon so if we force-quit top-level all die
        p.start()

    # Keep top-level connection alive so all children stay alive
    while True:
        time.sleep(0.1)