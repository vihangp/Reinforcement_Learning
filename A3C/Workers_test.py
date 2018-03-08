import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] (%(threadName)-10s) %(message)s')

def Worker(num):
    logging.debug('Starting')
    time.sleep(1)
    print(num)
    logging.debug('Exiting')
    return


threads = []
for i in range(4):
    t = threading.Thread(name="Worker_{}".format(i),target=Worker, args=(i,))
    threads.append(t)
    t.start()

