from multiprocessing import Pool, Queue, Process, Manager
import random
import signal
import time
num_books = 10

def writer(i, q):
    # Imitate CPU-bound work happening in writer
    delay = random.randint(1,10)
    time.sleep(delay)
    
    # Put the result into the queue
    t = time.time()
    print(f"I am writer {i}: {t}")
    q.put(t)

def init_worker():
    """
    Pool worker initializer for keyboard interrupt on Windows
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
def reader(i, q):
    """
    Queue reader worker
    """
    
    # Read the top message from the queue
    message = q.get()
    
    # Imitate CPU-bound work happening in reader
    time.sleep(3)
    print(f"I am reader {i}: {message}")
    
if __name__ == "__main__":
    # Create manager
    m = Manager()
    
    # Create multiprocessing queue
    q = m.Queue()    # Create a group of parallel writers and start them
    for i in range(num_writers):
        Process(target=writer, args=(i,q,)).start()
    
    # Create multiprocessing pool
    #p = Pool(num_readers, init_worker)    # Create a group of parallel readers and start them
    with Pool() as p:
        p.map(writer, range(num_books))
        
    # Number of readers is matching the number of writers
    # However, the number of simultaneously running
    # readers is constrained to the pool size
    readers = []
    for i in range(10):
        readers.append(p.apply_async(reader, (i,q,)))
    
    # Wait for the asynchrounous reader threads to finish
    try:
        [r.get() for r in readers]
    except:
        print("Interrupted")
        p.terminate()
        p.join()
        
        