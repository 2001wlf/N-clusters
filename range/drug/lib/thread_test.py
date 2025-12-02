import random
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED,as_completed
import threading
import time


def work(n):
    global id
    # print(id)
    # if n > 2:
    #     all_task.append(t.submit(work, n-1))
    with threading.Lock():
        tree.append(id)
        g.append(0)
        v.append(0)
        id += 1
    return random.randint(0, n)

def work_(n):
    global id
    # print(id)
    stack = [n]
    while stack:
        n = stack.pop(0)
        if n > 2:
            stack.append(n-1)
        tree.append(id)
        g.append(0)
        v.append(0)
        id += 1


if __name__ == '__main__':
    # global tree
    # global g
    # global v
    # global id
    # global threads
    tree = []
    g = []
    v = []
    id = 0
    threads = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers = 100000) as t:
        all_task = [t.submit(work, 100000)]
        for future in as_completed(all_task):
            data = future.result()
            print(data)
            if data > 2:
                all_task.append(t.submit(work, data))
    end_time = time.time()
    print("===========================")
    print(end_time-start_time)
    print(tree)

    tree = []
    g = []
    v = []
    id = 0
    start_time = time.time()
    work_(100000)
    end_time = time.time()
    print("===========================")
    print(end_time - start_time)
    # print(tree)

