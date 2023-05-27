import os 
import sys
import time 
import random 
import threading
import itertools as it 
t_list = ['electronics', 'home', 'beauty', 'fashion']
# source_list = ['_'.join(sorted(x)) for x in it.combinations(t_list, 2)] + ['_'.join(sorted(x)) for x in it.combinations(t_list, 3)]
source_list = ['_'.join(sorted(x)) for x in it.combinations(t_list, 4)]

target_list = ['book', 'grocery', 'pet', 'toy']

class Param:
    def __init__(self, model_name, source,):
        self.model_name = model_name 
        self.source = source 
    

class myThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        print(f'bash scripts/cross-domain/sub/{self.threadID}.sh')
        os.system(f'bash scripts/cross-domain/sub/{self.threadID}.sh')
        

def main():
    param_list = []
    for source in source_list:
        for model_name in range(5): 
            param = Param(model_name=model_name, source=source)
            param_list.append(param)
    num_params = len(param_list)
    random.seed(0)
    param_list = random.sample(param_list, num_params)
    num_batch = int(sys.argv[1])
    num_device = 8
    batch_size = num_params // num_batch
    os.system('rm -r ./scripts/cross-domain/sub')
    os.makedirs('./scripts/cross-domain/sub', exist_ok=True)
    for i, p in enumerate(param_list):
        f =  open(f'./scripts/cross-domain/sub/{i % num_batch}.sh', 'a') 
        f.write(f'bash scripts/cross-domain/multi_base.sh {p.source} {p.model_name} {i % num_device}\n')
        f.close()
    # thread_list = []
    # worker = int(sys.argv[2])
    # for i in range(num_device):
    #     thread = myThread(i + num_device * worker)
    #     thread.start()
    #     thread_list.append(thread)
    #     time.sleep(2)
    # for t in thread_list:
    #     t.join()



main()