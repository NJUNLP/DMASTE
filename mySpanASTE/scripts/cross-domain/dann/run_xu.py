import os 
import sys
import time 
import random 
import threading
source_list = ['14res', '15res', '16res', '14lap', '14lap', '14lap']
target_list = ['14lap', '14lap', '14lap', '14res', '15res', '16res']


class Param:
    def __init__(self, model_name, source, target, ad_steps):
        self.model_name = model_name 
        self.source = source 
        self.target = target 
        self.ad_steps = ad_steps 
    

class myThread(threading.Thread):
    def __init__(self, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
    def run(self):
        os.system(f'bash scripts/cross-domain/dann/sub/{self.threadID}.sh')
        print(f'bash scripts/cross-domain/dann/sub/{self.threadID}.sh')
        

def main():
    param_list = []
    for source, target in zip(source_list, target_list):
        for model_name in range(5): 
            for ad_steps in [1, 3, 5, 7, 10, 15, 20, 30, 50, 100]:
                param = Param(model_name=model_name, source=source, target=target, ad_steps=ad_steps)
                param_list.append(param)
    num_params = len(param_list)
    random.seed(0)
    param_list = random.sample(param_list, num_params)
    num_batch = int(sys.argv[1])
    num_device = 8
    batch_size = num_params // num_batch
    os.system('rm -r ./scripts/cross-domain/dann/sub')
    os.makedirs('./scripts/cross-domain/dann/sub', exist_ok=True)
    for i, p in enumerate(param_list):
        f =  open(f'./scripts/cross-domain/dann/sub/{i % num_batch}.sh', 'a') 
        f.write(f'bash scripts/cross-domain/dann/maste.sh {p.source} {p.target} {p.ad_steps} {p.model_name} {i % num_device}\n')
        f.close()
    thread_list = []
    worker = int(sys.argv[2])
    for i in range(num_device):
        thread = myThread(i + num_device * worker)
        thread.start()
        thread_list.append(thread)
        time.sleep(2)
    for t in thread_list:
        t.join()



main()