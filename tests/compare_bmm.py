import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity

points = torch.randn(4,6,88,16,44,3,1,device='cuda',dtype=torch.float32)
post_rots = torch.randn(4,6,1,1,1,3,3,device='cuda',dtype=torch.float32)

a,b,c,d,e,f,g = points.shape

for i in range(5): #warmup
    output = post_rots.matmul(points)
torch.cuda.synchronize()
t1 = time.time()
output = post_rots.matmul(points)
torch.cuda.synchronize()
print('forward time(s/iter): ',time.time()-t1)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    output = post_rots.matmul(points)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
output_tmp = output

## reshape to -1 dim to accelerate bmm
for i in range(5): #warmup
    #output = post_rots.matmul(points)
    output = post_rots.reshape(-1,3,3).matmul(points.view(a*b,c*d*e,f).permute(0,2,1))
    output = output.permute(0,2,1).view(a,b,c,d,e,f,g)
torch.cuda.synchronize()
t1 = time.time()
#output = post_rots.matmul(points)
output = post_rots.reshape(-1,3,3).matmul(points.view(a*b,c*d*e,f).permute(0,2,1))
output = output.permute(0,2,1).view(a,b,c,d,e,f,g)
torch.cuda.synchronize()
print('forward time(s/iter): ',time.time()-t1)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    #output = post_rots.matmul(points)
    output = post_rots.reshape(-1,3,3).matmul(points.view(a*b,c*d*e,f).permute(0,2,1))
    output = output.permute(0,2,1).view(a,b,c,d,e,f,g)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print(f'{torch.abs(output-output_tmp).sum() = }')
max_diff = (output - output_tmp).abs().max().item()
print(f"[max abs diff] = {max_diff:.6f}")
