## Sparse4D
To enable training in AMD ROCm platform, small modifications are needed:

1. Prepare data
2. Compile the deformable_aggregation CUDA op:
````
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
````
3. update motmetrics
````
pip install motmetrics==1.1.3

将 /usr/local/lib/python3.10/dist-packages/motmetrics/metrics.py 中
from collections import OrderedDict, Iterable
修改为
from collections import OrderedDict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

````
4. pip install pandas==1.1.5
5. Download pre-trained weights and Training
````
bash local_train.sh sparse4dv3_temporal_r50_1x8_bs6_256x704
````

We have also provided an [example](https://github.com/binding7012/Sparse4D) to provide out-of-the-box experience.
